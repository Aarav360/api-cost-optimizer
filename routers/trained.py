"""The trained router, F (PLAN_v4.md §8 3b and §13; Amendment A11, items 3-5, 9 and 10).

A request goes in, router scores for `haiku`, `sonnet` and `opus_low` come out. The steps, in order:

1. Template lines are removed from the prompt (`strip_template_lines`); blank lines stay. If nothing else is left the
   empty string is embedded and the request is counted.
2. The text is tokenized with the model's fast tokenizer and no truncation of its own. Over the window (8,192 tokens,
   CLS and SEP included) it keeps the head and the tail of its content tokens, half each (`fit_window`), and is counted.
3. `gte-modernbert-base`, at commit `MODEL_COMMIT`, runs on the CPU in float32; the CLS vector is the embedding.
4. The embedding is length-normalized, standardized with the settings' mean and scale, and the schema flag (1 if the
   request has an output schema, else 0) is appended. The schema itself is never embedded.
5. Three logistic heads, in numpy, give the router scores. Nothing after the backend rounds back to float32: the
   embedding is upcast to float64 (exact for float32) and stays there.

`score(request)` is `score_with(fold(request), request)`: a hash of the request picks one of five fold models, so test
runs the policy dev's out-of-fold router scores certified (item 9). `score_with` lives here and not on the `Router`
protocol, which is frozen with the decision rule (Amendment A10, item 1).

**Loading** (`load_backend`) is offline, from a local snapshot of the pinned commit (`SNAPSHOT`), from
`model.safetensors` only, never the repository's pickle. The five files the router reads are checked against the
settings' sha256s first, and a mismatch is refused whole with `SnapshotError`. The router never fetches:
`python -m analysis.fetch_model` does that once. `TrainedRouter` loads this way unless a `backend` is passed in, which
is for checks and for Stage 5 · 5's float64 drift measurement (`TransformersBackend.upcast`), never for Stage 8. The
fake backend lives in `analysis/`, outside this module's import closure.

**The settings file** (`SETTINGS_FILE`, `routers/trained_settings.json`) is plain JSON, no pickles:

    {"model": {"repository": "Alibaba-NLP/gte-modernbert-base", "commit": "<40 hex>",
               "files": {"config.json": "<sha256>", "model.safetensors": ..., "tokenizer.json": ...,
                         "tokenizer_config.json": ..., "special_tokens_map.json": ...}},
     "window": 8192,                          # tokens, special tokens included
     "compute_precision": "float32",
     "attention_implementation": "sdpa",      # or "eager"
     "threads": 1,                            # torch.set_num_threads
     "reference_compile": false,
     "solver": {"name": "newton-cholesky", "tol": 1e-10, "max_iter": 100},   # the fit's, recorded here
     "template_lines": ["<line, whitespace-trimmed>", ...],
     "standardizer": {"mean": [d floats], "scale": [d floats]},   # of the length-normalized train embeddings
     "c": {"haiku": <float>, "sonnet": <float>, "opus_low": <float>}}         # each head's C, recorded here

**The fold-weights file** (`FOLD_WEIGHTS_FILE`, `routers/trained_fold_weights.json`) holds five fold models, each
with a head per config over the d standardized embedding values and then the flag, so d + 1 coefficients:

    {"folds": [{"haiku": {"coef": [d + 1 floats], "intercept": <float>},
                "sonnet": {...}, "opus_low": {...}}, ... five in all]}

F-train's weights are in a file of their own that this module never loads. Each file's reader refuses a missing or
unknown key, a wrong width and a number that isn't finite, whole (`SettingsError`).

This module imports numpy, PyTorch and `transformers`, never scikit-learn, `analysis/` or `harness/`; fitting lives in
`analysis/`, so the freeze covers exactly what Stage 8 runs (item 10).
"""

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from routers.ladder import SCORED
from routers.request import Request

MODEL_REPOSITORY = "Alibaba-NLP/gte-modernbert-base"
MODEL_COMMIT = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
# The five files the router reads (item 4). `modules.json` and `1_Pooling/config.json` are never read: the pooling is
# this module's own code, and `pytorch_model.bin`, a pickle, is never loaded.
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
FOLDS = 5

_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = _ROOT / ".cache" / "gte-modernbert-base" / MODEL_COMMIT  # filled once by `analysis.fetch_model`
SETTINGS_FILE = Path(__file__).with_name("trained_settings.json")
FOLD_WEIGHTS_FILE = Path(__file__).with_name("trained_fold_weights.json")

ATTENTION_IMPLEMENTATIONS = ("sdpa", "eager")
SETTINGS_KEYS = {
    "model", "window", "compute_precision", "attention_implementation", "threads", "reference_compile", "solver",
    "template_lines", "standardizer", "c",
}  # fmt: skip


class SettingsError(ValueError):
    """A settings or fold-weights file that is refused whole."""


class SnapshotError(RuntimeError):
    """A model snapshot whose files don't match the settings' sha256s. Nothing is loaded."""


class EmbeddingBackend(Protocol):
    """The tokenizer and model, as the router uses them: one request at a time, so no padding enters (item 4)."""

    def encode(self, text: str) -> list[int]:
        """Token ids with the special tokens at each end (CLS first, SEP last), and no truncation."""

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        """The CLS vector of the model's last layer for these ids, in the backend's own precision."""


@dataclass(frozen=True)
class Settings:
    repository: str
    commit: str
    files: dict[str, str]  # file name -> sha256
    window: int
    compute_precision: str
    attention: str
    threads: int
    reference_compile: bool
    solver: dict[str, Any]
    template_lines: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    c: dict[str, float]


@dataclass(frozen=True)
class Head:
    coef: np.ndarray  # one logistic head's weights over the standardized embedding and then the flag
    intercept: float


FoldWeights = tuple[Mapping[str, Head], ...]  # five fold models, each a head per config in SCORED


def _fail(what: str) -> SettingsError:
    return SettingsError(what)


def _exact_keys(data: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != keys:
        raise _fail(f"{where} must have exactly the keys {sorted(keys)}")
    return data


def _numbers(values: Any, where: str) -> np.ndarray:
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    ):
        raise _fail(f"{where} must be a list of numbers")
    array = np.array(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise _fail(f"{where} must be finite")
    return array


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _fail(f"{where} must be a finite number")
    return float(value)


def parse_settings(data: Any) -> Settings:
    """The settings file's contents as `Settings`, or `SettingsError` for anything the shape above doesn't allow."""
    top = _exact_keys(data, SETTINGS_KEYS, "the settings")
    model = _exact_keys(top["model"], {"repository", "commit", "files"}, "the settings' model")
    if model["repository"] != MODEL_REPOSITORY or model["commit"] != MODEL_COMMIT:
        raise _fail(f"the settings must pin {MODEL_REPOSITORY} at {MODEL_COMMIT}")
    files = _exact_keys(model["files"], set(MODEL_FILES), "the settings' model files")
    if not all(isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef") for h in files.values()):
        raise _fail("each model file's sha256 must be 64 lowercase hexadecimal digits")
    window = top["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 4:
        raise _fail("the window must be an integer of at least 4 tokens")
    if top["compute_precision"] != "float32":
        raise _fail("the compute precision must be float32")
    if top["attention_implementation"] not in ATTENTION_IMPLEMENTATIONS:
        raise _fail(f"the attention implementation must be one of {ATTENTION_IMPLEMENTATIONS}")
    threads = top["threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise _fail("the thread count must be a positive integer")
    if top["reference_compile"] is not False:
        raise _fail("reference_compile must be false")
    solver = _exact_keys(top["solver"], {"name", "tol", "max_iter"}, "the settings' solver")
    lines = top["template_lines"]
    if not isinstance(lines, list) or not all(isinstance(x, str) and x and x == x.strip() for x in lines):
        raise _fail("template lines must be non-blank strings with no whitespace around them")
    standardizer = _exact_keys(top["standardizer"], {"mean", "scale"}, "the settings' standardizer")
    mean, scale = _numbers(standardizer["mean"], "the mean"), _numbers(standardizer["scale"], "the scale")
    if mean.shape != scale.shape or not (scale > 0).all():
        raise _fail("the mean and scale must have one length, and every scale must be positive")
    c = _exact_keys(top["c"], set(SCORED), "the settings' C")
    if not all(_number(v, "C") > 0 for v in c.values()):
        raise _fail("each C must be positive")
    return Settings(
        model["repository"], model["commit"], dict(files), window, "float32", top["attention_implementation"],
        threads, False, dict(solver), tuple(lines), mean, scale, {k: float(v) for k, v in c.items()},
    )  # fmt: skip


def parse_fold_weights(data: Any) -> FoldWeights:
    """The fold-weights file's contents: five fold models, each with a head per config."""
    folds = _exact_keys(data, {"folds"}, "the fold weights")["folds"]
    if not isinstance(folds, list) or len(folds) != FOLDS:
        raise _fail(f"the fold weights must hold {FOLDS} fold models")
    parsed = []
    for k, fold_model in enumerate(folds):
        heads = _exact_keys(fold_model, set(SCORED), f"fold model {k}")
        width = None
        out = {}
        for config in SCORED:
            head = _exact_keys(heads[config], {"coef", "intercept"}, f"fold model {k}'s {config} head")
            coef = _numbers(head["coef"], f"fold model {k}'s {config} coefficients")
            if width not in (None, len(coef)):
                raise _fail(f"fold model {k}'s heads must have one width")
            width = len(coef)
            out[config] = Head(coef, _number(head["intercept"], f"fold model {k}'s {config} intercept"))
        parsed.append(out)
    return tuple(parsed)


def read_settings(path: Path = SETTINGS_FILE) -> Settings:
    return parse_settings(json.loads(path.read_text()))


def read_fold_weights(path: Path = FOLD_WEIGHTS_FILE) -> FoldWeights:
    return parse_fold_weights(json.loads(path.read_text()))


def strip_template_lines(prompt: str, template_lines: Sequence[str]) -> str:
    """The prompt without its template lines: a line goes when, whitespace-trimmed, it matches an entry exactly.

    Blank lines stay, so paragraphs keep their shape. If a line was removed and only blank lines are left, the
    result is the empty string (item 5), not a stray newline. A prompt with no template line comes back unchanged.
    """
    listed = set(template_lines)
    kept = [line for line in prompt.split("\n") if line.strip() not in listed or not line.strip()]
    if len(kept) < prompt.count("\n") + 1 and not any(line.strip() for line in kept):
        return ""
    return "\n".join(kept)


def schema_flag(request: Request) -> float:
    """1 if the request has an output schema, else 0 (A9 item 1's "structured output requested" cue)."""
    return 0.0 if request.output_schema is None else 1.0


def canonical_text(request: Request) -> str:
    """The request as the hash reads it: sorted keys, no spaces, non-ASCII characters kept (item 9)."""
    payload = {"prompt": request.prompt, "output_schema": request.output_schema}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fold(request: Request) -> int:
    """The fold model that scores this request: its canonical text's sha256, read as an integer, mod 5."""
    return int(hashlib.sha256(canonical_text(request).encode("utf-8")).hexdigest(), 16) % FOLDS


def fit_window(ids: list[int], window: int) -> list[int]:
    """Token ids cut to the window: CLS and SEP stay, and the content keeps its head and its tail, half each.

    An odd number of content slots gives the head the extra token. Ids that fit are returned unchanged.
    """
    if len(ids) <= window:
        return ids
    content = window - 2
    head, tail = content - content // 2, content // 2
    return [ids[0], *ids[1 : 1 + head], *ids[len(ids) - 1 - tail : len(ids) - 1], ids[-1]]


def verify_snapshot(settings: Settings, snapshot_dir: Path = SNAPSHOT) -> None:
    """Check the five files against the settings' sha256s, refusing whole: every file that differs is named."""
    bad = []
    for name in MODEL_FILES:
        path = snapshot_dir / name
        if not path.is_file():
            bad.append(f"{name} (missing)")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != settings.files[name]:
            bad.append(f"{name} (sha256 differs)")
    if bad:
        raise SnapshotError(f"the model snapshot at {snapshot_dir} doesn't match the settings: {', '.join(bad)}")


class TransformersBackend:
    """`ModernBertModel` and its fast tokenizer, from the verified snapshot, on the CPU."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        input_ids = torch.tensor([ids])
        with torch.inference_mode():
            hidden = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).last_hidden_state
        return hidden[0, 0].numpy()

    def upcast(self) -> "TransformersBackend":
        """The same verified model in float64, for Stage 5 · 5's drift measurement. A copy: this one is unchanged."""
        return TransformersBackend(self.tokenizer, copy.deepcopy(self.model).double())


def load_backend(settings: Settings, snapshot_dir: Path = SNAPSHOT) -> TransformersBackend:
    """Verify the snapshot's five files, then load them offline: safetensors only, float32, on the CPU (item 4)."""
    verify_snapshot(settings, snapshot_dir)
    torch.set_num_threads(settings.threads)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    if not tokenizer.is_fast:
        raise SnapshotError("the snapshot's tokenizer isn't the fast tokenizer")
    config = ModernBertConfig.from_pretrained(snapshot_dir, local_files_only=True)
    config.reference_compile = False  # the pinned transformers has no compile path to turn off; set for the record
    loaded: Any = ModernBertModel.from_pretrained(
        snapshot_dir, config=config, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation=settings.attention, output_loading_info=True,
    )  # fmt: skip
    model, info = loaded
    if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
        raise SnapshotError(
            f"the weights don't fit the model: {info}"
        )  # a randomly initialized layer would score silently
    model.eval()
    return TransformersBackend(tokenizer, model)


@dataclass
class Counts:
    """Requests the router has met that item 5 says are counted (the totals go in the freeze record)."""

    emptied: int = 0  # template lines left nothing, so the empty string was embedded
    truncated: int = 0  # more tokens than the window, so head and tail were kept


class TrainedRouter:
    """F. `score(request)` and `score_with(fold, request)` give router scores; `embed` is the fit script's entry."""

    def __init__(
        self,
        settings: Settings,
        fold_weights: FoldWeights,
        *,
        backend: EmbeddingBackend | None = None,
        snapshot_dir: Path = SNAPSHOT,
    ) -> None:
        width = len(settings.mean) + 1  # the standardized embedding, then the flag
        if any(len(head.coef) != width for fold_model in fold_weights for head in fold_model.values()):
            raise SettingsError(f"every head needs {width} coefficients: the standardizer's length plus the flag")
        self.settings = settings
        self.fold_weights = fold_weights
        # A backend passed in is for checks and the float64 drift pass; the normal route always checks the files.
        self.backend: EmbeddingBackend = backend if backend is not None else load_backend(settings, snapshot_dir)
        self.counts = Counts()

    def embed(self, request: Request) -> np.ndarray:
        """The CLS vector of the request's stripped prompt, as the backend gives it, upcast to float64 (exactly)."""
        text = strip_template_lines(request.prompt, self.settings.template_lines)
        if text == "" and request.prompt != "":
            self.counts.emptied += 1
        ids = self.backend.encode(text)
        if len(ids) > self.settings.window:
            self.counts.truncated += 1
            ids = fit_window(ids, self.settings.window)
        return np.asarray(self.backend.cls_vector(ids), dtype=np.float64)

    def features(self, embedding: np.ndarray, request: Request) -> np.ndarray:
        """Length-normalize, standardize and append the flag: what the heads read."""
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.shape != self.settings.mean.shape:
            raise ValueError(f"an embedding of {vector.shape}, not the standardizer's {self.settings.mean.shape}")
        norm = float(np.linalg.norm(vector))
        if not norm > 0:
            raise ValueError("a zero embedding can't be length-normalized")
        standardized = (vector / norm - self.settings.mean) / self.settings.scale
        return np.append(standardized, schema_flag(request))

    def score_features(self, fold_index: int, features: np.ndarray) -> dict[str, float]:
        """The fold model's three logistic heads on features: a router score for each config."""
        if not 0 <= fold_index < FOLDS:
            raise ValueError(f"fold {fold_index} isn't one of the {FOLDS} fold models")
        heads = self.fold_weights[fold_index]
        return {
            config: _sigmoid(float(np.dot(heads[config].coef, features)) + heads[config].intercept) for config in SCORED
        }

    def score_with(self, fold_index: int, request: Request) -> dict[str, float]:
        """Router scores from the given fold model. Dev's table calls this with each item's fold from the fold file."""
        return self.score_features(fold_index, self.features(self.embed(request), request))

    def score(self, request: Request) -> dict[str, float]:
        """Router scores from the fold model the request's hash picks. Reads the request only (A9 item 1)."""
        return self.score_with(fold(request), request)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)  # a very negative z can't overflow here, as it would in exp(-z)
    return e / (1.0 + e)


@cache
def default_router() -> TrainedRouter:
    """F as committed: the settings and fold-weights files, the pinned snapshot through the file check."""
    return TrainedRouter(read_settings(), read_fold_weights())


def score(request: Request) -> dict[str, float]:
    """Router scores from the committed files. Stage 8 and S4 call this."""
    return default_router().score(request)
