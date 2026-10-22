"""Check the trained router's foundation, F (PLAN_v4.md §8 3b; Amendment A11, items 4, 5, 9 and 10).

Run from the project root:  uv run python -m analysis.check_trained_router

On synthetic requests and a fake embedding backend (analysis/fake_backend.py), with answers worked out by hand. Never
a test item, and no real settings file, fold weights or fit: this ticket fits nothing.

- template-line removal: a listed line removed, an unlisted one kept, blank lines kept, an emptied prompt embedded as
  the empty string and counted;
- the schema flag, with and without a schema;
- the hash: the canonical text, the fold from hand-computed sha256s, the same request always in the same fold, a
  schema's key order changing nothing;
- the window: a long text keeps the head and the tail of its content tokens, and is counted;
- the settings and fold-weights files: their shapes, and a refusal for each bad case;
- tamper refusal against the real loader: stand-in files with recorded sha256s, one changed byte refused whole;
- scoring through the fake backend: hand-worked router scores, `score` using the hash's fold and `score_with` the
  fold it is given, and nothing after the backend rounding its output back to float32;
- the backend seam: the normal constructor loads the pinned snapshot through the file check, never the fake;
- imports: the router's closure holds no scikit-learn, analysis or harness, and not the fake backend;
- the hook only (it needs the model snapshot, which CI lacks, fetched by `analysis.fetch_model`): a text over the
  window, in the model's own tokens, keeps exactly half the window's content tokens at each end and comes to exactly
  the window.
"""

import ast
import copy
import hashlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from analysis.fake_backend import CLS, SEP, FakeBackend
from analysis.freeze import closure
from harness.client import PROJECT_ROOT
from routers import trained
from routers.request import Request
from routers.trained import SnapshotError, TrainedRouter, parse_fold_weights, parse_settings

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def imported_packages(tree: ast.AST) -> set[str]:
    """The top-level packages a module imports (a relative import stays inside its package)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def passes(call: Callable[[], object]) -> bool:
    try:
        call()
    except Exception:
        return False
    return True


def close(got: dict[str, float], want: dict[str, float], tolerance: float) -> bool:
    return list(got) == list(want) and all(abs(got[c] - want[c]) <= tolerance for c in want)


# A two-dimensional embedding, so every router score can be worked out by hand.
TEMPLATE = "Boilerplate"
SCHEMA = {"b": 1, "a": 2}
FILE_NAMES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def settings_data(files: dict[str, str] | None = None, window: int = 8192) -> dict[str, Any]:
    return {
        "model": {
            "repository": trained.MODEL_REPOSITORY,
            "commit": trained.MODEL_COMMIT,
            "files": files or {name: "0" * 64 for name in FILE_NAMES},
        },
        "window": window,
        "compute_precision": "float32",
        "attention_implementation": "sdpa",
        "threads": 1,
        "reference_compile": False,
        "solver": {"name": "newton-cholesky", "tol": 1e-10, "max_iter": 100},
        "template_lines": [TEMPLATE, "Answer briefly."],
        "standardizer": {"mean": [0.5, 0.25], "scale": [0.25, 0.5]},
        "c": {"haiku": 1.0, "sonnet": 0.1, "opus_low": 10.0},
    }


def weights_data() -> dict[str, Any]:
    """Fold k's three heads over the features (embedding 0, embedding 1, flag), each with its own numbers."""
    folds = []
    for k in range(5):
        folds.append(
            {
                "haiku": {"coef": [0.5 * (k + 1), -0.25, 0.5], "intercept": 0.1 * k},
                "sonnet": {"coef": [-1.0, 1.0, 0.25], "intercept": -0.2 + 0.05 * k},
                "opus_low": {"coef": [0.0, 0.5, -1.0], "intercept": 0.3 - 0.1 * k},
            }
        )
    return {"folds": folds}


def make_router(backend: FakeBackend, window: int = 8192) -> TrainedRouter:
    return TrainedRouter(
        parse_settings(settings_data(window=window)), parse_fold_weights(weights_data()), backend=backend
    )


ALPHA = Request("alpha")
ALPHA_SCHEMA = Request("alpha", SCHEMA)
BETA = Request("beta")
WITH_TEMPLATE = Request(f"{TEMPLATE}\nalpha\n\n  {TEMPLATE}  ")


def main() -> int:
    print("-- template lines --")
    lines = tuple(trained.parse_settings(settings_data()).template_lines)
    check(trained.strip_template_lines("alpha", lines) == "alpha", "a prompt with no template line is unchanged")
    check(trained.strip_template_lines(f"{TEMPLATE}\nalpha", lines) == "alpha", "a listed line is removed")
    check(trained.strip_template_lines("Boilerplate 2\nalpha", lines) == "Boilerplate 2\nalpha",
          "an unlisted line that starts like a listed one is kept")  # fmt: skip
    check(trained.strip_template_lines(f"  {TEMPLATE}\t\nalpha", lines) == "alpha",
          "a listed line is removed whatever whitespace surrounds it")  # fmt: skip
    check(trained.strip_template_lines(f"alpha\n\n{TEMPLATE}\n\nbeta", lines) == "alpha\n\n\nbeta",
          "blank lines are kept, so paragraphs keep their shape")  # fmt: skip
    check(trained.strip_template_lines(f"{TEMPLATE}\nAnswer briefly.", lines) == "",
          "a prompt of nothing but template lines is left empty")  # fmt: skip
    check(trained.strip_template_lines(f"{TEMPLATE}\n\n{TEMPLATE}", lines) == "",
          "template lines around a blank line leave the empty string, not a stray newline")  # fmt: skip
    check(
        trained.strip_template_lines("  \n\n", lines) == "  \n\n",
        "blank lines alone, with nothing removed, are unchanged",
    )

    fake = FakeBackend({"alpha\n": (3.0, 4.0), "": (6.0, 8.0), "alpha": (3.0, 4.0)})
    router = make_router(fake)
    router.embed(WITH_TEMPLATE)
    check(fake.decode(fake.embedded[-1]) == "alpha\n", "the router embeds the prompt without its template lines")
    check(router.counts.emptied == 0, "a prompt with something left is not counted as emptied")
    router.embed(Request(f"{TEMPLATE}\nAnswer briefly."))
    check(fake.decode(fake.embedded[-1]) == "" and router.counts.emptied == 1,
          "an emptied prompt is embedded as the empty string and counted")  # fmt: skip
    router.embed(Request(f"{TEMPLATE}\nAnswer briefly."))
    check(router.counts.emptied == 2, "each emptied request is counted")

    print("\n-- the schema flag --")
    check(trained.schema_flag(ALPHA_SCHEMA) == 1.0, "a request with an output schema has flag 1")
    check(trained.schema_flag(ALPHA) == 0.0, "a request without one has flag 0")
    check(trained.schema_flag(Request("alpha", {})) == 1.0, "an empty schema is still a schema: flag 1")
    flag_fake = FakeBackend({"alpha": (3.0, 4.0)})
    make_router(flag_fake).score_with(0, ALPHA_SCHEMA)
    check(flag_fake.decode(flag_fake.embedded[-1]) == "alpha", "the schema itself is never embedded")

    print("\n-- the hash --")
    check(trained.canonical_text(ALPHA) == '{"output_schema":null,"prompt":"alpha"}',
          "the canonical text of a request without a schema equals the hand-written one")  # fmt: skip
    check(trained.canonical_text(ALPHA_SCHEMA) == '{"output_schema":{"a":2,"b":1},"prompt":"alpha"}',
          "the canonical text sorts keys and has no spaces")  # fmt: skip
    check(trained.canonical_text(Request("café\n")) == '{"output_schema":null,"prompt":"café\\n"}',
          "the canonical text keeps non-ASCII characters and escapes a newline")  # fmt: skip
    # Each expected fold is a hand-computed sha256 (printf ... | shasum -a 256) read as a number mod 5. 16 is 1 mod 5,
    # so that number is the sum of the hash's hexadecimal digits mod 5.
    check(trained.fold(ALPHA) == 4, "alpha's sha256 b15eb578... gives fold 4")
    check(trained.fold(ALPHA_SCHEMA) == 0, "alpha with a schema, sha256 f4224edc..., gives fold 0")
    check(trained.fold(Request("café")) == 3, "café, sha256 800de9f9..., gives fold 3")
    check(trained.fold(BETA) == 3, "beta, sha256 98cc27d4..., gives fold 3")
    check(trained.fold(Request(f"{TEMPLATE}\nalpha\n\n{TEMPLATE}")) == 1,
          "a request with template lines, sha256 44315a6f..., gives fold 1: the hash covers the whole request")  # fmt: skip
    check(all(trained.fold(ALPHA) == 4 for _ in range(3)), "the same request always gets the same fold")
    reordered = Request("alpha", {"a": 2, "b": 1})
    check(trained.fold(reordered) == trained.fold(ALPHA_SCHEMA), "reordering the schema's keys changes nothing")
    check(trained.canonical_text(reordered) == trained.canonical_text(ALPHA_SCHEMA),
          "nor does it change the canonical text")  # fmt: skip

    print("\n-- the window --")
    ids = [CLS, *range(100, 120), SEP]  # 20 content tokens
    kept = trained.fit_window(ids, 8)
    check(
        kept == [CLS, 100, 101, 102, 117, 118, 119, SEP], "over the window: the head and tail of the content, half each"
    )
    check(trained.fit_window(ids, 22) == ids, "a text of exactly the window is untouched")
    check(trained.fit_window(ids, 21) == [CLS, *range(100, 110), *range(111, 120), SEP],
          "an odd number of content slots gives the head the extra token")  # fmt: skip
    window_fake = FakeBackend({}, default=(3.0, 4.0))
    window_router = make_router(window_fake, window=8)
    window_router.embed(Request("x" * 20))
    check(len(window_fake.embedded[-1]) == 8 and window_router.counts.truncated == 1,
          "the router embeds a text over the window cut to it, and counts it")  # fmt: skip
    window_router.embed(Request("x" * 6))
    check(len(window_fake.embedded[-1]) == 8 and window_router.counts.truncated == 1,
          "a text that fits the window is not counted")  # fmt: skip

    print("\n-- the settings and fold-weights files --")
    settings = parse_settings(settings_data())
    check(settings.window == 8192 and settings.threads == 1 and settings.attention == "sdpa", "a settings file parses")
    check(settings.mean.tolist() == [0.5, 0.25] and settings.scale.tolist() == [0.25, 0.5], "with its standardizer")
    check(settings.mean.dtype == np.float64, "kept in float64")
    check(len(parse_fold_weights(weights_data())) == 5, "fold weights parse to five fold models")
    broken: list[tuple[str, Callable[[], object]]] = []

    def settings_without(*path: str) -> Callable[[], object]:
        def build() -> object:
            data = settings_data()
            target = data
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            return parse_settings(data)

        return build

    def settings_with(change: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        def build() -> object:
            data = settings_data()
            change(data)
            return parse_settings(data)

        return build

    def weights_with(change: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        def build() -> object:
            data = weights_data()
            change(data)
            return parse_fold_weights(data)

        return build

    broken += [
        ("a missing key", settings_without("window")),
        ("a missing model file", settings_without("model", "files", "config.json")),
        ("an unknown key", settings_with(lambda d: d.update(extra=1))),
        ("another commit", settings_with(lambda d: d["model"].update(commit="0" * 40))),
        ("a sha256 of the wrong length", settings_with(lambda d: d["model"]["files"].update({"config.json": "abc"}))),
        ("a window of one token", settings_with(lambda d: d.update(window=2))),
        ("a half precision", settings_with(lambda d: d.update(compute_precision="float16"))),
        ("an unknown attention implementation", settings_with(lambda d: d.update(attention_implementation="flash"))),
        ("reference compile on", settings_with(lambda d: d.update(reference_compile=True))),
        ("no thread", settings_with(lambda d: d.update(threads=0))),
        ("a blank template line", settings_with(lambda d: d["template_lines"].append("  "))),
        ("a template line with whitespace around it", settings_with(lambda d: d["template_lines"].append(" x "))),
        ("a standardizer of two lengths", settings_with(lambda d: d["standardizer"]["scale"].append(1.0))),
        ("a standardizer scale of zero", settings_with(lambda d: d["standardizer"]["scale"].__setitem__(0, 0.0))),
        ("a not-a-number mean", settings_with(lambda d: d["standardizer"]["mean"].__setitem__(0, float("nan")))),
        ("a missing C", settings_with(lambda d: d["c"].pop("sonnet"))),
        ("a C of zero", settings_with(lambda d: d["c"].update(haiku=0.0))),
    ]
    for what, call in broken:
        check(refused(trained.SettingsError, call), f"the settings file is refused for {what}")
    for what, call in [
        ("four fold models", weights_with(lambda d: d["folds"].pop())),
        ("a missing head", weights_with(lambda d: d["folds"][2].pop("sonnet"))),
        ("a head with an unknown config", weights_with(lambda d: d["folds"][2].update(opus={}))),
        ("a missing intercept", weights_with(lambda d: d["folds"][0]["haiku"].pop("intercept"))),
        (
            "an infinite coefficient",
            weights_with(lambda d: d["folds"][1]["haiku"]["coef"].__setitem__(0, float("inf"))),
        ),
    ]:
        check(refused(trained.SettingsError, call), f"the fold weights are refused for {what}")
    wrong_width = copy.deepcopy(weights_data())
    wrong_width["folds"][3]["opus_low"]["coef"].append(0.0)
    check(refused(trained.SettingsError, lambda: TrainedRouter(settings, parse_fold_weights(wrong_width), backend=FakeBackend({}))),
          "weights whose width isn't the standardizer's plus the flag are refused with the settings")  # fmt: skip

    print("\n-- tamper refusal, against the real loader --")
    with tempfile.TemporaryDirectory() as temp:
        snapshot = Path(temp)
        for name in FILE_NAMES:
            (snapshot / name).write_bytes(f"stand-in for {name}\n".encode())
        recorded = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in FILE_NAMES}
        stand_in = parse_settings(settings_data(files=recorded))
        check(
            passes(lambda: trained.verify_snapshot(stand_in, snapshot)),
            "stand-in files that match their recorded sha256s pass",
        )
        # Same length, one byte different: a refusal can't come from a size check.
        original = (snapshot / "config.json").read_bytes()
        (snapshot / "config.json").write_bytes(original[:-2] + b"X\n")
        check(refused(SnapshotError, lambda: trained.verify_snapshot(stand_in, snapshot)),
              "one changed byte in a file is refused")  # fmt: skip
        check(refused(SnapshotError, lambda: trained.load_backend(stand_in, snapshot)),
              "the loader refuses before it loads anything")  # fmt: skip
        check(refused(SnapshotError, lambda: TrainedRouter(stand_in, parse_fold_weights(weights_data()), snapshot_dir=snapshot)),
              "the normal constructor refuses a tampered snapshot")  # fmt: skip
        (snapshot / "tokenizer.json").write_bytes(b"also changed\n")
        try:
            trained.verify_snapshot(stand_in, snapshot)
        except SnapshotError as error:
            check("config.json" in str(error) and "tokenizer.json" in str(error) and "model.safetensors" not in str(error),
                  "the refusal is whole: it names every file that differs, and only those")  # fmt: skip
        (snapshot / "config.json").write_bytes(original)
        (snapshot / "tokenizer.json").write_bytes(b"stand-in for tokenizer.json\n")
        (snapshot / "model.safetensors").unlink()
        check(refused(SnapshotError, lambda: trained.verify_snapshot(stand_in, snapshot)), "a missing file is refused")
        (snapshot / "model.safetensors").write_bytes(b"stand-in for model.safetensors\n")
        check(passes(lambda: trained.verify_snapshot(stand_in, snapshot)), "restoring the bytes passes again")
        (snapshot / "pytorch_model.bin").write_bytes(b"a pickle the router never reads\n")
        check(passes(lambda: trained.verify_snapshot(stand_in, snapshot)) and "pytorch_model.bin" not in stand_in.files,
              "the repository's pickle is neither read nor listed among the five files")  # fmt: skip

    print("\n-- scoring through the fake backend --")
    # Hand-worked: (3, 4) has length 5, so (0.6, 0.8); standardized with mean (0.5, 0.25) and scale (0.25, 0.5) it is
    # (0.4, 1.1); then the flag. Fold 4's haiku head gives z = 2.5*0.4 - 0.25*1.1 + 0.5*0 + 0.4 = 1.125, and so on;
    # each router score is 1/(1+e^-z) from a 50-digit decimal calculation.
    alpha_fold4 = {"haiku": 0.7549149868676283, "sonnet": 0.6681877721681662, "opus_low": 0.610639233949222}
    alpha_schema_fold0 = {"haiku": 0.6046790847140093, "sonnet": 0.679178699175393, "opus_low": 0.46257015465625045}
    alpha_fold2 = {"haiku": 0.6283161882953663, "sonnet": 0.6456563062257955, "opus_low": 0.6570104626734988}
    beta_fold3 = {"haiku": 0.6952966693475654, "sonnet": 0.6570104626734988, "opus_low": 0.6341355910108006}
    vectors = {"alpha": (3.0, 4.0), "beta": (6.0, 8.0)}
    router = make_router(FakeBackend(vectors))
    check(
        close(router.score_with(2, ALPHA), alpha_fold2, 1e-12), "score_with scores with the fold it is given (fold 2)"
    )
    check(close(router.score_with(4, ALPHA), alpha_fold4, 1e-12), "and another fold gives other router scores (fold 4)")
    check(not close(router.score_with(2, ALPHA), router.score_with(4, ALPHA), 1e-3), "the two folds really differ")
    check(
        close(router.score_with(0, ALPHA_SCHEMA), alpha_schema_fold0, 1e-12),
        "the flag enters the heads (schema, fold 0)",
    )
    check(close(router.score(ALPHA), alpha_fold4, 1e-12), "score uses the hash's fold: alpha is in fold 4")
    check(close(router.score(ALPHA_SCHEMA), alpha_schema_fold0, 1e-12), "alpha with a schema is in fold 0")
    check(close(router.score(BETA), beta_fold3, 1e-12), "beta is in fold 3, and a vector's length doesn't matter")
    check(
        list(router.score(ALPHA)) == ["haiku", "sonnet", "opus_low"], "router scores are for haiku, sonnet and opus_low"
    )
    check(all(isinstance(v, float) for v in router.score(ALPHA).values()), "as plain floats")
    check(all(refused(ValueError, partial(router.score_with, f, ALPHA)) for f in (-1, 5)),
          "a fold outside 0 to 4 is refused")  # fmt: skip
    check(refused(ValueError, lambda: make_router(FakeBackend({"alpha": (0.0, 0.0)})).score(ALPHA)),
          "a zero embedding can't be length-normalized and is refused")  # fmt: skip
    check(refused(ValueError, lambda: make_router(FakeBackend({"alpha": (1.0, 2.0, 3.0)})).score(ALPHA)),
          "an embedding of the wrong width is refused")  # fmt: skip
    check(close(make_router(FakeBackend({"alpha": (3.0, 4.0)}, dtype=np.float32)).score(ALPHA), alpha_fold4, 1e-12),
          "a float32 backend's embedding is upcast exactly and scores the same")  # fmt: skip
    templated = make_router(FakeBackend({"alpha\n": (3.0, 4.0)}))
    check(close(templated.score_with(4, WITH_TEMPLATE), alpha_fold4, 1e-12),
          "a request with template lines is scored on what is left")  # fmt: skip
    features = router.features(np.array([3.0, 4.0]), ALPHA_SCHEMA)
    check(
        np.allclose(features, [0.4, 1.1, 1.0], atol=1e-15), "the features are the standardized embedding and the flag"
    )

    print("\n-- no rounding back to float32 --")
    # A float64 embedding that float32 can't hold: (3 + 4e-9, 4 - 3e-9) rounds to exactly (3, 4) in float32.
    wide = (3.0 + 4e-9, 4.0 - 3e-9)
    check(
        tuple(float(np.float32(v)) for v in wide) == (3.0, 4.0),
        "the test embedding does change when rounded to float32",
    )
    wide_fold4 = {"haiku": 0.7549149884032805, "sonnet": 0.6681877711926295, "opus_low": 0.6106392338065666}
    wide_router = make_router(FakeBackend({"alpha": wide}, dtype=np.float64))
    got = wide_router.score(ALPHA)
    check(close(got, wide_fold4, 1e-13), "a float64 backend's embedding gives the float64 hand-worked router scores")
    check(
        not close(got, alpha_fold4, 1e-10), "which differ from the float32-rounded embedding's, so rounding would show"
    )
    check(wide_router.embed(ALPHA).dtype == np.float64 and wide_router.embed(ALPHA).tolist() == list(wide),
          "the embedding function returns the backend's float64 vector untouched")  # fmt: skip

    print("\n-- the backend seam --")
    try:
        TrainedRouter(settings, parse_fold_weights(weights_data()), snapshot_dir=Path("/nonexistent/snapshot"))
        check(False, "the normal constructor, with no backend, loads the pinned snapshot")
    except SnapshotError:
        check(True, "the normal constructor, with no backend, goes to the snapshot and its file check")
    check(not any(name in vars(trained) for name in ("FakeBackend", "fake_backend")), "the router module holds no fake")

    print("\n-- imports --")
    sources = {relative: ast.parse((PROJECT_ROOT / relative).read_text()) for relative in closure("routers/trained.py")}
    imported = set().union(*(imported_packages(tree) for tree in sources.values()))
    check(
        {"torch", "transformers", "numpy"} <= imported,
        "the guard finds the imports it looks for: numpy, torch and transformers",
    )
    # scikit-learn is checked in the source, not at run time: `transformers` itself imports it when it is installed.
    forbidden = {"sklearn", "analysis", "harness"}
    check(
        imported_packages(ast.parse("import sklearn.linear_model\nfrom analysis import freeze\nimport harness"))
        == forbidden,
        "the guard finds an import of scikit-learn, analysis and harness in a module that has them",
    )
    check(
        not imported & forbidden,
        "the router's closure imports no scikit-learn, analysis or harness",
    )
    check("routers/trained.py" in sources and not any("fake_backend" in relative for relative in sources),
          "the fake backend is not in the router's import closure")  # fmt: skip
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, routers.trained; print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'analysis', 'harness'}))"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    check(probe.returncode == 0 and probe.stdout.strip() == "[]",
          f"importing the router loads no analysis or harness module, the fake backend included ({probe.stdout.strip()})")  # fmt: skip

    if trained.SNAPSHOT.joinpath("model.safetensors").exists():
        real_snapshot_checks()
    else:
        print("\n-- the real snapshot: skipped, none under .cache (fetch it with analysis.fetch_model) --")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def real_snapshot_checks() -> None:
    print("\n-- the real snapshot (the hook only) --")
    recorded = {name: hashlib.sha256((trained.SNAPSHOT / name).read_bytes()).hexdigest() for name in FILE_NAMES}
    real = parse_settings(
        settings_data(files=recorded)
    )  # hashes taken from the files: this check reads tokens, not hashes
    backend = trained.load_backend(real)
    tokenizer = backend.tokenizer
    window = real.window
    text = " ".join(f"word{i}" for i in range(window))  # far more than the window in tokens
    whole = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    check(len(whole) > window and whole[0] == tokenizer.cls_token_id and whole[-1] == tokenizer.sep_token_id,
          f"the long test text is over the window ({len(whole)} tokens) and starts with CLS and ends with SEP")  # fmt: skip
    half = (window - 2) // 2
    want = [whole[0], *whole[1 : 1 + half], *whole[-1 - half : -1], whole[-1]]
    ids = trained.fit_window(backend.encode(text), window)
    check(len(ids) == window, f"the cut text comes to exactly the window, {window} tokens, CLS and SEP included")
    check(ids == want and ids[1 : 1 + half] == whole[1 : 1 + half] and ids[-1 - half : -1] == whole[-1 - half : -1],
          f"it keeps exactly {half} content tokens at its head and {half} at its tail")  # fmt: skip
    check(backend.encode("") == [tokenizer.cls_token_id, tokenizer.sep_token_id], "the empty string is CLS and SEP")
    # Through the model itself on a small window (the full window takes half a minute on the CPU).
    small = parse_settings(settings_data(files=recorded, window=64))
    small_router = TrainedRouter(small, parse_fold_weights(weights_data()), backend=backend)
    vector = small_router.embed(Request(text))
    check(vector.shape == (768,) and vector.dtype == np.float64 and bool(np.isfinite(vector).all())
          and small_router.counts.truncated == 1,
          "the model embeds a cut text: 768 finite float64 values, counted as truncated")  # fmt: skip
    upcast = TrainedRouter(real, parse_fold_weights(weights_data()), backend=backend.upcast())
    short = Request("What is the capital of France?")
    gap = float(np.abs(upcast.embed(short) - small_router.embed(short)).max())
    check(
        0 < gap < 1e-4,
        f"the same snapshot upcast to float64 embeds the same text to float32's rounding error ({gap:.1e})",
    )


if __name__ == "__main__":
    sys.exit(main())
