"""Fit the trained router, F (PLAN_v4.md §8 3b; Amendment A11, items 5-8, 10, 12, 15 and 19).

Run from the project root:  uv run python -m analysis.fit_trained_router <command>

Three commands, in this order, and only the first two before the freeze commit:

- `caches`: embeds every train and dev request, one at a time, through the router's own `embed` (item 10), and writes
  the vectors under `results/trained_embeddings/` (not committed). It reads prompts and output schemas only: no
  outcome, and the router scores a timing pass would produce are never made here.
- `settings`: from the train caches and train outcomes only, learns the template lines (item 5), the standardizer and
  each config's C by cross-validation within train (item 7), runs the train signal check, and writes the settings file
  and F-train's weights (item 12). Prints what item 19 records.
- `folds`: the fit commit. Fits the five fold models from the frozen settings, the caches, the fold files and the
  outcomes of train and four dev folds each, and writes the fold-weights file. It reads dev outcomes and prints
  nothing about them: only each fold model's number of training rows and the file's sha256 (item 15).

Nothing before the freeze reads a dev outcome. The benchmark reader and the matrix loader are imported from outside the
freeze; the refit check guards them (item 15).
"""

import hashlib
import json
import sys
import warnings
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.folds import N_FOLDS, read_folds
from routers import trained
from routers.ladder import SCORED
from routers.request import Request, from_row
from routers.trained import FoldWeights, Head, Settings, TrainedRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "results" / "trained_embeddings"
F_TRAIN_WEIGHTS = PROJECT_ROOT / "routers" / "trained_f_train_weights.json"  # the deployable router never loads it

TEMPLATE_PERCENT = 2  # a template line is in at least this share of train prompts (item 5)
GRID = tuple(float(c) for c in np.logspace(-4, 2, 13))  # 13 log-spaced values; never widened (item 7)
SOLVER: dict[str, Any] = {"name": "newton-cholesky", "tol": 1e-10, "max_iter": 100}  # item 7; recorded in the settings
ATTENTION = "sdpa"
THREADS = 4  # set explicitly; the library default depends on the machine (item 4)
WINDOW = 8192


class FitError(RuntimeError):
    """A fit, cache or input the script refuses: a fit that didn't converge, a cache that isn't the benchmark's rows."""


# ---------------------------------------------------------------------------------------------------------------
# Template lines and the standardizer (items 5 and 7)


def line_counts(prompts: Sequence[str]) -> Counter[str]:
    """How many prompts hold each non-blank line, whitespace-trimmed: a line counts once per prompt."""
    counts: Counter[str] = Counter()
    for prompt in prompts:
        counts.update({line.strip() for line in prompt.split("\n") if line.strip()})
    return counts


def learn_template_lines(prompts: Sequence[str], percent: int = TEMPLATE_PERCENT) -> list[str]:
    """Every non-blank line found verbatim, whitespace-trimmed, in at least `percent`% of the prompts, sorted.

    The comparison is in whole numbers, so a line exactly at the cut is in.
    """
    return sorted(line for line, n in line_counts(prompts).items() if n * 100 >= len(prompts) * percent)


def length_normalized(embeddings: np.ndarray) -> np.ndarray:
    """Each row divided by its norm, with the expression the router's `features` uses."""
    rows = []
    for vector in embeddings:
        norm = float(np.linalg.norm(vector))
        if not norm > 0:
            raise FitError("a zero embedding can't be length-normalized")
        rows.append(vector / norm)
    return np.array(rows)


def standardizer(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the length-normalized embeddings, fitted once and shared (item 7)."""
    normalized = length_normalized(embeddings)
    mean, scale = normalized.mean(axis=0), normalized.std(axis=0)
    if not (scale > 0).all():
        raise FitError("an embedding dimension is constant on train, so it can't be standardized")
    return mean, scale


class _NoBackend:
    """For building feature rows: it can't embed."""

    def encode(self, text: str) -> list[int]:
        raise FitError("this router was built to read features, not to embed")

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        raise FitError("this router was built to read features, not to embed")


def feature_router(settings: Settings) -> TrainedRouter:
    return TrainedRouter(settings, (), backend=_NoBackend())


def feature_matrix(settings: Settings, embeddings: np.ndarray, requests: Sequence[Request]) -> np.ndarray:
    """What the heads read, one row per request, from the router's own `features` so fit and scorer can't differ."""
    router = feature_router(settings)
    return np.array([router.features(e, r) for e, r in zip(embeddings, requests, strict=True)])


# ---------------------------------------------------------------------------------------------------------------
# One logistic head (items 7 and 10)


def fit_model(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    """L2 logistic regression, an unpenalized intercept, no class weights, scikit-learn's newton-cholesky at
    tolerance 1e-10 and at most 100 iterations. A fit that doesn't converge is refused (FitError), whole."""
    if len(set(y.tolist())) != 2:
        raise FitError("a head needs both outcomes among its rows")
    model = LogisticRegression(C=c, solver=str(SOLVER["name"]), tol=float(SOLVER["tol"]),
                               max_iter=int(SOLVER["max_iter"]))  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as warning:
            raise FitError(
                f"the fit at C = {c} didn't converge in {SOLVER['max_iter']} iterations: {warning}"
            ) from None
    return model


def head_of(model: LogisticRegression) -> Head:
    return Head(model.coef_[0].copy(), float(model.intercept_[0]))


def fit_head(x: np.ndarray, y: np.ndarray, c: float) -> Head:
    return head_of(fit_model(x, y, c))


def out_of_fold(x: np.ndarray, y: np.ndarray, folds: np.ndarray, c: float) -> np.ndarray:
    """Each row's success estimate from a head fitted on the other folds, scikit-learn's own `predict_proba`."""
    p = np.zeros(len(y))
    for k in range(N_FOLDS):
        held = folds == k
        p[held] = fit_model(x[~held], y[~held], c).predict_proba(x[held])[:, 1]
    return p


def mean_fold_loss(p: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Log loss averaged over the folds (scikit-learn's cross-validation convention)."""
    return float(np.mean([log_loss(y[folds == k], p[folds == k], labels=[0, 1]) for k in range(N_FOLDS)]))


@dataclass(frozen=True)
class ChosenC:
    c: float
    losses: tuple[float, ...]  # mean log loss at each grid value, in grid order
    at_end: str  # "lower end", "upper end" or "" (inside the grid)


def choose_c(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> ChosenC:
    """The grid value with the lowest cross-validated mean log loss; a tie goes to the smaller C (more penalty).
    A C at either end is kept and reported, and the grid is not widened (item 7)."""
    losses = tuple(mean_fold_loss(out_of_fold(x, y, folds, c), y, folds) for c in GRID)
    best = int(np.argmin(losses))
    return ChosenC(GRID[best], losses, "lower end" if best == 0 else "upper end" if best == len(GRID) - 1 else "")


# ---------------------------------------------------------------------------------------------------------------
# Within-task concordance, as a point value (the bootstrap's version is `analysis.router_metrics`)


def concordance(scores: np.ndarray, y: np.ndarray, cells: Sequence[Hashable], tasks: Sequence[str]) -> dict:
    """Share of (success, failure) pairs within one cell whose scores are in the right order, ties counting half.

    Pooled (key None) and per task. A value is None where there is no pair. Pairs across cells are never counted
    (item 12): a cell is a task and a train fold.
    """
    members: dict[Hashable, list[int]] = {}
    for i, cell in enumerate(cells):
        members.setdefault(cell, []).append(i)
    won: dict[str | None, float] = {}
    pairs: dict[str | None, int] = {}
    for indices in members.values():
        task = tasks[indices[0]]
        up = np.array([scores[i] for i in indices if y[i] == 1])[:, None]
        down = np.array([scores[i] for i in indices if y[i] == 0])[None, :]
        credit = (up > down) + 0.5 * (up == down)
        for key in (None, task):
            won[key] = won.get(key, 0.0) + float(credit.sum())
            pairs[key] = pairs.get(key, 0) + credit.size
    keys = [None, *dict.fromkeys(tasks)]
    return {key: (won[key] / pairs[key] if pairs.get(key) else None) for key in keys}


# ---------------------------------------------------------------------------------------------------------------
# The train signal check (item 7) and the files


def signal_check(x: np.ndarray, y: np.ndarray, folds: np.ndarray, tasks: np.ndarray, c: float) -> dict:
    """Cross-validated log loss and within-task concordance of an intercept-only model, the task-rates model and F
    (a head at C), for one config. Rates and the head come from the other four train folds."""
    base, rates = np.zeros(len(y)), np.zeros(len(y))
    for k in range(N_FOLDS):
        held, rest = folds == k, folds != k
        base[held] = y[rest].mean()
        for task in set(tasks[held].tolist()):
            rates[held & (tasks == task)] = y[rest & (tasks == task)].mean()
    models = {"intercept only": base, "task rates": rates, "F": out_of_fold(x, y, folds, c)}
    cells = list(zip(tasks.tolist(), folds.tolist(), strict=True))
    return {name: {"log loss": mean_fold_loss(p, y, folds), "concordance": concordance(p, y, cells, tasks.tolist())}
            for name, p in models.items()}  # fmt: skip


def settings_data(
    files: Mapping[str, str], template_lines: Sequence[str], mean: np.ndarray, scale: np.ndarray, c: Mapping[str, float]
) -> dict[str, Any]:
    """The settings file's contents (see `routers/trained.py` for the shape)."""
    return {
        "model": {"repository": trained.MODEL_REPOSITORY, "commit": trained.MODEL_COMMIT, "files": dict(files)},
        "window": WINDOW, "compute_precision": "float32", "attention_implementation": ATTENTION, "threads": THREADS,
        "reference_compile": False, "solver": dict(SOLVER), "template_lines": list(template_lines),
        "standardizer": {"mean": mean.tolist(), "scale": scale.tolist()}, "c": dict(c),
    }  # fmt: skip


def json_text(data: Any) -> str:
    """Plain JSON floats, sorted keys, one value per line, so a diff and a hash are stable."""
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def heads_data(heads: Mapping[str, Head]) -> dict[str, Any]:
    return {config: {"coef": heads[config].coef.tolist(), "intercept": heads[config].intercept} for config in SCORED}


def fold_weights_text(models: Sequence[Mapping[str, Head]]) -> str:
    return json_text({"folds": [heads_data(heads) for heads in models]})


def f_train_weights_text(heads: Mapping[str, Head]) -> str:
    return json_text(heads_data(heads))


def read_f_train_weights(path: Path = F_TRAIN_WEIGHTS) -> FoldWeights:
    """F-train's one model in all five fold slots: the same arithmetic, and the hash still picks a slot (item 11)."""
    return trained.parse_fold_weights({"folds": [json.loads(path.read_text())] * trained.FOLDS})


def model_file_hashes(snapshot: Path = trained.SNAPSHOT) -> dict[str, str]:
    """The sha256 of each of the five model files, read from the snapshot."""
    hashes = {}
    for name in trained.MODEL_FILES:
        digest = hashlib.sha256()
        with (snapshot / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


# ---------------------------------------------------------------------------------------------------------------
# The embedding caches (item 10)


@dataclass(frozen=True)
class Cache:
    item_ids: list[str]
    embeddings: np.ndarray  # float64, exactly the float32 vectors `TrainedRouter.embed` gave
    tokens: list[int]  # the stripped text's length in tokens, special tokens included
    emptied: list[bool]  # the template lines left nothing, so the empty string was embedded


def provisional_settings(template_lines: Sequence[str], files: Mapping[str, str], width: int = 768) -> Settings:
    """Settings that can embed (the template lines, the window, the files) before the standardizer and C exist."""
    zeros, ones = np.zeros(width), np.ones(width)
    return trained.parse_settings(settings_data(files, template_lines, zeros, ones, dict.fromkeys(SCORED, 1.0)))


def embed_requests(router: TrainedRouter, requests: Sequence[Request]) -> Cache:
    """Embed each request on its own through the router's `embed` (no padding), noting its stripped length."""
    vectors, tokens, emptied = [], [], []
    for request in requests:
        before = router.counts.emptied
        vectors.append(router.embed(request))
        text = trained.strip_template_lines(request.prompt, router.settings.template_lines)
        tokens.append(len(router.backend.encode(text)))
        emptied.append(router.counts.emptied > before)
    return Cache([], np.array(vectors), tokens, emptied)


def cache_paths(split: str, directory: Path = CACHE_DIR) -> tuple[Path, Path]:
    return directory / f"{split}.npy", directory / f"{split}_items.json"


def write_cache(split: str, cache: Cache, directory: Path = CACHE_DIR) -> None:
    vectors, items = cache_paths(split, directory)
    np.save(vectors, cache.embeddings)
    items.write_text(json_text({"item_ids": cache.item_ids, "tokens": cache.tokens, "emptied": cache.emptied}))


def read_cache(split: str, directory: Path = CACHE_DIR) -> Cache:
    """The split's cache, refused (FitError) unless it holds exactly the reader's rows of that split, in order."""
    vectors, items = cache_paths(split, directory)
    if not (vectors.exists() and items.exists()):
        raise FitError(
            f"no {split} embedding cache under {directory}: run `python -m analysis.fit_trained_router caches`"
        )
    record = json.loads(items.read_text())
    embeddings = np.load(vectors)
    if record["item_ids"] != [row["item_id"] for row in read_rows(split=split)] or len(embeddings) != len(
        record["item_ids"]
    ):
        raise FitError(f"the {split} cache is not the benchmark's {split} rows, in order")
    return Cache(record["item_ids"], embeddings, record["tokens"], record["emptied"])


def train_prompts() -> list[str]:
    return [row["prompt"] for row in read_rows(split="train")]


def build_caches(directory: Path = CACHE_DIR) -> None:
    """Embed train, then dev. Template lines are learned from train prompts alone."""
    lines = learn_template_lines(train_prompts())
    router = TrainedRouter(provisional_settings(lines, model_file_hashes()), ())
    for split in ("train", "dev"):
        rows = read_rows(split=split)
        cache = embed_requests(router, [from_row(row) for row in rows])
        write_cache(split, replace(cache, item_ids=[row["item_id"] for row in rows]), directory)
        print(f"{split}: {len(rows)} requests embedded, {sum(cache.emptied)} emptied, "
              f"{router.counts.truncated} truncated so far")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# Train-side fit: everything item 19 records from train (no dev outcome)


@dataclass(frozen=True)
class TrainSide:
    settings: dict[str, Any]
    f_train: dict[str, Head]
    chosen: dict[str, ChosenC]
    signal: dict[str, dict]
    every_config_fails: dict[str, tuple[int, int]]  # task -> (items that every scored config fails, items)


def train_labels(matrix: results_matrix.Matrix, item_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Run-0 success per scored config for train items, in `item_ids` order. Train cells only."""
    train = matrix.split("train")
    return {config: np.array([train.cells[(i, config)].success for i in item_ids]) for config in SCORED}


def fit_train_side(matrix: results_matrix.Matrix | None = None) -> TrainSide:
    """Template lines, standardizer, C per config, the signal check and F-train, from train alone."""
    rows = read_rows(split="train")
    ids = [row["item_id"] for row in rows]
    requests = [from_row(row) for row in rows]
    tasks = np.array([row["source"] for row in rows])
    cache = read_cache("train")
    lines = learn_template_lines([r.prompt for r in requests])
    mean, scale = standardizer(cache.embeddings)
    fold_of = read_folds("train", None)
    folds = np.array([fold_of[i] for i in ids])
    y = train_labels(matrix if matrix is not None else results_matrix.load(), ids)
    shared = settings_data(model_file_hashes(), lines, mean, scale, dict.fromkeys(SCORED, 1.0))
    x = feature_matrix(trained.parse_settings(shared), cache.embeddings, requests)
    chosen = {config: choose_c(x, y[config], folds) for config in SCORED}
    signal = {config: signal_check(x, y[config], folds, tasks, chosen[config].c) for config in SCORED}
    f_train = {config: fit_head(x, y[config], chosen[config].c) for config in SCORED}
    fails = np.all([y[config] == 0 for config in SCORED], axis=0)
    every = {
        task: (int(fails[tasks == task].sum()), int((tasks == task).sum())) for task in dict.fromkeys(tasks.tolist())
    }
    final = settings_data(model_file_hashes(), lines, mean, scale, {c: chosen[c].c for c in SCORED})
    return TrainSide(final, f_train, chosen, signal, every)


def report(side: TrainSide) -> str:
    out = []
    for config in SCORED:
        chosen = side.chosen[config]
        out.append(
            f"{config}: C = {chosen.c:.6g}{' (at the ' + chosen.at_end + ' of the grid)' if chosen.at_end else ''}"
        )
        out.append("  mean log loss over the grid: " + ", ".join(f"{loss:.4f}" for loss in chosen.losses))
        for name, reading in side.signal[config].items():
            per_task = ", ".join(
                f"{t} {v:.3f}" for t, v in reading["concordance"].items() if t is not None and v is not None
            )
            pooled = reading["concordance"][None]
            out.append(f"  {name:15} log loss {reading['log loss']:.4f}, within-task concordance "
                       f"{'n/a' if pooled is None else f'{pooled:.4f}'} ({per_task})")  # fmt: skip
    out.append("train items that every scored config fails: " + ", ".join(
        f"{task} {n} of {total}" for task, (n, total) in side.every_config_fails.items()))  # fmt: skip
    return "\n".join(out)


def write_train_side(side: TrainSide) -> None:
    trained.SETTINGS_FILE.write_text(json_text(side.settings))
    F_TRAIN_WEIGHTS.write_text(f_train_weights_text(side.f_train))


# ---------------------------------------------------------------------------------------------------------------
# The fold fit (item 15). Written and checked before the freeze, run in the fit commit.


def fit_fold_models(
    train_x: np.ndarray, train_y: Mapping[str, np.ndarray], dev_x: np.ndarray, dev_y: Mapping[str, np.ndarray],
    dev_folds: np.ndarray, c: Mapping[str, float],
) -> tuple[list[dict[str, Head]], list[int]]:  # fmt: skip
    """Five fold models: fold model k learns from all of train and the dev rows outside fold k, each config's head at
    its own C (item 7). Returns the models and each one's number of training rows."""
    models, rows = [], []
    for k in range(N_FOLDS):
        keep = dev_folds != k
        x = np.vstack([train_x, dev_x[keep]])
        models.append({config: fit_head(x, np.concatenate([train_y[config], dev_y[config][keep]]), c[config])
                       for config in SCORED})  # fmt: skip
        rows.append(len(x))
    return models, rows


def fit_folds() -> None:
    """The fit commit's command. Reads dev outcomes and prints nothing about them (item 15)."""
    settings = trained.read_settings()
    matrix = results_matrix.load()
    inputs = {}
    for split in ("train", "dev"):
        rows = read_rows(split=split)
        ids = [row["item_id"] for row in rows]
        cache = read_cache(split)
        x = feature_matrix(settings, cache.embeddings, [from_row(row) for row in rows])
        cells = matrix.split(split).cells
        y = {config: np.array([cells[(i, config)].success for i in ids]) for config in SCORED}
        inputs[split] = (ids, x, y)
    dev_fold_of = read_folds("dev")
    dev_folds = np.array([dev_fold_of[i] for i in inputs["dev"][0]])
    models, n_rows = fit_fold_models(inputs["train"][1], inputs["train"][2], inputs["dev"][1], inputs["dev"][2], dev_folds,
                                   settings.c)  # fmt: skip
    text = fold_weights_text(models)
    trained.FOLD_WEIGHTS_FILE.write_text(text)
    for k, n in enumerate(n_rows):
        print(f"fold model {k}: {n} training rows")
    print(
        f"wrote {trained.FOLD_WEIGHTS_FILE.relative_to(PROJECT_ROOT)}, sha256 {hashlib.sha256(text.encode()).hexdigest()}"
    )


def main(argv: list[str]) -> int:
    commands = {"caches": build_caches, "folds": fit_folds}
    if argv == ["settings"]:
        side = fit_train_side()
        write_train_side(side)
        print(report(side))
        return 0
    if len(argv) != 1 or argv[0] not in commands:
        print("usage: python -m analysis.fit_trained_router caches | settings | folds")
        return 2
    commands[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
