"""Fit the bolstered Jev router, G+, and measure Jev's noise (PLAN_v4.md §8 3c+; Amendment A15, items 7, 10 and 19;
Amendment A11, items 7 and 8; Amendment A12, item 1; GitHub issue #30's formats of the weights file and the stored
record).

Run from the project root:  uv run python -m analysis.fit_jev_router [--overwrite]

It reads, from the committed `routers/jev_settings.json`, train's benchmark rows, train's committed folds, train's
run-0 outcomes (through `matrix.load_train`, so dev's cells never enter memory, A12 item 1), and the stored Jev answers
under `results/jev/` for train: `jev_bolstered_train.jsonl` (G+'s) and `jev_train.jsonl` (G's), each with the noise
sample's second pass (`pass` 2). It never reads a dev or test answer. Then it:

1. **Fit rows (item 7):** train items outside the past requests' source groups, less the items Jev refused on G+'s
   call (item 9). Both exclusions are counted and printed. Pass 1 of each stored record is the item's answer.
2. **Standardizes** the 16 features (`routers.jev.g_plus_features`, the deployable router's own code, so fit and
   scorer can't differ) with means and standard deviations fitted once on the fit rows.
3. **Chooses C** per config, by five-fold cross-validation over train's committed folds, by mean log loss, from the
   13 log-spaced values 1e-4 to 1e2. A C at either end is kept and reported, and the grid is never widened.
4. **Fits** one logistic regression per config (A11 item 7's settings: an L2 penalty, an unpenalized intercept, no
   class weights, `newton-cholesky` at tolerance 1e-10 and at most 100 iterations; a fit that doesn't converge is
   refused) and writes `routers/jev_weights.json`.
5. **The train signal check (item 19):** per config, cross-validated log loss and within-task concordance (pairs
   within one task and one train fold) of an intercept-only model, the task-rates model, G's raw router scores (log
   loss clipped to [0.005, 0.995] for this diagnostic only: a raw 1.00 on a failed item is infinite) and G+.
6. **D6 and the shift (item 10):** per router and config, the mean absolute difference and the mean signed difference
   (pass 2 less pass 1) between the noise sample's two passes' router scores. G+'s go through the fitted weights.

It prints what A15 item 19 records (the counts, each C and whether it sits at an end, the signal check and D6).
The weights file is not overwritten unless `--overwrite` is passed.

The model, the C choice and the concordance follow `analysis/fit_trained_router.py` (F's recipe, A11 items 7 and 12),
written again here so this script doesn't import torch.
"""

import hashlib
import json
import sys
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from analysis import build_jev_settings, collect_jev
from analysis.folds import N_FOLDS, read_folds
from routers import jev, jev_call
from routers.jev import (
    FEATURE_COUNT,
    GPlusWeights,
    GScores,
    Parsed,
    feature_names,
    g_plus_features,
    g_plus_scores,
    g_scores,
    parse_answers,
    parse_weights,
)
from routers.ladder import SCORED
from routers.request import from_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORD_DIR = PROJECT_ROOT / "results" / "jev"  # #36 writes <router>_<split>.jsonl here (not committed)
WEIGHTS_FILE = jev.WEIGHTS_FILE

GRID = tuple(float(c) for c in np.logspace(-4, 2, 13))  # 13 log-spaced values; never widened (A11 item 7)
SOLVER: dict[str, Any] = {"name": "newton-cholesky", "tol": 1e-10, "max_iter": 100}  # A11 item 7

G = "G"
G_PLUS = "G+"
SIGNAL_MODELS = ("intercept only", "task rates", G, G_PLUS)  # item 19's order


class JevFitError(ValueError):
    """An input or a fit the script refuses whole: a fit that didn't converge, a missing or repeated stored answer."""


# ---------------------------------------------------------------------------------------------------------------
# One logistic head (A11 item 7)


def fit_model(x: np.ndarray, y: np.ndarray, c: float, max_iter: int = int(SOLVER["max_iter"])) -> LogisticRegression:
    """L2 logistic regression, an unpenalized intercept, no class weights, scikit-learn's newton-cholesky at
    tolerance 1e-10. A fit that doesn't converge in `max_iter` iterations is refused (JevFitError), whole."""
    if len(set(y.tolist())) != 2:
        raise JevFitError("a head needs both outcomes among its rows")
    model = LogisticRegression(C=c, solver=str(SOLVER["name"]), tol=float(SOLVER["tol"]), max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as warning:
            raise JevFitError(f"the fit at C = {c} didn't converge in {max_iter} iterations: {warning}") from None
    return model


def cross_validated(x: np.ndarray, y: np.ndarray, folds: np.ndarray, c: float) -> np.ndarray:
    """Each row's success estimate from a head fitted on the other four train folds."""
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
    A C at either end is kept and reported, and the grid is not widened (A15 item 7)."""
    losses = tuple(mean_fold_loss(cross_validated(x, y, folds, c), y, folds) for c in GRID)
    best = int(np.argmin(losses))
    return ChosenC(GRID[best], losses, "lower end" if best == 0 else "upper end" if best == len(GRID) - 1 else "")


# ---------------------------------------------------------------------------------------------------------------
# The train signal check (A15 items 7 and 19)


def concordance(scores: np.ndarray, y: np.ndarray, cells: Sequence[Hashable], tasks: Sequence[str]) -> dict:
    """Share of (success, failure) pairs within one cell whose scores are in the right order, ties counting half.

    Pooled (key None) and per task; None where there is no pair. Pairs across cells never count (A11 item 12): a cell
    is a task and a train fold.
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


def signal_check(
    z: np.ndarray, y: np.ndarray, folds: np.ndarray, tasks: np.ndarray, c: float, g_raw: np.ndarray,
    clip: tuple[float, float] = (0.005, 0.995),
) -> dict:  # fmt: skip
    """One config's cross-validated log loss and within-task concordance for an intercept-only model, the task-rates
    model, G's raw router scores and G+ (a head at C on the standardized features `z`).

    Rates and the head come from the other four train folds. G's raw scores are clipped to `clip` for log loss only,
    a diagnostic's clip: concordance reads them raw.
    """
    base, rates = np.zeros(len(y)), np.zeros(len(y))
    for k in range(N_FOLDS):
        held, rest = folds == k, folds != k
        base[held] = y[rest].mean()
        for task in set(tasks[held].tolist()):
            rates[held & (tasks == task)] = y[rest & (tasks == task)].mean()
    models = {"intercept only": base, "task rates": rates, G: g_raw, G_PLUS: cross_validated(z, y, folds, c)}
    cells = list(zip(tasks.tolist(), folds.tolist(), strict=True))
    out = {}
    for name, p in models.items():
        loss_scores = np.clip(p, *clip) if name == G else p
        out[name] = {"log loss": mean_fold_loss(loss_scores, y, folds),
                     "concordance": concordance(p, y, cells, tasks.tolist())}  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------------------------
# D6 and the shift (A15 item 10)


@dataclass(frozen=True)
class Noise:
    d6: float  # mean absolute difference between the two passes' router scores
    shift: float  # mean signed difference, pass 2 less pass 1


def noise_difference(
    first: Mapping[str, Mapping[str, float]], second: Mapping[str, Mapping[str, float]], configs: Sequence[str]
) -> dict[str, Noise]:
    """Per config, over the items both passes scored: D6 and the shift between router scores by item and config."""
    if set(first) != set(second) or not first:
        raise JevFitError("the two passes must score the same, non-empty set of items")
    out = {}
    for config in configs:
        diffs = np.array([second[item][config] - first[item][config] for item in first])
        out[config] = Noise(float(np.abs(diffs).mean()), float(diffs.mean()))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Stored records (issue #30's format) and the answers in them


G_PLUS_RECORDS, G_RECORDS = "jev_bolstered", "jev"  # the records' router names (A15 item 9)


def read_records(path: Path, router: str, split: str) -> list[dict]:
    """The records of one JSON Lines file, through the collection script's own reader (so the two can't drift), with
    its refusal made the fit's own: a file that isn't `router`'s and `split`'s whole records is refused."""
    try:
        return collect_jev.read_records_file(path, router, split)
    except collect_jev.CollectError as error:
        raise JevFitError(str(error)) from error


Questions = Mapping[str, Mapping[str, object]]


def agreed(record: Mapping[str, Any], refused: bool) -> None:
    """The stored refusal flag and the parser's reading of the stored response must agree (item 9)."""
    if record["refused"] != refused:
        raise JevFitError(
            f"{record['item_id']} pass {record['pass']}: the stored refusal flag is {record['refused']}, but the "
            f"parser reads the stored response as refused = {refused}"
        )


def parsed_by_pass(records: Sequence[Mapping[str, Any]], questions: Questions) -> dict[int, dict[str, Parsed]]:
    """G+'s stored responses read by `routers/jev.py`'s own parser against the questions asked, by pass and item.

    The fit reads answers through the router's parser, as the tables do (A15 item 12), so it can't drift from it."""
    passes: dict[int, dict[str, Parsed]] = {1: {}, 2: {}}
    for record in records:
        parsed = parse_answers(record["response"], questions)
        agreed(record, parsed.refused)
        passes[record["pass"]][record["item_id"]] = parsed
    return passes


def g_scores_by_pass(records: Sequence[Mapping[str, Any]], questions: Questions) -> dict[int, dict[str, GScores]]:
    """G's router scores by pass and item: the router's own `g_scores` on each stored response (0 if refused)."""
    passes: dict[int, dict[str, GScores]] = {1: {}, 2: {}}
    for record in records:
        scored = g_scores(record["response"], questions)
        agreed(record, scored.refused)
        passes[record["pass"]][record["item_id"]] = scored
    return passes


# ---------------------------------------------------------------------------------------------------------------
# The fit


@dataclass(frozen=True)
class Counts:
    train: int  # train items
    excluded: int  # of them, in the past requests' source groups
    refused: int  # of the rest, those Jev refused on G+'s call
    g_refused: int  # of the fit rows, those Jev refused on G's call (kept: G scores 0)
    fit: int  # the fit rows


@dataclass(frozen=True)
class FitResult:
    counts: Counts
    item_ids: tuple[str, ...]  # the fit rows, in ID order
    labels: dict[str, np.ndarray]  # config -> run-0 success, 0 or 1, per fit row
    chosen: dict[str, ChosenC]
    models: dict[str, LogisticRegression]  # fitted on the fit rows' standardized features, for a check
    weights: dict[str, Any]  # the weights file's contents
    signal: dict[str, dict]  # config -> signal_check
    noise: dict[str, dict[str, Noise]]  # router ("G", "G+") -> config -> D6 and shift


def standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Means and standard deviations of the fit rows' features (item 7)."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if not (scale > 0).all():
        raise JevFitError("a feature is constant on the fit rows, so it can't be standardized")
    return mean, scale


def weights_data(features: Sequence[str], mean: np.ndarray, scale: np.ndarray, chosen: Mapping[str, ChosenC],
                 models: Mapping[str, LogisticRegression]) -> dict[str, Any]:  # fmt: skip
    """The weights file's contents, in issue #30's format."""
    return {
        "features": list(features),
        "standardizer": {"mean": mean.tolist(), "scale": scale.tolist()},
        "heads": {config: {"c": chosen[config].c, "coef": models[config].coef_[0].tolist(),
                           "intercept": float(models[config].intercept_[0])} for config in SCORED},
    }  # fmt: skip


def weights_text(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=1) + "\n"


def write_weights(text: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise JevFitError(f"{path.name} exists; pass --overwrite to replace it")
    path.write_text(text, encoding="utf-8")


def fit_from_files(
    settings_path: Path = build_jev_settings.SETTINGS_FILE,
    records_dir: Path = RECORD_DIR,
    benchmark_file: Path | None = None,
    store: Path | None = None,
    fold_of: Mapping[str, int] | None = None,
) -> FitResult:
    """G+'s fit, signal check and D6 from the settings file, train's rows and run-0 outcomes (train's cells alone),
    train's folds and the stored train answers. `benchmark_file`, `store` and `fold_of` are for a check to feed in."""
    settings_data = json.loads(settings_path.read_text(encoding="utf-8"))
    settings = jev_call.parse_settings(settings_data)
    rows, success = build_jev_settings.read_train(benchmark_file, store)
    folds_by_item = read_folds("train") if fold_of is None else fold_of
    plus_records = read_records(records_dir / "jev_bolstered_train.jsonl", G_PLUS_RECORDS, "train")
    plus_passes = parsed_by_pass(plus_records, jev_call.g_plus_questions(settings))
    g_passes = g_scores_by_pass(read_records(records_dir / "jev_train.jsonl", G_RECORDS, "train"),
                                jev_call.g_questions(settings))  # fmt: skip

    excluded = build_jev_settings.excluded_groups(settings_data, rows)
    outside = [r for r in rows if r["source_group"] not in excluded]
    for what, passes in (("G+'s", plus_passes), ("G's", g_passes)):
        if lacking := [r["item_id"] for r in outside if r["item_id"] not in passes[1]]:
            raise JevFitError(
                f"no stored pass-1 answer on {what} call for {len(lacking)} fit rows, such as {lacking[0]}"
            )
    kept = sorted((r for r in outside if not plus_passes[1][r["item_id"]].refused), key=lambda r: r["item_id"])
    # A refusal on G's call scores 0 there but keeps the row: G is only a column of the signal check.
    g_refused = sum(g_passes[1][r["item_id"]].refused for r in kept)
    counts = Counts(len(rows), len(rows) - len(outside), len(outside) - len(kept), g_refused, len(kept))

    # The deployable router's own features, so the fit and the scorer can't differ (item 7); the placeholder weights
    # only carry the feature order, which the standardizer and heads then follow.
    names = feature_names(settings)
    order = GPlusWeights(names, np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), {})
    x = np.array([g_plus_features(plus_passes[1][r["item_id"]].answers, from_row(r), settings, order) for r in kept])
    mean, scale = standardizer(x)
    z = (x - mean) / scale
    try:
        folds = np.array([folds_by_item[r["item_id"]] for r in kept])
    except KeyError as missing:
        raise JevFitError(f"item {missing} is not in train's fold file") from None
    tasks = np.array([r["source"] for r in kept])
    labels = {c: np.array([int(success[(r["item_id"], c)]) for r in kept]) for c in SCORED}
    g_raw = {c: np.array([g_passes[1][r["item_id"]].scores[c] for r in kept]) for c in SCORED}

    chosen = {c: choose_c(z, labels[c], folds) for c in SCORED}
    models = {c: fit_model(z, labels[c], chosen[c].c) for c in SCORED}
    signal = {c: signal_check(z, labels[c], folds, tasks, chosen[c].c, g_raw[c], settings.g_plus.clip)
              for c in SCORED}  # fmt: skip
    data = weights_data(names, mean, scale, chosen, models)

    # D6 (item 10): the noise sample's two passes, G+'s router scores through the fitted weights.
    fitted = parse_weights(json.loads(weights_text(data)), settings)
    row_of = {r["item_id"]: r for r in rows}
    sample = settings_data["noise_sample"]["item_ids"]
    for what, passes in (("G+'s", plus_passes), ("G's", g_passes)):
        for number in (1, 2):
            if lacking := [i for i in sample if i not in passes[number]]:
                raise JevFitError(f"no stored pass-{number} answer on {what} call for {len(lacking)} noise-sample "
                                  f"items, such as {lacking[0]}")  # fmt: skip

    def plus_score(item: str, number: int) -> dict[str, float]:
        a = plus_passes[number][item]
        return g_plus_scores(a.answers, a.refused, from_row(row_of[item]), settings, fitted)

    noise = {
        G: noise_difference({i: g_passes[1][i].scores for i in sample}, {i: g_passes[2][i].scores for i in sample}, SCORED),
        G_PLUS: noise_difference({i: plus_score(i, 1) for i in sample}, {i: plus_score(i, 2) for i in sample}, SCORED),
    }  # fmt: skip
    return FitResult(counts, tuple(r["item_id"] for r in kept), labels, chosen, models, data, signal, noise)


# ---------------------------------------------------------------------------------------------------------------
# What item 19 records


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def report(result: FitResult) -> str:
    """The counts, each C and whether it sits at an end, the signal check and D6, as item 19 records them."""
    n = result.counts
    lines = [f"train items {n.train}; in the past requests' groups {n.excluded}; refused on G+'s call {n.refused}; "
             f"fit rows {n.fit} (refused on G's call among them {n.g_refused})"]  # fmt: skip
    for config in SCORED:
        chosen = result.chosen[config]
        end = f" ({chosen.at_end} of the grid)" if chosen.at_end else ""
        lines += ["", f"{config}: C = {chosen.c:g}{end}; fit-row success rate {result.labels[config].mean():.4f}",
                  f"  {'model':16} {'log loss':>9} {'concordance':>12}  by task"]  # fmt: skip
        for name, row in result.signal[config].items():
            by_task = "  ".join(f"{task} {fmt(v)}" for task, v in row["concordance"].items() if task is not None)
            lines.append(f"  {name:16} {fmt(row['log loss']):>9} {fmt(row['concordance'][None]):>12}  {by_task}")
    lines += ["", "noise sample, pass 2 less pass 1: D6 (mean absolute difference), shift (mean signed difference)"]
    for router, by_config in result.noise.items():
        lines.append(
            f"  {router:3} " + "   ".join(f"{c} D6 {v.d6:.4f} shift {v.shift:+.4f}" for c, v in by_config.items())
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if WEIGHTS_FILE.exists() and "--overwrite" not in argv:
        print(f"{WEIGHTS_FILE.name} exists; pass --overwrite to refit and replace it")
        return 1
    result = fit_from_files()
    write_weights(weights_text(result.weights), WEIGHTS_FILE, overwrite=True)
    print(report(result))
    text = WEIGHTS_FILE.read_bytes()
    print(f"\nwrote {WEIGHTS_FILE.relative_to(PROJECT_ROOT)} (sha256 {hashlib.sha256(text).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
