"""Check the learning curve's numbers (PLAN_v4.md §8 3b; Amendment A11, items 12 and 15).

Run from the project root:  uv run python -m analysis.check_learning_curve

No walk is run at any size and nothing is written to the operating-points file. Never a test item.

- the nested subsamples, on hand-made rows: sizes, whole source groups, stratified by task and by split, nested,
  seeded, and refusals;
- the refits, on synthetic rows: each fold's rows per size worked out by hand, the 100% point equal to scikit-learn's
  own fit on all the fold model's rows, a fold model never reads its own fold's labels, and a smaller size differs;
- the readings, on four hand-worked items: log loss, concordance, the headline difference, F-train's point;
- the module reads no walk and no operating-points file;
- on dev (the hook only: it needs the embedding caches and the results matrix, about 30 s): the 100% point equals F's
  out-of-fold router scores exactly, and so do its readings; the fold models' training rows are item 20's at 100%;
  the sizes are nested quarters with the mix of train and dev rows kept; the headline is the readings' difference;
  the operating-points file is as it was. Nothing printed about an outcome beyond pass or fail.
"""

import ast
import hashlib
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from analysis import fit_trained_router as fit
from analysis import learning_curve as lc
from analysis.bootstrap import Bootstrap
from analysis.folds import N_FOLDS
from analysis.router_metrics import PairedEstimate
from routers import trained
from routers.ladder import SCORED

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refuses(evaluate: Callable[[], object], what: str, says: str = "") -> None:
    try:
        evaluate()
    except lc.CurveError as refusal:
        check(says in str(refusal), f"{what} ({refusal})")
    else:
        check(False, what)


MATRIX_FILE = Path(__file__).resolve().parent.parent / "results" / "dev_matrix.sqlite"


def imported_modules(source: str) -> set[str]:
    """Every module a source file imports, and every `from module import name` as module.name too."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found |= {node.module, *(f"{node.module}.{alias.name}" for alias in node.names)}
    return found


def is_less(got: PairedEstimate, larger: PairedEstimate, smaller: PairedEstimate) -> bool:
    """`got` is the larger reading's value less the smaller's."""
    if got.value is None or larger.value is None or smaller.value is None:
        return False
    return math.isclose(got.value, larger.value - smaller.value, abs_tol=1e-12)


def real_data_checks() -> None:
    """Dev: the hook only. Prints nothing about an outcome beyond pass or fail."""
    from analysis import trained_table
    from analysis.benchmark import read_rows
    from analysis.bootstrap import resample
    from analysis.folds import read_folds
    from analysis.operating_points import OPERATING_POINTS_FILE
    from analysis.plan import PLAN, amendment_item
    from analysis.router_metrics import log_loss, within_task_concordance

    print("\n-- on dev (the hook only) --")
    before = OPERATING_POINTS_FILE.read_bytes()
    curve, refits = lc.build()
    check(OPERATING_POINTS_FILE.read_bytes() == before, "the operating-points file is as it was after the refits")
    rows = read_rows(split="dev")
    ids = [row["item_id"] for row in rows]
    table = {e.item_id: e.router_scores for e in trained_table.dev_table().items}
    equal = all(np.array_equal(refits.scores[1.0][c], np.array([table[i][c] for i in ids])) for c in SCORED)
    check(equal, "the 100% point reproduces F's out-of-fold router scores exactly (every dev item, every config)")
    one_off = np.array([table[i]["haiku"] for i in ids])
    one_off[0] = np.nextafter(one_off[0], 1.0)
    check(not np.array_equal(refits.scores[1.0]["haiku"], one_off),
          "known positive: one router score off by a last bit is not equal")  # fmt: skip
    check(not np.array_equal(refits.scores[0.75]["haiku"], refits.scores[1.0]["haiku"]), "75% is not F")
    record = amendment_item(PLAN.read_text(), "A11", 20)
    recorded = tuple(int(n) for _, n in re.findall(r"fold model (\d), (\d+)", record))
    check(refits.rows[1.0] == recorded, f"the 100% point trains on item 20's rows: {refits.rows[1.0]}")
    check(all(abs(n - size * full) <= 0.02 * full for size, n in ((s, sum(r) / len(r)) for s, r in refits.rows.items())
              for full in [sum(refits.rows[1.0]) / N_FOLDS]),
          "each size trains on its share of the rows, within 2%")  # fmt: skip
    check(all(all(a <= b for a, b in zip(refits.rows[x], refits.rows[y], strict=True))
              for x, y in zip(list(refits.rows)[:-1], list(refits.rows)[1:], strict=True)),
          "each fold model trains on more rows at each size")  # fmt: skip

    # Readings: F's own, from the router-score table through the metrics module, are the 100% point's.
    from analysis import matrix as results_matrix

    dev = results_matrix.load().split("dev")
    draws = resample(dev)
    fold_of = read_folds("dev")
    cells = {i: (dev.items[i].task, fold_of[i]) for i in ids}
    for config in SCORED:
        success = {i: dev.cells[(i, config)].success for i in ids}
        scores = {i: table[i][config] for i in ids}
        direct_loss, direct_conc = (
            log_loss(draws, scores, success),
            within_task_concordance(draws, scores, success, cells),
        )
        point = curve.points[config][1.0]
        check(point.log_loss.pooled == direct_loss.pooled and point.concordance.pooled == direct_conc.pooled
              and point.log_loss.per_task == direct_loss.per_task,
              f"{config}: the 100% point's readings equal F's, with their intervals")  # fmt: skip
        step, low = curve.headline[config], curve.points[config][0.75]
        check(
            is_less(step.log_loss.pooled, point.log_loss.pooled, low.log_loss.pooled)
            and is_less(step.concordance.pooled, point.concordance.pooled, low.concordance.pooled),
            f"{config}: the headline is the 100% reading less the 75% reading",
        )
    f_train = {e.item_id: e.router_scores for e in trained_table.f_train_table().items}
    check(all(curve.f_train[c].labels == len(read_rows(split="train")) for c in SCORED) and len(f_train) == len(ids),
          "F-train's point is the one model refit on train's rows, read on the same dev items")  # fmt: skip


def units_of(layout: dict[tuple[str, str], list[int]]) -> list[lc.Unit]:
    """Rows for each (task, split) stratum: a list of source-group sizes, each group named for its stratum."""
    return [lc.Unit(task, split, f"{task}-{split}-g{g}")
            for (task, split), sizes in layout.items() for g, size in enumerate(sizes) for _ in range(size)]  # fmt: skip


def main() -> int:
    print("-- the nested subsamples --")
    # Four strata of single-item groups: 40, 20, 80 and 8 items. 25/50/75/100% of each is a whole number of items,
    # worked out by hand: 10/20/30/40, 5/10/15/20, 20/40/60/80 and 2/4/6/8.
    layout = {("a", "train"): [1] * 40, ("a", "dev"): [1] * 20, ("b", "train"): [1] * 80, ("b", "dev"): [1] * 8}
    units = units_of(layout)
    masks = lc.nested_subsamples(units)
    check(list(masks) == [0.25, 0.5, 0.75, 1.0], "the sizes are 25, 50, 75 and 100%")
    expected = {0.25: [10, 5, 20, 2], 0.5: [20, 10, 40, 4], 0.75: [30, 15, 60, 6], 1.0: [40, 20, 80, 8]}
    for size, want in expected.items():
        got = [int(sum(m for m, u in zip(masks[size], units, strict=True) if (u.task, u.split) == s)) for s in layout]
        check(got == want, f"{size:.0%}: items kept per stratum (a train, a dev, b train, b dev) {got} == {want}")
    check(bool(masks[1.0].all()), "100% keeps every row, so it is the whole training set")
    check(all(bool((masks[a] <= masks[b]).all()) for a, b in zip(list(masks)[:-1], list(masks)[1:], strict=True)),
          "each subsample is inside the next")  # fmt: skip
    again = lc.nested_subsamples(units)
    check(all(np.array_equal(masks[s], again[s]) for s in masks), "seeded: the same rows give the same subsamples")
    other = lc.nested_subsamples(units, seed=7)
    check(not np.array_equal(masks[0.5], other[0.5]), "known positive: another seed draws other rows")
    reversed_rows = lc.nested_subsamples(units[::-1])
    check(all(np.array_equal(masks[s], reversed_rows[s][::-1]) for s in masks),
          "the draw depends on the groups, not on the order the rows come in")  # fmt: skip

    # Groups of three items: 12 groups in a stratum, 36 items. 25% of 36 = 9 items = 3 groups, and so on. Whole groups.
    big = units_of({("a", "train"): [3] * 12, ("a", "dev"): [3] * 4})
    bigger = lc.nested_subsamples(big)
    for size in (0.25, 0.5, 0.75):
        kept: dict[str, int] = {}
        for u, m in zip(big, bigger[size], strict=True):
            kept[u.group] = kept.get(u.group, 0) + int(m)
        check(all(n in (0, 3) for n in kept.values()), f"{size:.0%}: every kept source group is kept whole")
    count = {s: int(bigger[s].sum()) for s in bigger}
    check(count == {0.25: 12, 0.5: 24, 0.75: 36, 1.0: 48}, f"groups of three: items kept {count}")

    # The cut is the prefix of the stratum's hash order closest to the share: sizes 5,1,1,1,1,1 (10 items), the
    # order worked out here from sha256 of the same text the module hashes.
    sizes = [5, 1, 1, 1, 1, 1]
    group_rows = units_of({("a", "train"): sizes})
    order = sorted(range(6), key=lambda g: hashlib.sha256(f"2026:curve:a:train:a-train-g{g}".encode()).hexdigest())
    cumulative = np.cumsum([sizes[g] for g in order])
    target = 0.5 * 10
    want_groups = int(np.argmin(np.abs(cumulative - target))) + 1
    got_items = int(lc.nested_subsamples(group_rows)[0.5].sum())
    check(got_items == int(cumulative[want_groups - 1]),
          f"a group bigger than the cut: the prefix nearest 5 of 10 items is kept ({got_items} items)")  # fmt: skip

    # A stratum with one group keeps it whole at every size, never nothing.
    one = lc.nested_subsamples(units_of({("a", "train"): [4], ("a", "dev"): [2, 2]}))
    check(all(int(one[s][:4].sum()) == 4 for s in one), "a stratum with a single group keeps it at every size")

    refuses(lambda: lc.nested_subsamples([]), "no rows is refused", "no rows")
    refuses(lambda: lc.nested_subsamples(units, sizes=(0.5, 0.25)), "sizes must increase", "increase")
    refuses(lambda: lc.nested_subsamples(units, sizes=(0.5, 0.75)), "the last size must be 100%", "100%")
    refuses(lambda: lc.nested_subsamples([lc.Unit("a", "train", "g"), lc.Unit("b", "train", "g")]),
            "a source group in two tasks is refused", "two")  # fmt: skip

    print("\n-- the refits, on synthetic rows --")
    rng = np.random.default_rng(5)
    width = 7  # six embedding columns and the flag
    settings = fit.provisional_settings([], dict.fromkeys(trained.MODEL_FILES, "0" * 64), width=width - 1)
    settings = trained.parse_settings(fit.settings_data(settings.files, [], settings.mean, settings.scale,
                                                        dict.fromkeys(SCORED, 1.0)))  # fmt: skip
    train_layout = {("a", "train"): [1] * 40, ("b", "train"): [1] * 20}
    dev_layout = {("a", "dev"): [1] * 20, ("b", "dev"): [1] * 10}
    train_units, dev_units = units_of(train_layout), units_of(dev_layout)
    train_x, dev_x = rng.normal(size=(60, width)), rng.normal(size=(30, width))
    # Each dev stratum is dealt over the five folds in turn: four of task a and two of task b in every fold.
    dev_folds = np.array([i % 5 for i in range(20)] + [i % 5 for i in range(10)])

    def labels(x: np.ndarray, shift: float) -> dict[str, np.ndarray]:
        return {c: (x[:, 0] * (k + 1) + rng.normal(size=len(x)) + shift > 0).astype(int) for k, c in enumerate(SCORED)}

    train_y, dev_y = labels(train_x, 0.0), labels(dev_x, 0.0)
    fitted = lc.refit_out_of_fold(settings, (train_x, train_y, train_units), (dev_x, dev_y, dev_units), dev_folds)
    # Fold k trains on train (a 40, b 20) and the dev rows outside k (a 16, b 8): 84 rows, of which a size keeps its
    # share of each stratum: 25% is a 10 + 5 of train and a 4 + 2 of dev.
    check(fitted.rows == {0.25: (21,) * 5, 0.5: (42,) * 5, 0.75: (63,) * 5, 1.0: (84,) * 5},
          f"rows per fold model at 25, 50, 75 and 100%: {fitted.rows}")  # fmt: skip
    worst = 0.0
    for k in range(N_FOLDS):
        held = dev_folds == k
        for c in SCORED:
            model = fit.fit_model(np.vstack([train_x, dev_x[~held]]),
                                  np.concatenate([train_y[c], dev_y[c][~held]]), 1.0)  # fmt: skip
            expected = model.predict_proba(dev_x[held])[:, 1]
            worst = max(worst, float(np.abs(fitted.scores[1.0][c][held] - expected).max()))
    check(worst < 1e-9, f"the 100% point is scikit-learn's own fit on all the fold model's rows ({worst:.1e})")
    check(all(float(np.abs(fitted.scores[0.25][c] - fitted.scores[1.0][c]).max()) > 1e-3 for c in SCORED),
          "known positive: 25% scores differ from 100%")  # fmt: skip
    flipped = {c: np.where(dev_folds == 0, 1 - v, v) for c, v in dev_y.items()}
    retrained = lc.refit_out_of_fold(settings, (train_x, train_y, train_units), (dev_x, flipped, dev_units), dev_folds)
    own = dev_folds == 0
    check(all(np.array_equal(retrained.scores[s][c][own], fitted.scores[s][c][own]) for s in fitted.scores for c in SCORED),
          "a fold model never reads its own fold's labels: flipping them changes nothing it scores")  # fmt: skip
    check(all(not np.array_equal(retrained.scores[1.0][c][~own], fitted.scores[1.0][c][~own]) for c in SCORED),
          "known positive: the other folds' scores do move, since they train on those labels")  # fmt: skip

    print("\n-- the readings, on four hand-worked items --")
    # One task, one cell. Successes 1, 0, 1, 0. At 75% the scores are .9, .6, .4, .2: the successes (.9, .4) against
    # the failures (.6, .2) are right in .9>.6, .9>.2, .4>.2 and wrong in .4<.6, so 3 of 4 pairs. At 100% they are
    # .9, .2, .8, .1: all 4 pairs. Log loss is the mean of -ln(p of the outcome): .9, .4, .4, .8 at 75% and
    # .9, .8, .8, .9 at 100%.
    ids = [f"i{i}" for i in range(4)]
    draws = Bootstrap(tuple(ids), ("t",) * 4, np.array([[1, 1, 1, 1]]))
    outcome = dict(zip(ids, [1, 0, 1, 0], strict=True))
    at75, at100 = [0.9, 0.6, 0.4, 0.2], [0.9, 0.2, 0.8, 0.1]
    refits = lc.Refits({0.75: dict.fromkeys(SCORED, np.array(at75)), 1.0: dict.fromkeys(SCORED, np.array(at100))},
                       {0.75: (3, 4), 1.0: (4, 4)})  # fmt: skip
    f_train_scores = dict(zip(ids, [0.5, 0.5, 0.5, 0.5], strict=True))
    curve = lc.measure(draws, refits, ids, dict.fromkeys(SCORED, outcome), dict.fromkeys(ids, ("t", 0)),
                       (3, dict.fromkeys(SCORED, f_train_scores), dict.fromkeys(ids, "t")))  # fmt: skip
    loss75 = -(math.log(0.9) + math.log(0.4) + math.log(0.4) + math.log(0.8)) / 4
    loss100 = -(math.log(0.9) + math.log(0.8) + math.log(0.8) + math.log(0.9)) / 4
    point75, point100 = curve.points["haiku"][0.75], curve.points["haiku"][1.0]
    check(math.isclose(point75.log_loss.pooled.value or 0, loss75) and math.isclose(point100.log_loss.pooled.value or 0, loss100),
          f"log loss at 75% and 100%: {loss75:.4f} and {loss100:.4f}")  # fmt: skip
    check(point75.concordance.pooled.value == 0.75 and point100.concordance.pooled.value == 1.0,
          "concordance at 75% and 100%: 3 of 4 pairs and 4 of 4")  # fmt: skip
    check(point75.labels == 3.5 and point100.labels == 4.0, "the labels are the fold models' mean training rows")
    step = curve.headline["opus_low"]
    check(math.isclose(step.log_loss.pooled.value or 0, loss100 - loss75) and step.concordance.pooled.value == 0.25,
          f"the headline is 100% less 75%: log loss {loss100 - loss75:+.4f}, concordance +0.25")  # fmt: skip
    f_train = curve.f_train["sonnet"]
    check(f_train.labels == 3.0 and math.isclose(f_train.log_loss.pooled.value or 0, math.log(2))
          and f_train.concordance.pooled.value == 0.5,
          "F-train is a separate point: its own rows, ln 2 for a score of .5, ties counting half")  # fmt: skip
    data = lc.as_data(curve)
    check(data["sizes"] == [0.75, 1.0] and data["headline"]["haiku"]["concordance"]["pooled"]["value"] == 0.25
          and data["points"]["haiku"]["0.75"]["labels"] == 3.5 and set(data["f_train"]) == set(SCORED),
          "as_data holds sizes, points per config and size, F-train and the headline as plain data")  # fmt: skip

    print("\n-- no walk, no operating-points file --")
    banned = {"analysis.walk", "analysis.rules_walk", "analysis.trained_walk", "analysis.operating_points"}
    found = imported_modules(Path(lc.__file__).read_text()) & banned
    check(not found, f"the module imports no walk and no operating-points code ({sorted(found)})")
    check(imported_modules("from analysis.walk import walk\nimport analysis.operating_points") >= {
          "analysis.walk", "analysis.operating_points"}, "known positive: the search finds those imports in a module that has them")  # fmt: skip

    if fit.cache_paths("dev")[0].exists() and fit.cache_paths("train")[0].exists() and MATRIX_FILE.exists():
        real_data_checks()
    else:
        print("\n(the caches or the dev matrix are missing: the dev section is skipped)")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
