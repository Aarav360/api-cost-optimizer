"""Check Stage 5's diagnostics on dev (PLAN_v4.md §8 3b; Amendment A11, items 12, 13 and 15).

Run from the project root:  uv run python -m analysis.check_trained_diagnostics

Needs the dev results matrix and the dev embedding cache, which CI doesn't have, so the hook runs it where they are
present, like check_trained_dev. It reads train and dev only, through the reader and the matrix loader, and never a
test item; it computes nothing a policy could be chosen from.

- the task-rates diagnostic's table, on a synthetic matrix worked out by hand: each fold's rates come from train and
  the other folds' dev labels, never the item's own fold (known positive: the rates with it differ), with the table's
  and the diagnostic's refusals;
- on dev: the table is that of the labels each fold model saw, worked out again from the raw matrix;
  its within-task concordance is exactly 0.5 per config (item 12), while the same rates pooled across folds fall
  below it (the known positive); the walks reproduce from their tables;
- the walks and tables of E, F and F-train equal the committed ones, and a diagnostic writes nothing: the
  operating-points file is byte for byte as it was, holds sections for E and F only, and no diagnostic module names
  the file's writer (known positive: F's walk module does);
- per-task warnings are A7 item 2's, on IFEval and MBPP+ only;
- same-target comparisons: each cell is the operating points' difference and reads "both fall back to A" where both do.
"""

import ast
import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from analysis import rules_table, trained_table
from analysis import trained_diagnostics as diagnostics
from analysis.matrix import Matrix
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    build_section,
    check_sections,
    read_section,
    section_hash,
)
from analysis.pareto import TASK_FIGURE_DATA
from analysis.router_metrics import within_task_concordance
from analysis.router_names import F_TRAIN, NAMED_ROUTERS, TASK_RATES, E, F
from analysis.task_warnings import task_warning
from analysis.trained_diagnostics import (
    COMPARISONS,
    DIAGNOSTICS,
    DiagnosticsError,
    task_rates_table,
)
from analysis.walk import ALL_STEPS_UP_TO
from harness.client import PROJECT_ROOT
from routers.ladder import SCORED

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refuses(error: type[Exception], call: Callable[[], object], what: str, text: str = "") -> None:
    try:
        call()
    except error as raised:
        check(text in str(raised), f"{what}: {raised}")
        return
    check(False, f"{what}: refused")


def synthetic() -> tuple[Matrix, Matrix, dict[str, int]]:
    """Train items t1..t6 and dev items d1..d5 over two tasks and two folds, hand-labelled.

    haiku succeeds on t1 t2 (task a) and t5 (task b), fails t3 t4 t6; on dev it succeeds on d1 d3 d5, fails d2 d4.
    sonnet always succeeds and opus_low never does, so their rates are 1 and 0 whatever the fold.
    """
    train_haiku = {"t1": 1, "t2": 1, "t3": 0, "t4": 0, "t5": 1, "t6": 0}
    dev_haiku = {"d1": 1, "d2": 0, "d3": 1, "d4": 0, "d5": 1}
    task = {"t1": "a", "t2": "a", "t3": "a", "t4": "a", "t5": "b", "t6": "b",
            "d1": "a", "d2": "a", "d3": "a", "d4": "b", "d5": "b"}  # fmt: skip
    rows = []
    for split, haiku in (("train", train_haiku), ("dev", dev_haiku)):
        for item_id, success in haiku.items():
            for config, s in (("haiku", success), ("sonnet", 1), ("opus_low", 0), ("opus", 1)):
                rows.append({"item_id": item_id, "split": split, "task": task[item_id], "source_group": item_id,
                             "config": config, "cost": 0.01, "success": s})  # fmt: skip
    matrix = Matrix.from_rows(rows)
    folds = {"d1": 0, "d2": 0, "d3": 1, "d4": 1, "d5": 0}
    return matrix.split("train"), matrix.split("dev"), folds


def main() -> int:
    print("-- the task-rates diagnostic's table, worked out by hand --")
    train, dev, folds = synthetic()
    table = task_rates_table(train, dev, folds)
    scores = {entry.item_id: entry.router_scores for entry in table.items}
    # Fold 0's items (d1 d2 task a, d5 task b) see train plus fold 1's dev items (d3 task a, d4 task b):
    # task a haiku (t1 t2 t3 t4 d3) = 3 of 5; task b (t5 t6 d4) = 1 of 3.
    # Fold 1's items (d3 task a, d4 task b) see train plus fold 0's (d1 d2 task a, d5 task b):
    # task a (t1 t2 t3 t4 d1 d2) = 3 of 6; task b (t5 t6 d5) = 2 of 3.
    expected = {"d1": 3 / 5, "d2": 3 / 5, "d5": 1 / 3, "d3": 3 / 6, "d4": 2 / 3}
    check(all(np.isclose(scores[i]["haiku"], v, rtol=0, atol=1e-15) for i, v in expected.items()),
          "each dev item's haiku rate is its task's rate over train and the other folds' dev labels")  # fmt: skip
    check(all(scores[i]["sonnet"] == 1.0 and scores[i]["opus_low"] == 0.0 for i in scores),
          "a config that always succeeds has rate 1 and one that never does 0, in every fold")  # fmt: skip
    # Known positive: the rate with the item's own fold left in is another number (task a: 4 of 9 across all labels).
    check(not np.isclose(scores["d1"]["haiku"], 4 / 9), "the rate with the item's own fold included would be 4 of 9")
    check(all(entry.overhead == 0.0 for entry in table.items) and len(table.items) == 5,
          "every overhead is an explicit $0, over all five dev items")  # fmt: skip
    refuses(DiagnosticsError, lambda: task_rates_table(train, dev, {k: v for k, v in folds.items() if k != "d3"}),
            "a dev item with no fold is refused", "d3")  # fmt: skip
    refuses(DiagnosticsError, lambda: task_rates_table(dev, dev, folds), "a dev matrix given as train is refused",
            "train")  # fmt: skip
    refuses(DiagnosticsError, lambda: task_rates_table(train, train, folds), "a train matrix given as dev is refused",
            "dev")  # fmt: skip
    real()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def val(x: float | None) -> float:
    assert x is not None
    return x


def brute_concordance(table, dev, cells) -> dict[str, float]:
    """Pooled within-cell concordance by a plain double loop over item pairs, ties half: the independent count."""
    out = {}
    for c in SCORED:
        won = pairs = 0.0
        for i in dev.items:
            for j in dev.items:
                if cells[i] != cells[j] or dev.cells[(i, c)].success != 1 or dev.cells[(j, c)].success != 0:
                    continue
                x, y = (diagnostics.scores_of(table, c)[k] for k in (i, j))
                won += 1.0 if x > y else 0.5 if x == y else 0.0
                pairs += 1
        out[c] = won / pairs
    return out


def calls_to(path: Path, name: str) -> bool:
    """Whether a module's syntax tree names `name` (a call, an import or an attribute), as the AST sees it."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.alias) and name in (node.name, node.asname):
            return True
    return False


def real() -> None:
    before = hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest()
    result = diagnostics.compute()
    dev, draws, tables = result.dev, result.draws, result.tables
    from analysis import matrix as results_matrix  # noqa: PLC0415 -- the train labels, for the independent rates

    train = results_matrix.load().split("train")

    print("\n-- the task-rates diagnostic on dev --")
    rates = tables[TASK_RATES]
    by_task_train = {
        t: [i for i, it in train.items.items() if it.task == t] for t in {it.task for it in dev.items.values()}
    }
    worst = 0.0
    for entry in rates.items:
        task, fold = dev.items[entry.item_id].task, result.fold_of[entry.item_id]
        seen = by_task_train[task] + [i for i, it in dev.items.items() if it.task == task and result.fold_of[i] != fold]
        for c in SCORED:
            labels = [(train if i in train.items else dev).cells[(i, c)].success for i in seen]
            worst = max(worst, abs(entry.router_scores[c] - sum(labels) / len(labels)))
    check(worst < 1e-12, f"every rate is its task's rate over train and the other four dev folds, worked out again "
                         f"from the matrix (largest gap {worst:.1e})")  # fmt: skip
    check(len({tuple(e.router_scores.values()) for e in rates.items}) <= 4 * 5,
          "a fold has one rate triple per task, so at most 4 tasks x 5 folds distinct triples")  # fmt: skip
    own = {t: [dev.cells[(i, "haiku")].success for i, it in dev.items.items() if it.task == t] for t in by_task_train}
    check(any(abs(e.router_scores["haiku"] - (sum(own[dev.items[e.item_id].task]) + sum(
        train.cells[(i, "haiku")].success for i in by_task_train[dev.items[e.item_id].task])) / (
        len(own[dev.items[e.item_id].task]) + len(by_task_train[dev.items[e.item_id].task]))) > 1e-9
        for e in rates.items), "known positive: the rates differ from the all-dev ones, so folds matter")  # fmt: skip
    distinct, steps = diagnostics.breakpoints(rates)
    check(ALL_STEPS_UP_TO < distinct <= 60 and steps <= 20,
          f"{distinct} distinct breakpoints, up to the 60 of 5 folds x 4 tasks x 3 configs and above 24, so the "
          f"quantile steps apply ({steps} steps)")  # fmt: skip

    print("\n-- within-task concordance and log loss --")
    for name in DIAGNOSTICS:
        cells = (
            diagnostics.cells_by_task(dev)
            if name == F_TRAIN
            else diagnostics.cells_by_task_and_fold(dev, result.fold_of)
        )
        brute = brute_concordance(tables[name], dev, cells)
        check(all(abs(val(result.concordance[name][c].pooled.value) - brute[c]) < 1e-12 for c in SCORED),
              f"{name}: pooled concordance equals a plain double loop over item pairs, per config")  # fmt: skip
    check(all(r.pooled.value == 0.5 and r.pooled.low == r.pooled.high == 0.5
              and all(e.value == 0.5 and e.low == e.high == 0.5 for e in r.per_task.values())
              for r in result.concordance[TASK_RATES].values()),
          "the task-rates diagnostic's concordance is exactly 0.5 per config, pooled and per task, interval included")  # fmt: skip
    pooled_across = {c: within_task_concordance(draws, diagnostics.scores_of(rates, c), diagnostics.successes_of(dev, c),
                                                diagnostics.cells_by_task(dev)).pooled.value for c in SCORED}  # fmt: skip
    check(all(val(v) < 0.5 for v in pooled_across.values()),
          "known positive: the same rates paired across folds fall below 0.5 ("
          + ", ".join(f"{c} {v:.3f}" for c, v in pooled_across.items()) + ")")  # fmt: skip
    loss = result.log_loss[TASK_RATES]
    for c in SCORED:
        p = np.clip(np.array([diagnostics.scores_of(rates, c)[i] for i in dev.items]), 2.2e-16, 1 - 2.2e-16)
        y = np.array([dev.cells[(i, c)].success for i in dev.items])
        by_math = sum(-math.log(a if b == 1 else 1 - a) for a, b in zip(p.tolist(), y.tolist())) / len(y)
        check(abs(val(loss[c].pooled.value) - by_math) < 1e-12 and loss[c].size[None] == 744,
              f"task rates, {c}: out-of-fold log loss {val(loss[c].pooled.value):.4f} equals math.log's mean over 744 items")  # fmt: skip
    check(all(math.isclose(val(result.concordance_gaps[g][c].pooled.value),
                           val(result.concordance[g[0]][c].pooled.value) - val(result.concordance[g[1]][c].pooled.value),
                           abs_tol=1e-12)
              and math.isclose(val(result.log_loss_gaps[g][c].pooled.value),
                               val(result.log_loss[g[0]][c].pooled.value) - val(result.log_loss[g[1]][c].pooled.value),
                               abs_tol=1e-12)
              for g in diagnostics.GAPS for c in SCORED),
          "the paired differences are the readings' differences, F less the task rates and F less F-train")  # fmt: skip

    print("\n-- the walks --")
    again = {
        E: rules_table.dev_table(), F: trained_table.dev_table(), F_TRAIN: trained_table.f_train_table(),
        TASK_RATES: task_rates_table(train, dev, result.fold_of),
    }  # fmt: skip
    for name, table in again.items():
        check(table == tables[name] and diagnostics.dev_walks(dev, table, draws) == result.walks[name],
              f"{name}: a table built again and walked again gives the same table and walks")  # fmt: skip
    for name in (E, F):
        router = NAMED_ROUTERS[name]
        committed = read_section(router)
        check(build_section(result.walks[name]) == committed
              and section_hash(build_section(result.walks[name])) == section_hash(committed),
              f"{name}'s walks here are its committed section, hash {section_hash(committed)[:12]}")  # fmt: skip
    for name in (F_TRAIN, TASK_RATES):
        certified = [w for by in result.walks[name].values() for w in by.values()]
        check(all(w.operating in w.certified and w.steps[0].threshold is None for w in certified),
              f"{name}: every operating point is A or a certified step, each walk starting at A")  # fmt: skip

    print("\n-- nothing is written to the operating-points file --")
    check(hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest() == before,
          "the file is byte for byte as it was before the diagnostics ran")  # fmt: skip
    check_ok = True
    try:
        check_sections(required=(NAMED_ROUTERS[E], NAMED_ROUTERS[F]))
    except ValueError:
        check_ok = False
    check(check_ok, "the allowlist still passes: sections for E and F only, none for F-train or a diagnostic")
    root = PROJECT_ROOT / "analysis"
    check(calls_to(root / "trained_walk.py", "write_section"), "known positive: F's walk module names the writer")
    check(not calls_to(root / "trained_diagnostics.py", "write_section")
          and not calls_to(root / "trained_diagnostics.py", "OPERATING_POINTS_FILE"),
          "the diagnostics module names neither the section writer nor the file")  # fmt: skip

    print("\n-- per-task warnings --")
    counts = {t: sum(1 for it in dev.items.values() if it.task == t) for t in by_task_train}
    committed = json.loads(TASK_FIGURE_DATA.read_text())["panels"]
    check(all(task_warning(t, n) == committed[t]["warning"] for t, n in counts.items())
          and {t for t in counts if task_warning(t, counts[t])} == {"ifeval", "mbppplus"},
          "the warning is the committed per-task figure's A7 item 2 text for IFEval and MBPP+, and none for the others")  # fmt: skip
    check(task_warning("mbppplus", 107) == "Only 107 dev items, below the plan's minimum of a few hundred: indicative"
          and task_warning("banking77", 240) is None,
          "hand-written: MBPP+ reads 'Only 107 dev items...', Banking77 reads nothing")  # fmt: skip

    print("\n-- same-target comparisons --")
    ok_cost = ok_reading = ok_order = True
    n_cells = 0
    for (a, b), by_mix in result.same_target.items():
        for mix, by_share in by_mix.items():
            for share, cell in by_share.items():
                n_cells += 1
                one, other = result.walks[a][mix][share], result.walks[b][mix][share]
                ok_reading &= cell.both_fall_back == (one.fell_back and other.fell_back)
                ok_reading &= (cell.reading == "both fall back to A") == cell.both_fall_back
                ok_reading &= (cell.cost is None) == cell.both_fall_back
                if cell.cost is not None and cell.quality is not None:
                    ok_cost &= math.isclose(cell.cost.value, one.operating.cost_per_1k - other.operating.cost_per_1k,
                                            abs_tol=1e-9)  # fmt: skip
                    ok_cost &= math.isclose(cell.quality.value, one.operating.quality - other.operating.quality,
                                            abs_tol=1e-12)  # fmt: skip
                    ok_order &= cell.cost.low <= cell.cost.high and cell.quality.low <= cell.quality.high
    check(n_cells == len(COMPARISONS) * 2 * 3, f"every comparison has a cell at each mix and target ({n_cells} cells)")
    check(ok_cost, "each cell's cost and quality are the first walk's operating point less the second's")
    check(ok_reading, "a cell reads 'both fall back to A', with no difference, exactly where both walks fell back")
    check(ok_order, "every interval runs from its low to its high")

    print("\n-- the data the chart script reads --")
    data = diagnostics.to_data(result)
    text = json.dumps(data)
    check(json.loads(text) == data, "to_data is plain JSON-ready data (it round-trips through json)")
    f_haiku = data["within_task_concordance"][F]["haiku"]  # type: ignore[index]
    check(f_haiku["pooled"]["value"] == result.concordance[F]["haiku"].pooled.value
          and f_haiku["per_task"]["mbppplus"]["warning"] == task_warning("mbppplus", 107)
          and f_haiku["per_task"]["banking77"]["warning"] is None,
          "F's haiku concordance in the data is the reading, with MBPP+'s warning and none on Banking77")  # fmt: skip
    both = data["same_target"]["F less E"]["realistic"]["100%"]  # type: ignore[index]
    check(both["reading"] == "both fall back to A" and both["cost_per_1k"] is None and both["quality"] is None,
          "a cell where both fall back to A reads so in the data, with no difference")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
