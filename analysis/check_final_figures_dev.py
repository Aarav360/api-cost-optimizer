"""Check the figures' code on dev's stored inputs against chart v4's committed JSONs (PLAN_v4.md Amendment A19, items 8 and 13;
issue #77, #90).

Run from the project root:  uv run python -m analysis.check_final_figures_dev

Needs the dev results matrix and the stored dev records in results/ (not committed), so the hook runs it and CI doesn't
(`HOOK_ONLY` in `analysis/check_guardrails.py`). It reads dev only; no router or gate runs and nothing is called.

`analysis/final_figures.py` builds a split's three figures' numbers from the tables the offline half evaluated at the frozen
operating points. Here it builds dev's, in two ways, and each is compared with `figures/pareto.json`, `pareto_tasks.json` and
`pareto_routed.json` as committed, key by key and digit by digit with no tolerance:

- **with dev's walked lines** (chart v4's own, `analysis/cascade_charts.py` and `analysis/router_panels.py`, from dev's gate-score
  table and router tables): every panel of all three JSONs equals the committed one, except the three keys a walk alone has in an
  operating point's row (`margin`, `steps_certified`, `steps_tried`), which a frozen row has no use for. So the same code that
  draws test, given dev's lines, reproduces chart v4 exactly: the references, the rows, the steps, the free checks' point, the
  routed panels;
- **with its own frozen lines**, which test gets: each step is a step of the committed line (the same dial, the same quality and
  cost, to the digit), A is first, the order is strictest first, and a frozen line has a step for each distinct dial of its
  section's points that routes and for no other.

The guard can fail: the same comparison on a copy with a digit changed, a key dropped or a key added finds each.
"""

import copy
import json
import sys
from itertools import pairwise
from typing import Any

from analysis import final_figures, offline_half
from analysis.cascade_charts import Line, lines_of
from analysis.chart_routers import ROUTERS
from analysis.check_final_run_dev import differences
from analysis.operating_points import read_section
from analysis.pareto import FIGURE_DATA, TASK_FIGURE_DATA
from analysis.routed_figure import ROUTED_FIGURE_DATA
from analysis.router_names import HAIKU, JEV_GATE, cascade_section
from analysis.router_panels import step_outcomes
from analysis.simulator import always, simulate
from routers.ladder import FRONTIER

problems: list[str] = []
WALK_ONLY = ("margin", "steps_certified", "steps_tried")  # what a walk has in a point's row and a frozen row does not


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def without_walk_keys(value: Any) -> Any:
    """The committed JSON with the walk's three keys taken out of every row that holds them."""
    if isinstance(value, dict):
        return {k: without_walk_keys(v) for k, v in value.items() if k not in WALK_ONLY}
    if isinstance(value, list):
        return [without_walk_keys(v) for v in value]
    return value


def walked_lines(inputs: offline_half.Inputs, real: Any) -> dict[str, Line]:
    """Chart v4's lines on dev: the fifteen cascades' (`lines_of`) and the four routers' (every step of their tables)."""
    matrix = inputs.matrix
    lines = lines_of(matrix, real.table, real.routers, real.cutoffs)
    frontier = simulate(matrix, always(matrix, FRONTIER))
    for router in ROUTERS:
        steps = step_outcomes(matrix, real.routers[router.name], frontier)
        lines[router.name] = Line(tuple((t, None, tuple(o)) for t, o in steps.items()))
    return lines


def main() -> int:
    print("-- the guard can fail --")
    base: dict[str, Any] = {"panels": {"a": {"x": 0.1, "y": [1, 2]}}}
    digit = copy.deepcopy(base)
    digit["panels"]["a"]["x"] += 2e-16
    dropped = copy.deepcopy(base)
    del dropped["panels"]["a"]["y"]
    added = copy.deepcopy(base)
    added["panels"]["a"]["z"] = 1
    check(all(len(differences(bad, base)) == 1 for bad in (digit, dropped, added)),
          "a digit changed, a key dropped and a key added are each a difference")  # fmt: skip

    print("\n-- dev's figures from dev's stored inputs --")
    from analysis import table_v4

    inputs = offline_half.dev_inputs()
    tables = json.loads(json.dumps(offline_half.build_tables(inputs)))
    real = table_v4.real_inputs()
    committed = {name: json.loads(path.read_text(encoding="utf-8"))
                 for name, path in (("headline", FIGURE_DATA), ("tasks", TASK_FIGURE_DATA), ("routed", ROUTED_FIGURE_DATA))}  # fmt: skip
    walked = final_figures.build(inputs, tables, walked_lines(inputs, real))
    for name in ("headline", "tasks", "routed"):
        built, held = getattr(walked, name), committed[name]
        found = differences(built["panels"], without_walk_keys(held["panels"]))
        check(
            not found, f"{name}: with dev's walked lines, every panel is the committed one, digit for digit {found[:3]}"
        )
        check(all(built[key] == held[key] for key in ("split", "run", "n", "price_snapshot")),
              f"{name}: the split, run, item count and price snapshot are the committed ones: dev, 0, {held['n']}")  # fmt: skip
    check(walked.headline["routers"].keys() == committed["headline"]["routers"].keys()
          and walked.headline["cascades"] == committed["headline"]["cascades"]
          and all(walked.headline["routers"][r][k] == committed["headline"]["routers"][r][k]
                  for r in walked.headline["routers"] for k in ("label", "operating_points_file", "operating_points_sha256")),
          "the routers' and cascades' labels and the operating-points file's section hashes are the committed ones")  # fmt: skip
    broken = copy.deepcopy(walked.headline)
    broken["panels"]["realistic"]["cascades"][cascade_section(HAIKU, JEV_GATE)]["steps"][1]["quality"] += 1e-12
    check(len(differences(broken["panels"], without_walk_keys(committed["headline"]["panels"]))) == 1,
          "known positive: one step's quality nudged by 1e-12 is found")  # fmt: skip

    print("\n-- dev's frozen lines against dev's walked lines --")
    frozen = final_figures.build(inputs, tables)
    routers = {r.name for r in ROUTERS}
    mismatched, extra, unordered, longer = [], [], [], []
    for mix in ("realistic", "uniform"):
        for section in final_figures.policy_sections():
            walked_steps = final_figures.steps_of(walked, mix, section)
            frozen_steps = final_figures.steps_of(frozen, mix, section)
            keyed = {
                (s["threshold"], s.get("cutoff")): s for s in walked_steps[1:]
            }  # A has no dial; the free checks' one step has none either
            if frozen_steps[0] != walked_steps[0] or any(
                keyed.get((s["threshold"], s.get("cutoff"))) != s for s in frozen_steps[1:]
            ):
                mismatched.append(f"{mix} {section}")
            held = read_section(section)
            routes = {(p.get("threshold"), p.get("cutoff")) for m in ("realistic", "uniform") for p in held[m].values()
                      if not p["fell_back"]}  # fmt: skip
            dials = [(s["threshold"], s.get("cutoff")) for s in frozen_steps[1:]]
            expected = routes | ({(None, None)} if section == final_figures.FREE_SECTION else set())
            if set(dials) != expected or len(dials) != len(expected):
                extra.append(f"{mix} {section}")
            if frozen_steps[0]["threshold"] is not None or any(
                final_figures.dial_order(a) < final_figures.dial_order(b) for a, b in pairwise(dials)
            ):
                unordered.append(f"{mix} {section}")
            if len(frozen_steps) > len(walked_steps):
                longer.append(f"{mix} {section}")
    check(not mismatched, f"every frozen step is a committed step of its line, to the digit {mismatched[:3]}")
    check(
        not extra,
        f"each frozen line has one step for each distinct dial of its section that routes, and no other {extra[:3]}",
    )
    check(not unordered, f"each frozen line starts at A and runs strictest first {unordered[:3]}")
    check(not longer, f"a frozen line is never longer than dev's walked one {longer[:3]}")
    check(
        len(routers) == 4 and len(final_figures.policy_sections()) == 19, "all nineteen policies' lines were compared"
    )

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
