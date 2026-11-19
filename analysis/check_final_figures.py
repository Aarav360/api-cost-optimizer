"""Check test's figures and the explorer's split switch on the rehearsal's made-up world (PLAN_v4.md Amendment A19, items 8 and
13; A17 item 23; issue #77, #90).

Run from the project root:  uv run python -m analysis.check_final_figures

A made-up world with made-up test tables (`analysis/rehearsal_world.py`), a stand-in sandbox and fake Jev and Anthropic clients:
no results/, no Docker, no model snapshot, no Jev or Anthropic call, nothing near the real store. CI runs it. The dev reproduction,
which needs results/, is `analysis/check_final_figures_dev.py`.

- **the lines by hand**: on a four-item matrix with outcomes written out, a policy's frozen line is A and then each distinct dial that
  routes, strictest first (a fall back adds none, however the points are listed), and its mix-weighted and per-task quality and cost
  are the hand-worked numbers; a cascade's line is ordered by cutoff;
- **the rehearsal builds test's figures** in the tables step, in files of their own beside `tables_test.json`: the headline, per-task and
  router-started figures as PNGs and JSONs, named for test, saying test in every caption, with the realistic note counting test
  items. Nothing is written to `figures/`, and chart v4's six dev files are byte for byte what they were (their sha256);
- **a cross is a frozen point**: each routed point's row sits on a step of its line, a fall back is at A, each dial a line holds is a
  dial of the operating-points file's section (read from the file, not from the tables), a line starts at A and runs strictest first,
  and the rows are the tables' own;
- **nothing is walked**: the whole go runs with every walk and every line re-derivation of chart v4 made to raise, and the figure code
  names none of them (a search, with a planted known positive);
- **a part the figures need and the tables lack** refuses before anything is drawn: no file is written;
- **the explorer** built from the rehearsal's test JSONs beside dev's opens on test, holds test's numbers as its JSON holds them, and
  names the split in every readout (`analysis/check_explorer.py`'s searches).
"""

import hashlib
import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from analysis import (
    cascade_charts,
    check_explorer,
    explorer,
    final_figures,
    gate_walk,
    offline_half,
    routed_walk,
    router_panels,
)
from analysis import walk as walk_module
from analysis.chart_routers import ROUTERS
from analysis.check_final_run import rehearsed
from analysis.operating_eval import Fixed, FrozenPoint
from analysis.operating_points import read_section
from analysis.pareto import FIGURES, PANELS
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, NAMED_ROUTERS, G, cascade_section, parse_cascade
from analysis.simulator import Outcome, load_mixes
from harness.client import PROJECT_ROOT

problems: list[str] = []
HAIKU_JEV = cascade_section(HAIKU, JEV_GATE)  # a pure cascade with a paid gate
JEV_FREE = cascade_section(NAMED_ROUTERS[G], FREE_GATE)  # a router-started cascade


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= tolerance


# Chart v4's six dev files as committed when this ticket started (sha256): the test figures are files of their own, so these never move.
DEV_FILES = {
    "pareto.json": "f5a96c4b802d9b019aff4e1c986fbcedd3a1089835232fa7277ce487ae21a30b",
    "pareto.png": "d4c089041f377915cef062f6d5735d399d16b6bb4adc72df4f239b1283d92b03",
    "pareto_tasks.json": "8076a4badc46b0e283dbd9d95b5f0748e0a576571d1465ec0f84e302f74385a1",
    "pareto_tasks.png": "a63716104002e132615e0aae0e3718e8fc8507123edce70a78a35d2cbe32e781",
    "pareto_routed.json": "87ecd4809a9e497e068c74ef91db39fd5b9fc2f38125433d20c19acb7ab86e79",
    "pareto_routed.png": "d079f693ca800606e991a6780c65d1b9bd087f5737b597eeda53d01303a2abde",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figures_folder() -> dict[str, str]:
    """Every file under figures/ and its hash: what a rehearsal must leave as it found it."""
    return {p.name: sha256(p) for p in sorted(FIGURES.iterdir()) if p.is_file()}


# ---- the lines by hand ----------------------------------------------------------------------------------------------

# Four items, two in task a and two in task b, each weighing .25 under the mix {a: .5, b: .5}. A (opus) is right on a1, a2 and b2 and
# wrong on b1, at $0.010 each: quality .75, $10.00 per 1,000. Two dials of a policy:
#   HIGH routes a1 to a cheap config ($0.002): right on a1, a2, b2, wrong on b1; costs $0.002 + 3 x $0.010: quality .75, $8.00 per 1,000.
#   LOW routes a1, a2 and b1 cheaply ($0.002 each) and is wrong on a2 and b1: quality .5, costs $0.016: $4.00 per 1,000.
# In task a on its own: A (1.0, $10.00), HIGH (1.0, $6.00), LOW (.5, $2.00). In task b: A (.5, $10.00), HIGH (.5, $10.00), LOW (.5, $6.00).
ITEMS, TASKS_OF = ("a1", "a2", "b1", "b2"), ("a", "a", "b", "b")
MIX = {"a": 0.5, "b": 0.5}


def outcomes(success: tuple[int, ...], costs: tuple[float, ...]) -> list[Outcome]:
    return [Outcome(i, t, i, c, s, ("opus",)) for i, t, s, c in zip(ITEMS, TASKS_OF, success, costs)]


A = outcomes((1, 1, 0, 1), (0.010, 0.010, 0.010, 0.010))
HIGH = outcomes((1, 1, 0, 1), (0.002, 0.010, 0.010, 0.010))
LOW = outcomes((1, 0, 0, 1), (0.002, 0.002, 0.002, 0.010))


def worked() -> None:
    print("-- a frozen line by hand: A, then each distinct dial that routes, strictest first --")
    fixed = Fixed({"opus": A})
    # Listed with the low dial first and a fall back between: the line is still A, HIGH, LOW.
    fixed.note({
        FrozenPoint("rules", "uniform", 0.97, 0.5, None, False): LOW,
        FrozenPoint("rules", "realistic", 0.99, None, None, True): A,
        FrozenPoint("rules", "realistic", 1.0, 0.9, None, False): HIGH,
        FrozenPoint("rules", "uniform", 0.99, 0.9, None, False): HIGH,
        FrozenPoint(HAIKU_JEV, "realistic", 1.0, None, 0.6, False): LOW,
        FrozenPoint(HAIKU_JEV, "uniform", 0.97, None, 0.8, False): HIGH,
        FrozenPoint(HAIKU_JEV, "uniform", 0.99, None, None, True): A,
    })  # fmt: skip
    inputs = SimpleNamespace(cache={"fixed": fixed}, matrix=None, extras={})
    lines = final_figures.frozen_lines(inputs, ["rules", HAIKU_JEV])  # type: ignore[arg-type]
    rules, jev = lines["rules"], lines[HAIKU_JEV]
    check([(t, c) for t, c, _ in rules.steps] == [(None, None), (0.9, None), (0.5, None)],
          "a router's line is A, its dial 0.9 and its dial 0.5 (highest threshold first), with no step for the fall back")  # fmt: skip
    check([(t, c) for t, c, _ in jev.steps] == [(None, None), (None, 0.8), (None, 0.6)],
          "a pure cascade's line is ordered by cutoff, highest first: A, 0.8, 0.6")  # fmt: skip
    check(
        list(rules.steps[0][2]) == A and list(rules.steps[1][2]) == HIGH,
        "A's step holds A's outcomes, and a dial's its own",
    )
    part = final_figures.router_mix_part(rules, MIX, {})
    wanted = [(None, 0.75, 10.0), (0.9, 0.75, 8.0), (0.5, 0.5, 4.0)]
    check(all(s["threshold"] == t and close(s["quality"], q) and close(s["cost_per_1k"], c)
              for s, (t, q, c) in zip(part["steps"], wanted)) and len(part["steps"]) == 3,
          "under the mix the steps are (.75, $10.00), (.75, $8.00) and (.5, $4.00), as worked by hand")  # fmt: skip
    task_a = final_figures.router_task_part(rules, "a", {})["steps"]
    task_b = final_figures.router_task_part(rules, "b", {})["steps"]
    check(all(close(s["quality"], q) and close(s["cost_per_1k"], c)
              for s, (q, c) in zip(task_a, [(1.0, 10.0), (1.0, 6.0), (0.5, 2.0)]))
          and all(close(s["quality"], q) and close(s["cost_per_1k"], c)
                  for s, (q, c) in zip(task_b, [(0.5, 10.0), (0.5, 10.0), (0.5, 6.0)])),
          "within task a the steps are (1.0, $10.00), (1.0, $6.00) and (.5, $2.00); within task b (.5, $10.00), (.5, $10.00) and (.5, $6.00)")  # fmt: skip
    only_fell_back = Fixed({"opus": A})
    only_fell_back.note({FrozenPoint("jev", "realistic", 1.0, None, None, True): A})
    bare = final_figures.frozen_lines(SimpleNamespace(cache={"fixed": only_fell_back}, matrix=None, extras={}), ["jev"])  # type: ignore[arg-type]
    check([(t, c) for t, c, _ in bare["jev"].steps] == [(None, None)],
          "a policy whose every point fell back has a line of A alone: nothing is evaluated that the file did not freeze")  # fmt: skip


def refusals() -> None:
    print("\n-- a part the figures need and the tables lack refuses before anything is drawn --")
    sections = final_figures.policy_sections()
    parts: dict[str, dict] = {key: {} for key in final_figures.required_parts(sections)}
    final_figures.check_parts(parts, sections)
    for gone in ("rules", JEV_FREE, f"{JEV_FREE}_tasks", "references"):
        short = {k: v for k, v in parts.items() if k != gone}
        try:
            final_figures.check_parts(short, sections)
            ok = False
        except final_figures.FigureError as error:
            ok = gone in str(error)
        check(ok, f"the tables without {gone!r} are refused, naming it")
    check(len(sections) == 19 and sections[:4] == [r.name for r in ROUTERS] and sections[4:] == list(CASCADES),
          "the figures read the four routers and the fifteen cascades, each by its part's key, with no line per policy")  # fmt: skip


# ---- the rehearsal --------------------------------------------------------------------------------------------------

WALKS = (
    (cascade_charts, "line_of"),
    (cascade_charts, "pure_candidates"),
    (cascade_charts, "routed_candidates"),
    (cascade_charts, "lines_of"),
    (router_panels, "step_outcomes"),
    (gate_walk, "pure_candidates"),
    (routed_walk, "routed_candidates"),
    (walk_module, "walk_candidates"),
)  # what chart v4 uses to re-derive a line's steps from the split's own tables, and the walks themselves


def walked(*_: object, **__: object) -> None:
    raise AssertionError("a walk or a line re-derivation ran on test's tables")


def png_ok(path: Path) -> bool:
    data = path.read_bytes()
    return data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000


def rehearsal() -> None:
    print("\n-- the rehearsal builds test's figures in the tables step, with nothing walked --")
    names = final_figures.names_of("test")
    before = figures_folder()
    with rehearsed() as r, ExitStack() as stack:
        for module, name in WALKS:
            stack.enter_context(mock.patch.object(module, name, walked))
        positive = False
        try:
            cascade_charts.line_of(None, [], [])  # type: ignore[arg-type]
        except AssertionError:
            positive = True
        check(positive, "known positive: with the walks patched, a call to one of chart v4's line builders raises")
        outcome, _, err = r.go()
        out = r.world.output
        check(
            outcome.stopped_at is None,
            f"the whole made-up run goes through with every walk made to raise {err[-200:]!r}",
        )
        files = {p.name for p in out.iterdir()}
        wanted = {name for pair in names.values() for name in pair}
        check(wanted <= files and wanted == {"pareto_test.png", "pareto_test.json", "pareto_tasks_test.png",
                                             "pareto_tasks_test.json", "pareto_routed_test.png", "pareto_routed_test.json"},
              "the output folder holds the headline, per-task and router-started figures, each a PNG and a JSON named for test")  # fmt: skip
        check(all(png_ok(out / png) for png, _ in names.values()), "each PNG is drawn")
        figs = {kind: json.loads((out / js).read_text(encoding="utf-8")) for kind, (_, js) in names.items()}
        tables = r.tables
        check(all(f["split"] == "test" and f["run"] == 0 and f["n"] == tables["n"] == 8 and f["built_by"] ==
                  "analysis/final_figures.py" and f["figure"] == names[k][0] and f["operating_points"] == "frozen"
                  for k, f in figs.items()), "the JSONs say test, run 0, the tables' item count, the figure's own name and frozen points")  # fmt: skip
        record = r.record
        step = next(s for s in record["gos"][-1]["steps"] if s["name"] == "tables")
        said = r.log
        check("tables: wrote the figures, " in said and all(name in said for pair in names.values() for name in pair),
              "the tables step says it wrote the figures, by name")  # fmt: skip
        check(sorted(step["detail"]["figures"]) == sorted(wanted),
              "and the record keeps their names")  # fmt: skip
        check(all("Test split" in f["caption"] and "Dev split" not in f["caption"] for f in figs.values())
              and "test items" in figs["headline"]["note"] and "dev items" not in figs["headline"]["note"],
              "every caption opens on the test split, and the realistic note counts test items")  # fmt: skip
        check(all("test items" in p["warning"] for p in figs["tasks"]["panels"].values() if p["warning"])
              and not any("on dev" in text for f in figs.values() for text in f["units"].values()),
              "the per-task warnings count test items, and no unit reads as dev's")  # fmt: skip
        check(r.tables["split"] == "test", "the tables are test's")

        print("\n-- a cross is a frozen point, a line is its dials --")
        parts = tables["parts"]
        crosses, on_step, at_a, dials_ok, ordered, rows_ok = 0, True, True, True, True, True
        for mix in PANELS:
            a_row = figs["headline"]["panels"][mix]["configs"]["opus"]
            for section in final_figures.policy_sections():
                figure = SimpleNamespace(headline=figs["headline"], routed=figs["routed"])
                steps = final_figures.steps_of(figure, mix, section)  # type: ignore[arg-type]
                if section in CASCADES:
                    where = (
                        figs["routed"]["panels"][mix]["routers"][parse_cascade(section)[0]]["lines"][section]
                        if parse_cascade(section)[0] != "haiku"
                        else figs["headline"]["panels"][mix]["cascades"][section]
                    )
                    drawn = where["operating_points"]
                else:
                    drawn = figs["headline"]["panels"][mix]["routers"][section]["operating_points"]
                rows_ok &= drawn == parts[section]["mixes"][mix]
                held = read_section(section)
                for target, row in drawn.items():
                    crosses += 1
                    q, c = row["quality"]["value"], row["cost_per_1k"]["value"]
                    if row["fell_back"]:
                        at_a &= close(q, a_row["quality"]["value"]) and close(c, a_row["cost_per_1k"]["value"])
                        continue
                    on_step &= any(close(s["quality"], q) and close(s["cost_per_1k"], c) for s in steps)
                routes = {
                    (p.get("threshold"), p.get("cutoff"))
                    for m in PANELS
                    for p in held[m].values()
                    if not p["fell_back"]
                }
                dials = [(s["threshold"], s.get("cutoff")) for s in steps[1:]]
                free = section == final_figures.FREE_SECTION
                dials_ok &= set(dials) == routes | ({(None, None)} if free else set()) and len(dials) == len(set(dials))
                ordered &= steps[0]["threshold"] is None and close(steps[0]["quality"], a_row["quality"]["value"]) and not any(
                    final_figures.dial_order(a) < final_figures.dial_order(b) for a, b in zip(dials, dials[1:]))  # fmt: skip
        check(crosses == 2 * 3 * 19, f"{crosses} crosses: each of the 19 policies' three points under both mixes")
        check(on_step, "each routed point's cross sits on a step of its line, to 1e-12")
        check(at_a, "each point the file says fell back is at A: its quality and cost are A's")
        check(dials_ok, "each line holds one step for each distinct dial of its section in routers/operating_points.json that routes, "
                        "and no other (the free checks' dial-less step apart)")  # fmt: skip
        check(ordered, "each line starts at A and runs strictest first")
        check(rows_ok, "the figures' rows are the tables' own rows, point by point")
        check(all(f["units"] and "walk" in f["units"].get("steps", "walk") for f in (figs["headline"], figs["tasks"])),
              "the units say a step is a frozen dial evaluated, never a walk")  # fmt: skip
        point = figs["headline"]["panels"]["realistic"]["cascades"][final_figures.FREE_SECTION]["point"]
        check(set(point) == {"quality", "cost_per_1k"} and point["quality"]["low"] <= point["quality"]["value"] <= point["quality"]["high"],
              "the free checks alone are one point with its intervals, as on dev")  # fmt: skip

        print("\n-- the explorer built from the rehearsal's test JSONs --")
        dev = explorer.load_figures()["dev"]
        test = (figs["headline"], figs["tasks"], figs["routed"])
        mixes = load_mixes()
        html = explorer.page({"dev": dev, "test": test}, mixes)  # type: ignore[arg-type]
        data = check_explorer.embedded(html) or {}
        check(data.get("open") == "test" and list(data.get("splits", {})) == ["test", "dev"],
              "the page opens on test and holds test and dev")  # fmt: skip
        before_problems = len(check_explorer.problems)
        check_explorer.check_split("test", data["splits"]["test"], test, "rehearsal ", inside=False)  # type: ignore[arg-type]
        check(len(check_explorer.problems) == before_problems,
              "its test views hold the rehearsal JSONs' numbers: configs, steps, operating points, cascades and the free point")  # fmt: skip
        shown = check_explorer.run_page(html, everything=True)
        if shown is None:
            print("SKIP  the page's readouts: Node is not installed")
        else:
            check(len(shown["readouts"]) == 12 and not check_explorer.unnamed(shown["readouts"])
                  and shown["title"] == "Router Pareto Explorer: test split",
                  "run in Node, all 12 views name their split in every readout, and it opens titled for test")  # fmt: skip

    check(
        figures_folder() == before,
        "nothing was written to figures/: the rehearsal's files are in its own output folder",
    )
    check(all(sha256(FIGURES / name) == digest for name, digest in DEV_FILES.items()),
          "chart v4's six dev files are byte for byte what they were: their sha256 are the committed ones")  # fmt: skip
    wrong = {**DEV_FILES, "pareto.json": "0" * 64}
    check(any(sha256(FIGURES / name) != digest for name, digest in wrong.items()),
          "known positive: a wrong hash for a dev file is found")  # fmt: skip

    print("\n-- the tables lack a policy's part: the step refuses and draws nothing --")
    real = offline_half.build_tables

    def without_a_part(inputs: offline_half.Inputs) -> dict:
        """The tables as built, less one cascade's part, so every part another ticket reads is there and only the figures lack one."""
        built = real(inputs)
        del built["parts"][JEV_FREE]
        return built

    with rehearsed() as r, mock.patch.object(offline_half, "build_tables", without_a_part):
        outcome, _, _ = r.go()
        left = {p.name for p in r.world.output.iterdir()}
        check(outcome.stopped_at == "tables" and outcome.error == "FigureError" and not any(n.startswith("pareto") for n in left),
              f"the run stops at the tables step with a FigureError, and no figure file is written ({outcome.error})")  # fmt: skip


def source() -> None:
    print("\n-- the figure code names no walk --")
    text = Path(final_figures.__file__).read_text(encoding="utf-8")
    banned = r"\b(walk_candidates|step_thresholds|pure_candidates|routed_candidates|cutoff_steps|quantile_steps|lines_of|line_of|" \
             r"pure_line|routed_line|step_outcomes|dev_walks|Walk)\b"  # fmt: skip
    code = re.sub(r'""".*?"""', "", text, flags=re.S)
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    check(
        re.search(banned, code) is None, "analysis/final_figures.py names no walk and no line re-derivation of chart v4"
    )
    check(re.search(banned, code + "\nfrom analysis.gate_walk import pure_candidates") is not None,
          "known positive: the same search finds an import of a walk's candidates")  # fmt: skip
    check(
        "analysis/final_figures.py" in str(PROJECT_ROOT.joinpath("analysis/final_figures.py")),
        "the search read the module itself",
    )


def main() -> int:
    worked()
    refusals()
    rehearsal()
    source()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
