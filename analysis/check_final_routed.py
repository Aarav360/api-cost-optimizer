"""Check the twelve router-started cascades on test (PLAN_v4.md Amendment A17, items 8, 10 and 26.4; A19, items 6, 8 and 13;
issue #77, #87).

Run from the project root:  uv run python -m analysis.check_final_routed

E, F, G and G+, each ending in the free checks, the Jev gate and the Sonnet gate, are evaluated at their frozen operating
points (`analysis/final_routed.py`). On made-up tables only: no results/, no Docker, no Jev or Anthropic call, and nothing near
the real store or a real test item. Every expected number is worked by hand below.

- **The traces.** Four made-up items, four routers' score tables (with the overhead each router charges) and a gate-score table
  hold a free-check failure on two answers, a gate score exactly at its cutoff, a score exactly at the threshold and a refused
  Jev call. For each of the twelve cascades, the traces `final_routed.routed_traces_of` builds equal the hand-worked ones: the
  configs called and the overhead, which is the router's plus the paid gate call made (none for an item the router sent to
  `opus`, none for an answer a free check rejected, the refused call's tokens charged).
- **The rows.** The part of each cascade, run on a made-up operating-points file and a matrix of four items, gives quality, cost
  per 1,000 and escalation rate worked by hand from those traces; a point that fell back is A; a threshold changes the rows.
- **The file's dials.** On the real operating-points file (made-up tables), each cascade's six rows carry exactly the file's
  threshold, cutoff and `fell_back`, read here as plain JSON.
- **Nothing is walked.** The parts are built with every walk function made to fail, and a known positive shows the same guard
  fires when a walk is called; no name of a walk is in the module.
- **Refusals** (whole): a point that routes but has no threshold, a free-checks cascade with a cutoff, a paid gate with none, a
  pure cascade's name, and a gate-score table with no Sonnet column.
- **The lines.** The twelve cascades are in `PARTS` in item 8's order, each with its committed dev counterpart and a per-task part
  with none, above the three parts that read them.
- **The rehearsal's own world.** On the made-up run of `analysis/check_final_run.py` (E and F scored by the real routers, G and G+
  through the fake Jev, both gates through their stand-ins), each routed point's quality and cost in the tables equal what a loop
  of this check's own gives from the loaded tables, and a known positive shows that loop finds a different result for a changed dial.
"""

import ast
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from analysis import final_routed, offline_half, routed_walk
from analysis.bootstrap import Bootstrap
from analysis.gate_table import GateRow, GateScoreTable, GateTableError, PaidScore
from analysis.matrix import Matrix
from analysis.operating_eval import FrozenPoint, FrozenPointError
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.router_names import (
    CASCADES,
    FREE_GATE,
    G_PLUS,
    HAIKU,
    JEV_GATE,
    NAMED_ROUTERS,
    SONNET_GATE,
    cascade_section,
)
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.simulator import Trace

problems: list[str] = []
BOLSTERED = NAMED_ROUTERS[G_PLUS]  # G+'s section name, typed in one place (check_jev_names)


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception] | tuple[type[Exception], ...], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= tolerance


# ---- the made-up world, worked by hand -----------------------------------------------------------------------------------

ITEMS = ("i1", "i2", "i3", "i4")
TASKS = ("banking77", "mmlu_pro", "ifeval", "mbppplus")  # one item each
MIXES = {
    "uniform": {"ifeval": 0.25, "banking77": 0.25, "mmlu_pro": 0.25, "mbppplus": 0.25},
    "realistic": {"mbppplus": 0.65, "banking77": 0.20, "ifeval": 0.10, "mmlu_pro": 0.05},
}  # A5 item 6, as the rehearsal's check writes them
PRICE = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
RIGHT = {
    "haiku": (1, 0, 1, 1),
    "sonnet": (0, 1, 0, 1),
    "opus_low": (1, 1, 0, 1),
    "opus": (1, 1, 0, 1),
}  # by item i1..i4

# Each router's scores for haiku, sonnet and opus_low, by item, and the overhead it charges each item. At threshold 0.5 the
# cheapest config scoring at least 0.5 is the start (F's i4 haiku sits exactly on it and counts).
ROUTERS = {
    "rules": ({"i1": (.9, .9, .9), "i2": (.2, .6, .9), "i3": (.2, .3, .6), "i4": (.1, .2, .3)}, 0.0),
    "trained": ({"i1": (.3, .9, .9), "i2": (.9, .9, .9), "i3": (.1, .1, .1), "i4": (.5, .5, .5)}, 0.0),
    "jev": ({"i1": (.9, .9, .9), "i2": (.9, .9, .9), "i3": (.1, .8, .8), "i4": (.1, .1, .1)}, 0.001),
    BOLSTERED: ({"i1": (.1, .1, .1), "i2": (.1, .6, .6), "i3": (.7, .7, .7), "i4": (.5, .9, .9)}, 0.002),
}  # fmt: skip
STARTS = {  # each router's pick at 0.5, by hand
    "rules": ("haiku", "sonnet", "opus_low", "opus"),
    "trained": ("sonnet", "haiku", "opus", "haiku"),
    "jev": ("haiku", "haiku", "sonnet", "opus"),
    BOLSTERED: ("opus", "sonnet", "haiku", "haiku"),
}
JEV_OVERHEAD, SONNET_OVERHEAD = 0.0001, 0.0005
CUTOFF = {FREE_GATE: None, JEV_GATE: 0.6, SONNET_GATE: 0.7}
THRESHOLD = 0.5
# The answers that fail a free check, and the paid gates' scores by (item, config): (Jev, Sonnet). None is a refused call.
FREE_FAILS = {("i2", "haiku"): "stop", ("i3", "sonnet"): "schema"}
SCORES: dict[tuple[str, str], tuple[float | None, float]] = {
    ("i1", "haiku"): (.8, .9), ("i1", "sonnet"): (.5, .6), ("i1", "opus_low"): (.9, .9),
    ("i2", "sonnet"): (.6, .7), ("i2", "opus_low"): (.2, .3),  # i2 sonnet sits exactly on both cutoffs
    ("i3", "haiku"): (.1, .2), ("i3", "opus_low"): (.65, .75),
    ("i4", "haiku"): (None, .8), ("i4", "sonnet"): (.7, .9), ("i4", "opus_low"): (.9, .1),
}  # fmt: skip


def gate_table(*, sonnet: bool = True) -> GateScoreTable:
    rows = []
    for item in ITEMS:
        for config in PRICE:
            failed = FREE_FAILS.get((item, config))
            jev = sonnet_entry = None
            if failed is None and (item, config) in SCORES:
                score, verdict = SCORES[item, config]
                jev = PaidScore(score, JEV_OVERHEAD, score is None)
                sonnet_entry = PaidScore(verdict, SONNET_OVERHEAD, False) if sonnet else None
            rows.append(GateRow(item, config, failed is None, failed, jev, sonnet_entry))
    return GateScoreTable.build(rows, {(item, config) for item in ITEMS for config in PRICE})


def router_table(section: str) -> RouterScoreTable:
    scores, overhead = ROUTERS[section]
    items = [ItemRouterScores(i, dict(zip(("haiku", "sonnet", "opus_low"), scores[i])), overhead) for i in ITEMS]
    return RouterScoreTable.build(items, list(ITEMS))


def trace(item: str, *configs: str, overhead: float = 0.0) -> tuple[str, tuple[str, ...], float]:
    return item, configs, overhead


J, S = JEV_OVERHEAD, SONNET_OVERHEAD
# The twelve cascades' traces at threshold 0.5 and each gate's cutoff, one list per cascade, worked from the tables above.
HAND: dict[tuple[str, str], list[tuple[str, tuple[str, ...], float]]] = {
    # E: starts haiku, sonnet, opus_low, opus; no router overhead
    ("rules", FREE_GATE): [trace("i1", "haiku"), trace("i2", "sonnet"), trace("i3", "opus_low"), trace("i4", "opus")],
    ("rules", JEV_GATE): [trace("i1", "haiku", overhead=J), trace("i2", "sonnet", overhead=J),
                          trace("i3", "opus_low", overhead=J), trace("i4", "opus")],
    ("rules", SONNET_GATE): [trace("i1", "haiku", overhead=S), trace("i2", "sonnet", overhead=S),
                             trace("i3", "opus_low", overhead=S), trace("i4", "opus")],
    # F: starts sonnet, haiku (fails the free checks), opus, haiku
    ("trained", FREE_GATE): [trace("i1", "sonnet"), trace("i2", "haiku", "opus"), trace("i3", "opus"),
                             trace("i4", "haiku")],
    ("trained", JEV_GATE): [trace("i1", "sonnet", "opus", overhead=J), trace("i2", "haiku", "opus"), trace("i3", "opus"),
                            trace("i4", "haiku", "opus", overhead=J)],  # a refused call is a rejection, charged its tokens
    ("trained", SONNET_GATE): [trace("i1", "sonnet", "opus", overhead=S), trace("i2", "haiku", "opus"), trace("i3", "opus"),
                               trace("i4", "haiku", overhead=S)],
    # G: overhead 0.001 an item; starts haiku, haiku (fails), sonnet (fails), opus
    ("jev", FREE_GATE): [trace("i1", "haiku", overhead=.001), trace("i2", "haiku", "opus", overhead=.001),
                         trace("i3", "sonnet", "opus", overhead=.001), trace("i4", "opus", overhead=.001)],
    ("jev", JEV_GATE): [trace("i1", "haiku", overhead=.001 + J), trace("i2", "haiku", "opus", overhead=.001),
                        trace("i3", "sonnet", "opus", overhead=.001), trace("i4", "opus", overhead=.001)],
    ("jev", SONNET_GATE): [trace("i1", "haiku", overhead=.001 + S), trace("i2", "haiku", "opus", overhead=.001),
                           trace("i3", "sonnet", "opus", overhead=.001), trace("i4", "opus", overhead=.001)],
    # G+: overhead 0.002 an item; starts opus, sonnet, haiku, haiku
    (BOLSTERED, FREE_GATE): [trace("i1", "opus", overhead=.002), trace("i2", "sonnet", overhead=.002),
                                   trace("i3", "haiku", overhead=.002), trace("i4", "haiku", overhead=.002)],
    (BOLSTERED, JEV_GATE): [trace("i1", "opus", overhead=.002), trace("i2", "sonnet", overhead=.002 + J),
                                  trace("i3", "haiku", "opus", overhead=.002 + J),
                                  trace("i4", "haiku", "opus", overhead=.002 + J)],
    (BOLSTERED, SONNET_GATE): [trace("i1", "opus", overhead=.002), trace("i2", "sonnet", overhead=.002 + S),
                                     trace("i3", "haiku", "opus", overhead=.002 + S),
                                     trace("i4", "haiku", overhead=.002 + S)],
}  # fmt: skip


def point(section: str, threshold: float | None = THRESHOLD, cutoff: float | None = None, fell_back: bool = False,
          mix: str = "realistic", share: float = 0.99) -> FrozenPoint:  # fmt: skip
    return FrozenPoint(section, mix, share, threshold, cutoff, fell_back)


def traces_of(router: str, gate: str, at: FrozenPoint) -> list[Trace]:
    inputs = final_routed.routed_traces_of(router_table(router), gate_table(), gate)
    return inputs(at)


def traces_by_hand() -> None:
    print("-- the twelve cascades' traces, against the hand-worked ones --")
    check(
        list(HAND) == [(r, g) for r in ROUTERS for g in CUTOFF],
        "the hand-worked traces are the twelve cascades, item 8's order",
    )
    for (router, gate), expected in HAND.items():
        at = point(cascade_section(router, gate), cutoff=CUTOFF[gate])
        got = traces_of(router, gate, at)
        check(len(got) == 4 and all(t.item_id == e[0] and t.configs == e[1] and close(t.overhead, e[2])
                                    for t, e in zip(got, expected)),
              f"{CASCADES[cascade_section(router, gate)]}: configs called and overhead, item by item "
              f"{[(t.configs, round(t.overhead, 6)) for t in got]}")  # fmt: skip
    # the starts the routers pick are read by hand above, so the guard below finds a wrong start
    check(all(HAND[r, FREE_GATE][k][1][0] == STARTS[r][k] for r in ROUTERS for k in range(4)),
          "the hand-worked traces start where each router's scores say at 0.5 (a score on the threshold counts)")  # fmt: skip
    built = traces_of("rules", FREE_GATE, point(cascade_section("rules", FREE_GATE)))
    check(
        built[0].configs != ("sonnet",),
        "known positive: a hand-worked trace with a wrong start (sonnet for i1) is found to differ",
    )
    other = traces_of("rules", FREE_GATE, point(cascade_section("rules", FREE_GATE), threshold=0.8))
    check([t.configs for t in other] == [("haiku",), ("opus_low",), ("opus",), ("opus",)],
          "a threshold of 0.8 changes E's starts: haiku, opus_low, opus, opus")  # fmt: skip


# ---- the rows, through the part ------------------------------------------------------------------------------------------


def hand_row(expected: list[tuple[str, tuple[str, ...], float]], mix: dict[str, float]) -> tuple[float, float, float]:
    """Quality, dollars per 1,000 and the share escalated, worked from a cascade's hand-worked traces: the last config called
    answers, every config called and the overhead are paid, and a second config is an escalation."""
    quality = cost = escalated = 0.0
    for position, (item, configs, overhead) in enumerate(expected):
        weight = mix[TASKS[position]]  # one item a task
        quality += weight * RIGHT[configs[-1]][position]
        cost += weight * (sum(PRICE[c] for c in configs) + overhead) * 1000
        escalated += weight if len(configs) > 1 else 0.0
        assert item == ITEMS[position]
    return quality, cost, escalated


def matrix() -> Matrix:
    return Matrix.from_rows([
        {"item_id": i, "split": "test", "task": t, "source_group": i, "config": c, "cost": PRICE[c], "success": RIGHT[c][k]}
        for k, (i, t) in enumerate(zip(ITEMS, TASKS)) for c in PRICE
    ])  # fmt: skip


DRAWS = Bootstrap(ITEMS, TASKS, np.array([[1, 1, 1, 1], [2, 1, 3, 1], [1, 2, 1, 2], [3, 1, 1, 2]]))
EXTRAS = {
    "rules_table": router_table("rules"),
    "trained_table": router_table("trained"),
    "jev_table": router_table("jev"),
    f"{BOLSTERED}_table": router_table(BOLSTERED),
    "gate_table": SimpleNamespace(table=gate_table()),
}
FELL = {"threshold": None, "cutoff": None, "fell_back": True}


def made_up_file(folder: Path, extra: dict[str, dict] | None = None) -> Path:
    """An operating-points file with the twelve sections: 100% falls back (to A), and 99% and 97% in both mixes sit on one
    threshold and the gate's cutoff, so the two mixes and two targets share their simulated outcomes."""
    sections = {}
    for router in ROUTERS:
        for gate in CUTOFF:
            dials = {"threshold": THRESHOLD, "cutoff": CUTOFF[gate], "fell_back": False}
            by_mix = {"100%": FELL, "99%": dials, "97%": dials}
            sections[cascade_section(router, gate)] = {"realistic": by_mix, "uniform": by_mix}
    sections.update(extra or {})
    path = folder / "operating_points.json"
    path.write_text(json.dumps(sections), encoding="utf-8")
    return path


def inputs_for() -> Any:
    return SimpleNamespace(cache={}, matrix=matrix(), draws=DRAWS, extras=EXTRAS)


def rows_by_hand() -> None:
    print("\n-- each cascade's rows through its part, against the hand-worked quality, cost and escalation --")
    a = {
        "uniform": (0.75, 10.0),
        "realistic": (0.9, 10.0),
    }  # A: always opus, by hand (right on i1, i2, i4; $0.01 an item)
    with tempfile.TemporaryDirectory() as folder:
        path = made_up_file(Path(folder))
        for router in ROUTERS:
            for gate in CUTOFF:
                section = cascade_section(router, gate)
                rows = final_routed.routed_part(section, path=path)(inputs_for())["mixes"]
                ok = True
                for mix, weights in MIXES.items():
                    quality, cost, escalated = hand_row(HAND[router, gate], weights)
                    for target in ("99%", "97%"):
                        row = rows[mix][target]
                        ok &= (row["fell_back"] is False and row["threshold"] == THRESHOLD and row["cutoff"] == CUTOFF[gate]
                               and close(row["quality"]["value"], quality) and close(row["cost_per_1k"]["value"], cost)
                               and close(row["escalation_rate"], escalated)
                               and close(row["savings_vs_A"]["per_1k"], a[mix][1] - cost))  # fmt: skip
                    fell = rows[mix]["100%"]
                    ok &= (fell["fell_back"] is True and fell["threshold"] is None and fell["cutoff"] is None
                           and close(fell["quality"]["value"], a[mix][0]) and close(fell["cost_per_1k"]["value"], a[mix][1])
                           and fell["escalation_rate"] == 0 and fell["savings_vs_A"]["share"] == 0)  # fmt: skip
                check(ok, f"{CASCADES[section]}: 99% and 97% are the cascade's hand-worked quality, cost and escalation under "
                          f"both mixes, and the 100% point that fell back is A ($0.01 an item)")  # fmt: skip
        # absolute anchors, worked straight from the tables (E/free: right on i1 i2 i4, $0.017 over the four items)
        rows = final_routed.routed_part(cascade_section("rules", FREE_GATE), path=path)(inputs_for())["mixes"]
        check(close(rows["uniform"]["99%"]["quality"]["value"], 0.75) and close(rows["uniform"]["99%"]["cost_per_1k"]["value"], 4.25)
              and close(rows["realistic"]["99%"]["quality"]["value"], 0.9)
              and close(rows["realistic"]["99%"]["cost_per_1k"]["value"], 7.2),
              "E/free: quality 0.75 and $4.25 per 1,000 under the uniform mix, 0.9 and $7.20 under the realistic, from the tables")  # fmt: skip
        rows = final_routed.routed_part(cascade_section("trained", JEV_GATE), path=path)(inputs_for())["mixes"]
        check(close(rows["uniform"]["99%"]["quality"]["value"], 0.75) and close(rows["uniform"]["99%"]["cost_per_1k"]["value"], 11.05)
              and close(rows["uniform"]["99%"]["escalation_rate"], 0.75) and close(rows["realistic"]["99%"]["escalation_rate"], 0.9),
              "F/Jev: quality 0.75, $11.05 per 1,000 and 3 of 4 escalated under the uniform mix (0.9 under the realistic)")  # fmt: skip
        # a threshold that moves: E/free at 0.8 in one point
        moved = {cascade_section("rules", FREE_GATE): {
            "realistic": {"100%": FELL, "99%": FELL, "97%": {"threshold": 0.8, "cutoff": None, "fell_back": False}},
            "uniform": {"100%": FELL, "99%": FELL, "97%": {"threshold": 0.8, "cutoff": None, "fell_back": False}},
        }}  # fmt: skip
        rows = final_routed.routed_part(cascade_section("rules", FREE_GATE), path=made_up_file(Path(folder), moved))(
            inputs_for()
        )["mixes"]
        check(rows["uniform"]["99%"]["fell_back"] and not rows["uniform"]["97%"]["fell_back"]
              and close(rows["uniform"]["97%"]["quality"]["value"], 0.75) and close(rows["uniform"]["97%"]["cost_per_1k"]["value"], 6.25)
              and close(rows["realistic"]["97%"]["cost_per_1k"]["value"], 7.9),
              "E/free at threshold 0.8: starts haiku, opus_low, opus, opus, so $6.25 per 1,000 uniform and $7.90 realistic")  # fmt: skip
        tasks = final_routed.routed_part(cascade_section("rules", FREE_GATE), per_task=True, path=path)(inputs_for())[
            "tasks"
        ]
        check(set(tasks) == set(TASKS) and all(set(tasks[t]) == {"realistic", "uniform"} for t in tasks)
              and all(tasks[t]["uniform"]["100%"]["mcnemar"] is None for t in tasks),
              "the per-task part holds each of the four tasks under both mixes, a fallen-back point with no McNemar row")  # fmt: skip
        check(close(tasks["mbppplus"]["uniform"]["99%"]["quality"]["value"], 1.0),
              "and E/free's MBPP+ quality is 1 (its one item routes to opus, right)")  # fmt: skip


# ---- the file's dials ---------------------------------------------------------------------------------------------------


def the_files_dials() -> None:
    print("\n-- thresholds, cutoffs and fall-backs are the operating-points file's --")
    held = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    routes = 0
    for section in final_routed.ROUTED:
        rows = final_routed.routed_part(section)(inputs_for())["mixes"]
        ok = True
        for mix in ("realistic", "uniform"):
            for target in ("100%", "99%", "97%"):
                row, entry = rows[mix][target], held[section][mix][target]
                ok &= (row["threshold"] == entry.get("threshold") and row["cutoff"] == entry.get("cutoff")
                       and row["fell_back"] == entry["fell_back"])  # fmt: skip
                routes += not entry["fell_back"]
        check(ok, f"{CASCADES[section]}: all six rows carry the file's threshold, cutoff and fell_back")
    check(routes == 46, f"{routes} of the twelve cascades' 72 points route in the file (A19 item 6's 54 of the cascades' 90, less the "
                        f"pure cascades' 8)")  # fmt: skip


# ---- nothing is walked --------------------------------------------------------------------------------------------------


def names_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }


WALKING = {
    "walk_candidates",
    "routed_walks",
    "routed_candidates",
    "step_thresholds",
    "pure_walks",
    "cascade_walks",
    "Walk",
}


def nothing_walked() -> None:
    print("\n-- nothing is walked on test --")

    def walked(*_: object, **__: object) -> None:
        raise AssertionError("a walk was called")

    names = names_in(Path(final_routed.__file__))
    check(not names & WALKING, f"the module names no walk function ({sorted(names & WALKING)})")
    check(bool(names_in(Path(routed_walk.__file__)) & WALKING),
          "known positive: the same search finds the walks' names in analysis/routed_walk.py")  # fmt: skip
    patches = [mock.patch("analysis.walk.walk_candidates", walked), mock.patch("analysis.routed_walk.routed_walks", walked),
               mock.patch("analysis.routed_walk.routed_candidates", walked), mock.patch("analysis.walk.step_thresholds", walked)]  # fmt: skip
    for p in patches:
        p.start()
    try:
        for section in final_routed.ROUTED:
            final_routed.routed_part(section)(inputs_for())
        built = True
    except AssertionError:
        built = False
    finally:
        for p in patches:
            p.stop()
    check(built, "all twelve parts build with every walk function made to fail")
    with patches[1]:
        check(refused(AssertionError, lambda: routed_walk.routed_walks(None, None, None, "free", None, None)),  # type: ignore[arg-type]
              "known positive: the patched walk raises when a walk is called")  # fmt: skip


# ---- refusals ------------------------------------------------------------------------------------------------------------


def refusals() -> None:
    print("\n-- refusals, each whole --")
    table = router_table("rules")
    gates = gate_table()
    free = cascade_section("rules", FREE_GATE)

    def build(
        gate: str, at: FrozenPoint, router: RouterScoreTable = table, gated: GateScoreTable = gates
    ) -> list[Trace]:
        return final_routed.routed_traces_of(router, gated, gate)(at)

    check(refused(FrozenPointError, lambda: build(FREE_GATE, point(free, threshold=None))),
          "a point that routes (the file says it did not fall back) but holds no threshold")  # fmt: skip
    check(refused(GateTableError, lambda: build(FREE_GATE, point(free, cutoff=0.5))),
          "a free-checks cascade holding a cutoff")  # fmt: skip
    check(refused(GateTableError, lambda: build(JEV_GATE, point(cascade_section("rules", JEV_GATE), cutoff=None))),
          "a paid gate with no cutoff")  # fmt: skip
    check(refused(GateTableError, lambda: build("sonnet-ish", point(free))), "a gate no amendment names")
    check(refused(GateTableError, lambda: build(SONNET_GATE, point(free, cutoff=0.7), gated=gate_table(sonnet=False))),
          "a gate-score table with no Sonnet column, for a cascade that ends in the Sonnet gate")  # fmt: skip
    check(not refused(Exception, lambda: build(JEV_GATE, point(free, cutoff=0.6), gated=gate_table(sonnet=False))),
          "known positive: the same table with no Sonnet column serves the Jev gate")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        path = made_up_file(Path(folder))
        check(refused(final_routed.FinalRoutedError, lambda: final_routed.routed_part(cascade_section(HAIKU, JEV_GATE), path=path)),
              "a pure cascade's section is no router-started cascade")  # fmt: skip
        check(refused(final_routed.FinalRoutedError, lambda: final_routed.routed_part("rules", path=path)),
              "a router's section is no cascade")  # fmt: skip
        missing = {**EXTRAS}
        del missing["jev_table"]
        without: Any = SimpleNamespace(cache={}, matrix=matrix(), draws=DRAWS, extras=missing)
        part = final_routed.routed_part(cascade_section("jev", FREE_GATE), path=path)
        check(refused(final_routed.FinalRoutedError, lambda: part(without)), "a router's table missing from the inputs")


# ---- the rehearsal's own world -------------------------------------------------------------------------------------------


def independent_row(
    inputs: Any, router: str, gate: str, threshold: float | None, cutoff: float | None, mix: dict[str, float]
):
    """Quality and dollars per 1,000 of one cascade point, worked from the loaded tables by a loop of its own (not the
    trace builder's): the cheapest config scoring at least the threshold starts, an answer stays if it passed the free checks
    and, with a paid gate, scored at least the cutoff; otherwise `opus` answers too. Overheads: the router's, and the gate's
    for each answer it judged. Returns None for a point that fell back, which the caller compares with A."""
    table, gates, matrix = (
        inputs.extras[final_routed.ROUTER_TABLES[router]],
        inputs.extras["gate_table"].table,
        inputs.matrix,
    )
    counts: dict[str, int] = {}
    for item in matrix.items.values():
        counts[item.task] = counts.get(item.task, 0) + 1
    quality = cost = 0.0
    for entry in table.items:
        start = next((c for c in ("haiku", "sonnet", "opus_low") if entry.router_scores[c] >= threshold), "opus")
        called, overhead = [start], entry.overhead
        if start != "opus":
            row = gates.row(entry.item_id, start)
            kept = row.free_passed
            if kept and gate != FREE_GATE:
                paid = row.jev if gate == JEV_GATE else row.sonnet
                overhead += paid.overhead
                kept = paid.score is not None and paid.score >= cutoff
            if not kept:
                called.append("opus")
        task = matrix.items[entry.item_id].task
        weight = mix[task] / counts[task]
        quality += weight * matrix.cells[entry.item_id, called[-1]].success
        cost += weight * (sum(matrix.cells[entry.item_id, c].cost for c in called) + overhead) * 1000
    return quality, cost


def rehearsal() -> None:
    print("\n-- the twelve cascades on the rehearsal's made-up world, recomputed from its loaded tables --")
    from analysis import check_final_run as ci
    from analysis.check_final_jev import unlocked_run

    held = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    with ci.rehearsed() as r:
        outcome, _, _ = r.go()
        check(outcome.stopped_at is None, "the whole made-up run goes through, the tables step included")
        parts = r.tables["parts"]
        inputs = offline_half.test_inputs(unlocked_run(r))
        check(all(s in parts and final_routed.tasks_key(s) in parts for s in final_routed.ROUTED),
              "the tables hold each cascade's part and its per-task part")  # fmt: skip
        for router in ROUTERS:
            for gate in CUTOFF:
                section = cascade_section(router, gate)
                ok, routed = True, 0
                for mix, weights in MIXES.items():
                    for target in ("100%", "99%", "97%"):
                        row, entry = parts[section]["mixes"][mix][target], held[section][mix][target]
                        if entry["fell_back"]:
                            ok &= (
                                row["fell_back"] is True
                                and row["quality"]["value"]
                                == parts["references"]["mixes"][mix]["configs"]["opus"]["quality"]["value"]
                            )
                            continue
                        routed += 1
                        quality, cost = independent_row(
                            inputs, router, gate, entry["threshold"], entry.get("cutoff"), weights
                        )
                        ok &= close(row["quality"]["value"], quality) and close(row["cost_per_1k"]["value"], cost)
                check(
                    ok,
                    f"{CASCADES[section]}: its {routed} routed points' quality and cost are the loaded tables', worked by a loop of its own",
                )
        row = parts[cascade_section("jev", JEV_GATE)]["mixes"]["realistic"]["99%"]
        entry = held[cascade_section("jev", JEV_GATE)]["realistic"]["99%"]
        moved = independent_row(
            inputs, "jev", JEV_GATE, 2.0, entry["cutoff"], MIXES["realistic"]
        )  # every item sent to opus
        check(not close(row["cost_per_1k"]["value"], moved[1]) and not entry["fell_back"],
              "known positive: the loop finds a different cost for a point whose threshold is changed")  # fmt: skip
        check(set(parts["target_met"]["points"]) >= set(final_routed.ROUTED),
              "target met covers all twelve with no line of its own")  # fmt: skip


# ---- the lines in the tables ---------------------------------------------------------------------------------------------


def the_lines() -> None:
    print("\n-- the twelve cascades' lines in the offline half --")
    keys = [part.key for part in offline_half.PARTS]
    wanted = [k for section in final_routed.ROUTED for k in (section, final_routed.tasks_key(section))]
    expected_order = [
        cascade_section(r, g)
        for r in ("rules", "trained", "jev", BOLSTERED)
        for g in (FREE_GATE, JEV_GATE, SONNET_GATE)
    ]
    check(
        list(final_routed.ROUTED) == expected_order,
        "ROUTED is E, F, G and G+, each with the free checks, Jev and Sonnet",
    )
    positions = [keys.index(k) for k in wanted if k in keys]
    check(len(positions) == 24 and positions == sorted(positions) and max(positions) < keys.index("flip_rates"),
          "all 24 parts are in PARTS, in order, above flip_rates, target met and the headline")  # fmt: skip
    by_key = {part.key: part for part in offline_half.PARTS}
    check(all(by_key[s].committed is not None for s in final_routed.ROUTED)
          and all(by_key[final_routed.tasks_key(s)].committed is None for s in final_routed.ROUTED),
          "each cascade's part names its committed dev numbers, and its per-task part has none (dev committed none)")  # fmt: skip
    extras = {extra.name for extra in offline_half.EXTRAS}
    check(set(final_routed.ROUTER_TABLES.values()) <= extras and "gate_table" in extras,
          "the router tables and the gate-score table are extras the parts read")  # fmt: skip


def main() -> int:
    traces_by_hand()
    rows_by_hand()
    the_files_dials()
    nothing_walked()
    refusals()
    the_lines()
    rehearsal()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
