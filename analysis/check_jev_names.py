"""Check that G and G+ are named, allowed and styled before any Jev number exists (PLAN_v4.md §8 and §12; Amendment A15,
items 14 and 15), and that Stage 7's fifteen cascades are too (Amendment A17, items 8, 23 and 25).

Run from the project root:  uv run python -m analysis.check_jev_names

Reads no results matrix and calls no Jev, so CI runs it. That every figure JSON and PNG of E and F rebuilds unchanged
is `check_stage4`'s and `check_stage5`'s, which need `results/`.

- the router names: G is the Jev router and G+ the bolstered Jev router, with the sections `jev` and `jev_bolstered` and
  a label each, in `analysis/router_names.py`, and E's and F's names and labels are as they were;
- the operating-points allowlist accepts both sections, singly and together, accepts them as required sections, and
  still refuses F-train and the diagnostics (known positive: it refuses F-train in a file that also holds G's section);
- the names are typed in one place: no module of `analysis/` spells `"jev_bolstered"` or `"G+"` as a string constant
  except `router_names.py`, the modules a freeze hashes (editing one means re-selecting G and G+, A15 item 13) and the
  checks that mirror the settings file's own keys or hold an expected value (known positives: the scan finds the
  literal in two hashed modules and in a temporary file, and passes a file that only mentions it in a docstring);
- the fifteen cascades: their labels (`haiku/free` to `G+/Sonnet`) and section keys (`cascade_{start}_{gate}`) are named
  once, in A17 item 8's order, in `analysis/router_names.py` beside `NAMED_ROUTERS` and not inside it; no module of
  `analysis/`, `routers/` or `harness/` spells one out as a string constant (known positive: the scan finds one in a
  temporary file); the allowlist accepts all fifteen sections and still refuses F-train and every diagnostic; and the
  three gates' styles (a colour, markers and a line each, shared by the pure cascade and the router-started line of
  that gate) share nothing with a router's and sit 15 apart in OKLab, in every view, from every other colour on the
  headline figure and on each router's panel (known positive: a colour 15 apart from nothing there is found);
- the chart's router list holds E, F, G and G+ in that order (Stage 6 · 12 added the last two; they were reserved before),
  and their two styles in `analysis/chart_routers.py` have a colour, a marker, a point marker and a line that no other
  router's shares,
  and a colour at least 15 apart (OKLab x100, in normal vision and under simulated protan, deutan and tritan vision)
  from every other colour the figures draw and from each other's (known positive: a teal that the configs' blue
  stands within 3 of).
"""

import ast
import itertools
import sys
import tempfile
from functools import partial
from pathlib import Path

from analysis.chart_routers import CASCADE_STYLES, E_STYLE, F_STYLE, G_PLUS_STYLE, G_STYLE, ROUTERS
from analysis.check_chart_routers import APART, gap
from analysis.check_jev_freeze import frozen_paths
from analysis.check_jev_settings_commit import COMMIT_FILES
from analysis.checking import refused
from analysis.operating_points import OperatingPointsError, check_sections, write_section
from analysis.pareto import SERIES_1, SERIES_2, SERIES_3
from analysis.router_names import (
    CASCADE_STARTS,
    CASCADES,
    F_TRAIN,
    FREE_GATE,
    G_PLUS,
    JEV_GATE,
    LABELS,
    NAMED_ROUTERS,
    SONNET_GATE,
    TASK_RATES,
    E,
    F,
    G,
    cascade_section,
    parse_cascade,
)

# What the scan looks for: the two names no router's code should spell out. "jev" is also a folder and a settings key,
# so a scan for it could not tell a router from those.
SPELLED = ("jev_bolstered", "G+")
# Modules that may spell them, each for a reason: the one place, the checks' expected values, and the checks that
# mirror the settings file's own keys, which the hashed `routers/jev_call.py` reads.
MAY_SPELL = {
    "router_names.py": "the one place",
    "check_jev_names.py": "expected values",
    "check_chart_routers.py": "expected values",
    "check_jev_settings.py": "the settings file's keys",
    "check_jev_settings_commit.py": "the settings file's keys",
    "check_jev_call.py": "the settings file's keys",
    "check_jev_client.py": "the settings file's keys",
}
HASHED_MODULES = {
    Path(p).name for p in (*frozen_paths(), *COMMIT_FILES) if p.startswith("analysis/") and p.endswith(".py")
}
# The fifteen cascades in A17 item 8's order, typed out by hand here (the expected values): the pure cascades first,
# then E, F, G and G+, each with free, Jev and Sonnet.
CASCADE_LABELS = [
    "haiku/free", "haiku/Jev", "haiku/Sonnet", "E/free", "E/Jev", "E/Sonnet", "F/free", "F/Jev", "F/Sonnet",
    "G/free", "G/Jev", "G/Sonnet", "G+/free", "G+/Jev", "G+/Sonnet",
]  # fmt: skip
CASCADE_KEYS = [
    "cascade_haiku_free", "cascade_haiku_jev", "cascade_haiku_sonnet",
    "cascade_rules_free", "cascade_rules_jev", "cascade_rules_sonnet",
    "cascade_trained_free", "cascade_trained_jev", "cascade_trained_sonnet",
    "cascade_jev_free", "cascade_jev_jev", "cascade_jev_sonnet",
    "cascade_jev_bolstered_free", "cascade_jev_bolstered_jev", "cascade_jev_bolstered_sonnet",
]  # fmt: skip
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def spelled_in(source: str, names: tuple[str, ...] = SPELLED) -> list[str]:
    """The `names` that a module's source holds as a string constant (a docstring that merely mentions one is a
    longer string, so it does not count)."""
    return sorted({n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and n.value in names})


def spellers(folder: Path, names: tuple[str, ...] = SPELLED) -> dict[str, list[str]]:
    """The modules under `folder` that spell one of `names` out, with the names."""
    found = {p.name: spelled_in(p.read_text(encoding="utf-8"), names) for p in sorted(folder.glob("*.py"))}
    return {name: spelled for name, spelled in found.items() if spelled}


def parses_keys(source: str) -> bool:
    """Whether a module takes a cascade's section key apart by hand: it holds the bare prefix `"cascade_"` as a string
    (or the part of an f-string before a start) or a start's whole prefix such as `"cascade_haiku_"`, where
    `router_names.parse_cascade` gives the start and the gate. Other keys that merely begin so (`cascade_units`) don't
    count."""
    prefixes = tuple(f"cascade_{start}_" for start in CASCADE_STARTS)
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and (n.value == "cascade_" or n.value.startswith(prefixes))
               for n in ast.walk(ast.parse(source)))  # fmt: skip


def main() -> int:
    print("-- the router names --")
    check((G, G_PLUS) == ("G", "G+"), "G is the Jev router and G+ the bolstered Jev router")
    check(NAMED_ROUTERS[G] == "jev" and NAMED_ROUTERS[G_PLUS] == "jev_bolstered",
          "their sections are `jev` and `jev_bolstered`")  # fmt: skip
    check(
        LABELS[G] == "G: Jev router" and LABELS[G_PLUS] == "G+: bolstered Jev router", "their labels are G's and G+'s"
    )
    check(NAMED_ROUTERS[E] == "rules" and NAMED_ROUTERS[F] == "trained"
          and LABELS[E] == "E: rules router" and LABELS[F] == "F: trained router",
          "E's and F's sections and labels are as they were")  # fmt: skip
    check(set(LABELS) == set(NAMED_ROUTERS) == {E, F, G, G_PLUS}, "every named router has a label, and no other does")
    check(len(set(NAMED_ROUTERS.values())) == 4 and len(set(LABELS.values())) == 4,
          "no two routers share a section or a label")  # fmt: skip

    print("\n-- the names are typed in one place --")
    found = spellers(Path(__file__).resolve().parent)
    check(spelled_in('G_PLUS = "G+"') == ["G+"] and spelled_in('x = ["jev_bolstered"]') == ["jev_bolstered"],
          "known positive: the scan finds each name as a string constant")  # fmt: skip
    check(spelled_in('"""A docstring that says "jev_bolstered" and G+ in passing."""') == [] and spelled_in("x = 1") == [],
          "and passes a docstring that only mentions one, and a module with neither")  # fmt: skip
    check({"build_jev_settings.py", "fit_jev_router.py"} <= set(found) <= {*HASHED_MODULES, *MAY_SPELL},
          "known positive: the real scan finds the literals in the hashed builder and fit script")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "typed_again.py").write_text('NAME = "jev_bolstered"\n')
        check(
            spellers(Path(folder)) == {"typed_again.py": ["jev_bolstered"]},
            "known positive: a new module that types a name is found",
        )
    stray = sorted(set(found) - set(MAY_SPELL) - HASHED_MODULES)
    check(stray == [], f"no other module under analysis/ spells a router name out {stray}")
    check("router_names.py" in found and set(MAY_SPELL) >= {"router_names.py"} and HASHED_MODULES >= {"fit_jev_router.py", "smoke_jev.py"},
          "the one place and the hashed modules the allowlist stands on are the ones found")  # fmt: skip

    print("\n-- the operating-points allowlist --")
    jev, bolstered = NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS]
    with tempfile.TemporaryDirectory() as folder:
        both = Path(folder) / "both.json"
        write_section(jev, {}, both)
        check_sections(both)
        check(True, "a file with G's section alone passes")
        write_section(bolstered, {}, both)
        check_sections(both, required=(jev, bolstered))
        check(True, "a file with G's and G+'s sections passes, with both asked for as required sections")
        only = Path(folder) / "only.json"
        write_section(bolstered, {}, only)
        check_sections(only)
        check(True, "G+'s section alone passes")
        check(refused(OperatingPointsError, lambda: check_sections(only, required=(jev, bolstered))),
              "a required section that is missing is refused")  # fmt: skip
        for name in (F_TRAIN, TASK_RATES, "jev_diagnostic"):
            other = Path(folder) / "other.json"
            other.write_text(both.read_text())
            write_section(name, {}, other)
            check(refused(OperatingPointsError, lambda: check_sections(other)),
                  f"known positive: {name!r} in a file that holds G's and G+'s sections is refused")  # fmt: skip

    print("\n-- the chart's router list and the two reserved styles --")
    check([r.name for r in ROUTERS] == [NAMED_ROUTERS[E], NAMED_ROUTERS[F], NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS]],
          "the router list holds E, F, G and G+ in that order, under their names in the operating-points file")  # fmt: skip
    check([r.style for r in ROUTERS] == [E_STYLE, F_STYLE, G_STYLE, G_PLUS_STYLE]
          and [r.label for r in ROUTERS] == [LABELS[E], LABELS[F], LABELS[G], LABELS[G_PLUS]],
          "each entry has its router's reserved style and its label from the names file")  # fmt: skip
    styles = {E: E_STYLE, F: F_STYLE, G: G_STYLE, G_PLUS: G_PLUS_STYLE}
    for field in ("color", "marker", "point_marker", "linestyle"):
        check(
            len({getattr(s, field) for s in styles.values()}) == 4, f"the four routers' styles differ in their {field}"
        )
    check(all(len(s.color) == 7 and s.color.startswith("#") for s in styles.values()),
          "each style's colour is a hex colour")  # fmt: skip
    others = {"the configs' blue": SERIES_1, "the envelope's orange": SERIES_2, "the oracle's green": SERIES_3,
              "E's colour": E_STYLE.color, "F's colour": F_STYLE.color}  # fmt: skip
    for letter, style in ((G, G_STYLE), (G_PLUS, G_PLUS_STYLE)):
        peers = {**others, **{f"{o}'s colour": styles[o].color for o in (G, G_PLUS) if o != letter}}
        for what, colour in peers.items():
            check(gap(style.color, colour) >= APART,
                  f"{letter}'s colour is {gap(style.color, colour):.1f} from {what}, at least {APART} in every view")  # fmt: skip
    teal = "#0e8a9a"
    check(gap(teal, SERIES_1) < APART, f"known positive: a teal ({teal}) is {gap(teal, SERIES_1):.1f} from the "
          "configs' blue, and the check says so")  # fmt: skip

    print("\n-- the fifteen cascades' names --")
    check(
        list(CASCADES.values()) == CASCADE_LABELS, "the labels run from haiku/free to G+/Sonnet in A17 item 8's order"
    )
    check(list(CASCADES) == CASCADE_KEYS, "the section keys are cascade_{start}_{gate}, in the same order")
    check(
        len(set(CASCADES)) == len(set(CASCADES.values())) == 15,
        "there are fifteen, each with a key and a label of its own",
    )
    check(not set(CASCADES) & {*NAMED_ROUTERS.values(), *LABELS.values()}
          and not set(CASCADES.values()) & {*NAMED_ROUTERS.values(), *LABELS.values()},
          "none shares a name with a router, and NAMED_ROUTERS and LABELS still hold the four routers alone")  # fmt: skip
    check(cascade_section("haiku", FREE_GATE) == "cascade_haiku_free"
          and cascade_section(NAMED_ROUTERS[G_PLUS], SONNET_GATE) == "cascade_jev_bolstered_sonnet"
          and cascade_section(NAMED_ROUTERS[F], JEV_GATE) == "cascade_trained_jev",
          "a start and a gate give their section key")  # fmt: skip
    check(refused(ValueError, lambda: cascade_section("opus", FREE_GATE))
          and refused(ValueError, lambda: cascade_section("haiku", "log-probabilities")),
          "a start or a gate no amendment names is refused")  # fmt: skip
    check(parse_cascade("cascade_jev_bolstered_sonnet") == ("jev_bolstered", "sonnet")
          and parse_cascade("cascade_haiku_free") == ("haiku", "free")
          and parse_cascade("cascade_jev_jev") == ("jev", "jev"),
          "a section key gives back its start and its gate, a start with an underscore in it included")  # fmt: skip
    check(all(cascade_section(*parse_cascade(key)) == key for key in CASCADE_KEYS),
          "parse_cascade undoes cascade_section for all fifteen")  # fmt: skip
    for bad in ("cascade_haiku_opus", "cascade_opus_free", "rules", "cascade_", "haiku/Jev", "cascade_jev"):
        check(refused(ValueError, partial(parse_cascade, bad)), f"known positive: {bad!r} is refused, not parsed")

    print("\n-- the cascades' names are typed in one place --")
    named = (*CASCADE_LABELS, *CASCADE_KEYS)
    check(spelled_in('x = "haiku/Jev"', named) == ["haiku/Jev"] and spelled_in('y = ["cascade_jev_free"]', named) == ["cascade_jev_free"],
          "known positive: the scan finds a label and a key as string constants")  # fmt: skip
    check(spelled_in('"""Says haiku/Jev and cascade_jev_free in passing."""', named) == [] and spelled_in("x = 1", named) == [],
          "and passes a docstring that only mentions them, and a module with neither")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "typed_again.py").write_text('KEY = "cascade_rules_sonnet"\nNAME = "G+/free"\n')
        check(spellers(Path(folder), named) == {"typed_again.py": ["G+/free", "cascade_rules_sonnet"]},
              "known positive: a new module that types a cascade's name is found")  # fmt: skip
    root = Path(__file__).resolve().parent.parent
    for folder_name in ("analysis", "routers", "harness"):
        # The checks that hold the expected values are this module's; nothing else may type a cascade's name.
        stray = sorted(set(spellers(root / folder_name, named)) - {"check_jev_names.py"})
        check(stray == [], f"no module under {folder_name}/ other than this check spells a cascade's name out {stray}")
    check(parses_keys('start = key.removeprefix("cascade_")') and parses_keys('p = key.startswith(f"cascade_{HAIKU}_")')
          and parses_keys('pure = key.startswith("cascade_haiku_")'),
          "known positive: the scan finds a key taken apart by its prefix, in a string and in an f-string")  # fmt: skip
    check(not parses_keys('x = data["cascade_units"]') and not parses_keys('"""Keys are cascade_{start}_{gate}."""'),
          "and passes another key that begins so, and a docstring that only describes the keys")  # fmt: skip
    for folder_name in ("analysis", "routers", "harness"):
        hand = sorted(p.name for p in (root / folder_name).glob("*.py")
                      if p.name not in {"router_names.py", "check_jev_names.py"} and parses_keys(p.read_text(encoding="utf-8")))  # fmt: skip
        check(hand == [], f"no module under {folder_name}/ takes a section key apart by hand, but parse_cascade {hand}")

    print("\n-- the allowlist and the cascades --")
    with tempfile.TemporaryDirectory() as folder:
        every = Path(folder) / "every.json"
        for key in CASCADE_KEYS:
            alone = Path(folder) / "alone.json"
            alone.unlink(missing_ok=True)
            write_section(key, {}, alone)
            check_sections(alone, required=(key,))
        check(True, "each of the fifteen sections passes alone, and as a required section")
        for name in (*NAMED_ROUTERS.values(), *CASCADE_KEYS):
            write_section(name, {}, every)
        check_sections(every, required=(*NAMED_ROUTERS.values(), *CASCADE_KEYS))
        check(True, "a file with the four routers' sections and all fifteen passes, with all nineteen required")
        for name in (
            F_TRAIN,
            TASK_RATES,
            "jev_diagnostic",
            "cascade_haiku_opus",
            "cascade_f_free",
            "cascade_jev_gate_free",
        ):
            other = Path(folder) / "other.json"
            other.write_text(every.read_text())
            write_section(name, {}, other)
            check(refused(OperatingPointsError, lambda: check_sections(other)),
                  f"known positive: {name!r} in a file that holds every named section is refused")  # fmt: skip
        sans = Path(folder) / "sans.json"
        write_section(CASCADE_KEYS[0], {}, sans)
        check(refused(OperatingPointsError, lambda: check_sections(sans, required=(CASCADE_KEYS[0], CASCADE_KEYS[1]))),
              "a required cascade section that is missing is refused")  # fmt: skip

    print("\n-- the cascades' chart styles --")
    gates = (FREE_GATE, JEV_GATE, SONNET_GATE)
    check(
        list(CASCADE_STYLES) == list(gates),
        "each gate has one style, shared by the pure cascade and the router-started line",
    )
    cascade_styles = [CASCADE_STYLES[g] for g in gates]
    router_styles = [E_STYLE, F_STYLE, G_STYLE, G_PLUS_STYLE]
    for field in ("color", "marker", "point_marker", "linestyle"):
        check(len({getattr(s, field) for s in cascade_styles}) == 3, f"the three gates' styles differ in their {field}")
        check(not {getattr(s, field) for s in cascade_styles} & {getattr(s, field) for s in router_styles},
              f"and share no {field} with a router's")  # fmt: skip
    check(all(len(s.color) == 7 and s.color.startswith("#") for s in cascade_styles), "each colour is a hex colour")
    check(not {s.marker for s in cascade_styles} & {"o"}, "no gate's step marker is the configs' circle")
    drawn = {"the configs' blue": SERIES_1, "the envelope's orange": SERIES_2, "the oracle's green": SERIES_3,
             **{f"{LABELS[r]}'s colour": styles[r].color for r in (E, F, G, G_PLUS)}}  # fmt: skip
    gate_colours = {f"the {g} gate's colour": CASCADE_STYLES[g].color for g in gates}
    # The headline draws the references, the four routers' crosses and the pure cascades; a router's panel draws the
    # router, A (the configs' blue), the envelope, and the three lines. The headline's set holds the panel's.
    headline = {**drawn, **gate_colours}
    pairs = list(itertools.combinations(headline.items(), 2))
    check(len(pairs) == 45, "the headline figure holds ten colours, so 45 pairs")
    for (first, colour_a), (second, colour_b) in pairs:
        if first in gate_colours or second in gate_colours:
            check(
                gap(colour_a, colour_b) >= APART,
                f"{first} is {gap(colour_a, colour_b):.1f} from {second}, at least {APART} in every view",
            )
    panel = {"the configs' blue": SERIES_1, "the envelope's orange": SERIES_2, **gate_colours}
    for letter in (E, F, G, G_PLUS):
        mine = {f"{LABELS[letter]}'s colour": styles[letter].color, **panel}
        worst = min(
            gap(mine[a], mine[b]) for a, b in itertools.combinations(mine, 2) if a in gate_colours or b in gate_colours
        )
        check(worst >= APART, f"{letter}'s panel: each of the three lines is at least {APART} ({worst:.1f}) from its router, A, "
              "the envelope and the other lines")  # fmt: skip
    nudged = "#e65a2c"
    check(gap(nudged, SERIES_2) < APART, f"known positive: the envelope's orange nudged to {nudged} is "
          f"{gap(nudged, SERIES_2):.1f} from it, and the check says so")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
