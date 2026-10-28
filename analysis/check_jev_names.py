"""Check that G and G+ are named, allowed and styled before any Jev number exists (PLAN_v4.md §8 and §12; Amendment A15,
items 14 and 15).

Run from the project root:  uv run python -m analysis.check_jev_names

Reads no results matrix and calls no Jev, so CI runs it. That every figure JSON and PNG of E and F rebuilds unchanged
is `check_stage4`'s and `check_stage5`'s, which need `results/`.

- the router names: G is the Jev router and G+ the bolstered Jev router, with the sections `jev` and `jev_bolstered` and
  a label each, in `analysis/router_names.py`, and E's and F's names and labels are as they were;
- the operating-points allowlist accepts both sections, singly and together, accepts them as required sections, and
  still refuses F-train and the diagnostics (known positive: it refuses F-train in a file that also holds G's section);
- the chart's router list holds E and F only, so G and G+ are not drawn until Stage 6 · 12, and their two styles are
  reserved in `analysis/chart_routers.py`: a colour, a marker, a point marker and a line that no other router's shares,
  and a colour at least 15 apart (OKLab x100, in normal vision and under simulated protan, deutan and tritan vision)
  from every other colour the figures draw and from each other's (known positive: a teal that the configs' blue
  stands within 3 of).
"""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from analysis.chart_routers import E_STYLE, F_STYLE, G_PLUS_STYLE, G_STYLE, ROUTERS
from analysis.check_chart_routers import APART, gap
from analysis.operating_points import OperatingPointsError, check_sections, write_section
from analysis.pareto import SERIES_1, SERIES_2, SERIES_3
from analysis.router_names import F_TRAIN, G_PLUS, LABELS, NAMED_ROUTERS, TASK_RATES, E, F, G

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except OperatingPointsError:
        return True
    return False


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
        check(refused(lambda: check_sections(only, required=(jev, bolstered))),
              "a required section that is missing is refused")  # fmt: skip
        for name in (F_TRAIN, TASK_RATES, "jev_diagnostic"):
            other = Path(folder) / "other.json"
            other.write_text(both.read_text())
            write_section(name, {}, other)
            check(refused(lambda: check_sections(other)),
                  f"known positive: {name!r} in a file that holds G's and G+'s sections is refused")  # fmt: skip

    print("\n-- the chart's router list and the two reserved styles --")
    check([r.name for r in ROUTERS] == [NAMED_ROUTERS[E], NAMED_ROUTERS[F]],
          "the router list still holds E and F only: G and G+ are drawn from Stage 6 · 12")  # fmt: skip
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

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
