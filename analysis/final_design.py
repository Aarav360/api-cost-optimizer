"""The numbers Amendment A19 declares about the final evaluation, computed from the committed tables (PLAN_v4.md;
Amendment A19, items 6 and 11; CODING_STANDARDS.md, "Numbers in documents").

Not a check: a module `analysis/check_final_design.py` (which holds A19's text against these numbers) and
`analysis/check_writeup.py` (which holds WRITEUP.md's skeleton against them) both import, so the expected misses and the
headline's dev numbers have one source and the write-up restates none of them. Reads no test item.

- `expected_misses()`: how many operating points route, and how many misses to expect on test (item 6).
- `headline()`: the cascade the headline's rule picks on dev and its point at the headline's mix and target.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

from analysis.operating_eval import MIXES
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.router_names import CASCADES, G_PLUS, NAMED_ROUTERS, G
from analysis.table_v4 import TABLE_V4
from harness.client import PROJECT_ROOT

TABLE_V3 = PROJECT_ROOT / "figures" / "table_v3.json"
HEADLINE_MIX, HEADLINE_TARGET = "realistic", "99%"  # item 6: "on the realistic mix at the 99% target"
NORMAL = NormalDist()


def points(by_mix: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """A policy's six operating points: two mixes, three quality targets."""
    return [point for mix in MIXES for point in by_mix[mix].values()]


def fresh_miss(dev_miss: float) -> float:
    """A point's chance of missing its target on a fresh draw of items, from the share of dev's resamples that miss it.
    Dev's estimate and the fresh one each carry the same sampling error, so their gap's is √2 times one (a normal
    approximation; it ignores the bias that choosing on dev adds)."""
    if dev_miss <= 0.0 or dev_miss >= 1.0:
        return dev_miss
    return NORMAL.cdf(NORMAL.inv_cdf(dev_miss) / 2**0.5)


@dataclass(frozen=True)
class Misses:
    """Item 6's counts: the operating points that route, and the misses expected over those tables v3 and v4 cover."""

    routed: int  # the cascades' operating points that route (not a fall back to A)
    cascades_every: int  # all the cascades' operating points: six each
    routers: dict[str, int]  # each named router's routing points, by letter
    covered: int  # the routing points tables v3 and v4 cover: the cascades', G's and G+'s
    dev_misses: float  # expected misses over them by dev's own resamples
    fresh_misses: float  # and allowing for a fresh draw of items
    at_margin: float  # a point that cleared by exactly the margin misses this share of dev's resamples
    fresh_at_margin: float  # and this share of fresh draws

    @property
    def routers_every(self) -> int:
        return 6 * len(self.routers)

    @property
    def total(self) -> int:
        return self.routed + sum(self.routers.values())


def expected_misses() -> Misses:
    """Item 6's counts, from the operating-points file and tables v3 and v4."""
    v3 = json.loads(TABLE_V3.read_text(encoding="utf-8"))["operating_points"]
    v4 = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    sections = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    routing = {name: sum(not p["fell_back"] for p in points(s)) for name, s in sections.items()}
    cascades = [name for name in sections if name in CASCADES]
    routers = {letter: routing[name] for letter, name in NAMED_ROUTERS.items()}
    covered = [p for letter in (G, G_PLUS) for p in points(v3[letter]) if not p["fell_back"]]
    covered += [p for c in v4["cascades"].values() for p in points(c["operating_points"]) if not p["fell_back"]]
    at_margin = NORMAL.cdf(-1.0)
    return Misses(
        routed=sum(routing[name] for name in cascades),
        cascades_every=6 * len(cascades),
        routers=routers,
        covered=len(covered),
        dev_misses=sum(1 - p["captured"]["met"] for p in covered),
        fresh_misses=sum(fresh_miss(1 - p["captured"]["met"]) for p in covered),
        at_margin=at_margin,
        fresh_at_margin=fresh_miss(at_margin),
    )


@dataclass(frozen=True)
class Headline:
    """The cascade item 6's rule picks on dev, with its point at the headline's mix and target."""

    best: str  # the cascade's section key
    saves: float  # share of A's cost saved there
    met: float  # share of dev's resamples in which it meets its target
    fell_back: bool


def headline() -> Headline:
    """The best cascade in table v4 and its dev point at `HEADLINE_MIX` and `HEADLINE_TARGET`."""
    v4 = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    best = v4["best_cascade"]["best"]
    point = v4["cascades"][best]["operating_points"][HEADLINE_MIX][HEADLINE_TARGET]
    return Headline(best, point["savings_vs_A"]["share"], point["captured"]["met"], point["fell_back"])
