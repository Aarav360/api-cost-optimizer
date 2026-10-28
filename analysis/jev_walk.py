"""The walks on dev for G and G+, and their sections of the operating-points file (PLAN_v4.md §8 3c and 3c+; Amendment
A9, items 9, 10 and 13; Amendment A15, items 13 and 14; issue #40).

Run from the project root:  uv run python -m analysis.jev_walk

After the dev collection and dev's tables (`analysis/jev_table.py`): walks each Jev router's dev table under both task
mixes for the three quality targets, with the walk Stage 4 built (`analysis/rules_walk.dev_walks`) and A9 item 9's
steps, builds the router's section and writes it under `NAMED_ROUTERS`, leaving every other router's section as it
was. G's name there is `jev`, G+'s is `jev_bolstered`.

The steps come from the router-score table alone, before any outcome is looked up. The comparisons that read outcomes
beyond the walks themselves (item 14) come after, in Stage 6 · 11.
"""

from dataclasses import dataclass

from analysis import matrix as results_matrix
from analysis.bootstrap import Bootstrap, resample
from analysis.jev_table import dev_table
from analysis.operating_points import build_section, write_section
from analysis.references import TARGETS
from analysis.router_names import G_PLUS, NAMED_ROUTERS, G
from analysis.router_panels import step_outcomes
from analysis.router_table import RouterScoreTable
from analysis.rules_walk import MIXES, dev_walks
from analysis.simulator import always, load_mixes, simulate, totals
from analysis.walk import Walk
from routers.ladder import FRONTIER

ROUTERS = (G, G_PLUS)  # the routers walked here, in the order the record reads them


@dataclass(frozen=True)
class Reading:
    """One operating point's numbers under its mix: where the walk put it and what it costs against A."""

    threshold: float | None  # None: falls back to A
    quality: float
    cost_per_1k: float
    savings_share: float  # 1 - cost / A's cost
    captured: float | None  # the captured fraction on dev, None where n/a


def table_of(router: str) -> RouterScoreTable:
    """A named router's dev table (`NAMED_ROUTERS`' section name is the records' router name)."""
    return dev_table(NAMED_ROUTERS[router])


def walks_of(router: str, draws: Bootstrap | None = None) -> dict[str, dict[float, Walk]]:
    dev = results_matrix.load().split("dev")
    return dev_walks(dev, table_of(router), draws or resample(dev))


def rebuilt_section(router: str) -> dict:
    """A router's section as the walk on dev gives it now, for the check against the committed file."""
    return build_section(walks_of(router))


def readings(router: str, walks: dict[str, dict[float, Walk]], draws: Bootstrap) -> dict[str, dict[float, Reading]]:
    """Each operating point's quality, cost, savings against A and captured fraction (dev, no interval)."""
    dev = results_matrix.load().split("dev")
    frontier = simulate(dev, always(dev, FRONTIER))
    steps = step_outcomes(dev, table_of(router), frontier)
    mixes = load_mixes()
    out: dict[str, dict[float, Reading]] = {}
    for mix in MIXES:
        savings = draws.savings(dev, frontier, mixes[mix])
        a_cost = totals(frontier, mixes[mix]).cost_per_1k
        out[mix] = {}
        for share in TARGETS:
            point = walks[mix][share].operating
            total = totals(steps[point.threshold], mixes[mix])
            out[mix][share] = Reading(
                point.threshold, total.quality, total.cost_per_1k, 1 - total.cost_per_1k / a_cost,
                savings.captured(steps[point.threshold], share).value,
            )  # fmt: skip
    return out


def main() -> None:
    dev = results_matrix.load().split("dev")
    draws = resample(dev)
    for router in ROUTERS:
        walks = walks_of(router, draws)
        write_section(NAMED_ROUTERS[router], build_section(walks))
        print(f"{router}:")
        for mix, by_share in readings(router, walks, draws).items():
            for share, r in by_share.items():
                result = walks[mix][share]
                where = "falls back to A" if r.threshold is None else f"threshold {r.threshold:.4f}"
                tried = len(result.steps) - 1
                print(f"  {mix:9} {share:4.0%} of A: {where}; {len(result.certified) - 1} of {tried} steps tried "
                      f"certified; quality {100 * r.quality:.1f}%, ${r.cost_per_1k:.2f} per 1k, savings against A "
                      f"{100 * r.savings_share:.1f}%, captured "
                      f"{'n/a' if r.captured is None else f'{r.captured:.2f}'}")  # fmt: skip


if __name__ == "__main__":
    main()
