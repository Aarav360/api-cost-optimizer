"""The walk on dev for the rules router, E (PLAN_v4.md §8; Amendment A9, items 9, 10 and 13).

Run from the project root:  uv run python -m analysis.rules_walk

The last of the three steps item 13 puts after the freeze: E's dev router-score table, the dev task-inference
diagnostic, then this. It walks E's table on dev under both task mixes for the three quality targets (§11), builds
E's section of the operating-points file and writes it, leaving every other router's section as it was. Chart and
table v1 (`analysis/pareto.py`) and the record (item 18) read the walks and that file, never a new selection.
"""

from analysis import matrix as results_matrix
from analysis.bootstrap import Bootstrap, resample
from analysis.matrix import Matrix
from analysis.operating_points import build_section, write_section
from analysis.references import TARGETS
from analysis.router_table import RouterScoreTable
from analysis.rules_table import dev_table
from analysis.simulator import load_mixes
from analysis.walk import Walk, walk

ROUTER = "rules"  # E's name in the operating-points file
MIXES = ("realistic", "uniform")  # the walk's task mixes, in the figure's panel order


def dev_walks(dev: Matrix, table: RouterScoreTable, draws: Bootstrap) -> dict[str, dict[float, Walk]]:
    """One walk per task mix and quality target on dev, keyed by mix name and then target share."""
    mixes = load_mixes()
    return {mix: {share: walk(dev, table, draws, mixes[mix], share) for share in TARGETS} for mix in MIXES}


def rebuilt_section() -> dict:
    """E's section as the walk on dev gives it now, for the check against the committed file."""
    dev = results_matrix.load().split("dev")
    return build_section(dev_walks(dev, dev_table(), resample(dev)))


def main() -> None:
    dev = results_matrix.load().split("dev")
    walks = dev_walks(dev, dev_table(), resample(dev))
    write_section(ROUTER, build_section(walks))
    for mix, by_share in walks.items():
        for share, result in by_share.items():
            operating = result.operating
            steps = len(result.certified) - 1  # A is certified without a step
            where = "falls back to A" if result.fell_back else f"threshold {operating.threshold:.4f}"
            print(f"{mix:9} {share:4.0%} of A: {where}; {steps} of {len(result.steps) - 1} steps tried certified; "
                  f"quality {100 * operating.quality:.1f}%, ${operating.cost_per_1k:.2f} per 1k")  # fmt: skip


if __name__ == "__main__":
    main()
