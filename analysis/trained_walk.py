"""The walk on dev for the trained router, F (PLAN_v4.md §8 3b; Amendment A11, items 9, 13 and 15).

Run from the project root:  uv run python -m analysis.trained_walk

After the fit commit and dev's tables (`analysis/trained_table.py`): walks F's out-of-fold table on dev under both
task mixes for the three quality targets, with the walk Stage 4 built and A9 item 9's quantile steps (F's 744 x 3
breakpoints are far above the 24 that would each be a step), builds F's section of the operating-points file and
writes it under `NAMED_ROUTERS["F"]`, leaving E's section as it was. Only F is walked here: F-train's and the
task-rates diagnostic's walks come after, as diagnostics (item 15), and never enter the file (item 13).
"""

from analysis import matrix as results_matrix
from analysis.bootstrap import resample
from analysis.operating_points import NAMED_ROUTERS, build_section, write_section
from analysis.rules_walk import dev_walks
from analysis.trained_table import dev_table

ROUTER = NAMED_ROUTERS["F"]  # F's name in the operating-points file


def rebuilt_section() -> dict:
    """F's section as the walk on dev gives it now, for the check against the committed file."""
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
