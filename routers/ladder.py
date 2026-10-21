"""The config ladder and the frontier (PLAN_v4.md §8; Amendment A8, item 5; Amendment A9, item 8).

Moved here from analysis/matrix.py, which re-exports it and asserts that it ends at the collection runner's
frontier, so this package never imports the harness.
"""

# The config ladder, cheapest to dearest (Amendment A8, item 5): the order of mean dev
# cost per request in every task and both mixes. "Cheapest config" and "one config up"
# always mean this order, and exact cost ties go to the lower config.
LADDER = ("haiku", "sonnet", "opus_low", "opus")
FRONTIER = LADDER[-1]  # runs on dev and test only (Amendment A1, item 3); a router never scores it
SCORED = LADDER[:-1]  # the configs below the frontier: what a router scores
