"""The Router protocol and the decision rule (PLAN_v4.md §8; Amendment A9, item 3)."""

from collections.abc import Mapping
from typing import Protocol

from routers.ladder import FRONTIER, SCORED
from routers.request import Request


class Router(Protocol):
    def score(self, request: Request) -> dict[str, float]:
        """A router score from 0 to 1 for each config below the frontier (`haiku`, `sonnet`, `opus_low`).

        Never for the frontier, which is the fallback and needs no score.
        """
        ...


def choose(router_scores: Mapping[str, float], threshold: float) -> str:
    """The cheapest config on the ladder whose router score is at least the threshold, else the frontier.

    A router score equal to the threshold counts. A dearer config scored below a cheaper one is never chosen
    over it: the ladder is walked from the bottom and the first config to reach the threshold wins.
    """
    for config in SCORED:
        if router_scores[config] >= threshold:
            return config
    return FRONTIER
