"""The request a router sees (PLAN_v4.md §8; Amendment A9, items 1 and 7).

A request is what every config receives for an item: its prompt and, if it has one, its output schema. The
conversion from a benchmark row reads those two stored fields, the ones the collection run sent, and nothing else,
so a router can't see an item's task, source group, split, `meta` or expected answer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    prompt: str
    output_schema: dict | None = None  # the API carries it as `output_config.format`


def from_row(row: dict) -> Request:
    """A benchmark row's request, from its `prompt` and `output_schema` only."""
    return Request(row["prompt"], row.get("output_schema"))
