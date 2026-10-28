"""G's and G+'s dev router-score tables, and their router-score quantiles (PLAN_v4.md §8 3c and 3c+; Amendment A15,
items 9, 10, 12, 13 and 14; issue #40).

Run from the project root:  uv run python -m analysis.jev_table

After the freeze commit and the dev collection (`analysis/collect_jev.py`): the first three dev steps of item 13 after
the collection. It reads no outcome and calls no Jev.

- A table is built from the stored dev records through the router's own parser (`collect_jev.record_scores`, which
  takes the path a live call takes: G's answers unchanged, G+'s through the frozen weights), so table and router can't
  drift apart (item 12).
- Each trace's overhead is the call's stored input tokens at the settings' price, $0.042 per million (item 10).
- A refused response is a router score of 0 for every config (item 9). The stored flag must agree with the response.
- Every dev item must have exactly one pass 1 record. The table's own refusal (`RouterScoreTable.build`) then covers
  the rest: a missing or repeated item, a score outside [0, 1], an overhead that isn't a number >= 0.
- The quantiles are of the table's router scores alone, per config: they read no outcome (item 14).
"""

import functools
from collections.abc import Mapping, Sequence

import numpy as np

from analysis import collect_jev
from analysis.benchmark import read_rows
from analysis.router_table import ItemRouterScores, RouterScoreTable
from routers import jev
from routers.ladder import SCORED
from routers.request import Request, from_row

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)  # the reported quantiles of a router's scores on dev (item 14)


class JevTableError(ValueError):
    """Stored records this module refuses whole: a missing, repeated, mislabelled or contradictory record."""


def build_table(
    router: str,
    records: Sequence[Mapping[str, object]],
    requests: Mapping[str, Request],
    item_ids: Sequence[str],
    loaded: collect_jev.Loaded,
    weights: jev.GPlusWeights | None = None,
) -> RouterScoreTable:
    """The table over `item_ids`, in that order, from `router`'s stored dev records, or a refusal.

    Refused whole: an unknown router, a record that isn't `router`'s dev pass 1 record, an item stored twice or not at
    all, a stored refusal flag that disagrees with the response, and whatever the table itself refuses."""
    if router not in collect_jev.ROUTERS:
        raise JevTableError(f"unknown router {router!r}")
    by_item: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record["router"] != router or record["split"] != "dev" or record["pass"] != 1:
            raise JevTableError(
                f"{record['item_id']} is a {record['router']} {record['split']} pass {record['pass']} record"
            )
        item = str(record["item_id"])
        if item in by_item:
            raise JevTableError(f"{item} is stored twice")
        by_item[item] = record
    if missing := [item for item in item_ids if item not in by_item]:
        raise JevTableError(f"{len(missing)} dev items have no stored {router} record, e.g. {missing[0]}")
    if extra := sorted(set(by_item) - set(item_ids)):
        raise JevTableError(f"{router} has records for items that are not dev's, e.g. {extra[0]}")
    entries = []
    for item in item_ids:
        record = by_item[item]
        try:
            scored = collect_jev.record_scores(record, requests[item], loaded, weights)
        except collect_jev.CollectError as error:
            raise JevTableError(str(error)) from error
        if record["refused"] != scored.refused:
            raise JevTableError(
                f"{item}: the stored refusal flag is {record['refused']}, but the parser reads the response as "
                f"refused = {scored.refused}"
            )
        entries.append(ItemRouterScores(item, scored.scores, collect_jev.overhead_dollars(record, loaded)))
    return RouterScoreTable.build(entries, list(item_ids))


@functools.cache
def dev_table(router: str) -> RouterScoreTable:
    """`router`'s table over every dev item, in the reader's order, from the stored dev records (a router is
    `collect_jev.ROUTERS`' "jev" for G or "jev_bolstered" for G+)."""
    loaded = collect_jev.load_settings()
    weights = jev.read_weights(loaded.call) if router == "jev_bolstered" else None
    rows = read_rows(split="dev")
    records = collect_jev.read_records(router, "dev")
    return build_table(
        router, records, {r["item_id"]: from_row(r) for r in rows}, [r["item_id"] for r in rows], loaded, weights
    )


def quantiles(table: RouterScoreTable) -> dict[str, dict[float, float]]:
    """Per config, `QUANTILES` of the table's router scores (numpy's linear quantiles; the minimum and maximum are the
    smallest and largest). Reads the table alone, so no outcome."""
    return {
        config: dict(zip(QUANTILES, np.quantile([e.router_scores[config] for e in table.items], QUANTILES).tolist()))
        for config in SCORED
    }


def share_at_or_above(table: RouterScoreTable, config: str, value: float) -> float:
    """The share of items whose router score for `config` is at least `value`: how much the scores bunch at the top."""
    return sum(e.router_scores[config] >= value for e in table.items) / len(table.items)


def call_lines() -> list[str]:
    """What the dev collection left behind, in the words A15 item 19 uses for train's, from the stored records alone:
    the counts, the first call's server `Date` (item 20 records it) and latency (item 10, dev's, as measured)."""
    from analysis import jev_train_report as report  # a report module, imported here as it needs the benchmark reader

    loaded = collect_jev.load_settings()
    rows = read_rows(split="dev")
    task_of = {row["item_id"]: row["source"] for row in rows}
    requests = {row["item_id"]: from_row(row) for row in rows}
    lines, dates = [], []
    for router in collect_jev.ROUTERS:
        records = collect_jev.read_records(router, "dev")
        counts = report.call_counts(records)
        rebuilt = report.rebuild(router, records, requests, loaded.call)
        calls, timing = report.record_lines(router, counts, rebuilt, report.latency(records, task_of))
        lines += [f"calls   {calls}", f"latency {timing}"]
        dates.append(report.first_date(records))
    first = min(dates).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return [*lines, f"first dev call, server Date: {first}"]


def main() -> None:
    for router, name in ((collect_jev.ROUTERS[0], "G"), (collect_jev.ROUTERS[1], "G+")):
        table = dev_table(router)
        refused = len(collect_jev.refused_items(collect_jev.read_records(router, "dev"), collect_jev.load_settings()))
        distinct = len({tuple(e.router_scores[c] for c in SCORED) for e in table.items})
        print(f"{name}'s dev router-score table: {len(table.items)} dev requests, {refused} refused, {distinct} distinct "
              f"triples of router scores; it passes the table's refusal")  # fmt: skip
        print(f"  router scores by quantile ({', '.join(f'{q:.0%}' for q in QUANTILES)}):")
        for config, reading in quantiles(table).items():
            print(f"  {config:9}: " + "  ".join(f"{v:.4f}" for v in reading.values()))
    print("\n" + "\n".join(call_lines()))


if __name__ == "__main__":
    main()
