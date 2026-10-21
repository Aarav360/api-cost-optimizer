"""McNemar's test of two policies within one task (PLAN_v4.md §4; Amendment A9, item 11).

Every task scores 1 or 0 (Amendment A5, item 5), so two policies' outcomes on the same items split into pairs both
get right or wrong and the discordant ones, where only one is right. The test is exact and two-sided: a binomial
on the discordant pairs, computed with `math.comb`. It is descriptive only: no multiplicity correction, and p > 0.05
never reads as "matches". Under the task mixes, where items weigh unequally, the paired bootstrap interval of the
difference is the test instead, so nothing here takes a mix. Showing "—" for a row that falls back to A is the
table's job.
"""

import math
from dataclasses import dataclass

from analysis.simulator import Outcome


class McNemarError(ValueError):
    """Outcomes McNemar's test refuses: not the same items, not in one task, or not scored 1 or 0. Nothing is tested."""


@dataclass(frozen=True)
class McNemar:
    b: int  # items only the first policy gets right
    c: int  # items only the second policy gets right
    p: float  # exact, two-sided, 1 where there are no discordant pairs
    b_favours: str  # the first policy's name
    c_favours: str  # the second policy's name


def mcnemar(first: list[Outcome], second: list[Outcome], first_name: str, second_name: str) -> McNemar:
    """Two policies' outcomes on the same items of one task: the two discordant counts and the exact p.

    With b and c the discordant counts and n = b + c, p = min(1, 2 · Σ_{k ≤ min(b, c)} C(n, k) / 2ⁿ), and 1 where
    n = 0. The counts name the policy each favours. Refuses (McNemarError) outcomes that aren't the same items,
    each once, that span more than one task, or whose success isn't 1 or 0.
    """
    if not first or not second:
        raise McNemarError("McNemar's test needs outcomes, and one set has no outcomes")
    by_first = {outcome.item_id: outcome for outcome in first}
    by_second = {outcome.item_id: outcome for outcome in second}
    if len(by_first) != len(first) or len(by_second) != len(second) or by_first.keys() != by_second.keys():
        raise McNemarError(f"the two policies' outcomes must be of the same items, each once, but there are "
                           f"{len(first)} outcomes for {len(by_first)} items and {len(second)} for {len(by_second)}, "
                           f"and {len(by_first.keys() ^ by_second.keys())} items are in only one set")  # fmt: skip
    if len(tasks := {outcome.task for outcome in (*first, *second)}) != 1:
        raise McNemarError(f"McNemar's test compares policies within one task, but the outcomes span {sorted(tasks)}")
    # NaN fails the comparison too, so a success that isn't a number is refused here.
    if bad := [o.success for o in (*first, *second) if not (o.success == 0 or o.success == 1)]:
        raise McNemarError(f"every success must be 0 or 1, and {len(bad)} aren't, e.g. {bad[0]}")
    b = sum(by_first[i].success == 1 and by_second[i].success == 0 for i in by_first)
    c = sum(by_first[i].success == 0 and by_second[i].success == 1 for i in by_first)
    n = b + c
    # Exact integers, divided once: the quotient is correctly rounded however large 2ⁿ is, as long as p fits a float.
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p = 1.0 if n == 0 else min(1.0, 2 * tail / 2**n)
    return McNemar(b, c, p, first_name, second_name)
