"""G and G+ on test: their drift check, their Jev calls, their router-score tables and what the tables step reports
(PLAN_v4.md Amendment A15, item 17; A19, items 2, 9 and 10; issue #77, #84).

**The drift check** (`drift_step`, A19 item 2.3) runs before any call on test. The noise sample, 100 train items (A15
item 10), is sent once more through the frozen G and G+, into the drift files of their own (`results/jev/jev_drift.jsonl`
and `jev_bolstered_drift.jsonl`, apart from Stage 7's). For each router and config it measures D6 (the noise Stage 6's two
train passes already had, by `fit_jev_router.noise_difference`, which is frozen and only imported), D8 (the resend's mean
absolute difference from Stage 6's first pass) and the shift (the mean signed difference). The shared rule
(`analysis/drift_rule.py`) refuses if a returned model ID isn't `jev-1.14.0`, if D8 > 3 x D6 or if a shift is larger than
D6 in size, before any test call; otherwise the run goes on with D6, D8, the shifts and the decisions that flip at each
operating threshold in its record. The resend goes through the frozen call building and the router's own client, so its
bodies are what the freeze hashed.

**The calls** (`calls_step`, A19 item 2.4) send every test request once through each router, with the frozen code, from the
unlocked handle's requests, into `jev_test.jsonl` and `jev_bolstered_test.jsonl`. A record is stored before the next call, so
a crash loses nothing, and a resume sends only the requests with no stored response (a stored response is never drawn
again). Both steps ask the ledger first (`analysis/final_ledger.py`, the $25 pool): a projection over the cap refuses before
the first call. The record keeps the first test call's server `Date` (the order evidence dev's record has). The refusal and
truncation counts are not said here: they go with the tables (A19 item 5), through `facts_record`.

**The tables** come from the stored responses through the parser (`analysis/jev_table.py`'s `build_table`, which dev's tables
use too, here for `split="test"`), and two parts evaluate each router at its frozen operating points
(`analysis/offline_half.py`). Nothing here reads a grade.

A rehearsal's world names a `JevStandIn` (`analysis/backends.py`); a world that names backends and no stand-in is refused,
so a rehearsal can't reach the real service. The go's world names none and gets the real settings, weights and key.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis import collect_jev, jev_train_report
from analysis.collect_jev import STAGE_8_SPLITS, Item
from analysis.drift_rule import JEV_PIN, DriftReport, apply_rule, decision_flips, measures_of
from analysis.final_ledger import check_projection
from analysis.final_run import ItemFailure, Run, StepRefusal, StepResult
from analysis.fit_jev_router import noise_difference
from analysis.gate_spend import jev_projected_cost
from analysis.jev_table import build_table
from analysis.operating_points import OPERATING_POINTS_FILE, read_section
from analysis.router_names import LETTER_OF
from analysis.router_table import RouterScoreTable
from analysis.world import World
from routers import jev
from routers.ladder import SCORED
from routers.request import Request

DRIFT_CHECK = "drift_jev_router"
CALLS_STEP = "calls_jev_router"
ROUTERS = collect_jev.ROUTERS  # G's and G+'s section names, `jev` and `jev_bolstered`
DRIFT, TEST = "drift", "test"  # the labels of the two Stage 8 files of each router


class JevStepRefusal(StepRefusal):
    """A Jev router step that can't run as asked: a world with no Jev, or stored records the check can't read."""


@dataclass(frozen=True)
class Setup:
    """What both steps need: the settings, G+'s weights, and how to open a client."""

    loaded: collect_jev.Loaded
    weights: jev.GPlusWeights
    client: Callable[[], jev.JevClient]
    train_results: Path  # where Stage 6's train records are read from


def setup_of(world: World) -> Setup:
    """The real settings, weights and key for the go's world, the world's own stand-in for a rehearsal's. A world that names
    backends and no stand-in is refused: only the go's world gets the real Jev."""
    backends = world.backends
    if backends is None:
        loaded = collect_jev.load_settings()
        return Setup(loaded, jev.read_weights(loaded.call), lambda: jev.JevClient(loaded.client), world.results)
    stand_in = backends.jev
    if stand_in is None:
        raise JevStepRefusal(
            "the world names backends but no Jev stand-in: only the go's world, which names none, gets the real Jev"
        )
    return Setup(stand_in.loaded, stand_in.weights, stand_in.client, stand_in.train_results or world.results)


def weights_of(router: str, setup: Setup) -> jev.GPlusWeights | None:
    return setup.weights if collect_jev.CALLS[router].weighted else None


def scores_of(
    router: str, records: Mapping[str, Mapping[str, Any]], requests: Mapping[str, Request], setup: Setup
) -> dict[str, dict[str, float]]:
    """`{item: {config: router score}}` of stored records, through the parser the live call takes."""
    weights = weights_of(router, setup)
    return {
        item: collect_jev.record_scores(record, requests[item], setup.loaded, weights).scores
        for item, record in records.items()
    }


def by_item(records: list[dict], pass_number: int) -> dict[str, dict]:
    return {r["item_id"]: r for r in records if r["pass"] == pass_number}


# The drift check


def drift_labels(router: str) -> dict[float, list[str]]:
    """Each frozen threshold of the router and the operating points that sit on it ("realistic 99%"); a point that fell back
    holds none."""
    labels: dict[float, list[str]] = {}
    for mix, targets in read_section(router, OPERATING_POINTS_FILE).items():
        for target, point in targets.items():
            if not point["fell_back"] and point["threshold"] is not None:
                labels.setdefault(point["threshold"], []).append(f"{mix} {target}")
    return labels


def train_passes(router: str, setup: Setup, items: list[Item]) -> tuple[dict, dict, dict]:
    """Stage 6's two stored train passes over the noise sample: the first pass's scores, the second's, and D6 by config
    (`fit_jev_router.noise_difference`, the frozen arithmetic A15 item 19 recorded D6 from). Refused (`JevStepRefusal`) if a
    pass is missing for a noise item."""
    requests = {item.item_id: item.request for item in items}
    train = collect_jev.read_records(router, "train", setup.train_results)
    first, second = by_item(train, 1), by_item(train, 2)
    for what, held in (("pass 1", first), ("pass 2", second)):
        if lacking := [i for i in requests if i not in held]:
            raise JevStepRefusal(f"{DRIFT_CHECK} ({LETTER_OF[router]}): {router} has no stored {what} for {len(lacking)} of the "
                                 f"noise sample's items, such as {lacking[0]}")  # fmt: skip
    first_scores = scores_of(router, {i: first[i] for i in requests}, requests, setup)
    second_scores = scores_of(router, {i: second[i] for i in requests}, requests, setup)
    return first_scores, second_scores, noise_difference(first_scores, second_scores, SCORED)


def measure_drift(router: str, setup: Setup, world: World, items: list[Item]) -> DriftReport:
    """One router's drift check on the stored drift records: D6 from train's two passes, D8 and the shift of the resend from
    the first, the flips, and the rule. Refuses (`DriftRefusal`) per `analysis/drift_rule.py`."""
    check = f"{DRIFT_CHECK} ({LETTER_OF[router]})"
    requests = {item.item_id: item.request for item in items}
    first_scores, _, noise = train_passes(router, setup, items)
    resent = by_item(collect_jev.read_records(router, DRIFT, world.results, STAGE_8_SPLITS), 1)
    if lacking := [i for i in requests if i not in resent]:
        raise JevStepRefusal(f"{check}: {router} has no stored drift resend for {len(lacking)} of the noise sample's items, "
                             f"such as {lacking[0]}")  # fmt: skip
    resend_scores = scores_of(router, {i: resent[i] for i in requests}, requests, setup)
    resend_noise = noise_difference(first_scores, resend_scores, SCORED)
    models: list[object] = []
    for record in resent.values():
        response = record["response"]
        models += [record["model"], response.get("model") if isinstance(response, dict) else None]
    flips = decision_flips(first_scores, resend_scores, drift_labels(router))
    return apply_rule(check, pin=JEV_PIN, model_ids=models, measures=measures_of(noise, resend_noise), flips=flips)


def drift_step(run: Run) -> StepResult:
    """A15 item 17's check: the noise sample sent once more through the frozen G and G+, into the drift files, and the shared
    rule applied to each router. A refusal stops the run before any test call; the resends stay stored, and a resume asks
    for none of them again."""
    check_projection(run)  # the cap refuses here, before the first resend
    setup = setup_of(run.world)
    items = collect_jev.noise_items(setup.loaded, run.world.benchmark)
    sent = stored = 0
    reports: dict[str, Any] = {}
    with setup.client() as client:
        for router in ROUTERS:
            current: list[str] = []
            try:
                summary = collect_jev.collect(
                    items, router=router, split=DRIFT, loaded=setup.loaded, client=client, results_dir=run.world.results,
                    splits=STAGE_8_SPLITS, before_call=current.append,
                )  # fmt: skip
            except Exception as error:
                raise ItemFailure(current[-1] if current else f"{router} {DRIFT}") from error
            sent, stored = sent + summary.new, stored + summary.skipped
            report = measure_drift(router, setup, run.world, items)
            reports[LETTER_OF[router]] = {**report.record(), "resent": summary.new, "stored": summary.skipped}
            run.say(describe(LETTER_OF[router], report))
    return StepResult(requests={"sent": sent, "stored": stored}, detail={"routers": reports})


def describe(letter: str, report: DriftReport) -> str:
    """The step's line: D6, D8 and the shift of each config, and the decisions that flip at each threshold: train items'
    numbers only."""
    measures = "; ".join(
        f"{config} D6 {m.d6:.4f} D8 {m.d8:.4f} shift {m.shift:+.4f}" for config, m in report.measures.items()
    )
    flipped = {f.threshold: len(f.items) for f in report.flips}
    return f"{DRIFT_CHECK}: {letter} on {', '.join(report.model_ids)}: {measures}; decisions that flip by threshold {flipped}"


# The projection and the calls on test


def pending(router: str, items: list[Item], split: str, results: Path) -> list[Item]:
    done = {r["item_id"] for r in collect_jev.read_records(router, split, results, STAGE_8_SPLITS)}
    return [item for item in items if item.item_id not in done]


def test_items(run: Run) -> list[Item]:
    return [Item(item_id, request) for item_id, request in run.unlocked().requests().items()]


def project(run: Run) -> float:
    """An upper bound on what G's and G+'s drift resends and test calls still cost: for each request with no stored
    response, its built body's bytes plus the fixed overhead as tokens (`gate_spend.jev_projected_cost`), at Jev's price. The
    $25 pool's projection (`final_ledger.PROJECTORS`)."""
    setup = setup_of(run.world)
    price = setup.loaded.client.price_per_million_input_tokens
    drift = collect_jev.noise_items(setup.loaded, run.world.benchmark)
    test = test_items(run)
    lengths: list[int] = []
    for router in ROUTERS:
        budget = setup.loaded.call.budget_bytes[router]
        build = collect_jev.CALLS[router].build
        for items, split in ((drift, DRIFT), (test, TEST)):
            lengths += [len(build(i.request, setup.loaded.call, budget_bytes=budget).body)
                        for i in pending(router, items, split, run.world.results)]  # fmt: skip
    return jev_projected_cost(lengths, price)


def first_test_call(results: Path) -> dict[str, str]:
    """The first test Jev call's server `Date`, as the header said it and as a time: the order evidence A19 item 10 asks for,
    read from the stored records (the earliest `collected_at` of G's and G+'s first records)."""
    firsts = []
    for router in ROUTERS:
        records = collect_jev.read_records(router, TEST, results, STAGE_8_SPLITS)
        if records:
            firsts.append((datetime.fromisoformat(str(records[0]["collected_at"])), router, records))
    if not firsts:
        raise JevStepRefusal("no test Jev call is stored")
    _, router, records = min(firsts, key=lambda found: found[0])
    return {
        "first_test_jev_date": str(records[0]["server_date"]),
        "first_test_jev_date_parsed": jev_train_report.first_date(records).isoformat(),
        "first_test_jev_router": LETTER_OF[router],
    }


def calls_step(run: Run) -> StepResult:
    """A19 item 2.4 for G and G+: every test request once through each router, through the handle, into the test files. Says
    only counts: the refusals and truncations are reported with the tables."""
    items = test_items(run)
    account = check_projection(run)  # a projection over the cap refuses before the first call
    setup = setup_of(run.world)
    sent = stored = 0
    with setup.client() as client:
        for router in ROUTERS:
            current: list[str] = []
            try:
                summary = collect_jev.collect(
                    items, router=router, split=TEST, loaded=setup.loaded, client=client, results_dir=run.world.results,
                    splits=STAGE_8_SPLITS, before_call=current.append,
                    progress=progress_of(run, LETTER_OF[router]),
                )  # fmt: skip
            except Exception as error:
                raise ItemFailure(current[-1] if current else f"{router} {TEST}") from error
            sent, stored = sent + summary.new, stored + summary.skipped
            run.say(f"{CALLS_STEP}: {LETTER_OF[router]}: {summary.new} sent, {summary.skipped} already stored, "
                    f"of {len(items)} test requests")  # fmt: skip
    return StepResult(
        requests={"sent": sent, "stored": stored}, detail={**first_test_call(run.world.results), "spend": account}
    )


def progress_of(run: Run, letter: str) -> Callable[[int, int], None]:
    """The collection's progress callback: a line every hundred calls, counts only."""

    def show(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            run.say(f"{CALLS_STEP}: {letter}: {done} of {total}")

    return show


# What the tables step reads


def test_table(router: str) -> Callable[[Run], RouterScoreTable]:
    """`router`'s router-score table over the handle's requests, from the stored test records through the parser."""

    def build(run: Run) -> RouterScoreTable:
        setup = setup_of(run.world)
        requests = run.unlocked().requests()
        records = collect_jev.read_records(router, TEST, run.world.results, STAGE_8_SPLITS)
        return build_table(
            router, records, requests, list(requests), setup.loaded, weights_of(router, setup), split=TEST
        )

    return build


def dev_table(router: str) -> Callable[[], RouterScoreTable]:
    def build() -> RouterScoreTable:
        from analysis import jev_table

        return jev_table.dev_table(router)

    return build


@dataclass(frozen=True)
class Facts:
    """G's and G+'s test calls as the stored records say: counts the run reports with the tables, not before (a refusal and a
    cut both carry something about the requests' outcomes, A19 item 5)."""

    calls: dict[str, int]
    refused: dict[str, int]
    cut: dict[str, int]
    mismatched: dict[str, int]  # stored body hashes that are not what the frozen code builds
    models: dict[str, list[str]]

    def record(self) -> dict[str, Any]:
        return {
            LETTER_OF[router]: {"calls": self.calls[router], "refused": self.refused[router], "cut": self.cut[router],
                                "hash_mismatches": self.mismatched[router], "models": self.models[router]}
            for router in ROUTERS
        }  # fmt: skip


def facts(run: Run) -> Facts:
    setup = setup_of(run.world)
    requests = run.unlocked().requests()
    calls, refused, cut, mismatched, models = {}, {}, {}, {}, {}
    for router in ROUTERS:
        records = collect_jev.read_records(router, TEST, run.world.results, STAGE_8_SPLITS)
        rebuilt = jev_train_report.rebuild(router, records, requests, setup.loaded.call)
        calls[router], cut[router], mismatched[router] = len(records), rebuilt.cut, len(rebuilt.mismatched)
        refused[router] = len(collect_jev.refused_items(records, setup.loaded))
        models[router] = sorted({str(r["model"]) for r in records})
    return Facts(calls, refused, cut, mismatched, models)


def facts_record(value: Facts | None) -> Mapping[str, Any]:
    return {} if value is None else value.record()
