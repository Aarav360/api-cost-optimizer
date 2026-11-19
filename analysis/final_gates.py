"""The free checks and the Jev gate on test, with the Jev gate's drift check (PLAN_v4.md Amendment A19, items 2, 7, 9, 10 and
13; A17 items 13, 14 and 26; A15 item 17; issue #77, #85).

Three steps of the final evaluation, in A19 item 2's order, and what builds test's gate-score table from what they store:

- **`drift_step`** (`drift_jev_gate`, before any call on test): the Jev gate's noise sample, the 100 `haiku` train answers
  Stage 7 sent twice, is sent a third time through the frozen gate call, into `results/gates/jev_gate_drift.jsonl` and
  nowhere else. D6 is Stage 7's own two passes (`analysis/gate_train.jev_noise`), D8 and the shift are the resend's against
  the first pass, and `analysis/drift_rule.py` (#84's, shared with the Jev routers' and the Sonnet gate's checks) applies A15
  item 17's rule: a model ID that isn't the pin, D8 above 3 x D6, or a shift larger than D6 in size refuses
  (`DriftRefusal`), with every call on test still unmade. Otherwise it reports D6, D8, the shifts and the decisions that flip
  at each frozen Jev gate cutoff (a cascade that ends in the Jev gate, `analysis/router_names.CASCADES`), and goes on. Each
  resend's body must hash to the one Stage 7 stored for that answer, so the frozen code is what is sent.
- **`free_checks_step`** (`calls_free_checks`): the three free checks on every test answer, whatever its config, through the
  handle's answers read, into `free_checks_test.jsonl`. The sandbox is the world's (a stand-in in CI, the real one in the
  hook and at the go).
- **`jev_gate_step`** (`calls_jev_gate`): the Jev gate on every answer that passed the free checks, into
  `jev_gate_test.jsonl`, under the ledger (`analysis/final_ledger.py`): the pending calls' cost is projected and refused over
  the cap before the first call, and a resume resends only the answers with no stored record.

**What the gates see.** A gate is given a `GateInput`: the request (prompt and schema), the answer's text and its stop reason,
all from the handle's `requests()` and `answers()` reads, which hold nothing else (A17 item 2, A19 item 4). This module never
reads the handle's graded items or an expected answer, and `analysis/check_final_gates.py` holds that by reading this file and
by running the steps on a made-up world with markers planted in what no gate may see.

**Nothing is shown before the tables.** The steps say how many answers they checked or sent and how many were already stored,
never how many passed, were refused or cut: a count of free-check rejections or Jev refusals is an outcome, and comes with
the tables (`gate_data_record`, an `Extra`'s record). A failure names the answer and the exception's class only.

**Test's gate-score table** (`test_table`, `build_test_table`) is built from the stored records through the gates' own parsers
and refused whole on each of A17 item 13's conditions: an answer missing or repeated, a gate score that isn't a number from 0
to 1, a paid score on an answer that failed a free check, an overhead that is negative or not finite, a call missing from a
passing answer, and a stored refusal flag that differs from the parser's. #86 adds the Sonnet gate's column through
`build_test_table(..., extra_paid=...)`.

**The pure cascades** `haiku/free` and `haiku/Jev` (`cascade_part`) are evaluated at their frozen operating points and cutoffs
with `analysis/operating_eval.py`: the file decides each point's dials and whether it fell back, and nothing is walked on
test. Run on dev's stored table, the same code reproduces table v4's rows for them (`committed_part`).
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis import collect_jev_gate, drift_rule
from analysis.collect_gates import (
    FREE,
    FREE_RECORD_FIELDS,
    GateCollectError,
    GateInput,
    gate_inputs,
    read_gate_records,
)
from analysis.collect_jev import append_record, trim_torn_line
from analysis.collect_jev_gate import STORED_FIELDS
from analysis.drift import DriftRefusal
from analysis.final_ledger import CALLS, LedgerRefusal, recorded_spend, refusal
from analysis.final_run import ItemFailure, Run, StepRefusal, StepResult
from analysis.gate_spend import JEV_OVERHEAD_TOKENS, jev_projected_cost
from analysis.gate_table import GateScoreTable, PaidScore, build_table, pure_traces
from analysis.gate_train import count_configs, difference, jev_noise
from analysis.operating_points import read_section
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, cascade_section, parse_cascade
from analysis.unlock import Handle
from routers import jev, jev_gate
from routers.cascade import configs_called
from routers.gates import CodeCheckSettings, Runner, free_checks, image_digest, read_gate_settings, sandbox_runner
from routers.ladder import LADDER

FREE_NAME, JEV_NAME = "free_checks", "jev_gate"  # `results/gates/{name}_{label}.jsonl`
TEST, DRIFT = "test", "drift"  # the two labels of Stage 8's files (`analysis/record_files.py`)
DRIFT_CHECK = "drift_jev_gate"  # the drift check's name: the step's, and the refusal's
STAND_IN_DIGEST = "sha256:" + "0" * 63 + "1"  # what a stand-in sandbox's free-check records carry as the image's digest
FREE_SECTION, JEV_SECTION = cascade_section(HAIKU, FREE_GATE), cascade_section(HAIKU, JEV_GATE)  # haiku/free, haiku/Jev
GATE_TABLE = "gate_table"  # the `Extra`'s name for test's and dev's gate-score table


class FinalGatesError(ValueError):
    """A stored file, a world or an answer these steps refuse. Nothing is sent."""


# ---- the files ---------------------------------------------------------------------------------------------------


def gate_file(name: str, label: str, results: Path) -> Path:
    """`results/gates/{name}_{label}.jsonl` for a Stage 8 label. A Stage 7 label is refused: this module writes none of those."""
    if label not in (TEST, DRIFT):
        raise FinalGatesError(f"{label!r} is not one of Stage 8's labels ({TEST!r}, {DRIFT!r})")
    return results / "gates" / f"{name}_{label}.jsonl"


def read_free_test(results: Path) -> list[dict]:
    """Test's stored free-check records in the order stored: the one reader of them."""
    return read_gate_records(gate_file(FREE_NAME, TEST, results), FREE_RECORD_FIELDS, TEST, FREE)


def read_jev(label: str, results: Path) -> list[dict]:
    """The stored Jev gate records of test or of the drift check (one pass each) in the order stored."""
    records = read_gate_records(gate_file(JEV_NAME, label, results), STORED_FIELDS, label, JEV_GATE)
    if any(type(r["refused"]) is not bool for r in records):
        raise GateCollectError("a stored Jev gate record has a refusal flag that isn't a bool")
    return records


# ---- what the gates are given ------------------------------------------------------------------------------------


def test_inputs(handle: Handle) -> list[GateInput]:
    """Every test answer with its request: the handle's two reads and nothing else. Refuses a repeated (item, config)."""
    requests = handle.requests()
    inputs = [GateInput(a.item_id, a.config, requests[a.item_id], a.answer, a.stop_reason) for a in handle.answers()]
    keys = [(i.item_id, i.config) for i in inputs]
    if len(set(keys)) != len(keys):
        raise FinalGatesError("the handle gives the same answer more than once")
    return inputs


def named(item: GateInput) -> str:
    return f"{item.item_id} {item.config}"


def stamp(run: Run) -> str:
    return run.clock().astimezone().isoformat(timespec="seconds")


# ---- the free checks ---------------------------------------------------------------------------------------------


def sandbox_of(run: Run, settings: CodeCheckSettings) -> tuple[Runner, str]:
    """The world's sandbox and the image's digest: a stand-in and its stated digest, or the real sandbox and Docker's."""
    backends = run.world.backends
    if backends is not None and backends.sandbox is not None:
        return backends.sandbox, backends.image_digest or STAND_IN_DIGEST
    return sandbox_runner, image_digest(settings.image)


def stored_free(inputs: Sequence[GateInput], results: Path) -> set[tuple[str, str]]:
    """The answers already checked, refusing a stored record for an answer that isn't one of `inputs`."""
    keys = {(i.item_id, i.config) for i in inputs}
    stored = {(r["item_id"], r["config"]) for r in read_free_test(results)}
    if stray := sorted(stored - keys):
        raise FinalGatesError(
            f"{len(stray)} stored free-check records are not for this unlock's answers, e.g. {stray[0]}"
        )
    return stored


def free_checks_step(run: Run) -> StepResult:
    """The free checks on every test answer, one record each, stored as they are made. A rerun skips stored answers; a
    sandbox that fails after its retries ends the run naming the answer, and what was stored before it stays valid."""
    inputs = test_inputs(run.unlocked())
    results = run.world.results
    settings = read_gate_settings().code_check
    runner, digest = sandbox_of(run, settings)
    path = gate_file(FREE_NAME, TEST, results)
    trim_torn_line(path)
    stored = stored_free(inputs, results)
    todo = [i for i in inputs if (i.item_id, i.config) not in stored]
    for item in todo:
        try:
            fields = free_checks(item.request, item.answer, item.stop_reason, settings, digest=digest,
                                 now=lambda: stamp(run), run=runner, sleep=time.sleep)  # fmt: skip
        except Exception as error:
            raise ItemFailure(named(item)) from error
        key = {"item_id": item.item_id, "config": item.config, "run": 0, "split": TEST, "gate": FREE, "pass": 1}
        append_record(path, key | fields)
    run.say(f"free checks: {len(todo)} answers checked, {len(stored)} already stored")
    return StepResult(requests={"sent": len(todo), "stored": len(stored)})


# ---- the Jev gate: calls under the ledger ------------------------------------------------------------------------


@dataclass(frozen=True)
class Pending:
    item: GateInput
    call: jev_gate.GateCall


def jev_client_of(run: Run) -> AbstractContextManager[Any]:
    """The Jev client: the world's factory for a rehearsal, the real client (key from the environment, the pinned
    endpoint) for the go's world, which names no backends. A world that names backends and no Jev client is refused."""
    backends = run.world.backends
    if backends is None:
        return jev.JevClient(jev.read_client_settings())
    if backends.jev_gate is None:
        raise StepRefusal("the world names backends but no Jev client: only the go's world, which names none, gets the "
                          "real one")  # fmt: skip
    client: AbstractContextManager[Any] = backends.jev_gate()
    return client


def budget_of(settings: jev_gate.JevGateSettings) -> int:
    if settings.budget_bytes is None or settings.budget_bytes < 1:
        raise FinalGatesError("the gate settings hold no byte budget")
    return settings.budget_bytes


def pending_calls(
    inputs: Sequence[GateInput], stored: set[tuple[str, str]], settings: jev_gate.JevGateSettings
) -> list[Pending]:
    """The frozen gate call for each input that has no stored record yet, built before any is sent."""
    budget = budget_of(settings)
    return [Pending(i, jev_gate.gate_call(i.request, i.answer, settings, budget_bytes=budget))
            for i in inputs if (i.item_id, i.config) not in stored]  # fmt: skip


def projected_dollars(pending: Sequence[Pending]) -> float:
    """An upper bound on what the pending calls cost: each body's bytes as tokens, plus the fixed overhead, at Jev's price."""
    price = jev.read_client_settings().price_per_million_input_tokens
    return jev_projected_cost([len(p.call.body) for p in pending], price)


def check_cap(run: Run, pending: Sequence[Pending]) -> None:
    """Refuse (LedgerRefusal) before the first call if Stage 8's recorded spend plus these calls is over the $25 cap."""
    words = refusal(recorded_spend(run.world.results, run.world.output), {CALLS: projected_dollars(pending)})
    if words:
        raise LedgerRefusal(words)


def send_pending(run: Run, client: Any, path: Path, label: str, pending: Sequence[Pending]) -> None:
    """Send each pending call and store its record before the next, in `label`'s file (pass 1). A stop from the client
    ends the run naming the answer; what was stored before it stays valid."""
    for entry in pending:
        item = entry.item
        try:
            result = client.call(entry.call.body, entry.call.sha256, f"Jev gate {label} {named(item)}")
            key = {
                "item_id": item.item_id,
                "config": item.config,
                "run": 0,
                "split": label,
                "gate": JEV_GATE,
                "pass": 1,
            }
            extra = {"answer_cut_bytes": entry.call.answer_cut_bytes, "prompt_cut_bytes": entry.call.prompt_cut_bytes}
            append_record(path, key | result.to_record() | extra | {"collected_at": stamp(run)})
        except Exception as error:
            raise ItemFailure(named(item)) from error


def jev_gate_step(run: Run) -> StepResult:
    """The Jev gate on every test answer that passed the free checks, one call each, stored as it comes. Needs every answer's
    free-check record first. The pending calls are projected against the cap before the first one; a rerun sends only the
    answers with no stored record."""
    inputs = test_inputs(run.unlocked())
    results = run.world.results
    settings = jev_gate.read_settings()
    free = {(r["item_id"], r["config"]): bool(r["passed"]) for r in read_free_test(results)}
    if unchecked := [k for k in ((i.item_id, i.config) for i in inputs) if k not in free]:
        raise FinalGatesError(f"{len(unchecked)} answers have no free-check record, e.g. {unchecked[0]}")
    path = gate_file(JEV_NAME, TEST, results)
    trim_torn_line(path)
    stored = {(r["item_id"], r["config"]) for r in read_jev(TEST, results)}
    callable_inputs = [i for i in inputs if free[(i.item_id, i.config)]]
    pending = pending_calls(callable_inputs, stored, settings)
    check_cap(run, pending)
    if pending:
        with jev_client_of(run) as client:
            send_pending(run, client, path, TEST, pending)
    already = len(callable_inputs) - len(pending)
    run.say(f"Jev gate: {len(pending)} calls sent, {already} already stored; "
            f"${projected_dollars(pending):.4f} was projected for them")  # fmt: skip
    return StepResult(requests={"sent": len(pending), "stored": already})


def project(run: Run) -> float:
    """The Jev gate's projected spend, for the ledger's whole-run projection (an upper bound, in the calls pool): the drift
    resends with no stored record, each at the input tokens Stage 7 was billed for the same body plus the fixed overhead,
    and a call for each test answer with no stored call, leaving out those a stored free check rejected. Before the free
    checks have run it assumes every answer passes, so it never lets through what the cap would stop. A results folder with
    no Stage 7 noise pass has no drift sample to project (the drift step itself refuses)."""
    results = run.world.results
    price = jev.read_client_settings().price_per_million_input_tokens
    train = collect_jev_gate.read_records("train", results)
    first = {(r["item_id"], r["config"]): r for r in train if r["pass"] == 1}
    done = {(r["item_id"], r["config"]) for r in read_jev(DRIFT, results)}
    resend_tokens = sum(
        collect_jev_gate.input_tokens(first[key]) + JEV_OVERHEAD_TOKENS
        for key in drift_sample(results)
        if key not in done
    )
    inputs = test_inputs(run.unlocked())
    rejected = {(r["item_id"], r["config"]) for r in read_free_test(results) if not r["passed"]}
    stored = {(r["item_id"], r["config"]) for r in read_jev(TEST, results)} | rejected
    pending = pending_calls(inputs, stored, jev_gate.read_settings())
    return resend_tokens * price / 1e6 + projected_dollars(pending)


# ---- the Jev gate's drift check ----------------------------------------------------------------------------------


def drift_sample(results: Path) -> list[tuple[str, str]]:
    """The noise sample as Stage 7 sent it: the (item, config) pairs with a pass 2 in `jev_gate_train.jsonl`, in the order
    stored. Empty if Stage 7 stored none (the hook's real-data check holds it against `analysis/noise_samples.py`'s rule)."""
    return [(r["item_id"], r["config"]) for r in collect_jev_gate.read_records("train", results) if r["pass"] == 2]


def drift_inputs(run: Run, sample: Sequence[tuple[str, str]]) -> list[GateInput]:
    """The sample's train answers with their requests: the world's stand-in, or the gates' own readers on train."""
    backends = run.world.backends
    if backends is not None and backends.gate_train is not None:
        found = list(backends.gate_train(sample))
    else:
        wanted = set(sample)
        found = [i for i in gate_inputs("train") if (i.item_id, i.config) in wanted]
    by_key = {(i.item_id, i.config): i for i in found}
    if len(by_key) != len(found) or set(by_key) != set(sample):
        raise FinalGatesError("the train answers don't match the noise sample: some are missing or repeated")
    return [by_key[key] for key in sample]


def cutoff_labels() -> dict[float, list[str]]:
    """Each frozen cutoff of a cascade that ends in the Jev gate, with the operating points that sit on it: the pure
    cascade's walked ones and the fixed one the router-started cascades share. A point that fell back holds none."""
    labels: dict[float, list[str]] = {}
    for section, label in CASCADES.items():
        if parse_cascade(section)[1] != JEV_GATE:
            continue
        for mix, by_target in read_section(section).items():
            for target, point in by_target.items():
                if not point["fell_back"] and point.get("cutoff") is not None:
                    labels.setdefault(float(point["cutoff"]), []).append(f"{label} {mix} {target}")
    return labels


def accepted(start: str, score: float | None, cutoff: float) -> bool:
    """The deployable cascade decision (`routers/cascade.py`) for an answer that passed the free checks: kept, not escalated."""
    return configs_called(start, True, score, cutoff) == (start,)


def gate_flips(
    first: Mapping[tuple[str, str], float | None],
    resend: Mapping[tuple[str, str], float | None],
    labels: Mapping[float, Sequence[str]],
) -> tuple[drift_rule.Flip, ...]:
    """At each frozen cutoff, the answers whose decision (kept or escalated) differs between Stage 7's first pass and the
    resend. A refused call (no score) is below every cutoff. Highest cutoff first."""
    if set(first) != set(resend):
        raise ValueError("the first pass and the resend must score the same answers")
    return tuple(
        drift_rule.Flip(cutoff, tuple(labels[cutoff]),
                        tuple(f"{item} {config}" for item, config in first
                              if accepted(config, first[item, config], cutoff) != accepted(config, resend[item, config], cutoff)))
        for cutoff in sorted(labels, reverse=True)
    )  # fmt: skip


def scores_of(
    records: Sequence[Mapping[str, Any]], settings: jev_gate.JevGateSettings
) -> dict[tuple[str, str], float | None]:
    return {
        (str(r["item_id"]), str(r["config"])): jev_gate.read_response(r["response"], settings).score for r in records
    }


def model_ids(records: Sequence[Mapping[str, Any]]) -> list[object]:
    """Every model ID a resend came back with: the stored field and the response's own."""
    return [
        m
        for r in records
        for m in (r["model"], r["response"].get("model") if isinstance(r["response"], dict) else None)
    ]


def drift_report(
    train: Sequence[Mapping[str, Any]],
    resent: Sequence[Mapping[str, Any]],
    settings: jev_gate.JevGateSettings,
    labels: Mapping[float, Sequence[str]],
) -> drift_rule.DriftReport:
    """The check's rule on Stage 7's two passes (`train`) and the resend (`resent`): D6 from the passes, D8 and the shift
    from the first pass and the resend, per config, and the flips at each cutoff. Raises `DriftRefusal` on the rule."""
    sample = {(str(r["item_id"]), str(r["config"])) for r in train if r["pass"] == 2}
    mine = [r for r in train if (str(r["item_id"]), str(r["config"])) in sample]
    d6 = jev_noise(mine, settings)
    first = scores_of([r for r in mine if r["pass"] == 1], settings)
    resend = scores_of(resent, settings)
    if set(resend) != sample or set(first) != sample:
        raise FinalGatesError("the resend and Stage 7's first pass are not both over the noise sample")
    d8 = {
        config: difference({k[0]: first[k] for k in sample if k[1] == config},
                           {k[0]: resend[k] for k in sample if k[1] == config})
        for config in d6
    }  # fmt: skip
    measures = drift_rule.measures_of(d6, d8)
    return drift_rule.apply_rule(DRIFT_CHECK, pin=drift_rule.JEV_PIN, model_ids=model_ids(resent), measures=measures,
                                 flips=gate_flips(first, resend, labels))  # fmt: skip


def drift_step(run: Run) -> StepResult:
    """A15 item 17 for the Jev gate (A17 item 26.2): the noise sample resent once into `jev_gate_drift.jsonl`, D8 measured
    against Stage 7's first pass. Stops the run on the shared rule with nothing called on test. Says train's numbers only."""
    results = run.world.results
    settings = jev_gate.read_settings()
    train = collect_jev_gate.read_records("train", results)
    sample = drift_sample(results)
    if not sample:
        raise FinalGatesError("Stage 7 stored no noise pass for the Jev gate: there is nothing to resend")
    first = {(r["item_id"], r["config"]): r for r in train if r["pass"] == 1}
    inputs = drift_inputs(run, sample)
    path = gate_file(JEV_NAME, DRIFT, results)
    trim_torn_line(path)
    stored = {(r["item_id"], r["config"]) for r in read_jev(DRIFT, results)}
    calls = pending_calls(inputs, set(), settings)
    if differing := [
        named(c.item) for c in calls if c.call.sha256 != first[(c.item.item_id, c.item.config)]["body_sha256"]
    ]:
        raise DriftRefusal(DRIFT_CHECK, f"the frozen gate call builds a different body than Stage 7 stored for "
                                        f"{len(differing)} of the {len(calls)} noise-sample answers, e.g. {differing[0]}; "
                                        f"nothing was called on test")  # fmt: skip
    pending = [c for c in calls if (c.item.item_id, c.item.config) not in stored]
    check_cap(run, pending)
    if pending:
        with jev_client_of(run) as client:
            send_pending(run, client, path, DRIFT, pending)
    report = drift_report(train, read_jev(DRIFT, results), settings, cutoff_labels())
    said = "; ".join(f"{c}: D6 {m.d6:.4f}, D8 {m.d8:.4f}, shift {m.shift:+.4f}" for c, m in report.measures.items())
    run.say(f"{DRIFT_CHECK}: {len(calls)} answers resent ({len(pending)} calls sent, {len(stored)} already stored); {said}; "
            f"decisions that flip at the frozen cutoffs {[len(f.items) for f in report.flips]}")  # fmt: skip
    return StepResult(requests={"sent": len(pending), "stored": len(stored)},
                      detail={**report.record(), "answers": len(calls)})  # fmt: skip


# ---- test's gate-score table -------------------------------------------------------------------------------------


def build_test_table(
    free: Sequence[Mapping[str, Any]],
    jev_records: Sequence[Mapping[str, Any]],
    answers: Sequence[tuple[str, str]],
    settings: jev_gate.JevGateSettings,
    client: jev.ClientSettings,
    extra_paid: Mapping[str, Mapping[tuple[str, str], PaidScore]] | None = None,
) -> GateScoreTable:
    """The gate-score table of test from the stored records, through the gates' own parsers (A17 item 13). Refused whole
    (GateCollectError, GateTableError) on a missing or repeated answer, a score off [0, 1], a paid score on an answer that
    failed a free check, a negative or non-finite overhead, a passing answer with no Jev call, and a stored refusal flag that
    the response doesn't bear out. `extra_paid` adds another paid gate's column (#86's Sonnet gate)."""
    passed = {(str(r["item_id"]), str(r["config"])) for r in free if r["passed"]}
    column = collect_jev_gate.column(passed, jev_records, settings, client)
    return build_table(free, set(answers), paid={JEV_GATE: column, **(extra_paid or {})})


def test_table(run: Run) -> GateScoreTable:
    """Test's table from the stored free-check and Jev gate records, over the handle's answers."""
    return test_data(run).table


@dataclass(frozen=True)
class GateData:
    """A split's gate-score table with the records it came from (the counts the tables report come from the records)."""

    table: GateScoreTable
    free: tuple[Mapping[str, Any], ...]
    jev: tuple[Mapping[str, Any], ...]


def test_data(run: Run) -> GateData:
    inputs = test_inputs(run.unlocked())
    results = run.world.results
    free, calls = read_free_test(results), read_jev(TEST, results)
    table = build_test_table(free, calls, [(i.item_id, i.config) for i in inputs], jev_gate.read_settings(),
                             jev.read_client_settings())  # fmt: skip
    return GateData(table, tuple(free), tuple(calls))


def dev_data() -> GateData:
    """Dev's table, as the dev run built it, with its stored free-check and Jev gate records."""
    from analysis import gate_dev_run
    from analysis.collect_gates import read_free_records

    return GateData(gate_dev_run.dev_table(), tuple(read_free_records("dev")),
                    tuple(collect_jev_gate.read_records("dev")))  # fmt: skip


def gate_data_record(data: GateData | None) -> Mapping[str, Any]:
    """What the record keeps and the tables step says: A17 item 28's counts per config, free-check rejections by the check
    that failed first, Jev calls, refusals (a rejection) and truncations. Outcomes, so shown with the tables and not before."""
    if data is None:
        return {}
    counts = count_configs(data.free, data.jev, [], LADDER)
    return {config: {k: v for k, v in asdict(c).items() if not k.startswith("sonnet")} for config, c in counts.items()}


# ---- the pure cascades at their frozen points --------------------------------------------------------------------


def tasks_key(section: str) -> str:
    """The key of a cascade's per-task part: dev committed no numbers for these, so they are a part of their own."""
    return f"{section}_tasks"


def cascade_part(section: str, *, per_task: bool = False) -> Callable[[Any], dict]:
    """A pure cascade's part (`haiku/free`, `haiku/Jev`): its frozen points evaluated on the split with
    `analysis/operating_eval.py` and the gate-score table (`GateData`) of the `gate_table` extra. The part holds the mixes' rows
    (table v4's, which dev committed), or with `per_task` the per-task rows, which dev committed no numbers for, so they are a
    part of their own. The file's `fell_back` decides each point, a cutoff is the file's, and nothing is walked."""
    start, gate = parse_cascade(section)
    if start != "haiku":
        raise FinalGatesError(f"{section} is not a pure cascade")

    def build(inputs: Any) -> dict:
        from analysis import pareto
        from analysis.cascade_comparisons import escalation_rate
        from analysis.operating_eval import evaluate, frozen_points, mix_rows, shared_fixed, shared_savings, task_rows
        from analysis.simulator import load_mixes

        table: GateScoreTable = inputs.extras[GATE_TABLE].table
        fixed = shared_fixed(inputs.cache, inputs.matrix)
        points = frozen_points(section)
        outcomes = evaluate(inputs.matrix, fixed["opus"], points, lambda point: pure_traces(table, gate, point.cutoff))
        if per_task:
            return {"tasks": task_rows(inputs.draws, fixed, outcomes, list(pareto.TASKS))}
        mixes = load_mixes()
        rows = mix_rows(inputs.draws, lambda mix: shared_savings(inputs.cache, inputs.matrix, inputs.draws, mix),
                        fixed, outcomes, mixes)  # fmt: skip
        for (
            point,
            found,
        ) in outcomes.items():  # a cascade's row holds its cutoff (None if it fell back) and its escalation
            row = rows[point.mix][point.target]
            row.setdefault("cutoff", None)
            row["escalation_rate"] = 0 if point.fell_back else escalation_rate(found, mixes[point.mix])
        return {"mixes": rows}

    return build


MIX_KEYS = ("threshold", "cutoff", "fell_back", "target", "quality", "cost_per_1k", "difference_from_A", "savings_vs_A",
            "captured", "routing", "escalation_rate", "matched_cost")  # fmt: skip


def committed_part(section: str) -> Callable[[], dict]:
    """The numbers table v4 committed for the cascade on dev: the keys a frozen row has. A walk's margin and the steps it
    tried are the walk's and aren't reproduced."""

    def committed() -> dict:
        from analysis.table_v4 import TABLE_V4

        held = json.loads(TABLE_V4.read_text(encoding="utf-8"))["cascades"][section]["operating_points"]
        return {"mixes": {mix: {target: {key: point[key] for key in MIX_KEYS if key in point}
                                for target, point in by.items()} for mix, by in held.items()}}  # fmt: skip

    return committed
