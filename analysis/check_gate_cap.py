"""Check Stage 7's $25 cap on gate calls, across both paid collectors (PLAN_v4.md §9; Amendment A17, item 18; A6 item 6).

Run from the project root:  uv run python -m analysis.check_gate_cap

On synthetic data only: no results/ folder, no Docker, no Jev or Anthropic call, no key. A17 item 18 caps Stage 7's gate
calls at $25 in total, so each paid collector must refuse a submission when the spend already recorded in
`results/gates/` (every split, smoke included, both gates, the latency calls too) plus the projected cost of what is
about to go out is over $25. The Sonnet collector's own per-submission cap (#64) stays; this is the cumulative one.

- the recorded spend is worked out by hand from made-up records: Jev's input tokens at its price, the Sonnet batches at
  list price less the batch discount, the standard-API latency calls at list price; a cut-short last line is dropped, an
  unreadable one or a record with no billed tokens refuses the count whole, and files that are not gate records
  don't count;
- the Sonnet batch collector and the latency caller are fed their bad case (spend recorded in other splits and the
  other gate, plus the projection, over $25: nothing is sent) and their boundary (exactly $25 goes through);
- the Jev collector is fed its bad case too: nothing is called when the projection tips the total over, and the
  projection is the body's bytes plus a fixed overhead at one token per byte, at Jev's price.
"""

import json
import sys
import tempfile
from pathlib import Path

from analysis import collect_gates, collect_jev_gate, gate_spend
from analysis import collect_sonnet_gate as sonnet_collect
from analysis.check_gates import BENCH, write_store
from analysis.check_jev_gate import free_check, run_collect, same
from analysis.check_jev_settings import write_benchmark
from analysis.check_sonnet_gate import free_record
from analysis.checking import refused
from analysis.collect_gates import GateCollectError, GateInput
from analysis.fake_anthropic import FakeAnthropic, message_dict
from analysis.fake_jev import FakeJev
from harness.client import PROJECT_ROOT
from routers import jev, sonnet_gate
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def write_lines(directory: Path, name: str, usages: list[tuple[int, int]], *, torn: str = "") -> None:
    """A gate file under `directory/gates` whose records hold only what the spend reads: billed tokens."""
    path = directory / "gates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps({"usage": {"input_tokens": i, "output_tokens": o}}) + "\n" for i, o in usages)
    path.write_text(lines + torn, encoding="utf-8")


def prior_spend_of_24(directory: Path) -> None:
    """$21 of Jev calls in the smoke split and $3 of Sonnet batch in train: two gates, two splits, and neither is the
    split the submission below goes to. 500,000,000 input tokens at $0.042 per million is $21; 600,000 output tokens at
    $10 per million is $6 at list price and $3 at the batch discount."""
    write_lines(directory, "jev_gate_smoke.jsonl", [(500_000_000, 38)])
    write_lines(directory, "sonnet_gate_train.jsonl", [(0, 600_000)])


GATED = [
    GateInput(f"b{n}", "haiku", Request(f"prompt b{n}", None), f"answer b{n} SCORE={n}0", "end_turn") for n in range(7)
]


def sonnet_jobs() -> list[sonnet_collect.Job]:
    settings = sonnet_gate.load_settings()
    return sonnet_collect.build_jobs(GATED, [free_record(i.item_id, i.config, None) for i in GATED], settings)


def spend_checks() -> None:
    print("-- the recorded spend --")
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        check(gate_spend.recorded_spend(directory).total == 0.0, "no gates folder is no spend")
        write_lines(directory, "jev_gate_train.jsonl", [(1_000_000, 40), (500_000, 40)])
        write_lines(directory, "jev_gate_smoke.jsonl", [(2_000_000, 40)])
        write_lines(directory, "sonnet_gate_dev.jsonl", [(1_000_000, 100_000)], torn='{"usage": {"input_tok')
        write_lines(directory, "sonnet_gate_latency.jsonl", [(500_000, 200_000)])
        write_lines(directory, "sonnet_gate_latency_smoke.jsonl", [(0, 100_000)])
        (directory / "gates" / "free_checks_train.jsonl").write_text('{"usage": {"input_tokens": 9e9}}\n')
        (directory / "gates" / "sonnet_batches.json").write_text("{}")
        spend = gate_spend.recorded_spend(directory)
        # Jev: 3.5M tokens at $0.042 per million = $0.147. Sonnet batch: 1M in and 100k out is $2 + $1 = $3 at list
        # price, $1.50 at the 50% batch discount. Latency (standard, list price): 500k in and 200k out is $1 + $2 = $3,
        # and 100k out is $1.
        check(
            abs(spend.jev - 0.147) < 1e-9
            and abs(spend.sonnet_batch - 1.5) < 1e-9
            and abs(spend.sonnet_standard - 4.0) < 1e-9
            and abs(spend.total - 5.647) < 1e-9,
            "Jev at its price, batches at list price less the discount, latency calls at list price: $5.647",
        )
        check(spend.files == 5, "five files count: both gates, train, dev, smoke and the latency calls (not the "
              "free checks, not the batch log)")  # fmt: skip
        write_lines(directory, "sonnet_gate_dev.jsonl", [(1_000_000, 100_000)], torn="not json\n")
        check(refused(gate_spend.GateSpendError, lambda: gate_spend.recorded_spend(directory)),
              "a complete line that is not JSON refuses the count: the spend can't be known")  # fmt: skip
        (directory / "gates" / "sonnet_gate_dev.jsonl").write_text('{"usage": {"input_tokens": 1}}\n')
        check(refused(gate_spend.GateSpendError, lambda: gate_spend.recorded_spend(directory)),
              "a record with no billed output tokens refuses it too: a call is charged what it was billed")  # fmt: skip
        (directory / "gates" / "sonnet_gate_dev.jsonl").write_text(
            '{"usage": {"input_tokens": -1, "output_tokens": 2}}\n'
        )
        check(refused(gate_spend.GateSpendError, lambda: gate_spend.recorded_spend(directory)),
              "and so does a negative count")  # fmt: skip
    check(gate_spend.over_cap(24.0, 1.0) is None and gate_spend.over_cap(24.0, 1.0001) is not None
          and gate_spend.over_cap(25.0, 0.0) is None and gate_spend.STAGE_CAP_USD == 25.0,
          "the cap is $25: exactly $25 is allowed, any more is not")  # fmt: skip
    check(abs(gate_spend.jev_projected_cost([1000, 2000], 0.042) - 4024 * 0.042 / 1e6) < 1e-15
          and gate_spend.JEV_OVERHEAD_TOKENS == 512,
          "a Jev call is projected as its body's bytes plus 512 tokens, one token per byte, at $0.042 per million")  # fmt: skip
    record = json.loads((PROJECT_ROOT / "analysis" / "gates_smoke_record.json").read_text(encoding="utf-8"))
    measured = record["measured"]["jev_fixed_overhead_tokens"]

    def errs_high(measured: object, projected: int = gate_spend.JEV_OVERHEAD_TOKENS) -> bool:
        return isinstance(measured, int) and 0 < measured < projected

    check(errs_high(measured),
          f"the projection's overhead is above the {measured} tokens the smoke test measured, so it errs high")  # fmt: skip
    check(not errs_high(600) and not errs_high(512) and not errs_high(None),
          "known positive: a measured overhead at or above the projection's, or none, fails it")  # fmt: skip


def sonnet_checks() -> None:
    print("\n-- the Sonnet batch collector --")
    jobs = sonnet_jobs()
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        prior_spend_of_24(directory)
        fake = FakeAnthropic()
        over = refused(sonnet_collect.CapExceeded, lambda: sonnet_collect.submit(
            jobs, split="dev", client=fake, cost_per_check=0.2, max_usd=100.0, results_dir=directory))  # fmt: skip
        check(over and not fake.created and not sonnet_collect.batch_log_path(directory).exists(),
              "$24 recorded in other splits and the other gate plus 7 at $0.20 is $25.40: over $25, nothing is sent "
              "although the per-submission cap of $100 is not")  # fmt: skip
        edge = sonnet_collect.submit(jobs[:4], split="dev", client=fake, cost_per_check=0.25, max_usd=100.0,
                                     results_dir=directory)  # fmt: skip
        check(edge.sent == 4 and len(fake.created) == 1, "4 at $0.25 brings it to exactly $25: allowed")
        # The batch just sent has no stored result yet, so the recorded spend is still $24; sending the other 3 at
        # $0.40 is a projection of $1.20 on top of it.
        check(refused(sonnet_collect.CapExceeded, lambda: sonnet_collect.submit(
            jobs, split="dev", client=fake, cost_per_check=0.4, max_usd=100.0, results_dir=directory)),
              "a rerun for the other 3 at $0.40 is $25.20 on the recorded $24: refused")  # fmt: skip
        check(abs(gate_spend.recorded_spend(directory).total - 24.0) < 1e-9, "and the refusal recorded no spend")
        loose = sonnet_collect.submit(jobs, split="dev", client=fake, cost_per_check=0.1, max_usd=100.0,
                                      results_dir=directory, stage_cap_usd=30.0)  # fmt: skip
        check(loose.sent == 3, "the stage cap is a parameter a check can move, and the default is the plan's $25")

    print("\n-- the latency caller --")
    probe = jobs[:3]
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        write_lines(directory, "jev_gate_dev.jsonl", [(300_000_000, 38)])  # $12.60
        write_lines(directory, "sonnet_gate_latency_smoke.jsonl", [(0, 1_240_000)])  # $12.40 at list price
        fake = FakeAnthropic(respond=lambda params: message_dict('{"probability_correct": 77}', usage=(300, 40)))
        settings = sonnet_gate.load_settings()
        over = refused(sonnet_collect.CapExceeded, lambda: sonnet_collect.run_latency(
            probe, split="train", client=fake, settings=settings, cost_per_check=0.2, max_usd=100.0,
            results_dir=directory))  # fmt: skip
        check(over and not fake.calls,
              "$25 recorded plus 3 calls at $0.20 is over $25: no call is made, whatever the per-run cap")  # fmt: skip
        made = sonnet_collect.run_latency(probe, split="train", client=fake, settings=settings, cost_per_check=0.0,
                                          max_usd=100.0, results_dir=directory)  # fmt: skip
        check(made == 3, "at exactly $25.00 recorded, calls projected at $0 go through")


def jev_checks() -> None:
    print("\n-- the Jev collector --")
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        bench = write_benchmark(directory, BENCH)
        store = directory / "store.sqlite"
        write_store(store)
        inputs = collect_gates.gate_inputs("train", bench, store)[:1]
        free_check(inputs, "train", directory)
        # $24.99999 recorded: output tokens at $10 per million, 2,499,999 of them, in a latency file.
        write_lines(directory, "sonnet_gate_latency.jsonl", [(0, 2_499_999)])
        with FakeJev(script=[same(0.9)]) as fake:
            over = refused(GateCollectError, lambda: run_collect(fake, directory, inputs, "train", budget_bytes=5000))
            check(over and fake.received == [] and not collect_jev_gate.jev_path("train", directory).exists(),
                  "$24.99999 recorded plus the call's projected cost tips the total over $25: nothing is called")  # fmt: skip
        write_lines(directory, "sonnet_gate_latency.jsonl", [(0, 2_400_000)])
        with FakeJev(script=[same(0.9)]) as fake:
            summary = run_collect(fake, directory, inputs, "train", budget_bytes=5000)
            check(summary.new == 1 and len(fake.received) == 1, "with $24 recorded, the same call is made")
        price = jev.read_client_settings().price_per_million_input_tokens
        check(abs(gate_spend.recorded_spend(directory).jev - 1000 * price / 1e6) < 1e-12,
              "and the stored call is counted at its billed input tokens and Jev's price on the next count")  # fmt: skip


def main() -> int:
    spend_checks()
    sonnet_checks()
    jev_checks()
    return 0


if __name__ == "__main__":
    code = main()
    if problems:
        print(f"\n{len(problems)} checks failed")
        sys.exit(1)
    print("\nall checks passed")
    sys.exit(code)
