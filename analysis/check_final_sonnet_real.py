"""Check the Sonnet gate's drift check against Stage 7's stored records (PLAN_v4.md Amendment A15, item 17; A17 items 17, 26 and 28;
A19, item 13; issue #77, #86).

Run from the project root:  uv run python -m analysis.check_final_sonnet_real

Needs `results/` (the train matrix and the stored train gate records, not committed), so the hook runs it and CI doesn't
(`HOOK_ONLY` in `analysis/check_guardrails.py`). It reads train only and calls nothing: no Anthropic call, no gate, no test item.

- the sample the drift check resends, read from Stage 7's stored standard-API calls, is the 80 `haiku` answers that
  `analysis/noise_samples.py`'s rule draws from train's free-check records, 20 per task;
- D6 and the shift, recomputed from the stored batch verdicts and standard-API calls, are A17 item 28's committed numbers (D6 0.0266
  and shift +0.0006 for `haiku`, over 77 answers with 3 left out), which the plan's text is searched for; the unfloored mean is
  above the floor, so the floor changes nothing here;
- the frozen request, built from train's answers as the drift step builds it, hashes to the one Stage 7 stored for each of the 80, in
  the standard-API file and in the batch file, so the resend would send what Stage 7 sent;
- the frozen cutoffs the flips are counted at are the three distinct ones of the Sonnet gate's cascades (0.77, the fixed one the
  router-started cascades share, and haiku/Sonnet's 0.5 and 0.3); a stored verdict compared with itself flips nothing (the rule's
  known zero);
- the projection: the 80 resends and the whole test batch (the benchmark manifest's test counts, three gated configs) at
  twice dev's cost per verdict come to under $16 in all, well under the $25 cap, and no answer or outcome of test is read.
"""

import json
import sys
from types import SimpleNamespace

from analysis import collect_sonnet_gate, final_gates, final_sonnet, gate_dev_run, gate_train
from analysis.collect_gates import read_free_records
from analysis.noise_samples import noise_lists, read_noise_settings
from analysis.router_names import SONNET_GATE
from harness.client import PROJECT_ROOT
from routers import sonnet_gate

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    results = PROJECT_ROOT / "results"
    settings = sonnet_gate.load_settings()
    batch = collect_sonnet_gate.read_sonnet_records("train", results)
    latency = collect_sonnet_gate.read_latency_records("train", results)
    sample = [(str(r["item_id"]), str(r["config"])) for r in latency]
    drawn = noise_lists(read_free_records("train", results), read_noise_settings())["sonnet"]
    check(len(sample) == 80 and sorted(sample) == sorted(drawn) and len(set(sample)) == 80,
          "the stored standard-API calls are the 80 answers the plan's drawing rule draws from train's free-check records")  # fmt: skip

    noise = gate_train.sonnet_drift(batch, latency)
    haiku = noise["haiku"]
    check(set(noise) == {"haiku"} and haiku.pairs == 77 and haiku.unscored == 3
          and round(haiku.d6, 4) == 0.0266 and round(haiku.shift, 4) == 0.0006 and round(haiku.mean_abs, 4) == 0.0266
          and haiku.mean_abs > gate_train.SONNET_FLOOR,
          f"D6 and the shift recomputed from the stored calls: {haiku.d6:.4f} and {haiku.shift:+.4f} over {haiku.pairs} ({haiku.unscored} left out)")  # fmt: skip
    plan = (PROJECT_ROOT / "PLAN_v4.md").read_text(encoding="utf-8")
    check("D6 0.0266 shift +0.0006 over 77 answers (3 left out)" in plan,
          "the plan's text holds the committed 0.0266, +0.0006, 77 and 3 the recomputation is held to")  # fmt: skip

    run = SimpleNamespace(world=SimpleNamespace(backends=None))
    inputs = final_gates.drift_inputs(run, sample)  # type: ignore[arg-type]  # a stand-in for the Run: only the world is read
    jobs = final_sonnet.drift_jobs(inputs, settings)
    first = {(str(r["item_id"]), str(r["config"])): r["request_sha256"] for r in latency}
    in_batch = {(str(r["item_id"]), str(r["config"])): r["request_sha256"] for r in batch}
    same = [j for j in jobs if j.sha256 == first[(j.item_id, j.config)] == in_batch[(j.item_id, j.config)]]
    check(len(jobs) == 80 and len(same) == 80,
          f"the frozen request built from train's answers hashes to Stage 7's stored one, in both files, for {len(same)} of 80")  # fmt: skip

    labels = final_gates.cutoff_labels(SONNET_GATE)
    fixed = gate_dev_run.read_cutoffs()["sonnet"]
    check(sorted(labels) == [0.3, 0.5, fixed] and fixed == 0.77 and len(labels[fixed]) == 9
          and labels[0.5] == ["haiku/Sonnet realistic 97%"] and labels[0.3] == ["haiku/Sonnet uniform 97%"],
          f"three distinct frozen cutoffs, the fixed {fixed} shared by the router-started cascades' 9 points")  # fmt: skip
    scores = final_sonnet.scores_of([r for r in batch if (str(r["item_id"]), str(r["config"])) in set(sample)])
    flips = final_gates.gate_flips(scores, scores, labels)
    check(all(not f.items for f in flips) and len(flips) == 3, "known zero: a pass compared with itself flips nothing")

    pending_dollars = 80 * final_sonnet.dollars_per_check(batch=False)
    manifest = json.loads((PROJECT_ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    test_items = sum(task["splits"]["test"] for task in manifest["tasks"].values())  # counts only: no item, no answer
    batch_dollars = 3 * test_items * final_sonnet.dollars_per_check(batch=True)
    check(test_items == 743 and pending_dollars < 1.5 and pending_dollars + batch_dollars < 16.0,
          f"the resends are projected at ${pending_dollars:.4f} and the whole test batch at ${batch_dollars:.2f}, under the cap")  # fmt: skip
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
