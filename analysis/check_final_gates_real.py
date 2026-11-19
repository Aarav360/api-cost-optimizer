"""Check the Jev gate's drift check against Stage 7's stored records (PLAN_v4.md Amendment A15, item 17; A17 items 17, 26 and 28;
A19, item 13; issue #77, #85).

Run from the project root:  uv run python -m analysis.check_final_gates_real

Needs `results/` (the train matrix and the stored train gate records, not committed), so the hook runs it and CI doesn't
(`HOOK_ONLY` in `analysis/check_guardrails.py`). It reads train only and calls nothing: no Jev call, no gate, no test item.

- the noise sample the drift check resends, read from Stage 7's stored second pass, is the 100 `haiku` answers that
  `analysis/noise_samples.py`'s rule draws from train's free-check records, 25 per task;
- D6 and the shift, recomputed from the stored first and second passes, are A17 item 28's committed numbers (D6 0.0144 and
  shift +0.0005 for `haiku`, over 100 answers with none left out), which the plan's text is searched for;
- the frozen gate call, built from train's answers as the drift step builds it, hashes to the body Stage 7 stored for each of
  the 100, so the resend would send what Stage 7 sent;
- the frozen cutoffs the flips are counted at are the six distinct ones of the Jev gate's cascades, and the fixed cutoff is among
  them; a stored resend of the first pass itself, scored against itself, flips nothing (the rule's known zero);
- the drift step's projection of its resends is under a dollar.
"""

import sys
from types import SimpleNamespace

from analysis import collect_jev_gate, final_gates, gate_dev_run
from analysis.collect_gates import read_free_records
from analysis.gate_train import jev_noise
from analysis.noise_samples import noise_lists, read_noise_settings
from harness.client import PROJECT_ROOT
from routers import jev, jev_gate

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    results = PROJECT_ROOT / "results"
    settings = jev_gate.read_settings()
    print("\n-- the noise sample and its numbers --")
    train = collect_jev_gate.read_records("train", results)
    sample = final_gates.drift_sample(results)
    drawn = noise_lists(read_free_records("train", results), read_noise_settings())["jev"]
    check(len(sample) == 100 and sorted(sample) == sorted(drawn) and len(set(sample)) == 100,
          "the stored noise pass is the 100 answers the plan's drawing rule draws from train's free-check records")  # fmt: skip

    noise = jev_noise([r for r in train if (r["item_id"], r["config"]) in set(sample)], settings)
    haiku = noise["haiku"]
    check(set(noise) == {"haiku"} and haiku.pairs == 100 and haiku.unscored == 0
          and round(haiku.d6, 4) == 0.0144 and round(haiku.shift, 4) == 0.0005,
          f"D6 and the shift recomputed from the stored passes: {haiku.d6:.4f} and {haiku.shift:+.4f} over {haiku.pairs}")  # fmt: skip
    plan = (PROJECT_ROOT / "PLAN_v4.md").read_text(encoding="utf-8")
    check("0.0144" in plan, "the plan's text holds the committed 0.0144 the recomputation is held to")

    print("\n-- the frozen call and the cutoffs --")
    run = SimpleNamespace(world=SimpleNamespace(backends=None))
    inputs = final_gates.drift_inputs(run, sample)  # type: ignore[arg-type]  # a stand-in for the Run: only the world is read
    first = {(r["item_id"], r["config"]): r for r in train if r["pass"] == 1}
    calls = final_gates.pending_calls(inputs, set(), settings)
    same = [c for c in calls if c.call.sha256 == first[(c.item.item_id, c.item.config)]["body_sha256"]]
    check(len(calls) == 100 and len(same) == 100,
          f"the frozen call built from train's answers hashes to Stage 7's stored body for {len(same)} of 100")  # fmt: skip

    labels = final_gates.cutoff_labels()
    fixed = gate_dev_run.read_cutoffs()["jev"]
    check(len(labels) == 6 and fixed in labels and len(labels[fixed]) == 19,
          f"six distinct frozen cutoffs, the fixed {fixed:.4f} shared by the router-started cascades' 19 points")  # fmt: skip
    scores = final_gates.scores_of([first[k] for k in sample], settings)
    flips = final_gates.gate_flips(scores, scores, labels)
    check(all(not f.items for f in flips) and len(flips) == 6, "known zero: a pass compared with itself flips nothing")

    print("\n-- the projection --")
    price = jev.read_client_settings().price_per_million_input_tokens
    tokens = sum(collect_jev_gate.input_tokens(first[k]) + 512 for k in sample)
    check(tokens * price / 1e6 < 1.0, f"the resends are projected at ${tokens * price / 1e6:.4f}, well under a dollar")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
