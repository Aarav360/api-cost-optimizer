# AI API Cost Optimizer — Plan v4

| | |
|---|---|
| Version | v4 |
| Status | Final — frozen before Stage 0, then changed only by the amendments below |
| Date | 2026-10-12 |
| Amended | 2026-10-14: Amendment A1, implementation decisions (below); 2026-10-15: Amendment A2, Haiku effort; 2026-10-16: Amendments A3, pilot decisions, A4, larger pools tested, and A5, benchmark frozen; 2026-10-18: Amendment A6, collection-run decisions; 2026-10-19: Amendment A7, Stage 0–1 records completed, and A8, Stage 3 decisions; 2026-10-20: Amendment A9, Stage 4 decisions |
| Supersedes | [PLAN_v3.md](docs/history/PLAN_v3.md) — earlier versions in [docs/history/](docs/history/) |

## What changed from v3

1. **Threshold safety margin:** every operating point must clear its quality target by a stated margin (default ≥ 1 standard error); thresholds may be selected by cross-validation over train+dev.
2. **Drift control:** pinned, dated model versions and one shared price snapshot; raw test answers are collected alongside dev into a sealed store and scored only in Stage 8.
3. **Self-consistency gate moved to extension X2**, because it needs temperature > 0 samples the matrix doesn't contain.
4. **Auto-gradable core:** the 3–4 core task types use objective evaluators only; judge-graded tasks and judge validation move to extension X1.
5. **Oracle computation specified:** a multiple-choice knapsack LP solved with per-item convex hulls and a greedy upgrade order.
6. **Oracle scope corrected:** the oracle frontier bounds cascades as well as direct routers; "% of oracle savings" applies to cascade rows.
7. **Jev access:** prefer TypeSafe's API (`POST /v1/systemone`, `jev-1.13.0`); OpenRouter's Decisions API is alpha and lists a 32K context.
8. **Caching-aware costing (S1 note):** caches are per model, so routing and escalation split cache reuse; account for this when costing shared-prefix or multi-turn traffic.

## Version history

| Version | Date | Summary |
|---|---|---|
| v1 | 2026-10-09 | Original draft: architecture, phased build, experiments, Jev as router |
| v2 | 2026-10-09 | Results matrix + offline evaluation; oracle and random baselines; trained router; quality gate; fairness rules; curves; statistics; stretch re-scope |
| v3 | 2026-10-09 | Sealed test split; matched-quality oracle frontier; convex random reference; 0–1 item scoring; offline-cascade boundary; 3–4 core tasks; Jev docs update |
| v4 | 2026-10-12 | Final: changes 1–8 above |
| v4-A1 | 2026-10-14 | Amendment A1: model lineup, 40/30/30 splits, cost controls |
| v4-A2 | 2026-10-15 | Amendment A2: `haiku` config runs at high effort |
| v4-A3 | 2026-10-16 | Amendment A3: pilot decisions: core tasks, structured outputs, output caps, budget |
| v4-A4 | 2026-10-16 | Amendment A4: IFBench and LiveCodeBench tested as additions, not adopted; `haiku` cap 64,000 |
| v4-A5 | 2026-10-16 | Amendment A5: benchmark and splits frozen; cross-fitting for the trained router; task mixes |
| v4-A6 | 2026-10-18 | Amendment A6: collection-run decisions: noise runs and how they're used, sealed test store, batch safeguards; the collection run's record |
| v4-A7 | 2026-10-19 | Amendment A7: pilot latency and cost per item recorded; claims on the mixes, per-task warnings for IFEval and MBPP+; freeze hashes; MBPP+ reference check under the grader's limits |
| v4-A8 | 2026-10-19 | Amendment A8: Stage 3 decisions: frontier stays `opus`; list-price costs; the simulator's trace interface; config ladder; oracle edge cases; paired bootstrap and paired safety margin; chart v0 |
| v4-A9 | 2026-10-20 | Amendment A9: Stage 4 decisions: routers see the request; router-score tables; a train-fitted rules router; the walk over breakpoints; McNemar's test; the freeze; chart v1 |

## Amendment A1 — implementation decisions (2026-10-14)

Made before Stage 0, so no data has been collected and the freeze still holds. Each item below is also applied in the section it names.

1. **Model lineup (answers open question 1).** Anthropic only, four configs, each with its effort level set explicitly:

   | Config | Model ID | Effort | Role |
   |---|---|---|---|
   | `haiku` | `claude-haiku-5-5` | high (A2; was medium, its default) | cheap |
   | `sonnet` | `claude-sonnet-5-5` | high (its default) | medium |
   | `opus_low` | `claude-opus-5-5` | low | medium alternative |
   | `opus` | `claude-opus-5-5` | medium (its default) | frontier, always the fallback |

   - **Price snapshot, 2026-10-12** ($ per 1M input / output tokens): Haiku 5.5 $0.10 / $0.50 (prompts ≤100K), Sonnet 5.5 $2 / $10, Opus 5.5 $4 / $20. The Message Batches API is 50% off.
   - **Pinning:** per the Claude API reference, these plain IDs are the pinned versions; no dated suffix exists. Log the `model` each response reports.
   - **No `temperature`:** these models reject it. The repeated-subset noise check (§6) is the only control for run-to-run variation.
   - **No refusal fallbacks:** don't enable server-side refusal fallbacks, so every row comes from its own config's model. Record `stop_reason`; a `refusal` or `max_tokens` stop scores as a failure and is counted.
2. **Splits: 40 / 30 / 30** (train / dev / test), replacing 50 / 20 / 30 (§5). Dev is 1.5× larger, so threshold selection is less noisy.
3. **Frontier skipped on train.** The `opus` config runs on dev and test only (§6). The trained router needs no frontier labels because frontier is the fallback; `opus_low` still runs on train. Anything that needs frontier outcomes (router thresholds, gate cutoffs, references A–D) is selected or computed on dev only, so cross-validation over train+dev is dropped (§8).
4. **Gates (§9).** The Jev, schema, and code-execution gates run on every answer: train/dev now, test in Stage 8. Jev costs about $1 in total. The verifier gate uses Sonnet 5.5 and runs on dev answers only (test in Stage 8). It is the comparison that makes the Jev-gate result meaningful, and the fallback general-purpose gate if Jev access fails.
5. **Batch API for bulk runs (§6).** The collection run (dev matrix and sealed test answers) and the verifier-gate runs use the Message Batches API. Key results by `custom_id`, never by order. Batch rows have no meaningful latency (`latency_s = null`): per-config latency comes from the pilot (standard API) and the Stage 8 live replay, and offline latency estimates use those.
6. **Response store.** Save every API response on disk, keyed by model, effort, prompt hash, and run number, and reuse it on re-runs. Re-running analysis or fixing scoring code never pays for a call twice.
7. **Output cap.** Set `max_tokens` per config from the pilot's observed output lengths, with headroom. Hitting it scores as a failure (item 1).
8. **Development runs** use at most 20 pilot items on `haiku` until the code works.
9. **Budget (answers open question 2).** Working ceiling: $150. Estimate under these settings, assuming ~400 input tokens per item until the pilot measures real counts:

   | Size | Short reasoning | Long reasoning |
   |---|---|---|
   | Plan-size (4 tasks × 600 items) | ~$84 | ~$184 |
   | Small (3 tasks × 300 items) | ~$38 | ~$86 |

   If the pilot's token counts put the plan-size estimate over $150, use the small size or drop whichever config the pilot shows is redundant (`sonnet` or `opus_low`).
10. **Open question 3 (deliverable) stays open.** Working assumption: write-up first, with S4 as a stretch goal.

## Amendment A2 — Haiku effort (2026-10-15)

Made before any pilot or collection calls, so the freeze still holds. Applied in the A1 lineup table and `configs/models.yaml`.

1. **The `haiku` config runs at high effort instead of medium (its default).** Haiku 5.5 is expected to gain substantial quality going from medium to high at little added cost: thinking tokens bill as output at Haiku's $0.50 per 1M, so 1,000 extra thinking tokens per item across the plan-size benchmark (2,400 items) adds about $1.20 at standard prices, $0.60 through the Batches API. The A1 budget estimate is effectively unchanged.

## Amendment A3 — Pilot decisions (2026-10-16)

Stage 0's decisions, made from the pilot before any benchmark item is drawn or any collection call is made, so the freeze on Stages 1–8 still holds. Applied in §5 and `configs/models.yaml`.

**The pilot.** 20 items from each of the six candidate datasets (`data/pilot.jsonl`) ran through all four configs on the standard API with a 32,000-token output cap. Banking77, MMLU-Pro, MATH and CRUXEval ran twice, in free text and as structured outputs (item 2), so 800 calls in all. A supplement of 20 MATH Level 5 items (`data/pilot_math_level5.jsonl`) ran as structured outputs only. Together they cost $4.72. No call errored, was refused, or hit the output cap, and every response reported its pinned model ID. Rows with answers are in `results/pilot/` (not committed). Every evaluator was first checked against the reference answers of the full candidate pools (`harness/check_evaluators.py`; for MBPP+, `harness/check_sandbox.py`, Amendment A7), and the pilot's format failures and every item that all four configs got wrong were read by hand.

Correct answers out of 20 (Banking77 and MMLU-Pro as structured outputs):

| Task | `haiku` | `sonnet` | `opus_low` | `opus` |
|---|---|---|---|---|
| IFEval | 14 | 18 | 18 | 20 |
| Banking77 | 10 | 10 | 13 | 14 |
| MMLU-Pro | 12 | 14 | 13 | 14 |
| MBPP+ | 17 | 17 | 18 | 17 |
| MATH (structured) | 20 | 20 | 20 | 20 |
| MATH Level 5 (structured) | 20 | 20 | 20 | 20 |
| CRUXEval (both formats) | 20 | 20 | 20 | 20 |

1. **Core tasks: IFEval, Banking77, MMLU-Pro, MBPP+.** CRUXEval and MATH are dropped as saturated (§5, ceiling effects): every config answered every item correctly, so there is no routing decision to study. MATH's main pilot items were mostly Levels 2–4, so the Level 5 supplement tested whether hard MATH still separates the configs, under a rule fixed before it ran: Level 5 MATH would be the fourth task unless every config scored 19/20 or better, and MBPP+ otherwise. Every config scored 20/20, so MBPP+ is the fourth task. With 20 items per cell, gaps of 2–3 items are within noise; only saturation is a clear signal.
2. **Banking77 and MMLU-Pro answer as structured outputs.** Their requests set `output_config.format` to a JSON schema (`loaders.structured_variant`): an enum of the item's labels or option letters as `answer`, after a short `explanation` field for MMLU-Pro. IFEval (its instructions are about the free-text shape) and MBPP+ (code stays in a code block) remain free text. On the 320 paired answers of the four tasks that ran in both formats, structured outputs were right where free text was wrong 13 times and the reverse twice, and format failures fell from 10 to 0. All six of Haiku's free-text MATH format failures held the correct answer without the requested `\boxed{}`, so free text understated a cheap config for a reason a production app would remove by requesting JSON. Structured outputs also produced fewer output tokens, though each request carries about 540 more input tokens of injected format instructions. No schema field asks for reasoning, because a `reasoning` field can draw a `reasoning_extraction` refusal.
3. **Output caps (`max_tokens`, collection run):** `haiku` 32,000 (raised to 64,000 by Amendment A4); `sonnet`, `opus_low`, `opus` 16,000. Each is at least 3× the config's longest pilot output on the four chosen tasks (`haiku` 10,637 on IFEval; `sonnet` 4,739; `opus_low` 3,089; `opus` 4,235). Only generated tokens are billed, so the headroom costs nothing (Amendment A1, item 7).
4. **Budget.** From the pilot's measured costs, the collection run's model calls (Batches API; train without `opus`, dev and test with it) come to about $30 for 600 items of each chosen task, and about $26 at the item counts available (item 5). This replaces the A1 estimate ($84–184), which assumed longer outputs. It leaves the $150 ceiling ample room for the Sonnet verifier gate, the noise-subset re-runs, Jev, and the Stage 8 live replay.
5. **Open for Stage 1: two pools are smaller than the plan size.** IFEval has 541 items and MBPP+ 377 (after the A1 exclusion), so 521 and 357 remain outside the pilot, short of 600 per task; at 40/30/30, IFEval's test split would hold about 156 items, below §5's "a few hundred test items per task type before making per-task claims". Whether to accept this, adjust those tasks' splits, or add items from similar sources is decided before the Stage 1 freeze.
6. **Known item noise, kept.** Some Banking77 queries fit more than one label (on 5 pilot items all four configs agreed on a label other than the key), some MMLU-Pro answer keys look wrong (e.g. `mmlu_pro:4346`), and two MBPP+ items' hidden tests encode quirks of the reference solution that the task statement doesn't give (`mbppplus:448`, `mbppplus:620`). This noise lowers every config's score alike rather than favoring one. It is reported in the write-up, not corrected.

## Amendment A4 — Larger pools tested, not adopted (2026-10-16)

Made in Stage 0, before any benchmark item is drawn or any collection call is made, so the freeze on Stages 1–8 still holds. Applied in `configs/models.yaml`.

**The test.** A3 item 5 left IFEval (521 items outside the pilot) and MBPP+ (357) short of 600. Two sources of the same task types were tried as additions:

- **IFBench** (AI2, 2025): 300 prompts using 58 new checkable instruction types, newer than IFEval. Data under ODC-BY, checker code under Apache 2.0, vendored at a pinned commit in `harness/ifbench/` with five marked patches.
- **LiveCodeBench** releases v5 and v6: 342 contest problems from LeetCode and AtCoder, dated 2024-09 to 2025-04, the newest available. They are graded in the sandbox by `harness/livecodebench_runner.py`, which follows LiveCodeBench's grading rules. The hidden tests are stored as pickles, so they are decoded by an unpickler that refuses to load any code.

Both graders were first checked by `harness/check_evaluators.py`: every item loads and builds, and hand-written right, wrong, crashing and looping answers score as expected. Then 20 items of each (`data/pilot_added_sources.jsonl`) ran through all four configs in free text, for $3.89. That brings Stage 0's spending to $8.61.

| Source | `haiku` | `sonnet` | `opus_low` | `opus` |
|---|---|---|---|---|
| IFBench | 10 | 7 | 8 | 9 |
| LiveCodeBench | 18 | 20 | 20 | 20 |

1. **Neither source is added.**
   - **LiveCodeBench is saturated.** `sonnet`, `opus_low` and `opus` solved all 20 problems, including the six hard AtCoder ones, and each solution passed every test of its problem. `haiku`'s two misses were hard problems on which it hit the 32,000-token output cap. These problems predate the models' training data, so their recency doesn't keep them out of it.
   - **IFBench doesn't separate the configs.** Every config scored 35–50%, and `haiku` scored highest. The failures were read by hand and are genuine misses of the strict rules.
   - **IFBench also drew refusals.** 8 of its 80 responses were classifier refusals in the `bio` category on benign prompts, 4 of them from `opus`. Refusals are an effect this study doesn't measure.

   Under §5's ceiling-effects rule, and given that the extra items carry no routing signal, adding either source would mainly add noise.
2. **The code stays, unused.** The loaders, the vendored checkers, the runner and the pilot file remain so this test can be reproduced. Neither source enters Stage 1.
3. **IFEval and MBPP+ use every available item.** How their splits handle the smaller pools is still open for Stage 1 (A3 item 5).
4. **`haiku`'s output cap is 64,000** (from 32,000 in A3). On hard problems `haiku` thinks past 32,000 tokens: both LiveCodeBench misses above, and up to 29,950 tokens on IFBench. A cap stop scores as a failure, but only generated tokens are billed, so a 64,000-token response costs $0.032 at `haiku`'s price. The other caps are unchanged.

## Amendment A5 — Benchmark and splits frozen (2026-10-16)

Stage 1's decisions, made before any collection call, so the freeze on Stages 2–8 still holds. Applied in §5, §8 and `configs/task_mix.yaml`. `harness/build_benchmark.py` builds the benchmark and `harness/check_benchmark.py` checks it.

1. **Items.** `data/benchmark.jsonl` holds 2,478 items:
   - IFEval: all 521 items outside the pilots.
   - MBPP+: all 357 items outside the pilots.
   - Banking77 and MMLU-Pro: 800 each, seeded draws from the 3,056 and 12,012 items outside the pilots. They are stored in their structured-output form (A3).

   Every item in a `data/pilot*.jsonl` file is excluded. `mbppplus:255`, whose tests need about 2.2 GB of memory, stays in and is graded with a 3 GB limit; its reference solution passes at 3 GB.
2. **Groups.** Items whose question text matches after normalizing case, punctuation and spacing share a `source_group`, and a group never straddles splits. That covers 17 MMLU-Pro pairs (the same question with different answer options; one is an 1,800-character legal scenario) and one MBPP+ pair with identical task text.
3. **Splits: 40/30/30 for every task**, stratified within each task by Banking77 intent and MMLU-Pro subject, so every split holds all 77 intents.

   | Task | Train | Dev | Test |
   |---|---|---|---|
   | IFEval | 208 | 157 | 156 |
   | Banking77 | 320 | 240 | 240 |
   | MMLU-Pro | 320 | 240 | 240 |
   | MBPP+ | 143 | 107 | 107 |

   The small tasks keep the standard split. To keep the trained router from running short of training data on them, it uses cross-fitting over dev instead (§8, 3b). Test-heavier splits (20/40/40, 30/35/35) were considered; their test error bars differed by only about ±1 point.
4. **Noise subset:** 75 dev items, about 10% of each task's dev split, listed in the manifest. Each runs three times in Stage 2 (§6).
5. **Scoring and success rules.** Every item scores 1 or 0, so quality and success coincide.
   - IFEval: every instruction is followed under Google's strict checks.
   - Banking77: the JSON `answer` names the reference intent, ignoring case.
   - MMLU-Pro: the JSON `answer` is the reference letter.
   - MBPP+: the last Python code block passes every hidden test in the sandbox, within 20 s and 512 MB (3 GB for `mbppplus:255`).
   - Any task: a refusal or a `max_tokens` stop is a failure. A reply missing its requested format is a failure, also counted separately as a format failure.
6. **Task mixes** (`configs/task_mix.yaml`):
   - **Uniform:** 25% per task.
   - **Realistic:** MBPP+ 65%, Banking77 20%, IFEval 10%, MMLU-Pro 5%.

   **Source.** The realistic weights come from the Anthropic Economic Index, January 2026 report: first-party API traffic sampled 2025-11-13 to 2025-11-20. Each task stands in for its closest occupational category:

   | Task | Category | Share of API traffic |
   |---|---|---|
   | MBPP+ | Computer & Mathematical | 52% |
   | Banking77 | Office & Administrative Support | 15% |
   | IFEval | Arts, Design, Entertainment & Media | 6% |
   | MMLU-Pro | Educational Instruction | 4% |

   Those shares cover 77% of traffic. They are renormalized to 100% and rounded to the nearest 5, because the mapping is approximate.

   **How results combine.** Quality is the weighted mean of the task means. Cost is the weighted mean cost per request, and savings are computed from those weighted dollar costs, never by averaging per-task savings.

   **Caveat.** The realistic mix is two-thirds coding, while MBPP+ is easier than typical coding traffic, so the realistic headline likely overstates coding savings. Per-task results are always reported.
7. **Freeze.** The build is seeded and reproduces its files byte for byte. `data/benchmark_manifest.json` records the counts, the groups and the noise subset. It also records the sha256 of `data/benchmark.jsonl`, of each split's ID list, of `configs/task_mix.yaml` and of every pilot file. `harness/check_benchmark.py` verifies them before the Stage 2 collection run, and again in Stage 8 before test is scored.
8. **Budget.** The pilot's per-item costs put the collection run's model calls at these counts at about $30: IFEval $13.77, Banking77 $7.32, MMLU-Pro $6.74, MBPP+ $2.00. The noise subset's extra runs add a few dollars.

## Amendment A6 — Collection-run decisions (2026-10-18)

Stage 2's decisions, made before any collection call, so the freeze on Stages 2–8 still holds. Applied in §6 and `harness/collect.py`.

1. **Noise subset: three runs in all.** The 75 noise-subset items (A5 item 4) run through every config as run 0 with the rest of dev, then twice more as runs 1 and 2: 600 extra requests, about $2–3. Every strategy, reference and threshold uses run 0 only. Runs 1 and 2 feed only the noise measure.
2. **How the noise measure is used.** Without `temperature`, each stored answer is one draw, and a lucky or unlucky draw looks the same as a real strength or weakness. The hindsight oracle is the most exposed: it sends each item to the cheapest config that succeeded on run 0, so a lucky draw counts as savings no deployable system could rely on. Fixed before any data:
   - The **flip rate** is the share of (item, config) pairs in the noise subset whose success differs across the three runs. It is reported per config and per task next to the confidence intervals.
   - If any config's flip rate is 5% or more, the write-up and charts describe the oracle curve as an optimistic upper bound.
3. **What may be seen of test before Stage 8.** For test, the collection run reports only request outcomes (succeeded, errored, expired, canceled), how many responses report a model other than the pinned one, and total cost. It reports no stop reasons, per-config token counts, answers or grades.
4. **Where test answers live.** Test results are written only to `results/test_sealed.sqlite`, never to the JSON response store. Once every test request has a result, the file is made read-only and its sha256 is recorded, with the batch IDs and counts, in the committed `data/collection_manifest.json`. `harness/collect.py` is the only module that names the path. Stage 8 verifies the hash before scoring.
5. **Batch request IDs.** Each request's `custom_id` is its item ID with `:` replaced by `-`, then the config and run, e.g. `mmlu_pro-4346_opus_low_r0`. Results are matched by `custom_id`, never by order (A1 item 5).
6. **Safeguards.**
   - Each batch's ID is saved as soon as the batch is created, so a re-run never resubmits a request that is stored or still in flight. Only errored and expired requests are resubmitted.
   - Submission runs `check_benchmark` first (A5 item 7) and refuses to go ahead if the cost projected from pilot rates exceeds a stated cap.
   - Before the collection run, at most 20 pilot items run on `haiku` through the Batches API (A1 item 8).

**The collection run (2026-10-18).** The smoke run (20 pilot items on `haiku`, $0.004) went through end to end. Then all 9,521 requests ran as three batches, one per split, and every one succeeded the first time, in about 10 minutes, so nothing was resent. The batch IDs are in `data/collection_manifest.json`.

- **Cost:** $29.37 (train $7.57, dev with the noise runs $12.02, test $9.78), against $32.03 projected. That brings spending to about $38 of the $150 ceiling.
- **Models:** every response reports its pinned model ID.
- **Sealed store:** 2,972 test responses, sha256 `2ea23b7e20a847d2f18a8efa2b1d19c4205b10145ddd52e69f84b405fa2d2d58`, read-only.
- **Train/dev stops:** none of the 6,549 scored rows is a format failure.
  - `mmlu_pro:6586` (dev, noise subset), a benign question on where the Lassa and Ebola viruses come from, drew a `bio` refusal from `opus` and `opus_low` on all three runs, before any output. `haiku` and `sonnet` answered it correctly every time.
  - `sonnet` hit its 16,000-token cap once, on `mmlu_pro:9170`.
- **Noise measure (item 2):** flip rates are `haiku` 5/75 (6.7%), `sonnet` 3/75 (4.0%), `opus_low` 4/75 (5.3%) and `opus` 3/75 (4.0%). IFEval flips most. `haiku` and `opus_low` are at or over 5%, so the oracle curve is described as an optimistic upper bound.
- **Evaluator hand check (§14):** `harness/review_collection.py` sampled 20 run-0 wrong answers per task. All 80 were graded correctly:
  - Banking77 and MMLU-Pro: the grader read the JSON `answer` exactly.
  - IFEval: each failed instruction was confirmed from the answer text.
  - MBPP+: none of the 116 MBPP+ failures is a `NameError`, `SyntaxError` or import error, so the grader never picked the wrong code block. One sampled item is item noise like A3 item 6: `mbppplus:800` asks to remove all whitespace, but its tests expect newlines kept.

## Amendment A7 — Stage 0–1 records completed (2026-10-19)

From a review of Stages 0 and 1, made after the collection run and before Stage 3. It changes no item, split, mix, evaluator, output cap or stored answer, so the A5 hashes and the A6 sealed store stand. Applied in A3, §5, §10, §11 and `harness/check_sandbox.py`.

1. **Pilot latency and cost per item.** Stage 0 was to record both (§14), and offline latency estimates use the pilot's (A1 item 5), but A3 gave only totals. From the pilot's rows (`results/pilot/pilot.rows.jsonl`, not committed): the four core tasks in the formats the collection run uses, 20 items per task and config, on the standard API with 8 calls in flight. Latency runs from request to last token, SDK retries included.

   Latency in seconds, median (p90):

   | Task | `haiku` | `sonnet` | `opus_low` | `opus` |
   |---|---|---|---|---|
   | All four (80 calls) | 2.8 (10.8) | 2.5 (10.6) | 3.1 (11.7) | 3.7 (16.9) |
   | IFEval | 6.9 (15.5) | 10.0 (19.6) | 11.1 (19.3) | 14.0 (32.8) |
   | Banking77 | 2.0 (2.8) | 1.5 (1.8) | 1.7 (2.6) | 2.2 (3.7) |
   | MMLU-Pro | 4.0 (6.9) | 3.0 (4.7) | 3.4 (8.4) | 4.5 (6.3) |
   | MBPP+ | 1.6 (3.6) | 2.1 (3.8) | 2.3 (4.7) | 3.4 (5.5) |

   Cost per item in dollars, standard API (the Batches API is half):

   | Task | `haiku` | `sonnet` | `opus_low` | `opus` |
   |---|---|---|---|---|
   | IFEval | 0.00110 | 0.01291 | 0.02107 | 0.02964 |
   | Banking77 | 0.00028 | 0.00408 | 0.00816 | 0.00963 |
   | MMLU-Pro | 0.00040 | 0.00401 | 0.00739 | 0.00842 |
   | MBPP+ | 0.00020 | 0.00274 | 0.00443 | 0.00635 |

   These per-item costs reproduce A5 item 8's projection to the cent.
2. **Claims rest on the two mixes; IFEval and MBPP+ per-task results carry a warning.** §5 asks for "a few hundred test items per task type before making per-task claims". A5 kept 40/30/30, so IFEval has 156 test items and MBPP+ 107 (A3 item 5 read IFEval's as short), while Banking77 and MMLU-Pro have 240 each. Fixed before Stage 3, so before any routing result exists:
   - **Headline claims are made on the two mixes.** Quality-target verdicts ("met" or "not met", §11) are given only for the mixes: a single task's confidence interval is several points wide, too wide to confirm a target within 1–3 points of frontier quality.
   - **Per-task results are always shown**, with their confidence intervals, in the per-task tables (§10) and the per-task chart (§12), so a mix can't hide a task where routing does badly (§4).
   - **IFEval and MBPP+ carry a warning** wherever their per-task results appear, e.g. "Only 107 test items, below the plan's minimum of a few hundred; indicative." The write-up draws no per-task conclusion from them. Banking77 and MMLU-Pro carry no warning.
   - **The realistic-mix headline carries a note** that the mix is two-thirds MBPP+, so it leans on the smallest test split and its confidence interval is wider than the uniform mix's. It sits beside A5 item 6's caveat.

   This only narrows what is claimed.
3. **Freeze hashes, recorded outside the manifest.** The benchmark's hashes lived only in `data/benchmark_manifest.json`, and nothing hashed the manifest itself, which holds the noise subset. At the A5 freeze:
   - `data/benchmark.jsonl`: `7b2ab45dca7efae470b541e4e08cc680fdd672ef6555d39b61a72cb0455cf0d2`
   - `data/benchmark_manifest.json`: `c67caa9493c9344aab40d4fcbab99f1a982f62e2e57780280e43bab94db77921`

   Stage 8 compares both files against these values, as well as running `check_benchmark` (A5 item 7), before test is scored.
4. **MBPP+ reference check under the grader's limits.** A5 item 1 says `mbppplus:255`'s reference solution passes at 3 GB, but no committed check ran it there. `harness/check_sandbox.py` ran every reference at the sandbox default of 512 MB, so it reported 376/378, failing `mbppplus:255` and the excluded `mbppplus:590`, and `harness/check_evaluators.py` covers only pilot items. `check_sandbox` now uses the grader's own time and memory limits (`evaluators.CODE_TIMEOUT_S` and `CODE_MEMORY`) and skips excluded items: 377/377 pass, and the 18 escape checks still pass (2026-10-19). `mbppplus:255` is in test, so no collection-run score depended on it.

## Amendment A8 — Stage 3 decisions: offline simulator and references (2026-10-19)

Stage 3's decisions, from a design review made before any Stage 3 code, router or threshold exists. Each fixed config's quality and cost on dev had been summarized; nothing else had been computed. They use train and dev only and change no item, split, mix, evaluator, stored answer or the sealed store. Applied in §7, §8, §12, §13 and the new `GLOSSARY.md`.

1. **The frontier stays `opus`.** On dev, `opus_low` matches `opus`'s quality for about 80% of its cost (run 0, list prices):

   | Config | Uniform quality | Uniform $ / 1k | Realistic quality | Realistic $ / 1k |
   |---|---|---|---|---|
   | `haiku` | 84.5% | 0.43 | 83.3% | 0.29 |
   | `sonnet` | 86.1% | 5.81 | 84.5% | 3.80 |
   | `opus_low` | 89.7% | 9.61 | 87.8% | 6.56 |
   | `opus` | 89.5% | 12.01 | 86.7% | 8.04 |

   The quality gap is 0.15 points under the uniform mix and 1.0 under the realistic mix, mostly two MBPP+ items out of 107, so it is likely noise (item 12 records its interval). Making `opus_low` the frontier after seeing dev would be a post-hoc choice, so A1 stands. The finding is reported, and the random-mixture envelope already credits it: `opus_low` is one of its corners, so always-`opus_low` sets savings that any router has to beat.
2. **Costs at list price.** The stored `cost_usd` includes the Batches API discount (A1 item 5). The simulator recomputes each row's cost from its tokens at the 2026-10-12 list prices, checks that this is twice the stored cost, and charges router and gate overhead at list price too. A live router decides per request, so overhead is compared with model calls at the prices both would pay. The batch discount stays a provider-side lever, modeled in S1.
3. **Vocabulary.** `GLOSSARY.md` fixes the terms used from Stage 3 on: a **policy** is anything that decides, per item, which configs to call (fixed configs, routers, cascades; not "system" or "strategy"); a **reference** is A–D; the **config ladder** is the fixed cheap-to-dear order (not "tier"); a **trace** is a policy's record for one item. Older wording in this plan stands where it is written.
4. **The simulator's interface**, shared by Stages 4–7, and by Stage 8 once test is scored:
   - A policy hands the simulator one trace per item: the configs it called, in order, and its overhead in dollars. The answer returned is always the last config's. A router calls one config, a cascade climbs the config ladder (calling configs in its order, skipping any it likes), and a fixed config calls itself on every item.
   - The simulator returns one outcome per item (task, source group, cost, success, configs called), and totals per mix and per task.
   - It refuses a set of traces that misses or repeats an item, names an unknown config, calls a config twice on one item, calls none, calls configs down the ladder (paying for a dearer config, then returning a cheaper one's answer), calls `opus` on a train item (A1 item 3), or has an overhead that is negative or not a finite number, which would make a policy look cheaper or spoil its totals. Nothing is evaluated partially.
   - It reads no gate verdicts, router scores or latencies: policies turn those into traces. Estimated latency comes in Stage 7, from A7 item 1's pilot latencies summed along each trace.
   - The oracle and the envelope are computed directly, not from traces.
5. **Config ladder: `haiku` < `sonnet` < `opus_low` < `opus`.** A router's "cheapest config whose score ≥ t" (§8) and a cascade's "next config up" (§9) use this fixed order, which is the order of mean dev cost per request in every task and both mixes. Per item the order varies (`sonnet` costs more than `opus_low` on 4.8% of dev items, `opus_low` more than `opus` on 12.8%), and no policy knows an item's cost in advance. Exact cost ties go to the lower config; on dev there are 234, all between `opus_low` and `opus` (209 on Banking77, where the two often give identical token counts).
6. **Oracle edge cases** (§7):
   - **Weights.** Each item weighs its task's mix weight divided by that task's dev item count (A5 item 6). Under a mix the oracle may trade quality between tasks, since policies are judged on the mix too. Per-task results use a per-task oracle.
   - **Hull.** On dev, `haiku` is the cheapest config on every item, so each item's hull has at most two points: `haiku` and the cheapest config that succeeds. 46 dev items fail on every config and stay on `haiku` with score 0. On 28, `opus` fails and a cheaper config succeeds.
   - **Ties** go to the lower config (item 5). This changes which config is reported, never a number.
   - **Targets out of range.** A target at or below the quality of sending every item to its cheapest config gets that cost. A target above the oracle's maximum quality is "n/a (infeasible)". The §11 targets are feasible on dev under both mixes.
   - **Value.** The LP's value is reported, fractional last upgrade included.
   - **Invariant.** A policy that meets a target can't cost less than the oracle at that target, so `captured_fraction ≤ 1` is asserted, not assumed.
   - The oracle uses run 0 only and is described as an optimistic upper bound (A6 item 2).
7. **Both references answer both ways.** The oracle and the envelope each give the cost of reaching a quality target (for `captured_fraction`, §7) and the quality reachable at a given cost (for comparisons at matched cost, §8). A quality outside a reference's range gives "n/a", and so does a cost below its cheapest point. A cost past its costliest corner reaches its maximum quality, since a bigger budget needn't be spent; otherwise a router costing more than the oracle's costliest corner, which on dev is little more than always-`haiku`'s cost, could never be compared with it at matched cost. The envelope is the upper concave hull of the four fixed-config points under a mix. It is computed in linear cost, because mixing odds are linear in cost, and drawn as a curve on the log axis. On dev it is one segment, from `haiku` to `opus_low`.
8. **Paired bootstrap, built in Stage 3** (§4). Source groups are resampled within each task and the mix weights applied afterwards: 2,000 seeded resamples, percentile 95% intervals. The same resamples serve every policy and reference, so differences from the frontier and captured fractions get intervals, and the oracle is re-solved on each resample. McNemar's test arrives with the first router in Stage 4.
   - **Captured-fraction intervals.** Each resample recomputes the quality targets from its own frontier quality, so they stay paired. A resample where the policy misses the target, or where the oracle saves nothing, counts as "n/a", which ranks below every number. The interval is taken over all resamples, so its lower bound reads "n/a" when more than 2.5% of them are n/a, and the share of resamples that meet the target is reported beside it. Dropping those resamples would inflate the interval.
9. **Safety margin: one paired standard error** (§8). "One standard error of the quality estimate" means one standard error of the policy's quality difference from the frontier on dev, from item 8's bootstrap, per mix and per candidate threshold.
   - **Why paired.** Quality targets are relative to frontier quality on the same items (§11), so the luck of which items were drawn moves the frontier and the target together. Whether a target is met depends on the difference.
   - **Why it matters.** On the realistic mix, where MBPP+'s 107 dev items carry 65% of the weight, a fixed config's own standard error is about 2.4 points, but its paired standard error against `opus` is 1.2–1.8 points. A 2.4-point margin would ask for dev quality of 89.1%, 88.3% and 86.5% at the 100/99/97% targets: above every config's quality for the first two, and about `opus`'s own for the third.
   - **Kept at one, knowing what it does to captured-fraction intervals.** A threshold chosen to clear its target by one paired standard error falls short of it in about one resample in six (a normal tail one standard error out), far more than the 2.5% that leaves item 8's lower bound a number. So a policy's captured-fraction interval will usually read "n/a" at its lower bound, and the share of resamples that meet the target, reported in a column of its own, says how often the policy would meet it. A lower bound that is a number would need a margin of about two standard errors (one and a half isn't enough), which on dev puts the 99% target, as well as the 100% one, above every config's quality, and cuts the savings at every operating point.
   - Fixed before any router exists.
10. **Chart v0** (§12): dev, run 0, n = 744, list prices of 2026-10-12.
    - **Headline figure.** Realistic and uniform panels. x is cost per 1,000 requests on a log scale; y is mix-weighted quality. The four configs are points with their own 95% intervals on both axes, A (`opus`) and B (`haiku`) labeled. C is the envelope and D the oracle, labeled an optimistic upper bound; neither has a band. Dotted lines mark the three §11 targets. The caption says differences between policies are judged by the paired intervals in the table, and the realistic panel carries A7 item 2's note.
    - **Per-task figure.** A 2×2 of the four configs with intervals and each task's own oracle and envelope. No target lines, since verdicts are given for the mixes only (A7 item 2). The IFEval and MBPP+ panels carry A7's warning with their dev counts (157 and 107).
    - **Table v0.**
      - Per mix, for each config: quality and cost per 1k with intervals, savings against A, and the paired quality difference from A.
      - Per mix, at each target: the oracle's cost and savings, the envelope's cost and captured fraction, and the oracle's routing mix.
      - Per mix, once: the oracle's maximum quality and the number of items every config fails.
      - Per task: each config's quality and cost with intervals, and the oracle's maximum quality. No target rows.
    - **Files.** Each figure is a PNG with a JSON of every plotted number, committed under `figures/` and overwritten as later stages add to the chart; git keeps the history. They hold aggregates of dev only.
11. **Code** (§13). `analysis/` is a package of scripts run like the harness (`uv run python -m analysis.<script>`), adding numpy and matplotlib. Pandas and notebooks are dropped, since the harness works with plain rows.
    - `analysis/check_simulator.py` checks the greedy oracle against a brute-force whole-item knapsack on small random matrices, and the envelope and trace accounting against hand-worked cases.
    - Of the results stores, the analysis code reads only `results/dev_matrix.sqlite` (and the frozen benchmark, for each item's source group). It refuses rows outside train and dev, and doesn't name the sealed store (A6 item 4).
12. **Record.** Chart v0's headline numbers on dev (run 0, n = 744, list prices of 2026-10-12), from `figures/pareto.json`. Intervals are percentile 95% from 2,000 paired resamples, and `analysis/check_simulator.py` compares every number here with that file.

    | | Realistic | Uniform |
    |---|---|---|
    | A (`opus`): quality, $ per 1k | 86.7% (81.8 to 91.1), $8.04 | 89.5% (87.2 to 91.9), $12.01 |
    | B (`haiku`): quality, $ per 1k | 83.3% (78.2 to 87.9), $0.29 | 84.5% (81.6 to 87.2), $0.43 |
    | `opus_low` − `opus`: paired quality difference | +1.0 points (−1.3 to +3.6) | +0.2 points (−1.2 to +1.6) |
    | D: maximum quality | 91.5% | 93.6% |
    | D: $ per 1k at 100%, 99%, 97% of A's quality | $0.40, $0.37, $0.31 | $0.65, $0.59, $0.51 |
    | D: savings against A there | 95.0%, 95.4%, 96.1% | 94.6%, 95.1%, 95.8% |
    | C: captured fraction there | 0.38, 0.54, 0.85 | 0.24, 0.37, 0.65 |

    - The paired interval of the `opus_low` − `opus` gap contains 0 under both mixes, so item 1's reading stands: the gap is likely noise, and the frontier stays `opus`.
    - The rest of table v0, every config's row and the per-task figure's numbers among them, is in `figures/pareto.json` and `figures/pareto_tasks.json`.

## Amendment A9 — Stage 4 decisions: the rules router and threshold selection (2026-10-20)

Stage 4's decisions, from a design review made after chart v0 and before any Stage 4 code, rule or router score existed. Nothing new was computed for them: the dev numbers they cite are Stage 3's. They use train and dev only and change no item, split, mix, evaluator, stored answer or the sealed store, so the A5 and A7 hashes and the A6 sealed store stand. Items 1–3 and 9–12 apply to every router from here on; items 4–6 and 13 are the rules router's. Applied in §4, §5, §8, §12, §13 and `GLOSSARY.md`. Two record items follow later, as A7 item 3 and A8 item 12 did: item 17, the freeze hashes, in the freeze commit (item 13), and item 18, E's operating points and the hash of its section, when Stage 4 finishes.

1. **A router sees the request** (§8). A request is what every config receives for an item: its prompt and, if it has one, its output schema, which the API carries as `output_config.format` (the collection run sends no system prompt). A router sees what the model sees, minus the settings that depend on the config (effort, `max_tokens`). The task, source, source group, split, `meta` and expected answer, and anything else the model never sees, stay off limits. §8's `score(prompt)` becomes `score(request)`.
   - A live router, the S4 proxy and Jev's Variant 1 ("State: the request") all sit where the request is.
   - On this benchmark only Banking77 and MMLU-Pro have schemas, and both repeat the label or option list already in the prompt, so the schema matters as a general "structured output requested" cue.
2. **Vocabulary.** `GLOSSARY.md` adds **request**, **router score**, **threshold**, **breakpoint**, **router-score table**, **walk** and **operating point**. A router score is never called bare "score", which is an item's quality score, nor "probability", which implies calibration that Jev's needn't have. A router reads the request, not only the prompt. "Cutoff" is kept for gates, so a later "gate score" can follow the same pattern.
3. **The deployable router and its router-score table** (§8). In Stage 5 no single model scores every dev item (cross-fitting, A5), and Jev's responses are paid for and stored, so selection reads a table, not a router.
   - **The router.** `score(request)` returns a router score for `haiku`, `sonnet` and `opus_low`. The frontier is never scored: it is the fallback, and needs no train labels (A1 item 3). Stage 8 and S4 call this.
   - **The decision rule.** A request goes to the cheapest config on the config ladder (A8 item 5) whose router score is at least the threshold, or to the frontier if none is.
   - **The table.** A router-score table holds a router's router scores and overhead, in dollars at list price, for every item of a split. The walk and the curves read only the table, never a router: a router's traces at any threshold are built from it. E builds its table by calling its own `score(request)` on every request, so the two can't drift apart. Stage 5 builds its from out-of-fold predictions, each fold's model scoring its held-out items. Stage 6 builds its from Jev's stored responses, each item's overhead from its tokens.
   - **Refusal.** A table is refused whole, as traces are (A8 item 4), if it misses or repeats an item, scores the frontier, leaves out a config below it, has a router score that isn't a number from 0 to 1, or has an overhead that is negative or not finite.
   - **Test.** Stage 8 builds test's table the same way, after the freeze. Stage 5 decides which model scores test, one refit on train and dev or the fold models; the table supports either.
4. **The rules router's router scores are train success rates** (§8 3a). Hand-written rules put each request in a bucket (item 5). A bucket's router score for `haiku`, `sonnet` and `opus_low` is that config's run-0 success rate on the bucket's train items. The minimum bucket size, 30 train items, is fixed here, before any rate is computed; a smaller bucket folds into its type's bucket. Rules are written from train prompts only, and dev only selects thresholds, which narrows §5's "dev: tune rules".
   - **Why train.** Hand-set numbers, as in §8's sketch, can't be set fairly: they come from intuition or from looking at success rates somewhere. Rates estimated on dev would fit router scores, thresholds and margins on the same items, so dev quality would look better than it is, and test would miss targets more often than A8 item 9's one in six.
   - **E against F.** Both learn from the same train labels, so the comparison is hand-written cues against embeddings, with the labels held equal.
   - **Expect about one config per task.** At 30 items, a rate near 85% has a standard error of about 6.5 points, more than most config gaps within a task on dev (1–5 points).
   - **Absolute rates.** Train has no frontier runs, so router scores are success rates, not gaps to the frontier. A hard bucket where every config is weak goes to `opus` even if `opus` does no better there. That follows from §8's rule, and the write-up says so.
5. **Buckets.** Four inferred types with one difficulty cue each, so up to eight buckets.
   - **Type checks**, in this order: an output schema, whose answer being a set of letters marks multiple choice and a set of other labels classification; then a lettered option list (multiple choice); then a request to classify or categorize with a list of labels (classification); then a request for a function or a code block (code). Anything else is open-ended, so every request has a bucket. Code comes after the option list because some MMLU-Pro computer-science questions contain code.
   - **Multiple choice:** a yes/no "needs a calculation" flag (digits with operators or units, or LaTeX), since MMLU-Pro's physics, chemistry and engineering questions carry numbers and units but rarely LaTeX. Its two buckets' sizes depend on how common the flag is.
   - **Classification and code:** prompt length in characters, split at the type's train median. Token counts would need an API call, and the stored `input_tokens` come from the response.
   - **Open-ended:** the number of sentences with general constraint markers ("must", "at least", "do not", "exactly", "end your response with"), split at the train median. An IFEval answer counts only if every instruction is followed, so each constraint is one more chance to fail, and IFEval is where routing has the most room (`haiku` 83.4% on dev, against its own oracle's 98.1%).
   - **Generic cues.** The rules look for cues real traffic would also show, never this benchmark's template lines or dataset-specific words such as Banking77's label names, so S4 could reuse them. The task-inference diagnostic, how often the inferred type is the item's task, is expected near 100% here and is described as holding "on this benchmark's templates", not as evidence that the cues transfer.
   - Medians come from train prompts, which involves no outcomes.
6. **Overhead and latency** (§8: "router cost and latency are added to every request"). E runs locally, so every trace's overhead is an explicit $0. Its latency is measured once over the dev requests, which shows no outcomes: median and p90, with the patterns compiled at import, warm-up excluded and the machine noted. It feeds Stage 7's latency estimate. No per-item latency field is added until Jev, whose latency varies by request.
7. **Keeping routers off test items** (§5–6, A6). A router can't know an item's split, so the guard sits where requests are made.
   - **One benchmark reader** on the analysis side yields train and dev items only, and a request for a test item is refused whole. `analysis/matrix.py` takes its source groups from it instead of opening the file itself. A check shows the refusal on a synthetic test item, never a real one.
   - **The conversion from a benchmark row to a request** lives in `routers/` (item 8) and builds each request from the stored `prompt` and `output_schema`, the fields the collection run sent.
   - **An allowlist check** searches tracked files for both `BENCHMARK_FILE` and `benchmark.jsonl`, so a path written out by hand is caught too. It allows the four harness modules that name the file (`build_benchmark`, `check_benchmark`, `collect` and `review_collection`), the reader, `data/benchmark_manifest.json` (whose hash keys hold the name) and Markdown files, and must find `harness/build_benchmark.py` as its known positive.
   - **A8 item 11 is widened:** analysis code may also read the benchmark's prompts and output schemas for train and dev items, through the reader only.
   - **Stage 8's unlock is specified now and built in Stage 8**, so no code able to produce test requests exists before then. The hashes it checks are all valid today, so they couldn't enforce the timing. It is a separate function, not a flag on the reader. It verifies A7 item 3's two hashes and runs `check_benchmark`, checks the sealed store's hash through `harness/collect.py`'s own path, and checks the freeze hashes (item 13), which is where §14's freeze of policies, thresholds and margins is enforced. If any check fails, it refuses everything.
8. **Code layout** (§13).
   - **`routers/` is deployable.** It imports nothing from `analysis/` or `harness/`, so S4 could load it, and `harness/collect.py`, which names the sealed store, never enters its imports. It holds the config ladder (moved from `analysis/matrix.py`, which re-exports it and still asserts it ends at `harness/collect.py`'s frontier), the request, the `Router` protocol, the decision rule, the row-to-request conversion, the rules router and its rates file. Fitting is a pure function there, taking requests and train successes as plain data.
   - **`analysis/`** holds the reader, the router-score table and its refusal, the walk and operating points, McNemar's test, and chart and table v1.
9. **Threshold selection: the walk** (§8). Each threshold is its own policy, with its own safety margin (A8 item 9). Thresholds are tried in a fixed sequence (fixed-sequence testing, as in Learn-then-Test), so trying many doesn't raise the chance of missing a target on test.
   - **Breakpoints.** An item's breakpoints are its router scores that are strictly higher than every cheaper config's router score for it: the thresholds at which its config changes. A score that can never change a decision, such as `sonnet` scored below `haiku` on the same item, isn't one.
   - **Steps.** If the router's dev items have 24 or fewer distinct breakpoints (E's eight buckets × 3 configs), each is a step. Otherwise the steps are the 0%, 5%, …, 95% quantiles of its pooled dev breakpoints with duplicates removed, at most 20: the first moves about 5% of decisions, and the last reaches the bottom of the router's range. Steps come from the router-score table alone, before any outcome is looked up, which is what keeps the guarantee.
   - **The walk**, one per task mix and quality target, starts at always-frontier (A) and tries the steps from the highest threshold to the lowest. At a step, each item goes to the cheapest config whose router score is at least the step's threshold. A step clears if its dev quality less the target is at least its margin. The walk stops at the first step that doesn't clear, and every step before it is certified.
   - **The operating point** is the cheapest certified step under that mix, which isn't always the last: per-item cost isn't monotone on the ladder (`sonnet` costs more than `opus_low` on 4.8% of dev items, A8 item 5), and fixed-sequence testing certifies every step before the first failure together.
   - **Rejected.** Taking the cheapest threshold that clears, among all of them, picks the luckiest of many, so it would miss on test more often than one in six. A fixed grid of 0.05 is too coarse for scores that bunch: with dev's per-task success rates as router scores (all twelve between 0.82 and 0.94), it gives four policies against twelve steps on distinct values, and at t = 0.85 four tasks change config at once.
   - **Cascades.** A cascade's order of steps (router thresholds and gate cutoffs together) is fixed when cascades are designed, in Stage 7.
10. **Six operating points per router**, one per task mix and quality target, since quality targets are defined per mix. Where the walk certifies nothing cheaper than A, the operating point is A itself: no router call and no overhead, captured fraction 0.00 (A meets the target and saves nothing), labeled "falls back to A". A router that never changes a decision isn't worth calling, and for Jev, sending every item to `opus` through the router would cost more than A.
11. **McNemar's test** (§4), which A8 item 8 left to this stage.
    - Exact and two-sided: a binomial on the discordant pairs, computed with `math.comb`, with no scipy. Every task scores 1 or 0 (A5 item 5), so it compares items' successes.
    - Per task, every policy row against A: the fixed configs (so `opus_low` against `opus` gets one too) and every operating point. It runs on dev now and on test in Stage 8. Reported as the two discordant counts and p, with "—" for a row that falls back to A.
    - None under the mixes, where items weigh unequally. There the paired bootstrap interval of the difference (A8 item 8) is the test.
    - Descriptive only: no multiplicity correction, never used to select a threshold, and A7 item 2's warning applies to IFEval and MBPP+. McNemar tests for a difference, so p > 0.05 never reads as "matches `opus`"; whether a policy loses no material quality is answered by its paired interval against the target. It also treats items as independent, ignoring the source groups the bootstrap keeps together. The effect is negligible, and the write-up says so.
12. **Matched cost uses A8 item 7's reading** (§8: "compared at matched cost"). Past a reference's costliest corner, the quality it reaches at a cost is its top quality (A8 item 7, as the Stage 3 review settled it). So E is compared with the envelope even where E costs more than always-`opus_low`, at which the envelope ends on dev ($6.56 per 1k realistic, $9.61 uniform). To certify the 100% realistic target, E needs dev quality of about 86.7% plus a margin of about 1.2–1.8 points, at or above `opus_low`'s 87.8%, so it will probably send some items to `opus` and cost more than that. E costing more than always-`opus_low` for no more quality is the result the comparison exists to show. This design review had planned a separate "quality within a budget" query for it; A8 item 7's revision made that unnecessary.
    - Below a reference's cheapest corner the reading stays n/a. On dev that can't happen to E: `haiku` is the cheapest config on every dev item (A8 item 6), and E has no overhead, so E can't cost less than B. That holds on dev only. On a test item where another config happens to cost less than `haiku` it is possible, if unlikely, and reads n/a.
13. **The freeze** (§8: "record the margin, selection method, and chosen thresholds before Stage 8"; §14).
    - **Rates file.** Beside the rules router: each bucket's router scores and train count, and the cut points (the train medians), so Stage 8 uses exactly these numbers and never re-reads train.
    - **Hashes** (sha256, as in A7 item 3): the rates file, and every `routers/` module in the rules router's import closure, `routers/__init__.py` included, plus the row-to-request conversion's module if the closure doesn't hold it. A check works the closure out from the imports and fails if the recorded list differs, so a module imported later can't sit outside the freeze.
    - **Order, provable in git.** Cues may be tuned on train prompts, train outcomes and the train task-inference diagnostic. The freeze comes before any E number on dev, in this order: the freeze, then the dev router-score table, then the dev task-inference diagnostic, then the walk. Simulating E's dev traces reveals dev outcomes as surely as the walk does, and the dev diagnostic uses dev task labels, so neither may come first. The freeze commit, holding the rules router, the rates file and the hashes (item 17), comes before any commit that holds an E number on dev, so the order can be checked in the log.
    - **Train rates are optimistic** if cues were tuned on train outcomes: they are measured on the items the cues were fitted to, so the buckets look more separated than they are. That can make the walk less efficient but can't break its guarantee, which rests on dev alone. The train rates aren't read as unbiased.
    - **After the freeze**, the modules E's hashes cover are never edited. Stages 5–7 add files to `routers/`; an edit to a frozen one means re-selecting E's operating points under a new amendment.
    - **One operating-points file**, shared by every router, holds for each router, mix and target the threshold, its margin, the selection method, the walk's steps and whether it fell back. Stage 8 reads thresholds only from it. E's section, written with sorted keys, gets its own sha256 when Stage 4 finishes (item 18), and the whole file is hashed once Stage 7 has finished adding to it. A hash catches any change, where a check against the record's rounded numbers would pass a threshold that moved in the fourth decimal.
14. **Chart v1** (§12), overwriting chart v0's files (A8 item 10).
    - **Headline figure.** E's steps are markers joined by straight lines in linear cost, drawn as a curve on the log axis like the envelope: mixing two adjacent steps at fixed odds traces that line. The steps are the same on both panels, since they come from router scores, not the mix. The curve runs in walk order from A, which is E's first point since it has no overhead. A step can cost more than the one before it, so the line may double back; it isn't hulled.
    - **Crosses** with 95% intervals on both axes mark E's operating points only, each labeled with its target. Points that coincide share one cross with a combined label ("99%, 97%"; "100%: falls back to A").
    - **No band**, for A8 item 10's reasons. This changes §12's "one curve per router and cascade, with CI bands": a router's uncertainty is shown by the crosses at its operating points and by the table's paired intervals.
    - **Per-task figure.** E's steps in each task as a line with markers, with no operating points, since verdicts are given for the mixes only (A7 item 2).
    - Called v1 so that A8 item 12 keeps naming v0.
15. **Table v1** adds rows and keys to table v0 and changes none of v0's, so the check on A8 item 12 keeps passing.
    - **Per mix, at each operating point:** target, threshold, steps certified and whether it fell back to A; quality and cost per 1k with intervals; margin; paired quality difference from A with its interval; savings against A; captured fraction with its interval and the share of resamples that meet the target (A8 item 8); and the routing mix.
    - **At matched cost:** the envelope's quality at the operating point's cost (item 12), and E's paired quality difference from it with an interval. Each resample rebuilds the envelope from its own config points, as the envelope's captured fraction does.
    - **Per task:** each operating point's quality and cost with intervals, and McNemar's test (item 11). The fixed configs' rows get McNemar too.
16. **Build order within Stage 4** (§14). The tickets run in parallel worktrees, branched from one Stage 4 branch and merged back into it.
    - **Wave 0:** this amendment, the glossary entries and the parent issue, before any code.
    - **Wave 1, in parallel:** the shared foundation (the ladder's move, re-exported from `analysis/matrix.py` so existing imports keep working; the request, the `Router` protocol and the decision rule; the row-to-request conversion; the reader, with `matrix.py` switched to it; the allowlist check; the router-score table and its refusal), and McNemar's test, which needs only the simulator's outcomes.
    - **Wave 2, in parallel, after the foundation:** the rules router on train only (cues, medians, fitted rates, the train diagnostic and latency over the dev requests), ending with the freeze commit; and the walk with the operating-points file, checked on hand-worked synthetic tables.
    - **Wave 3:** the dev router-score table, the dev diagnostic, the first walk on dev, chart and table v1, E's operating points and its section's hash (item 18) with a check on them, and the README.
    - Each parallel ticket owns its new check script and wires it into the pre-commit hook and CI. A check that needs `results/`, such as "the committed rates match a refit from the train matrix", runs in the hook only, like `check_simulator`. The cues on synthetic requests and the rates file's shape run in CI too.

---

# 1. Goal

Build and evaluate a system that decides how each AI request is processed in order to:

**Minimize API cost while keeping output quality close to always using the strongest available model.**

> How much can we reduce LLM inference cost without materially reducing answer quality — and which decision strategy gets closest to the best possible tradeoff?

A successful result might look like (illustrative, not a target):

- Baseline quality: 92%
- Optimized quality: 91% (95% CI ±1.2)
- Cost reduction: 55%
- Captures 70% of hindsight-oracle savings at the same quality target

The real numbers are the research output.

---

# 2. Core Idea

Instead of sending every request to the frontier model, put a decision layer in front of the models:

```text
                 User Request
                      |
                      v
                    Router
          (model tier + effort level)
             /        |        \
          Cheap     Medium    Frontier
             \        |        /
              v       v       v
                   Response
                      |
                      v
                 Quality Gate
                 /          \
             pass            fail
               |               |
            return     escalate to next tier
```

Cost levers, in order of project focus:

1. **Route** each request to the cheapest model config likely to succeed.
2. **Escalate** when the cheap answer isn't good enough (cascade).
3. **Provider-side discounts** — prompt caching and batch APIs (stretch S1).
4. **Avoid calls** — semantic cache (stretch S2).
5. **Send fewer tokens** — context optimization (stretch S3).

A "model config" is a `(model, reasoning effort)` pair. A medium model at high effort may beat a frontier model at low effort on cost per correct answer.

---

# 3. Prior Art

Read before building; use for baselines; cite in the write-up. From memory — confirm before citing:

- **FrugalGPT** (Stanford, 2023) — LLM cascades: try cheap models first, escalate on low confidence. Essentially the cascade in Phase 4.
- **RouteLLM** (LMSYS, 2024) — open-source trained routers between a strong and a weak model. Usable as a trained-router baseline.
- **RouterBench** (Martian, 2024) — benchmark and dataset for evaluating routers.
- **Commercial routers** — OpenRouter's Auto Router, Not Diamond, Martian.

What this project adds: rules vs. trained vs. Jev routers head to head, Jev as a quality gate, effort level as a routing dimension, and full accounting of router and gate costs.

---

# 4. Metrics

## Cost

Record per model call: provider, model version, effort level; input, cached input, output, and reasoning tokens; cost (from the shared dated price snapshot, plus provider-reported cost where available); latency.

Every system pays for **all** its calls:

```text
system_cost = router calls + model calls (every tier tried) + gate calls
```

Headline cost metrics:

```text
cost per 1,000 requests
cost per successful answer
savings vs. always-frontier at a preselected quality target
% of hindsight-oracle savings captured at that same target
```

## Quality

The core study uses only task types with **objective evaluators**, so it needs no LLM judge. Judge-graded tasks are extension X1.

| Core-eligible task | Quality metric |
|---|---|
| Classification | Item accuracy; dataset F1 as a separate diagnostic |
| Multiple choice | Correct answer |
| Math / short-answer reasoning | Normalized answer match against reference |
| Information extraction | Exact match / token F1 |
| Structured output | Schema valid + field-level correctness |
| Coding | Hidden unit tests passed |

**Headline score.** Every item gets a score in `[0, 1]`: `1/0` for correct/incorrect tasks, partial credit (e.g., token F1) normalized to `[0, 1]`. Each task also has a predeclared binary **success rule**, used to train routers and measure gate errors. Compute each task's mean item score, then combine task means with the two predeclared mixes (§5). Never average dataset-level F1 with item-level scores. Report F1 and other task-specific metrics separately, and always show per-task results so the aggregate can't hide a regression.

## Statistics

All systems run on the same items, so comparisons are **paired**:

- 95% confidence intervals for paired quality and cost differences: bootstrap within task types, keep near-duplicates and shared-source groups together, then apply the task weights.
- McNemar's test for binary-scored tasks ("is system X different from frontier?"), per task and descriptive only (Amendment A9 item 11).
- A 2-point gap on a few hundred items may be noise. No claim without a CI.

---

# 5. Benchmark Design

The benchmark decides the headline number: savings depend heavily on the mix of easy and hard items.

## Item format

```python
{
    "item_id": 152,
    "source": "gsm8k",
    "source_group": "gsm8k-0152",  # paraphrases, shared documents, shared templates share a group
    "task_type": "math",           # reporting only — routers never see this
    "split": "dev",                # train | dev | test
    "prompt": "...",
    "expected_answer": "...",      # evaluator only — routers and gates never see this
    "evaluation_method": "exact_match"
}
```

## Core task types

Choose **3–4** in Stage 0 from these objective-evaluator candidates.

**Chosen (Amendment A3): IFEval (instruction following), Banking77 (classification), MMLU-Pro (multiple choice), MBPP+ (coding).** Banking77 and MMLU-Pro answer as structured outputs. CRUXEval and MATH were dropped as saturated.

| Task | Candidate datasets |
|---|---|
| Classification | intent / sentiment sets (e.g., Banking77) |
| Multiple choice | MMLU-Pro, GPQA |
| Math | GSM8K, MATH |
| Coding | HumanEval+ / MBPP+, LiveCodeBench (less contaminated) |
| Extraction | SQuAD-style QA |
| Structured output | custom JSON-schema tasks |

Judge-graded tasks (summarization, open-ended) and long-context sets are outside the core; see X1 and S3.

Watch for:

- **Contamination** — prefer newer sets where possible.
- **Ceiling effects** — use the pilot to replace saturated sets before locking. If a locked task later saturates, keep it and report the limited routing signal.

## Splits

```text
train  40%   fit the trained router and the rules router's rates (no frontier runs; Amendments A1, A9)
dev    30%   tune thresholds, gate cutoffs (not rules; Amendment A9)
test   30%   sealed until Stage 8
```

- Stratify by task type; keep each source group in one split; freeze splits with a file hash before routing work begins.
- Within a task, stratify by Banking77 intent and MMLU-Pro subject. The frozen benchmark is `data/benchmark.jsonl` with its manifest (Amendment A5).
- Pilot items are separate from all three splits.
- The frontier config runs only on dev and test, so thresholds are selected on dev (§8, Amendment A1).
- Raw test answers are collected with the dev matrix but sealed (§6). Before Stage 8, nothing reads, scores, or inspects test answers, grades, or gate verdicts.

## Size

- Enough items for overall quality CIs around ±1–2 points; the pilot gives the variance.
- A few hundred test items per task type before making per-task claims. IFEval (156) and MBPP+ (107) fall short: their per-task results are shown with a warning and draw no conclusions, and quality-target verdicts are given only for the mixes (Amendment A7).

## Task mix

Report under two mixes, documented before Stage 8:

1. **Uniform** — each task type weighted equally.
2. **Realistic** — a weighting that resembles a plausible production workload.

Fixed in Amendment A5 (`configs/task_mix.yaml`):
- Uniform: 25% per task.
- Realistic: MBPP+ 65%, Banking77 20%, IFEval 10%, MMLU-Pro 5%, from first-party API traffic shares in the Anthropic Economic Index.

Re-weighting is offline and needs no new model calls.

---

# 6. Phase 1 — Evaluation Harness and Results Matrices

Build this before any optimizer.

```text
item × model config → call model (temp 0 where supported) → store raw answer → score (train/dev only) → store row
```

**Pinning.** Every config uses a pinned, dated model version (a snapshot ID, never a floating alias). One price snapshot, recorded in config, is used for both dev and test costs. Log the model ID each response reports. For the Claude lineup, the plain model IDs are the pinned versions (Amendment A1).

**Bulk runs use the Message Batches API** (50% off; Amendment A1). Batch rows store `latency_s = null`. Per-config latency comes from the pilot and the Stage 8 live replay.

**One collection run, two stores.**

| Store | Contents | Scored |
|---|---|---|
| `results/dev_matrix.sqlite` | every train/dev item × config (frontier: dev only): answer, tokens, cost, stop reason, quality, success | immediately |
| `results/test_sealed.sqlite` | every test item × config: raw answer, tokens, stop reason | Stage 8 only |

Collecting test answers in the same run as dev means both matrices come from the same model versions, so model or price drift before Stage 8 can't skew the comparison. The analysis code gets no path to the sealed store. Record its hash at collection time and verify it in Stage 8. Router and gate calls on test happen in Stage 8, after those components are frozen.

Dev matrix row (illustrative values):

```python
{
    "item_id": 152,
    "model": "model-small-2026-10-01",
    "effort": "low",
    "run": 0,
    "answer": "...",
    "input_tokens": 850,
    "cached_input_tokens": 0,
    "output_tokens": 220,
    "reasoning_tokens": 0,
    "cost_usd": 0.0031,
    "latency_s": 0.82,
    "quality": 0.91,
    "success": True,
    "price_snapshot": "2026-10-12"
}
```

Notes:

- **Four configs** (Amendment A1): `haiku`, `sonnet`, `opus_low`, `opus`. Add more only if the pilot and budget support them, up to about 5–6.
- **Nondeterminism:** the Claude 5.5 models reject `temperature`, so it isn't set. Re-run ~10% of dev 3× to measure run-to-run noise (three runs in all, run 0 included; Amendment A6).

Output (illustrative):

```text
Config A: Quality 84%   Cost/1k $2.10
Config B: Quality 90%   Cost/1k $6.40
Config C: Quality 94%   Cost/1k $18.40
```

---

# 7. Phase 2 — Reference Baselines (offline)

These come straight from the matrix, with **no new API calls**:

| Baseline | Definition | What it tells you |
|---|---|---|
| Always-frontier | every item → strongest config | quality reference |
| Always-cheapest | every item → cheapest config | the other extreme |
| Random mixtures | configs picked at fixed odds, ignoring the prompt | what you get with no routing intelligence |
| **Hindsight oracle** | per-item choices made after seeing every config's score | the most any router or cascade could save |

## Random mixtures

Mixing two configs at fixed odds traces the straight line between their cost–quality points. With three or more configs, the reference is the best fixed mix at each cost: the convex envelope of the fixed-config points, computed in linear cost (Amendment A8 item 7). Include router overhead when comparing a real system. A system below this envelope still routes requests; it just does worse than ignoring the prompt.

## Hindsight oracle

The oracle sees every config's score on every item and pays nothing to decide. It's a ceiling, not a deployable system.

- **Per target:** for a given quality target, it finds the cheapest per-item choices whose weighted average quality meets the target. Sweeping the target traces the oracle curve.
- **No imaginary answers:** if every config fails an item, the oracle takes the cheapest config and records its actual score.
- **Can beat frontier:** cheaper configs sometimes succeed where frontier fails, so the oracle may exceed frontier quality. Savings are therefore compared only **at the same quality target**.

**How to compute it.** This is a multiple-choice knapsack problem, relaxed to a linear program:

1. For each item, plot every config as a point with score (weighted by the item's task-mix weight) on the x-axis and cost on the y-axis. Drop dominated configs (cost more, score no higher), then keep the lower convex hull: drop any point that lies above the line between its neighbors.
2. Start every item at its cheapest hull point.
3. Each step up an item's hull is an upgrade with ratio `Δscore / Δcost`. Apply upgrades across all items in decreasing order of that ratio until aggregate quality reaches the target. The last upgrade may be fractional (a mixture of two configs), which is what makes this an LP.

**What it bounds.** When each escalation re-sends the original prompt, a cascade returns one stored answer but pays for every config it tried. The oracle could choose that same answer for that item at no greater cost. So the oracle frontier bounds **cascades as well as direct routers**, and "% of oracle savings" applies to cascade rows. A cascade that passes earlier answers or feedback forward makes new requests, and the bound doesn't cover it.

**Savings metrics:**

```text
oracle_savings(target) = frontier_cost − oracle_cost(target)
captured_fraction      = (frontier_cost − system_cost) / oracle_savings(target)
```

Report `captured_fraction` only when the system meets the target and `oracle_savings > 0`; otherwise report "n/a". Edge cases (weights, ties, targets out of range, the `captured_fraction ≤ 1` invariant) are fixed in Amendment A8 item 6.

Every router and cascade is then scored by lookup: for each item, read the cost and quality of the row for each config it used. Each policy hands the offline simulator one trace per item, and costs are at list price (Amendment A8 items 2 and 4).

---

# 8. Phase 3 — Routers

## Common interface

```python
class Router(Protocol):
    def score(self, request: Request) -> dict[str, float]:
        """For each config below the frontier, its router score: an estimate that the config succeeds."""
```

Shared decision rule: **pick the cheapest config on the ladder whose router score ≥ t, else the frontier.** Sweeping `t` traces a cost–quality curve. Routers are compared at matched cost or matched quality, not as single points. A request is the prompt plus its output schema, and each router's scores for a split are kept in a router-score table, from which its traces at any threshold are built (Amendment A9 items 1 and 3).

## Fairness rules

- Routers see only the request, the prompt and its output schema: no `task_type`, `expected_answer`, or metadata (Amendment A9 item 1).
- Router cost and latency are added to every request (for the rules router, $0 and a measured latency; Amendment A9 item 6).
- Thresholds are selected before Stage 8 and never changed on test.

## Selecting operating thresholds

For each headline quality target (§11), choose `t` where estimated quality **clears the target by a stated margin** — default: at least one standard error of the quality estimate, read as one paired standard error of the policy's quality difference from the frontier (Amendment A8 item 9). A threshold where dev quality only just meets the target would miss on test about half the time from noise alone. The same rule applies to gate cutoffs in cascades (§9).

- Select on dev; for the trained router, on its out-of-fold dev predictions (cross-fitting, Amendment A5). v4 also allowed cross-validation over train+dev; Amendment A1 drops it because the frontier config doesn't run on train, and every threshold replay needs frontier outcomes for the items it escalates. The larger 30% dev split offsets most of the lost precision.
- Record the margin, selection method, and chosen thresholds before Stage 8.
- The walk down from always-frontier over a router's breakpoints, the six operating points and the fallback to A: Amendment A9 items 9 and 10. What is recorded and frozen: item 13.

## 3a — Rules router

Do NOT start with Jev. First establish a dumb baseline. Features computed from the prompt:

- inferred task type (code fences, "classify", answer options, JSON schema present, …)
- prompt length / input tokens
- code present; structured output requested
- apparent reasoning complexity (multi-step cues, math notation)

```python
if looks_like_classification(prompt):
    return {"cheap": 0.9, "medium": 0.95, "frontier": 0.97}
elif contains_code(prompt):
    return {"cheap": 0.3, "medium": 0.6, "frontier": 0.9}
...
```

Rules give only a few operating points, not a smooth curve. That's fine for a dumb baseline.

The sketch's numbers are illustrative: each bucket's router scores are its train success rates, and the buckets and cues are fixed in Amendment A9 items 4 and 5.

## 3b — Trained router

Labels come from the predeclared success rule applied to the train matrix: for each item and config, did it succeed?

```text
prompt → embedding → one classifier per non-frontier config → P(config succeeds)
(frontier is the fallback, so it needs no classifier and no train labels; Amendment A1)
```

- Start with logistic regression on a sentence embedding. Cheap and hard to beat.
- **Cross-fitting (Amendment A5):** split dev into k folds (e.g., 5). Train on train plus k−1 folds, predict the held-out fold, and rotate until every dev item has a prediction from a router that never saw it. Thresholds are selected on these out-of-fold predictions. Test is untouched.
- Optionally add a RouteLLM router as a second trained baseline.

This is Jev's real competition: Jev has no training on your data, while this router learns your exact workload.

## 3c — Jev router

**About Jev** (checked 2026-10-12 against [TypeSafe's model docs](https://docs.typesafe.ai/models) and [OpenRouter's Jev page](https://openrouter.ai/blog/insights/what-is-jev/); re-check before the pilot):

- **Primary access — TypeSafe's API:** `POST /v1/systemone` with the pinned model ID `jev-1.13.0`. The aliases `jev-latest` and `jev-preview` currently point to the same model; don't use aliases in experiments.
- **Fallback — OpenRouter's Decisions API:** `https://openrouter.ai/api/alpha/decisions` (alpha), model `typesafe/jev-1.13`, listed with a 32K context. Use only if TypeSafe's API is unavailable, and log the snapshot ID each response returns.
- **Output:** structured decisions, not prose. Answer types: **Choice** (named options), **Score** (ordered rubric), **Noul** (probability a yes/no statement is true), each with probabilities. Multiple questions are evaluated in parallel in one call.
- **Limits:** 64K tokens per request for `state` plus all questions; 32K for `state` plus the longest question. Published rate limits are 100K tokens/s and 80 requests/s, subject to change.
- **Price:** $0.042 per million input tokens; output tokens free. Measure latency yourself; don't rely on published claims.

Two formulations to test:

```text
Variant 1 — Choice
  State:    the request
  Question: "What is the minimum model tier likely to answer this correctly?"
  Options:  cheap | medium | frontier
  → probability per option → cumulative P(tier sufficient)

Variant 2 — Noul per tier (one call, parallel questions)
  "A small, inexpensive model will answer this request correctly."   → P(yes)
  "A mid-sized model will answer this request correctly."            → P(yes)
  → frontier assumed sufficient
```

Variant 2 maps directly onto the shared decision rule. Requests over the 32K state limit need a documented truncation policy (e.g., keep the head, the tail, and the final instruction).

## 3d — Commercial router (optional)

Route dev prompts through one commercial router (e.g., OpenRouter Auto) and log its picks. Decide before Stage 8 whether it belongs in the final comparison; if so, run it on test once. It gives an external live reference point.

---

# 9. Phase 4 — Quality Gate and Cascades

```text
Router picks starting config → model answers → gate ──pass──→ return
                                                 |
                                               fail
                                                 v
                          next config up → gate → … → frontier (always accepted)
```

Cascades can recover from routing mistakes, **but only if the gate is good**. A gate that escalates too often costs more than always-frontier.

## The leakage rule

The gate receives only `(prompt, answer)` — never `expected_answer` or the benchmark's hidden tests. This is easy to violate by accident inside a harness that holds the answer key.

## Core gate options

| Gate | Extra cost | Works for | Caveat |
|---|---|---|---|
| Format / schema check | ~0 | structured output | catches only malformed answers |
| Execute code / self-written tests | low | coding | must not use the benchmark's hidden tests |
| Token log-probabilities | ~0 | providers that expose them | not universally available |
| **Jev Noul:** "This answer correctly and completely addresses the request." | very low | any | the experiment |
| Sonnet 5.5 verifier | medium-tier call; dev/test only (Amendment A1) | any | can eat the savings |

The Sonnet verifier is a system component whose accuracy is measured against ground truth like any other gate; it is not an evaluator. It is the comparison for the Jev gate, and the fallback general-purpose gate if Jev access fails. The self-consistency gate is extension X2.

Jev may fit better as a gate than as a router: judging a `(question, answer)` pair is a bounded yes/no decision, which is what Noul is built for.

## Offline cascades

Run each gate **once** on every train/dev `(item, config)` answer (the Sonnet verifier on dev only; Amendment A1) and store the verdict and probability. Cascades are then simulated offline:

```text
cost    = router + Σ (model cost + gate cost) for each config tried
quality = quality of the answer finally returned
```

This is valid only when each escalation sends the **original prompt** and gets an independent answer. Passing earlier answers, errors, or feedback forward makes a new request that needs fresh calls. Simulated latency is the sum of sequential router, model, and gate latencies; a small live replay of the chosen policies checks cost and end-to-end latency. Test gate verdicts are generated only in Stage 8.

## Gate metrics

- **False accept rate:** passed a wrong answer (costs quality)
- **False escalate rate:** rejected a correct answer (costs money)
- **Calibration plot:** gate probability vs. actual success

---

# 10. The Experiments

Development experiments run offline from the dev matrix and stored gate verdicts, except those marked *live*. In Stage 8, the frozen policies are applied to the newly scored test matrix. Test curves are descriptive only and never used to pick thresholds.

| ID | System | Output |
|---|---|---|
| A | Always-frontier | point |
| B | Always-cheapest | point |
| C | Random fixed-config mixtures | reference envelope |
| D | Hindsight oracle | upper-bound curve |
| E | Rules router | few points |
| F | Trained router | curve |
| G | Jev router (Choice, Noul variants) | curves |
| H | Commercial router *(live, optional)* | point |
| I | Best router + cascade, per core gate | curves |
| J | I + provider caching/batch *(stretch S1, modeled)* | curves |
| K | + semantic cache / context optimizer *(stretch S2–S3)* | curves |

Final results table (test split, realistic mix; repeat for the uniform mix and per task, with IFEval's and MBPP+'s marked indicative per Amendment A7; each policy at its pre-selected threshold):

| System | Quality (95% CI) | Cost / 1k | Savings vs A | % of oracle savings at matched quality | Estimated p50 latency |
|---|---:|---:|---:|---:|---:|
| A Frontier only | | | 0% | 0% | |
| B Cheapest only | | | | — | |
| D Oracle at target | | | | 100% | — |
| E Rules | | | | | |
| F Trained | | | | | |
| G Jev | | | | | |
| I Best cascade | | | | | |

Show measured live-replay latency separately from the offline estimate.

---

# 11. Reporting Quality Targets

"Quality ≥ 95% of baseline" sounds strict, but at a baseline of 93 it allows a 4.6-point drop, which most people would call material. Instead, pre-select several targets and report each frozen policy's result on test, relative and absolute:

```text
at 100% of frontier quality:  savings ?   (Δ 0.0 pts)
at  99% of frontier quality:  savings ?   (Δ ≈ −0.9 pts)
at  97% of frontier quality:  savings ?   (Δ ≈ −2.8 pts)
```

Thresholds come from §8, with the safety margin. If a policy misses its target on test, mark it **not met** and report its achieved quality and cost. Verdicts are given for the two mixes only; per-task results are shown without one (Amendment A7). Never sweep thresholds on test to claim savings at an exact target.

---

# 12. Main Visualizations

## Pareto chart — the intellectual center of the project

```text
Quality
  ^
  |            .-- hindsight oracle frontier
95|                        * Frontier
  |          /-- trained router curve
92|       /-- Jev curve
  |     /---------------- random-mixture envelope
89|   /
85| * Cheap
  +-----------------------------------> Cost / 1k (log)
```

- One curve per router and cascade, with CI bands (for routers, crosses at their operating points instead; Amendment A9 item 14)
- Oracle curve and frontier point as references; random-mixture envelope as the prompt-independent floor
- Chart v0's contents, its table and where its files go: Amendment A8 item 10
- Chart v1 and table v1, which add the rules router: Amendment A9 items 14 and 15

> For a given budget, what's the highest quality obtainable? For a required quality, what's the lowest cost?

## Supporting charts

- Per-task quality vs. cost (where routing helps and where it hurts)
- Gate calibration plots
- Routing mix at each chosen operating point

---

# 13. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| Model access | Anthropic API directly for all model tiers, with the Message Batches API for bulk runs (Amendment A1); Jev via TypeSafe's API (`POST /v1/systemone`) |
| Storage | SQLite: dev matrix, sealed test store, gate verdicts |
| Analysis | scripts with numpy + matplotlib (Amendment A8; was pandas + notebooks) |
| Trained router | sentence embeddings + scikit-learn |
| Stretch | FAISS or pgvector (S2); FastAPI proxy (S4); Streamlit or Next.js (S5) |

Start with 2–3 providers at most.

```text
data/benchmark.jsonl          frozen, hashed
configs/models.yaml           pinned model versions + dated price snapshot
harness/                      runner, evaluators
routers/                      request, ladder, decision rule; rules.py, trained.py, jev.py (Amendment A9)
gates/                        schema.py, code_exec.py, jev_gate.py, verifier.py
analysis/                     offline simulator, oracle LP, bootstrap, plots
figures/                      charts on dev, each with a JSON of its numbers (Amendment A8)
results/dev_matrix.sqlite
results/test_sealed.sqlite    no analysis-code access until Stage 8
```

---

# 14. Build Order

Run experiments at **every** stage on dev and update the Pareto chart each time. Stage 8 is the first time anything scores test answers or calls routers and gates on test. Freeze the benchmark, task weights, evaluators, policies, thresholds, margins, and analysis code before Stage 8.

### Core

| Stage | Work | Done when |
|---|---|---|
| 0 | **Pilot:** ~20 separate items per candidate task type through all configs | 3–4 objective-evaluator task types chosen; evaluators sanity-checked; model versions pinned; per-item cost, per-config latency, and `max_tokens` caps recorded; full-run budget checked against the $150 ceiling |
| 1 | Build benchmark + splits | source groups, scoring and success rules, and mix weights documented; dataset and splits frozen with hashes |
| 2 | Harness + collection run (Batches API) | every train/dev item × config scored and stored (frontier on dev only); raw test answers in the sealed store with hash recorded; evaluator parse failures reviewed on a hand-checked sample; noise subset re-run |
| 3 | Offline simulator + references A–D | Pareto chart v0 on dev with the oracle curve (LP) and random-mixture envelope |
| 4 | Rules router (E) | on chart |
| 5 | Trained router (F) | curve on chart |
| 6 | Jev router (G), both variants | curves on chart; measured Jev latency and cost logged |
| 7 | Gates + cascades (I) | gate verdicts stored (Jev, schema, code: train/dev; Sonnet verifier: dev); calibration plots; policies, thresholds, and margins selected and frozen |
| 8 | **Final evaluation** + write-up | sealed-store hash verified; test answers scored; router and gate calls on test run once; frozen policies evaluated with CIs under both mixes; small live replay checks cost and latency; README |

**The project is complete at Stage 8.** Every added feature is measured against the previous best system.

---

# 15. Extensions and Stretch Goals

## Extensions (after the core study)

**X1 — Judge-graded tasks.** Add summarization and open-ended tasks scored by an LLM judge with a fixed rubric. Use a judge from a different model family than the frontier tier where possible, to limit self-preference bias. Validate the judge against ~50–100 hand-graded items before trusting it. Judge cost is an experiment cost, not a system cost.

**X2 — Self-consistency gate.** Accept an answer when n independent samples agree. The temperature-0 matrix holds one answer per `(item, config)`, so this needs n extra samples per pair at temperature > 0 — roughly n× the matrix cost for each gated config. Suited to short answers.

## Stretch

**S1 — Provider-side levers.** Prompt caching (discounted repeated prompt prefixes) and batch APIs (discounted async work); check current pricing. Model them as multipliers on a documented workload profile (e.g., 60% of input tokens are a shared prefix; 30% of requests are batchable), then verify with a small live test of cached-token counts and costs.

*Caching interacts with routing.* Caches are scoped to a single model, and on Anthropic changing the top-level effort setting mid-conversation also invalidates the conversation cache. A router that spreads one conversation's turns, or requests that share a document, across configs therefore splits the cache, and an escalation pays full price for the prefix on the stronger model. The core cost model prices each call independently. That's fine for single-turn items with no shared prefix, but it overstates savings for multi-turn or shared-prefix traffic. When modeling S1 (and in S4), track each config's cache separately and charge escalations the cold-prefix cost. Make sure "frontier at low effort" is one of the configs (§6): it keeps all traffic in one cache and is the simple alternative any cascade has to beat.

**S2 — Semantic cache.** Reuse answers for sufficiently similar past requests. Benchmark items are distinct by design, so evaluate on a realistic trace with natural repetition (e.g., a public real-user chat log). Be conservative ("weather in Paris today" ≠ "…tomorrow"). Track hit rate, cost avoided, and wrong-hit rate.

**S3 — Context optimizer.** Trim long prompts: dedupe, retrieve relevant chunks, summarize old context, drop irrelevant documents. Needs a long-context task set (e.g., long-document QA). Rewriting context per request can break prefix caching and raise cost. Measure tokens removed, net savings after the optimizer's own cost and lost cache discounts, and quality change.

**S4 — Live proxy.** FastAPI, OpenAI-compatible endpoint running the best policy.

**S5 — Dashboard.** Show only measured numbers.

---

# 16. Risks and Open Questions

## Risks

| Risk | Mitigation |
|---|---|
| Jev API, pricing, or behavior changes | pin `jev-1.13.0` via TypeSafe; log the returned model ID |
| OpenRouter's Jev endpoint is alpha | TypeSafe's API is primary |
| Jev claims don't hold on this workload | measure everything ourselves |
| Jev token limits (64K request; 32K state + longest question) | enforce limits; documented truncation policy |
| Model versions or prices change before Stage 8 | pinned dated versions; one price snapshot; test answers collected with dev |
| A pinned model used by a router or gate is retired before Stage 8 | document the substitution and re-run the affected dev steps |
| Run-to-run nondeterminism | temp 0 where possible; repeated subset |
| Benchmark contamination / saturation | newer datasets; replace saturated sets in the pilot |
| Headline number driven by task mix | predeclared mixes; per-task reporting |
| Thresholds miss targets on test from noise | safety margin; larger 30% dev split (Amendment A1) |
| API budget overrun | Stage 0 pilot estimates cost first |
| Overfitting rules or thresholds | select on dev; report on sealed test |
| Offline cascades differ from live behavior | independent escalations; small live replay |

## Open questions (decide before Stage 0)

1. ~~Which models fill the cheap / medium / frontier tiers?~~ **Decided (A1):** Anthropic lineup, four configs.
2. ~~What's the total API budget?~~ **Decided (A1):** working ceiling of $150.
3. Is the deliverable a research write-up, a usable proxy, or both? (Decides whether S4 is core.)

---

# 17. What Makes This a Strong Project

Weak version:

> "Built a service that routes prompts to different LLMs."

Strong version:

> "Built and evaluated an adaptive multi-model inference system that minimizes API cost subject to a quality constraint. Compared rules-based, trained-classifier, and Jev routers plus confidence-gated cascades across multiple task types, against hindsight-oracle and random-mixture references."

With a defensible result:

> "Reduced inference cost X% vs. an always-frontier baseline at Y% of baseline quality (sealed test set, 95% CI), capturing Z% of hindsight-oracle savings at matched quality."

That shows AI engineering, experimental design, statistics, evaluation, and cost reasoning, and it holds up to an informed interviewer's questions.

---

# 18. The Principle

**Do not design the project around proving that Jev is good.** Design it around:

> What is the best way to make model-routing and escalation decisions?

Every component is a hypothesis:

- Jev might beat the trained router. It might not.
- It might be a mediocre router but an excellent gate.
- It might work well for classification and poorly for predicting hard reasoning tasks.
- Cascades might lose to plain routing once gate costs are counted.

Any of those results is interesting. The finished project answers a real question:

> **Which combination of routing, escalation, and cost levers gives the best cost–quality tradeoff for LLM inference — and how close does it get to the oracle?**
