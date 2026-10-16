# AI API Cost Optimizer — Plan v4

| | |
|---|---|
| Version | v4 |
| Status | Final — frozen for Stage 0 |
| Date | 2026-10-12 |
| Amended | 2026-10-14: Amendment A1, implementation decisions (below); 2026-10-15: Amendment A2, Haiku effort; 2026-10-16: Amendments A3, pilot decisions, and A4, larger pools tested |
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

**The pilot.** 20 items from each of the six candidate datasets (`data/pilot.jsonl`) ran through all four configs on the standard API with a 32,000-token output cap. Banking77, MMLU-Pro, MATH and CRUXEval ran twice, in free text and as structured outputs (item 2), so 800 calls in all. A supplement of 20 MATH Level 5 items (`data/pilot_math_level5.jsonl`) ran as structured outputs only. Together they cost $4.72. No call errored, was refused, or hit the output cap, and every response reported its pinned model ID. Rows with answers are in `results/pilot/` (not committed). Every evaluator was first checked against the reference answers of the full candidate pools (`harness/check_evaluators.py`), and the pilot's format failures and every item that all four configs got wrong were read by hand.

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
- McNemar's test for binary-scored tasks ("is system X different from frontier?").
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
train  40%   fit the trained router (no frontier runs; Amendment A1)
dev    30%   tune rules, thresholds, gate cutoffs
test   30%   sealed until Stage 8
```

- Stratify by task type; keep each source group in one split; freeze splits with a file hash before routing work begins.
- Pilot items are separate from all three splits.
- The frontier config runs only on dev and test, so thresholds are selected on dev (§8, Amendment A1).
- Raw test answers are collected with the dev matrix but sealed (§6). Before Stage 8, nothing reads, scores, or inspects test answers, grades, or gate verdicts.

## Size

- Enough items for overall quality CIs around ±1–2 points; the pilot gives the variance.
- A few hundred test items per task type before making per-task claims.

## Task mix

Report under two mixes, documented before Stage 8:

1. **Uniform** — each task type weighted equally.
2. **Realistic** — a weighting that resembles a plausible production workload.

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
- **Nondeterminism:** the Claude 5.5 models reject `temperature`, so it isn't set. Re-run ~10% of dev 3× to measure run-to-run noise.

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

Mixing two configs at fixed odds traces the straight line between their cost–quality points. With three or more configs, the reference is the best fixed mix at each cost: the convex envelope of the fixed-config points. Include router overhead when comparing a real system. A system below this envelope still routes requests; it just does worse than ignoring the prompt.

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

Report `captured_fraction` only when the system meets the target and `oracle_savings > 0`; otherwise report "n/a".

Every router and cascade is then scored by lookup: for each item, read the cost and quality of the row for each config it used.

---

# 8. Phase 3 — Routers

## Common interface

```python
class Router(Protocol):
    def score(self, prompt: str) -> dict[str, float]:
        """For each config, estimated P(this config succeeds)."""
```

Shared decision rule: **pick the cheapest config whose score ≥ t.** Sweeping `t` traces a cost–quality curve. Routers are compared at matched cost or matched quality, not as single points.

## Fairness rules

- Routers see only `prompt` — no `task_type`, `expected_answer`, or metadata.
- Router cost and latency are added to every request.
- Thresholds are selected before Stage 8 and never changed on test.

## Selecting operating thresholds

For each headline quality target (§11), choose `t` where estimated quality **clears the target by a stated margin** — default: at least one standard error of the quality estimate. A threshold where dev quality only just meets the target would miss on test about half the time from noise alone. The same rule applies to gate cutoffs in cascades (§9).

- Select on dev. v4 also allowed cross-validation over train+dev; Amendment A1 drops it because the frontier config doesn't run on train, and every threshold replay needs frontier outcomes for the items it escalates. The larger 30% dev split offsets most of the lost precision.
- Record the margin, selection method, and chosen thresholds before Stage 8.

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

## 3b — Trained router

Labels come from the predeclared success rule applied to the train matrix: for each item and config, did it succeed?

```text
prompt → embedding → one classifier per non-frontier config → P(config succeeds)
(frontier is the fallback, so it needs no classifier and no train labels; Amendment A1)
```

- Start with logistic regression on a sentence embedding. Cheap and hard to beat.
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

Final results table (test split, realistic mix; repeat for the uniform mix and per task; each policy at its pre-selected threshold):

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

Thresholds come from §8, with the safety margin. If a policy misses its target on test, mark it **not met** and report its achieved quality and cost. Never sweep thresholds on test to claim savings at an exact target.

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

- One curve per router and cascade, with CI bands
- Oracle curve and frontier point as references; random-mixture envelope as the prompt-independent floor

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
| Analysis | pandas + notebooks; matplotlib |
| Trained router | sentence embeddings + scikit-learn |
| Stretch | FAISS or pgvector (S2); FastAPI proxy (S4); Streamlit or Next.js (S5) |

Start with 2–3 providers at most.

```text
data/benchmark.jsonl          frozen, hashed
configs/models.yaml           pinned model versions + dated price snapshot
harness/                      runner, evaluators
routers/                      rules.py, trained.py, jev.py
gates/                        schema.py, code_exec.py, jev_gate.py, verifier.py
analysis/                     offline simulator, oracle LP, bootstrap, plots
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
