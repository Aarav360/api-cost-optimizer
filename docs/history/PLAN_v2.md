# AI API Cost Optimizer — Plan v2

| | |
|---|---|
| Version | v2 |
| Date | 2026-10-09 |
| Supersedes | [PLAN.md](PLAN.md) (v1) |

## What changed from v1

- **Full results matrix + offline evaluation.** Every benchmark item is run once through every model config; every router and cascade is then scored by lookup instead of new API calls.
- **New baselines:** oracle router (upper bound on savings) and random router (lower bound on "is this routing at all").
- **New contender:** a trained router (classifier on our own matrix), so Jev is compared against something stronger than hand-written rules.
- **Quality gate is now a first-class design problem** with explicit options, metrics, and a leakage rule. Jev is tested as a *gate* as well as a router.
- **Fairness rules:** routers see only the prompt (no `task_type` label); train / dev / test splits; tuning only on dev.
- **Curves, not points.** Routers output probabilities; sweeping a threshold traces a cost–quality curve. Savings are reported at several quality levels instead of one "95% of baseline" cutoff.
- **Statistics:** paired bootstrap confidence intervals, per-task reporting, results under more than one task mix.
- **New cost levers:** reasoning effort as a routing dimension; provider prompt caching and batch APIs.
- **Re-scoped:** semantic cache, context optimizer, live proxy, and dashboard are now stretch goals, because the benchmark can't meaningfully measure the first two.
- **Added:** prior art, pilot run, risks and open questions, "done when" criteria per stage.

---

# 1. Goal

Build and evaluate a system that decides how each AI request should be processed in order to:

**Minimize API cost while keeping output quality close to always using the strongest available model.**

The central question:

> How much can we reduce LLM inference cost without materially reducing answer quality — and which decision strategy gets closest to the best possible tradeoff?

A successful result might look like (illustrative, not a target):

- Baseline quality: 92%
- Optimized quality: 91% (95% CI ±1.2)
- Cost reduction: 55%
- Captures 70% of the savings an oracle router could achieve

The real numbers are the research output.

---

# 2. Core Idea

Instead of sending every request to the most powerful model:

```text
Request
   |
   v
Frontier model (GPT / Claude / Gemini)
```

put a decision layer between the application and the models:

```text
                      User Request
                           |
                           v
                  AI Cost Optimizer
                           |
                        Router
              (model tier + effort level)
                   /       |       \
                Cheap    Medium    Frontier
                   \       |       /
                    v      v      v
                       Response
                           |
                           v
                     Quality Gate
                           |
                     Good enough?
                      /        \
                    Yes         No
                     |           |
                  Return      Escalate to
                              next tier
```

Cost levers, roughly in order of how much this project will focus on them:

1. **Route** each request to the cheapest model config likely to succeed.
2. **Escalate** when the cheap answer isn't good enough (cascade).
3. **Use provider-side discounts** — prompt caching and batch APIs.
4. **Avoid calls entirely** — semantic cache (stretch).
5. **Send fewer tokens** — context optimization (stretch).

A "model config" is a `(model, reasoning effort)` pair, not just a model. A medium model at high effort may beat a frontier model at low effort on cost per correct answer.

---

# 3. Prior Art

This problem has published work. Read it before building, use it for baselines, and cite it in the write-up.

From memory — confirm details before citing:

- **FrugalGPT** (Stanford, 2023) — LLM cascades: try cheap models first, escalate on low confidence. Essentially the cascade in Phase 4.
- **RouteLLM** (LMSYS, 2024) — open-source trained routers (e.g., matrix factorization, BERT classifier) between a strong and a weak model. Usable directly as a trained-router baseline.
- **RouterBench** (Martian, 2024) — benchmark and dataset for evaluating routers.
- **Commercial routers** — OpenRouter's Auto Router, Not Diamond, Martian. Any of these can be an extra baseline.

What this project adds: a head-to-head comparison of rules vs. trained vs. Jev routers, Jev as a quality gate, the effort-level dimension, and honest accounting of router and gate costs.

---

# 4. Metrics

## Cost

For every model call, record:

- provider, model, effort level
- input tokens, cached input tokens, output tokens, reasoning tokens
- cost (computed from a dated price snapshot; also the provider-reported cost where available)
- latency

Every system's cost includes **all** calls it makes:

```text
system_cost = router calls + model calls (all tiers tried) + gate calls
```

A cascade that tries cheap → medium → frontier pays for all three. A Jev router adds its own cost and latency to every request.

Headline cost metrics:

```text
cost per 1,000 requests
cost per correct answer
savings vs. always-frontier at a given quality level
% of oracle savings captured
```

## Quality

No single metric works for every task, so each task type gets an evaluator:

| Task | Quality metric |
|---|---|
| Classification | Accuracy / F1 |
| Multiple choice | Correct answer |
| Math / short-answer reasoning | Exact match against reference |
| Information extraction | Exact match / F1 |
| Coding | Hidden unit tests passed |
| Structured output | Schema valid + field-level correctness |
| Summarization | LLM judge with rubric |
| Open-ended response | LLM judge with rubric |

LLM judge rules:

- Use a judge from a **different model family** than the frontier tier where possible, to limit self-preference bias.
- Validate the judge against ~50–100 hand-graded items before trusting it.
- Judge cost is an experiment cost, not a system cost.

## Statistics

Every system runs on the same items, so comparisons are **paired**:

- Report 95% confidence intervals via paired bootstrap over items.
- For binary-scored tasks, McNemar's test for "is system X different from frontier."
- A 2-point quality gap on a few hundred items is usually noise. Don't claim it without a CI.

---

# 5. Benchmark Design

The benchmark decides the headline number. Savings depend heavily on the mix of easy and hard items, so this deserves care.

## Item format

```python
{
    "item_id": 152,
    "source": "gsm8k",
    "task_type": "math",          # for reporting only — routers never see this
    "split": "dev",               # train | dev | test
    "prompt": "...",
    "expected_answer": "...",     # evaluator only — routers and gates never see this
    "evaluation_method": "exact_match"
}
```

## Sources

Prefer existing datasets over hand-authoring. Candidates:

| Task | Candidates |
|---|---|
| Classification | intent / sentiment sets (e.g., Banking77) |
| Multiple choice | MMLU-Pro, GPQA |
| Math | GSM8K, MATH |
| Coding | HumanEval+ / MBPP+, LiveCodeBench (less contaminated) |
| Extraction | SQuAD-style QA |
| Structured output | custom JSON-schema tasks |
| Summarization | news summarization sets + judge |
| Open-ended | Arena-Hard / MT-Bench-style prompts + judge |
| Long-context (for stretch S3) | long-document QA sets |

Watch for:

- **Contamination** — popular benchmarks are likely in training data. Prefer newer sets where possible.
- **Ceiling effects** — if every tier scores 98%, the task tells you nothing about routing. Drop or replace saturated sets.

## Splits

```text
train  50%   fit the trained router
dev    20%   tune rules, thresholds, gate cutoffs
test   30%   locked; touched once for final numbers
```

Splits are stratified by task type and frozen (with a file hash) before any routing work begins.

## Size

- Enough items that overall quality CIs are around ±1–2 points. A pilot will tell you the variance.
- At least a few hundred test items per task type if you want per-task claims.

## Task mix

Report results under at least two mixes, chosen and documented **before** looking at results:

1. **Uniform** — each task type weighted equally.
2. **Realistic** — a weighting that resembles a plausible production workload.

Re-weighting happens offline, so this is free.

---

# 6. Phase 1 — Evaluation Harness and the Results Matrix

Build this before any optimizer.

```text
benchmark item × model config
            |
            v
     call model (temp 0 where supported)
            |
            v
     capture answer
            |
            v
     evaluate answer
            |
            v
     store one row
```

The output is the project's central artifact: **the results matrix** — every item × every model config.

```python
{
    "item_id": 152,
    "model": "model-small",
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
    "price_snapshot": "2026-10-09"
}
```

Notes:

- Start with **3 configs** (cheap / medium / frontier). Add effort variants once the pipeline works. Keep the total to about 5–6.
- **Nondeterminism:** use temperature 0 where the API allows it (some reasoning models don't). Re-run a ~10% subset 3× to measure run-to-run noise.
- **Prices change:** store a dated price snapshot in config and compute cost from it, so results stay reproducible.

Output of this phase:

```text
Config A: Quality 84%   Cost/1k $2.10
Config B: Quality 90%   Cost/1k $6.40
Config C: Quality 94%   Cost/1k $18.40
```

---

# 7. Phase 2 — Reference Baselines (offline)

With the matrix in place, these need **no new API calls**:

| Baseline | Definition | What it tells you |
|---|---|---|
| Always-frontier | every item → strongest config | quality reference |
| Always-cheapest | every item → cheapest config | the other extreme |
| Random | random config per item, at a chosen mix | anything below this line isn't routing |
| **Oracle** | each item → cheapest config that got it right | maximum possible routing savings |

Random routing at varying mixes traces a straight line between the cheap and frontier points. A real router has to sit **above** that line.

The oracle often **exceeds** frontier quality, because cheap models sometimes get right what frontier gets wrong. The gap between oracle and frontier is the size of the prize.

Every later router is scored by lookup: for each item, find the row for the config the router chose and read its cost and quality.

---

# 8. Phase 3 — Routers

## Common interface

Every router gets the same input and produces the same output shape:

```python
class Router(Protocol):
    def score(self, prompt: str) -> dict[str, float]:
        """For each config, estimated P(this config answers correctly)."""
```

Decision rule, shared by all routers:

```text
pick the cheapest config whose score ≥ t
```

Sweeping `t` from 0 to 1 traces a **cost–quality curve** for each router. Routers are compared on these curves at matched cost or matched quality, not as single points.

**Fairness rules:**

- Routers see only `prompt`. No `task_type`, no `expected_answer`, no metadata.
- Thresholds and rules are tuned on `dev`. Final numbers come from `test`.
- Router cost and latency are added to every request.

## 3a — Rules router

Do NOT start with Jev. First establish a dumb baseline.

Features computed from the prompt itself:

- inferred task type (keyword/regex heuristics: code fences, "classify", answer options, JSON schema present, …)
- prompt length / input tokens
- whether code is present
- whether structured output is requested
- apparent reasoning complexity (multi-step cues, math notation)

```python
if looks_like_classification(prompt):
    return {"cheap": 0.9, "medium": 0.95, "frontier": 0.97}
elif contains_code(prompt):
    return {"cheap": 0.3, "medium": 0.6, "frontier": 0.9}
...
```

Rules produce only a few distinct operating points, not a smooth curve. That's fine. It's the dumb baseline.

## 3b — Trained router

Labels come straight from the matrix: for each `train` item and each config, did it answer correctly?

```text
prompt → embedding → one classifier per config → P(config correct)
```

- Start with logistic regression on a sentence embedding. Cheap and hard to beat.
- Optionally drop in a RouteLLM router as a second trained baseline.

This is Jev's real competition. Jev decides with no training on your data; this router learns your exact workload.

## 3c — Jev router

**About Jev** (as of Oct 2026, from launch coverage; vendor claims are not independently verified):

- TypeSafe AI's first model, launched Sept 15, 2026; early access.
- Returns structured decisions, not prose. Three answer types: **Choice** (pick among named options), **Score** (ordered rubric), **Noul** (probability that a yes/no statement is true). Answers come with probabilities.
- Evaluates multiple declared questions in parallel within one call.
- Available via TypeSafe's API, OpenRouter, and Vercel AI Gateway (Jev 1.13).
- Vendor-claimed: 70–500 ms latency; $0.042 per million input tokens; output tokens free.
- Context window: OpenRouter lists 32K; another listing says 64K. **Check current docs.**

Two formulations to test:

```text
Variant 1 — Choice
  State:    the request
  Question: "What is the minimum model tier likely to answer this correctly?"
  Options:  cheap | medium | frontier
  → probability per option; convert to cumulative P(tier sufficient)

Variant 2 — Noul per tier (one call, parallel questions)
  "A small, inexpensive model will answer this request correctly."   → P(yes)
  "A mid-sized model will answer this request correctly."            → P(yes)
  → frontier assumed sufficient
```

Variant 2 maps directly onto the shared decision rule.

Practicalities:

- Requests longer than Jev's context window need a truncation policy (e.g., keep the head and tail plus the final instruction). Decide and document it.
- Measure Jev's actual latency and cost yourself. Don't use the vendor numbers.

## 3d — Commercial router (optional)

Route the benchmark through one commercial router (e.g., OpenRouter Auto) and log what it picked. This gives an external reference point. It's live calls, not offline, so run it on `test` only.

---

# 9. Phase 4 — Quality Gate and Cascades

Routers will make mistakes. A cascade lets the system recover:

```text
Router picks starting config
        |
        v
  model answers
        |
        v
   quality gate ──pass──→ return
        |
      fail
        |
        v
   next config up ──→ gate ──→ … ──→ frontier (always accepted)
```

This is potentially more powerful than perfect up-front prediction, **but only if the gate is good**. A gate that escalates too often makes the system *more* expensive than always-frontier.

## The leakage rule

The gate never sees `expected_answer` or the benchmark's hidden tests. The gate interface only receives `(prompt, answer)`. This is easy to violate by accident inside a harness that has the answer key right there.

## Gate options

| Gate | Extra cost | Works for | Caveat |
|---|---|---|---|
| Format / schema check | ~0 | structured output | only catches malformed answers |
| Execute code / self-written tests | low | coding | must not use the benchmark's hidden tests |
| Token log-probabilities | ~0 | providers that expose them | not universally available |
| Self-consistency (n samples agree) | n× cheap call | short answers | multiplies cheap-tier cost |
| **Jev Noul:** "This answer correctly and completely addresses the request." | very low (vendor-claimed) | any | the experiment |
| LLM judge (medium model) | medium-tier call | any | can eat the savings |

Jev may fit better as a gate than as a router: judging a `(question, answer)` pair is a bounded yes/no decision, which is what Noul is built for.

## Offline cascades

Run each gate **once** on every `(item, config)` answer already in the matrix and store the verdict (and probability). Then any cascade — any router, any gate, any threshold — can be simulated offline:

```text
cost    = router + Σ (model cost + gate cost) for each config tried
quality = quality of the answer finally returned
```

## Gate metrics

- **False accept rate:** gate passed a wrong answer (costs quality)
- **False escalate rate:** gate rejected a correct answer (costs money)
- **Calibration plot:** gate probability vs. actual correctness

---

# 10. Stretch S1 — Provider-Side Cost Levers

Low-risk savings that need no routing intelligence:

- **Prompt caching.** Major providers discount input tokens for repeated prompt prefixes (system prompts, shared documents). Discounts and minimums vary by provider. Check current pricing.
- **Batch APIs.** Discounted asynchronous processing for latency-tolerant work. Check current pricing.

These depend on the workload, not the benchmark. Model them as multipliers on a documented workload profile:

```text
workload profile:
  60% of input tokens are a shared prefix
  30% of requests are latency-tolerant (batchable)
```

Then verify with a small live test (e.g., actually enable caching on a repeated system prompt and confirm the cached-token counts and costs).

---

# 11. Stretch S2 — Semantic Cache

```text
New request → search previous requests
                 |
                 +-- sufficiently similar → reuse result
                 +-- not similar → continue
```

**Why it's a stretch:** benchmark items are distinct by design, so the hit rate will be ~0%, or it will just equal however many paraphrases you planted. Cache value depends entirely on real traffic repetition.

If you build it:

- Evaluate on a realistic trace with natural repetition (e.g., a public real-user chat log dataset), not the benchmark.
- Be conservative. Similar questions don't always have the same answer ("weather in Paris today", "…tomorrow").
- Track hit rate, cost avoided, and **wrong-hit rate** (a reused answer that's wrong for the new request).

Stack: embeddings + FAISS, or Postgres + pgvector.

---

# 12. Stretch S3 — Context Optimizer

```text
17,000-token prompt → context optimizer → 8,000-token relevant context → LLM
```

Techniques: remove duplicate context, retrieve relevant chunks, summarize older context, drop irrelevant documents, trim conversation history.

**Why it's a stretch:** most benchmark prompts are short. This needs the long-context subset.

Watch the interaction with prompt caching: rewriting context per request breaks prefix caching and can **raise** cost. Test it.

Measure: tokens removed, cost saved (net of the optimizer's own cost and any lost cache discount), quality change.

---

# 13. The Experiments

All experiments except those marked *live* run offline from the matrix and stored gate verdicts.

| ID | System | Output |
|---|---|---|
| A | Always-frontier | point |
| B | Always-cheapest | point |
| C | Random routing | line |
| D | Oracle | point |
| E | Rules router | few points |
| F | Trained router | curve |
| G | Jev router (Choice, Noul variants) | curves |
| H | Commercial router *(live, optional)* | point |
| I | Best router + cascade, per gate type | curves |
| J | I + provider caching/batch (modeled) | curves |
| K | + semantic cache / context optimizer *(stretch)* | curves |

Final results table (test split, realistic mix — repeat for uniform mix and per task):

| System | Quality (95% CI) | Cost / 1k | Savings vs A | % of oracle savings | p50 latency |
|---|---:|---:|---:|---:|---:|
| A Frontier only | | | 0% | 0% | |
| B Cheapest only | | | | — | |
| D Oracle | | | | 100% | |
| E Rules | | | | | |
| F Trained | | | | | |
| G Jev | | | | | |
| I Best cascade | | | | | |

For curve systems, report the operating point at each quality level in §15.

---

# 14. The Main Visualizations

## Pareto chart (the intellectual center of the project)

```text
Quality
  ^
  |              ★ Oracle
95|                          ● Frontier
  |           ╱── Trained router curve
92|        ╱── Jev curve
  |      ╱─────────────────── random line
89|    ╱
  |  ╱
85| ● Cheap
  |
  +--------------------------------------> Cost / 1k (log)
```

- One curve per router; shaded CI bands
- Oracle and frontier as reference points; random as the floor line

> For a given budget, what is the highest quality obtainable?
> For a required quality, what is the lowest achievable cost?

## Supporting charts

- Per-task quality vs. cost (where does routing help, where does it hurt?)
- Gate calibration plots
- Routing mix at the chosen operating point (% to each config)

---

# 15. Reporting the Quality Constraint

"Quality ≥ 95% of baseline" sounds strict, but at a baseline of 93 it allows a 4.6-point drop, which most people would call material.

Instead of one threshold, report savings at several levels, both relative and absolute:

```text
at 100% of frontier quality:  savings ?   (Δ 0.0 pts)
at  99% of frontier quality:  savings ?   (Δ ≈ −0.9 pts)
at  97% of frontier quality:  savings ?   (Δ ≈ −2.8 pts)
```

Readers pick the tradeoff they care about.

---

# 16. Tech Stack

Keep it simple.

| Layer | Choice |
|---|---|
| Language | Python |
| Model access | OpenRouter for one API across tiers + Jev; direct provider APIs where needed (prompt caching, batch, log-probabilities) |
| Storage | SQLite (results matrix, gate verdicts) |
| Analysis | pandas + notebooks; matplotlib for the Pareto chart |
| Trained router | sentence embeddings + scikit-learn |
| Semantic cache *(stretch)* | embeddings + FAISS, or Postgres + pgvector |
| Live proxy *(stretch)* | FastAPI, OpenAI-compatible endpoint |
| Dashboard *(stretch)* | Streamlit (speed) or Next.js (full-stack demo) |

Start with 2–3 providers at most.

Suggested layout:

```text
data/benchmark.jsonl          frozen, hashed
configs/models.yaml           configs + dated price snapshot
harness/                      runner, evaluators, judge
routers/                      rules.py, trained.py, jev.py
gates/                        schema.py, jev_gate.py, judge_gate.py, …
analysis/                     offline simulator, bootstrap, plots
results/matrix.sqlite
```

---

# 17. Build Order

Run experiments at **every** stage, on `dev`, and update the Pareto chart each time. Only Stage 8 touches `test`.

### Core

| Stage | Work | Done when |
|---|---|---|
| 0 | **Pilot:** ~20 items per task type through all configs | evaluators sanity-checked; per-item cost known; full-run budget estimated |
| 1 | Build benchmark + splits | dataset frozen with hash; mix weights documented |
| 2 | Harness + full matrix run | every item × config stored; judge validated against hand labels; noise subset re-run |
| 3 | Offline simulator + baselines A–D | Pareto chart v0 with oracle and random line |
| 4 | Rules router (E) | on chart |
| 5 | Trained router (F) | curve on chart |
| 6 | Jev router (G), both variants | curves on chart; measured Jev latency/cost logged |
| 7 | Gates + cascades (I) | gate verdicts stored for every matrix row; calibration plots; best cascade per router |
| 8 | **Final evaluation on test** + write-up | results table with CIs under both mixes; README |

**The project is complete at Stage 8.**

### Stretch

| Stage | Work |
|---|---|
| S1 | Provider caching / batch levers (J) |
| S2 | Semantic cache on a realistic trace |
| S3 | Context optimizer on the long-context subset |
| S4 | Live FastAPI proxy running the best config |
| S5 | Dashboard (show only measured numbers) |

Every added feature is measured against the previous best system.

---

# 18. Risks and Open Questions

## Risks

| Risk | Mitigation |
|---|---|
| Jev is early access; API, pricing, or behavior may change | pin the version (e.g., 1.13); log it per call |
| Jev vendor claims don't hold on this workload | measure everything ourselves |
| Jev context window unclear (32K vs 64K) | check docs; truncation policy |
| Model prices change mid-project | dated price snapshot |
| Run-to-run nondeterminism | temp 0 where possible; repeat subset to measure noise |
| LLM judge bias | different-family judge; validate against hand labels |
| Benchmark contamination / saturation | newer datasets; drop saturated tasks |
| Headline number driven by task mix | pre-registered mixes; per-task reporting |
| API budget overrun | Stage 0 pilot estimates cost before the full run |
| Overfitting rules/thresholds | tune on dev, report on locked test |

## Open questions (decide before Stage 0)

1. Which models fill the cheap / medium / frontier tiers? Same provider family, or mixed?
2. What's the total API budget?
3. Is the end deliverable a research write-up, a usable proxy, or both? (This decides whether S4 is core.)

---

# 19. What Makes This a Strong Project

The weak version:

> "Built a service that routes prompts to different LLMs."

The strong version:

> "Built and evaluated an adaptive multi-model inference system that minimizes API cost subject to a quality constraint. Compared rules-based, trained-classifier, and Jev routers plus confidence-gated cascades across eight task types, benchmarked against oracle and random baselines."

With a concrete, defensible result:

> "Reduced inference cost X% vs. an always-frontier baseline at Y% of baseline quality (held-out test set, 95% CI), capturing Z% of oracle-achievable savings."

That shows AI engineering, experimental design, statistics, evaluation, backend work, and cost reasoning, and it holds up to an informed interviewer's questions.

---

# 20. The Principle

**Do not design the project around proving that Jev is good.**

Design it around:

> What is the best way to make model-routing and escalation decisions?

Every component is a hypothesis:

- Jev might beat the trained router. It might not.
- It might be a mediocre router but an excellent gate.
- It might work well for classification and poorly for predicting hard reasoning tasks.
- Cascades might lose to plain routing once gate costs are counted.

Any of those results is interesting. The finished project answers a real question:

> **Which combination of routing, escalation, and cost levers gives the best cost–quality tradeoff for LLM inference — and how close does it get to the oracle?**
