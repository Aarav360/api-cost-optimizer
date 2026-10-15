# AI API Cost Optimizer

## 1. Goal

Build a system that dynamically chooses how an AI request should be processed in order to:

**Minimize API cost while keeping output quality close to the quality of always using the strongest available model.**

The central question is:

> How much can we reduce LLM inference cost without materially reducing answer quality?

A successful result might look like:

- Baseline quality: 92%
- Optimized quality: 90–92%
- Cost reduction: 50–80%
- Lower or comparable latency

The exact numbers are not predetermined; discovering the actual tradeoff is part of the project.

---

# 2. Core Idea

Instead of sending every request to the most powerful model:

```text
Request
   |
   v
GPT / Claude / Gemini frontier model
```

create an intelligent layer between the application and the models:

```text
                      User Request
                           |
                           v
                  AI Cost Optimizer
                           |
          ┌────────────────┼─────────────────┐
          |                |                 |
       Cache            Router         Prompt Optimizer
          |                |
          |           Choose model
          |        /       |       \
          |     Cheap    Medium    Frontier
          |        \       |       /
          |         \      |      /
          └──────────→ Response
                         |
                         v
                  Quality / confidence
                       check
                         |
                   Good enough?
                    /       \
                  Yes        No
                   |          |
                Return     Escalate
                           to stronger
                              model
```

The system therefore optimizes several things:

1. Avoid unnecessary API calls.
2. Avoid unnecessarily expensive models.
3. Reduce unnecessary tokens.
4. Escalate difficult requests when cheaper models are insufficient.

---

# 3. The Two Main Metrics

## Cost

This is relatively straightforward.

For every request, record:

- model
- provider
- input tokens
- output tokens
- input-token cost
- output-token cost
- total request cost
- latency

Then calculate:

```text
total_cost
average_cost_per_request
cost_per_successful_request
```

The most useful metric will probably be:

```text
Cost per request while maintaining target quality
```

---

## Quality

Quality cannot be measured with one universal metric.

Instead, create several categories of tasks and use an appropriate evaluator for each.

### Example benchmark

| Task | Quality metric |
|---|---|
| Classification | Accuracy / F1 |
| Multiple choice | Correct answer |
| Information extraction | Exact match / F1 |
| Coding | Unit tests passed |
| Structured output | Schema + factual correctness |
| Short-answer reasoning | Reference answer / judge |
| Summarization | LLM judge or human rubric |
| Open-ended response | Rubric-based LLM judge |

Every request in your benchmark should therefore contain something like:

```python
{
    "prompt": "...",
    "task_type": "classification",
    "expected_answer": "...",
    "evaluation_method": "exact_match"
}
```

This becomes the foundation for testing the optimizer.

---

# 4. Establish the Baseline First

Before building optimization logic, run the entire benchmark through your strongest model.

For example:

```text
Baseline = always use strongest model
```

Measure:

```text
Quality: 93.2%
Cost: $18.40
Latency: 2.4 sec/request
```

This is the system your optimizer must beat on cost while remaining near its quality.

You can then define a constraint such as:

```text
Quality must remain >= 95% of baseline quality.
```

So if baseline quality is 93:

```text
minimum acceptable quality ≈ 88.4
```

You can make that threshold stricter later.

---

# 5. Phase 1 — Build the Evaluation Harness

Do this before building the optimizer.

Build a Python program that can:

```text
benchmark request
       |
       v
call selected model
       |
       v
capture answer
       |
       v
evaluate answer
       |
       v
record:
quality
cost
latency
tokens
```

Store each experiment in something simple initially:

```text
SQLite / Postgres
```

Example record:

```python
{
    "request_id": 152,
    "task": "coding",
    "model": "model-small",
    "input_tokens": 850,
    "output_tokens": 220,
    "cost": 0.0031,
    "latency": 0.82,
    "quality": 0.91
}
```

At the end, your harness should produce results such as:

```text
Model A:
Quality: 84%
Cost: $2.10

Model B:
Quality: 90%
Cost: $6.40

Model C:
Quality: 94%
Cost: $18.40
```

---

# 6. Phase 2 — Build a Simple Rules-Based Router

Do NOT start with Jev.

First establish a dumb baseline.

For example:

```python
if task_type == "classification":
    use_small_model()

elif task_type == "coding":
    use_large_model()

elif token_count > 10000:
    use_medium_model()
```

You could use features such as:

- task type
- prompt length
- number of input tokens
- whether code is present
- whether structured output is requested
- apparent reasoning complexity

Then evaluate:

```text
Always frontier
vs.
Rules-based routing
```

This establishes whether routing itself helps.

---

# 7. Phase 3 — Add Jev as an Intelligent Router

Now introduce Jev.

Jev is designed to accept unstructured state and produce constrained decisions such as choices, scores, and probabilities. TypeSafe specifically describes routing and branching as target use cases.

Give Jev the incoming request and ask it to select:

```text
CHEAP
MEDIUM
FRONTIER
```

Conceptually:

```text
Request:
"Debug this distributed transaction implementation..."

Question:
What minimum model tier is likely to successfully solve this request?

Choices:
- cheap
- medium
- frontier
```

Jev returns a structured decision and confidence.

Then:

```text
Jev
 |
 +-- cheap ------> Small model
 |
 +-- medium -----> Mid model
 |
 +-- frontier ---> Strong model
```

Now compare:

```text
Rules Router
vs.
Jev Router
```

This is one of the project's main experiments.

---

# 8. Phase 4 — Add Escalation

Routing will inevitably make mistakes.

Instead of requiring the router to be perfect, allow the system to recover.

Example:

```text
Jev predicts CHEAP
       |
       v
Cheap model answers
       |
       v
Answer confidence/evaluation
       |
       +-- acceptable → return
       |
       +-- uncertain → medium model
                          |
                          +-- still uncertain → frontier
```

This creates a **cascade**.

That is potentially much more powerful than trying to perfectly predict the required model beforehand.

You can test:

```text
Jev routing alone
vs.
Jev routing + escalation
```

---

# 9. Phase 5 — Semantic Cache

Before making any model call:

```text
New Request
     |
     v
Search previous requests
     |
     +-- sufficiently similar → reuse result
     |
     +-- not similar → continue
```

Start with embeddings and a vector database.

Possible stack:

```text
Embedding model
      ↓
FAISS / pgvector
```

You need to be conservative here because two semantically similar questions do not always have the same answer.

Track:

```text
cache hit rate
cost avoided
quality of cached responses
```

---

# 10. Phase 6 — Prompt / Context Optimizer

Long prompts can be unnecessarily expensive.

Add a preprocessing layer that identifies unnecessary context.

Example:

```text
17,000-token prompt
       |
       v
context optimizer
       |
       v
8,000-token relevant context
       |
       v
LLM
```

Possible techniques:

- remove duplicate context
- retrieve only relevant chunks
- summarize older context
- eliminate irrelevant documents
- trim conversation history

Measure:

```text
tokens removed
cost saved
quality change
```

Do not assume compression is beneficial. Test it experimentally.

---

# 11. Phase 7 — Build the Full Optimizer

Eventually:

```text
                         REQUEST
                            |
                            v
                     Semantic Cache
                      /          \
                    HIT          MISS
                     |             |
                  return           v
                            Context Optimizer
                                   |
                                   v
                              JEV Router
                            /      |      \
                         cheap   medium   frontier
                           |       |         |
                           v       v         v
                         Model   Model      Model
                            \      |       /
                             \     |      /
                              v    v     v
                            Quality Gate
                               |
                    ┌──────────┴──────────┐
                    |                     |
                 sufficient           insufficient
                    |                     |
                  return               escalate
                                            |
                                      stronger model
```

---

# 12. The Important Experiments

Your project should compare multiple systems rather than just present one architecture.

## Experiment A — Baseline

```text
Always strongest model
```

This gives maximum-quality reference performance.

---

## Experiment B — Cheapest model

```text
Always cheapest model
```

Shows the opposite extreme.

---

## Experiment C — Rules router

```text
Hand-written routing rules
```

---

## Experiment D — Jev router

```text
Jev chooses the model
```

---

## Experiment E — Jev + escalation

```text
Jev chooses initial model
+
weak answers escalate
```

---

## Experiment F — Full optimizer

```text
Cache
+
context optimization
+
Jev routing
+
escalation
```

Then create a table:

| System | Quality | Cost | Savings | Latency |
|---|---:|---:|---:|---:|
| Frontier only | 94.1 | $100 | 0% | 2.8s |
| Cheapest only | 78.3 | $12 | 88% | 0.6s |
| Rules router | ? | ? | ? | ? |
| Jev router | ? | ? | ? | ? |
| Jev + escalation | ? | ? | ? | ? |
| Full optimizer | ? | ? | ? | ? |

The `?` values are your actual research results.

---

# 13. The Most Important Visualization

Plot:

```text
Quality
  ^
  |
95|                ● Frontier
  |
92|          ● Optimizer
  |
89|     ●
  |
85| ● Cheap
  |
  +----------------------------> Cost
```

You are looking for the **Pareto frontier**:

> For a given amount of money, what is the highest quality obtainable?

Or equivalently:

> For a required quality level, what is the lowest achievable cost?

This is really the intellectual center of the project.

---

# 14. Dashboard

Once the underlying system works, build a simple dashboard.

Show:

### Overall

```text
Requests:        12,430
Baseline cost:   $182
Actual cost:     $71
Savings:         61%
Quality change:  -1.4%
```

### Model routing

```text
Small model:      52%
Medium model:     31%
Frontier model:   17%
```

### Optimization savings

```text
Routing:             $63 saved
Caching:             $28 saved
Prompt reduction:    $20 saved
```

### Task performance

```text
Classification   96%
Coding           87%
Extraction       94%
Reasoning        89%
```

That gives you a very demo-able product.

---

# 15. Suggested Tech Stack

Keep this relatively simple.

### Backend

```text
Python
FastAPI
```

### Database

Initially:

```text
SQLite
```

Then optionally:

```text
Postgres
```

### Semantic cache

```text
embeddings + FAISS
```

or:

```text
Postgres + pgvector
```

### Providers

Start with only 2–3 model tiers.

For example:

```text
cheap
medium
frontier
```

Do not begin by integrating ten providers.

### Router

```text
Rules
+
Jev
```

### Dashboard

Either:

```text
Streamlit
```

for speed,

or:

```text
React / Next.js
```

if you want the project to demonstrate full-stack ability.

---

# 16. Recommended Build Order

I would build it in this exact order:

### Stage 1
Create benchmark dataset.

### Stage 2
Build evaluation harness.

### Stage 3
Run multiple models and establish cost/quality curves.

### Stage 4
Implement rules-based routing.

### Stage 5
Implement Jev routing.

### Stage 6
Add escalation.

### Stage 7
Add semantic caching.

### Stage 8
Add context/prompt optimization.

### Stage 9
Run complete experiments.

### Stage 10
Build dashboard.

### Stage 11
Write results and README.

This ordering matters because every additional feature should be measured against the previous system.

---

# 17. What Makes This a Strong Project

The weak version is:

> “Built a service that routes prompts to different LLMs.”

A much stronger version is:

> “Built and evaluated an adaptive multi-model inference router that minimizes API cost subject to a quality constraint. Compared rules-based, Jev-based, and cascaded routing strategies across multiple task classes, incorporating semantic caching, context reduction, and confidence-based escalation.”

Even better if your experiment produces a concrete result:

> “Reduced inference cost 63% relative to an always-frontier baseline while retaining 98% of baseline benchmark quality.”

That becomes a very strong resume bullet because it shows:

- AI engineering
- algorithms
- experimental design
- API/backend development
- evaluation
- model routing
- cost optimization
- data analysis
- systems thinking

---

# 18. One Important Principle

**Do not design the project around proving that Jev is good.**

Design it around:

> What is the best way to make model-routing decisions?

Jev is one hypothesis.

It might beat your rules.

It might not.

It might work exceptionally well for classification but poorly for predicting difficult reasoning tasks.

That result itself is interesting.

TypeSafe currently describes Jev as an early-access model designed around structured probabilistic decisions rather than free-form generation, so treating it experimentally is especially appropriate rather than assuming its vendor claims will transfer directly to your workload.

The finished project should therefore answer a real question:

> **Which combination of routing, caching, context reduction, and escalation gives the best cost-quality tradeoff for LLM inference?**