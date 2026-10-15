# API Cost Optimizer

How much can you cut LLM inference cost without materially reducing answer quality, and which
decision strategy gets closest to the best possible tradeoff?

This project measures that on Anthropic's Claude 5.5 models. It compares routers (which pick a
model per request) and cascades (which try a cheap model first and escalate when a quality gate
fails) against two references: a hindsight oracle that shows the most any strategy could save,
and random mixtures that show what you get with no routing intelligence at all.

**Status:** Stage 0 (pilot) in progress. There are no results yet.

## Approach

- **One results matrix, then offline evaluation.** Every benchmark item runs once through every
  model config. Routers and cascades are then scored by lookup, with no new API calls, so every
  strategy is compared on exactly the same answers.
- **References, not just baselines.** Besides always-frontier and always-cheapest, the matrix
  gives a hindsight oracle (a multiple-choice knapsack relaxed to a linear program) and the
  envelope of random mixtures. Systems report the fraction of the oracle's savings they capture
  at the same quality target.
- **Routers, simplest first:** a rules baseline, then a classifier trained on prompt embeddings,
  then Jev (TypeSafe's decision model). Routers are compared at matched cost or matched quality,
  not as single points.
- **Gates and cascades:** schema and code-execution checks, Jev as a gate, and a Sonnet verifier.
- **A sealed test split.** Items are split 40 / 30 / 30 into train / dev / test. Test answers are
  collected in the same run as dev but aren't scored until the final stage. Every operating
  threshold is chosen on dev and must clear its quality target by a stated margin (by default,
  one standard error).

The full design, including every decision and its reason, is in [PLAN_v4.md](PLAN_v4.md).

## Model configs

A config is a (model, effort) pair, pinned in [configs/models.yaml](configs/models.yaml) with one
shared price snapshot (2026-10-12, USD per 1M input / output tokens).

| Config | Model | Effort | Price |
|---|---|---|---|
| `haiku` | `claude-haiku-5-5` | high | $0.10 / $0.50 |
| `sonnet` | `claude-sonnet-5-5` | high | $2 / $10 |
| `opus_low` | `claude-opus-5-5` | low | $4 / $20 |
| `opus` | `claude-opus-5-5` | medium (frontier) | $4 / $20 |

Bulk runs use the Message Batches API at half price. The working budget is $150.

## Pilot candidate datasets

The pilot runs 20 items from each candidate through every config, then 3–4 task types are locked
in, dropping any that every model aces.

| Dataset | Task type | Scoring | License |
|---|---|---|---|
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) ([copy used](https://huggingface.co/datasets/mteb/banking77)) | classification | exact label | CC BY 4.0 |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | multiple choice | answer letter | MIT |
| [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | math | answer equivalence | MIT |
| [CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) | code reasoning (output prediction) | Python literal | MIT |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | instruction following | strict instruction checks | Apache 2.0 |
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | coding | run the tests | Apache 2.0 |

Dataset revisions are pinned in [harness/loaders.py](harness/loaders.py), so an item ID always
means the same item.

## Running it

Requires [uv](https://docs.astral.sh/uv/), Docker (for the code sandbox), and an Anthropic API
key in a `.env` file at the project root (`ANTHROPIC_API_KEY=...`; it's gitignored).

```bash
uv sync
uv run python -m harness.smoke_test        # one Haiku call: checks the key and the response store
uv run python -m harness.sandbox --build   # build the pinned sandbox image
uv run python -m harness.check_sandbox     # escape checks + every MBPP+ reference solution
uv run python -m harness.build_pilot       # regenerate data/pilot.jsonl (same seed, same items)
```

Every API response is cached on disk under `results/` (not committed), keyed by a hash of the
full request, so re-running analysis never pays for the same call twice.

Model-written code runs in a fresh Docker container per execution: no network, a read-only file
system, none of your files, an unprivileged user, and caps on memory, processes, time and output.

## Layout

```text
PLAN_v4.md           study design (frozen for Stage 0, with amendments A1 and A2)
configs/models.yaml  pinned model configs and price snapshot
harness/             API client, response store, dataset loaders, code sandbox, pilot builder
sandbox/             Dockerfile for the code sandbox image
data/pilot.jsonl     the 120 Stage 0 pilot items (20 per candidate dataset)
docs/history/        earlier versions of the plan
```

## Progress

- [x] Study plan frozen (v4, amendments A1 and A2)
- [x] Harness: API client, on-disk response store, pinned model configs
- [x] Dataset loaders at pinned revisions, and 120 pilot items
- [x] Docker code sandbox (18/18 escape checks; 376/378 MBPP+ reference solutions pass)
- [ ] Evaluators for the six candidate datasets
- [ ] Pilot run: lock 3–4 task types, set output caps, check the budget
- [ ] Stages 1–8: benchmark and splits, collection run, references, routers, gates and cascades,
      final evaluation on the sealed test split
