# API Cost Optimizer

A study of how much LLM inference cost can be cut without materially reducing answer quality, by
deciding per request which Claude model config answers it. Decisions behind these terms are the
amendments in [PLAN_v4.md](PLAN_v4.md).

## Items and data

**Item**:
One benchmark prompt with its expected answer, belonging to one task and one split.
_Avoid_: question, example, sample

**Task**:
One of the four core benchmark sources (IFEval, Banking77, MMLU-Pro, MBPP+), each with its own
success rule.
_Avoid_: dataset (when meaning the task), category

**Task mix**:
A predeclared weighting of the tasks (uniform or realistic) under which quality and cost are
combined.
_Avoid_: workload, blend

**Results matrix**:
Every item's stored answer, cost and success for every config, from one collection run.
_Avoid_: results table, dataset

**Flip rate**:
The share of (item, config) pairs in the noise subset whose success differs across the three
runs.
_Avoid_: noise rate, variance

## Configs

**Config**:
A (model, effort) pair: `haiku`, `sonnet`, `opus_low` or `opus`.
_Avoid_: model (when the effort matters), tier

**Frontier**:
The `opus` config: the quality reference, and the config every cascade ends on.
_Avoid_: strongest model, best model, baseline

**Config ladder**:
The fixed order of configs from cheapest to dearest (`haiku`, `sonnet`, `opus_low`, `opus`) that
"cheapest config" and "one config up" refer to.
_Avoid_: tier, tier list

## Policies

**Policy**:
A deployable rule that decides, for each item, which configs to call and which answer to return.
Fixed configs, routers and cascades are policies.
_Avoid_: system, strategy, decision strategy

**Router**:
A policy that reads only the prompt and calls exactly one config per item.

**Cascade**:
A policy that calls configs up the config ladder, returning the first answer its gate accepts, or
the frontier's.
_Avoid_: escalation chain

**Gate**:
A check that sees only the prompt and an answer and decides whether to accept the answer.
_Avoid_: verifier (except the Sonnet verifier gate), judge

**Trace**:
The record of what a policy did on one item: the configs it called, in order, and its overhead.
The answer returned is always the last config's.
_Avoid_: decision, route

**Overhead**:
What a policy pays per item beyond model answers: its router and gate calls.
_Avoid_: router cost (when gates are included)

## References and metrics

**Reference**:
A fixed point or curve that policies are measured against: always-frontier, always-cheapest, the
random-mixture envelope and the hindsight oracle.
_Avoid_: baseline (for the oracle and the envelope)

**Random-mixture envelope**:
The best quality reachable at each cost by picking configs at fixed odds without reading the
prompt.
_Avoid_: random baseline, random router

**Hindsight oracle**:
The cheapest per-item choice of configs that meets a quality target, chosen after seeing every
config's success on every item. A ceiling, not a policy.
_Avoid_: optimal router, perfect router

**Quality target**:
A required quality, stated as a share of frontier quality under a task mix (100%, 99%, 97%).
_Avoid_: quality threshold (thresholds belong to routers and gates)

**Safety margin**:
How far a policy's dev quality must clear a quality target when its threshold is chosen: one
standard error of its paired quality difference from the frontier.
_Avoid_: buffer, cushion

**Captured fraction**:
The share of the hindsight oracle's savings at a quality target that a policy meeting that target
achieves.
_Avoid_: % of oracle savings, efficiency
