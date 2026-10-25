# API Cost Optimizer

A study of how much LLM inference cost can be cut without materially reducing answer quality, by
deciding per request which Claude model config answers it. Decisions behind these terms are the
amendments in [PLAN_v4.md](PLAN_v4.md).

## Items and data

**Item**:
One benchmark prompt with its expected answer, belonging to one task and one split.
_Avoid_: question, example, sample

**Request**:
What every config receives for an item: its prompt and, if it has one, its output schema. All a
router sees. An API request is a request plus one config's settings.
_Avoid_: input, query

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
A policy that reads only the request and calls exactly one config per item.

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

## Routers

**Router score**:
A router's estimate, for one request and one config below the frontier, that the config would
succeed.
_Avoid_: score (alone), confidence, probability

**Threshold**:
The value router scores are compared with. A request goes to the cheapest config on the ladder
whose router score is at least the threshold, or to the frontier if none is.
_Avoid_: cutoff, quality threshold

**Breakpoint**:
A threshold at which some item's config changes: a router score for a request that is higher than
every cheaper config's router score for it. A router's walk steps are taken from its breakpoints.
_Avoid_: cut point (the boundary of a rules router's bucket)

**Router-score table**:
A router's router scores and overhead for every item of a split, from which its traces at any
threshold are built.
_Avoid_: predictions

## The trained router

**Template line**:
A non-blank line that recurs verbatim in at least 2% of train prompts, such as a label list or a
fixed instruction. The trained router removes template lines from a request before embedding it.
_Avoid_: boilerplate, prefix

**Cross-fitting**:
Training the trained router on train plus every dev fold but one, scoring the held-out fold, and
rotating, so every dev item's router scores come from a model that never saw its label.
_Avoid_: cross-validation (which chooses settings within train)

**Fold model**:
One of the trained router's five models under cross-fitting: fitted on train and every dev fold but
one, it gives the held-out fold's router scores. On test, a hash of the request picks the fold model
that scores it.

**Out-of-fold router score**:
A router score for a dev item from the fold model that never saw its label. The trained router's dev
router-score table holds only these.
_Avoid_: out-of-fold prediction, cross-validated score

**F-train**:
The trained router refit on train alone with F's settings. A diagnostic that compares embeddings
with E's cues on the same labels; it never gets test operating points.
_Avoid_: train-only F

**Task-rates diagnostic**:
Each config's success rate on an item's task, from the trained router's labels, used as router
scores and walked like a router. It reads the task, so it is a diagnostic, never a router.
_Avoid_: task router, task-only router

**Within-task concordance**:
The share of (success, failure) pairs from the same task, and for cross-fitted router scores the
same fold, whose router scores are in the right order, ties counting half. 0.5 means the scores
carry nothing beyond the task.
_Avoid_: AUC (which pools pairs across tasks)

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
_Avoid_: quality threshold (thresholds belong to routers, cutoffs to gates)

**Safety margin**:
How far a policy's dev quality must clear a quality target when its threshold is chosen: one
standard error of its paired quality difference from the frontier.
_Avoid_: buffer, cushion

**Captured fraction**:
The share of the hindsight oracle's savings at a quality target that a policy meeting that target
achieves.
_Avoid_: % of oracle savings, efficiency

**Diagnostic**:
A number computed to explain a result, declared before the freeze, that never selects a setting or
threshold and never becomes a policy with test operating points. It may read what a router can't,
such as the task.
_Avoid_: baseline (for F-train or the task-rates diagnostic)

## Selecting thresholds

**Walk**:
The fixed sequence of steps in which a router's thresholds are tried for one task mix and quality
target, from always-frontier down, each step a lower threshold. It stops at the first step whose
dev quality doesn't clear the quality target by the safety margin. Every step before that is
certified. A cascade's order of steps is fixed when cascades are designed.

**Operating point**:
A policy at the thresholds selected for one task mix and quality target: the cheapest step its
walk certifies, or always-frontier where the walk certifies nothing cheaper.
_Avoid_: setting, configuration
