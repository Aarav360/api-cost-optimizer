# Write Up

<!-- skeleton: the declared parts of the write-up, committed before the go (PLAN_v4.md, Amendment A19, item 11). The
headings below are the declared parts, in their order, and so are the sentences that carry a number: `analysis/check_writeup.py`
holds each number against the committed tables and holds the headings against this file's skeleton commit. After the
run nothing declared is removed or reordered, and sections may be added. Text in {braces} is filled in after the run:
an agent fills in the numbers, and the discussion is the researcher's. The dated notes this draws on are in NOTES.md. -->

## Summary

### The headline

On the realistic mix at the 99% target, {policy}, the best cascade by A17 item 24's rule, saves {savings} of A's cost
[{low}, {high}] at a quality of {quality}, {margin} [{low}, {high}] from the target's share of A's quality on the same
items (target {met | not met | falls back to A}), and meets that target in {share} of test's resamples.

The headline is a rule, declared before test was read, not a policy: the best cascade by A17 item 24's rule, against A,
on the realistic mix at the 99% target. On dev that is haiku/Jev, and on dev that point saved 37.6% of A's cost and met
its target in 89.65% of dev's resamples. If a refusal path re-selects the cascades, the headline follows the rule and no
new choice is made. The uniform mix and the 97% target are reported beside it, never in its place. If it misses on test,
the headline says not met, with its quality and cost, and moves to no other target or policy. A policy that does better
on test is reported as a finding, under the heading below that says it was found after test was seen.

The realistic mix is two-thirds MBPP+ (65%), the smallest test split, so the headline leans on it and its interval is
wider than the uniform mix's (A7 item 2).

### The answers to the declared questions

One line for each question under Results, in that order, each in the form {answer}, with the six cells' differences in
cost and in quality and their intervals in Results, not here.

1. haiku/Jev against G: {answer}
2. haiku/Jev against G+: {answer}
3. The other cascades against the routers: {answer}
4. What the gate adds to a router: {answer}
5. What the router adds to a cascade: {answer}
6. What a paid gate adds over the free checks: {answer}
7. The Jev gate against the Sonnet gate: {answer}

The replication summary: {answer}

## Design

A is always-frontier, the reference every cost and quality is measured against. The policies are the routers E (rules),
F (trained), G (Jev) and G+ (bolstered Jev), and the cascades that end on the frontier, each started at `haiku` or at a
router's pick and gated by the free checks, the Jev gate or the Sonnet gate (PLAN_v4.md §8 and Amendment A17). Each
policy runs at the operating points its walk selected on dev, for the uniform and the realistic task mix and for quality
targets of 100%, 99% and 97% of A's quality; the frozen operating points are in `routers/operating_points.json`. The
final evaluation runs test once, in the order Amendment A19 item 2 fixes. {anything else the study's design needs said}

## Expected misses

A quality target is a bar, and the safety margin keeps a policy's dev quality one standard error above it, so a policy
that cleared its target on dev can still miss it on test. Several misses are expected, and some are the margin working
as designed, not a failure of the policy.

- 72 operating points route: 54 of the cascades' 90 and 18 of the routers' 24 (E 4, F 3, G 6, G+ 5). The rest fall back
  to A, which meets every target trivially and is never counted as met.
- Over the 65 that tables v3 and v4 cover (the cascades', G's and G+'s), dev's own resamples give about 3 expected
  misses. Allowing for test's own sampling noise gives about 6, since a point that cleared by exactly the margin misses
  15.9% of dev's resamples but about 24% of fresh draws of test's items.
- Choosing on dev pushes the count higher still, and E's and F's 7 points aren't counted.
- The misses will come in clusters, since points share items and some are the same step (G/Jev's 99% and 97%, for one).

## Results

### The headline in full

{the headline's point on test under both mixes and all three targets: quality, cost per 1k, savings against A, whether
the target is met, and the share of resamples meeting it, beside dev's}

### Misses against the expectation

{how many of the operating points that route missed their target on test, against the expectation above, and which share
a step}

### The declared questions

The same-target comparisons are 51 in all (A17 item 22). They are descriptive, with no multiplicity correction: about
one interval in twenty excludes 0 by chance. Each is read in the six mix-and-target cells, for the difference in cost per
1k and in quality, with its paired interval. The write-up leads with these questions in this order, not with whichever
intervals came out significant.

#### haiku/Jev against G

Comparisons: 1. {the six cells' differences, with intervals}

#### haiku/Jev against G+

Comparisons: 1. {the six cells' differences, with intervals}

#### The other cascades against the routers

Comparisons: 10. {the six cells' differences, with intervals}

#### What the gate adds to a router

Comparisons: 12. {the six cells' differences, with intervals}

#### What the router adds to a cascade

Comparisons: 12. {the six cells' differences, with intervals}

#### What a paid gate adds over the free checks

Comparisons: 10. {the six cells' differences, with intervals}

#### The Jev gate against the Sonnet gate

Comparisons: 5. {the six cells' differences, with intervals}

### The replication summary

For each dev comparison whose interval excluded 0, whether test's interval has the same sign and also excludes 0. Cost
and quality are counted apart, and a comparison counts once for each mix-and-target cell where its dev interval
excluded 0.

| | Dev intervals excluding 0 | Same sign on test | Same sign, and excluding 0 on test |
|---|---|---|---|
| Difference in cost per 1k | {n} | {n} | {n} |
| Difference in quality | {n} | {n} | {n} |

### Every policy at its operating points

Every frozen policy at its dev-selected operating points, under both mixes and per task: {table}. IFEval's and MBPP+'s
per-task results carry their warning wherever they appear, and the write-up draws no per-task conclusion from them (A7
item 2). IFEval: Only 156 test items, below the plan's minimum of a few hundred: indicative. MBPP+: Only 107 test items,
below the plan's minimum of a few hundred: indicative.

### The hindsight oracle and the random-mixture envelope

{the oracle and the envelope on test, and the captured fraction at each operating point}, built from the test matrix as
on dev (A8 item 4). Test has no repeat runs, so dev's flip rates carry over, and the oracle curve is an optimistic upper
bound (A6 item 2).

### The live replay

{the live replay's answers, graded with the same graders and shown beside the offline estimate with their n}. They are
shown, not judged (A17 item 26.5), and its first call and any reconnects are reported apart from the rest.

## Found after test was seen

Anything outside the headline, the 51 comparisons and the replication summary appears only under this heading, and each
entry says it was found after test was seen: a per-task pattern, a new metric, a ranking by test savings, a policy that
does better on test than the headline's. {none yet}

## Discussion

{the researcher's discussion}

## Limitations

### Selection on dev

Every policy's thresholds and cutoffs, the operating points and the headline's cascade itself (the best of 15 by A17
item 24's rule) were chosen on dev. Test's numbers for them are fresh; dev's are the record of how they were chosen, not
an estimate of how they perform. Choosing on dev also pushes the expected misses higher than the count above.

### IFEval and MBPP+ are only indicative

Test holds 156 IFEval items and 107 MBPP+ items, below the plan's minimum of a few hundred per task (§5), so their
per-task results are indicative and no per-task conclusion rests on them. Headline claims are made on the two mixes.

### One run of test

Test was run once: each item has one answer from each config, with no repeat runs, and a crash or a refusal stops the
final evaluation. Each answer is one draw, so a lucky or unlucky draw looks the same as a real strength or weakness
(A6 item 2).

### The oracle is an optimistic upper bound

The hindsight oracle sends each item to the cheapest config that succeeded on its one run, so a lucky draw counts as
savings no deployable policy could rely on. Its curve is an optimistic upper bound, as A6 item 2 says.

## Future work

### A benchmark where the configs are further apart

On this benchmark `haiku` sits close to the frontier: on dev it scores 83.3% against the frontier's 86.7% on the
realistic mix and 84.5% against 89.5% on the uniform mix, which leaves a router little to gain. Stage 0 had already tried
harder single-turn tasks and found them saturated or not separating the configs: MATH at its hardest level,
LiveCodeBench, CRUXEval and IFBench (A3 and A4). The next study would look further afield: expert-level problems (GPQA
Diamond, Humanity's Last Exam), agentic coding (SWE-bench Verified, Terminal-Bench), tool use (τ-bench) and human
preference (LMArena, AlpacaEval), each at a cost per item the design can afford, since every config runs on every item.

The gap has to be uneven and predictable, not just big. A router helps only when some requests are fine on `haiku`,
others need the frontier, and the prompt reveals which; if the frontier were better on every request by the same margin,
no router could beat random mixing. So the study would choose for realism, not for where routers win. It would reuse this
pipeline unchanged, since router-score tables, the walk, the envelope, the oracle and the freeze work on any table of
item by config outcomes; it would run a small pilot under a rule fixed in advance for what counts as separating the
configs, and freeze the new benchmark under its own amendment before any router sees it.
