# Write Up

<!-- skeleton: the declared parts of the write-up, committed before the go (PLAN_v4.md, Amendment A19, item 11). The
headings below are the declared parts, in their order, and so are the sentences that carry a number: `analysis/check_writeup.py`
holds each number against the committed tables and holds the headings against this file's skeleton commit. After the
run nothing declared is removed or reordered, and sections may be added. Text in {braces} is filled in after the run:
an agent fills in the numbers, and the discussion is the researcher's. The dated notes this draws on are in NOTES.md. -->

## Summary

### The headline

On the realistic mix at the 99% target, haiku/Jev, the best cascade by A17 item 24's rule, saves 47.3% of A's cost, at $3.96 per 1k [$3.43, $4.57] against A's $7.51, at a quality of 90.6% [86.8%, 94.1%], +0.64 points [-1.14, +2.35] from the target's share of A's quality on the same items (target met), and meets that target in 77.8% of test's resamples.

The headline is a rule, declared before test was read, not a policy: the best cascade by A17 item 24's rule, against A,
on the realistic mix at the 99% target. On dev that is haiku/Jev, and on dev that point saved 37.6% of A's cost and met
its target in 89.65% of dev's resamples. If a refusal path re-selects the cascades, the headline follows the rule and no
new choice is made. The uniform mix and the 97% target are reported beside it, never in its place. If it misses on test,
the headline says not met, with its quality and cost, and moves to no other target or policy. A policy that does better
on test is reported as a finding, under the heading below that says it was found after test was seen.

The realistic mix is two-thirds MBPP+ (65%), the smallest test split, so the headline leans on it and its interval is
wider than the uniform mix's (A7 item 2).

### The answers to the declared questions

One line for each question under Results, in that order, each a plain reading of its cells' counts, with the six cells' differences in
cost and in quality and their intervals in Results, not here.

1. haiku/Jev against G: over 6 cells, 0 with both points at A and so no difference; in the other 6, the first policy's cost per 1k was lower in 4 (2 with an interval excluding 0), higher in 2 (2) and equal in 0, and its quality was lower in 3 (0 with an interval excluding 0), higher in 3 (0) and equal in 0.
2. haiku/Jev against G+: over 6 cells, 0 with both points at A and so no difference; in the other 6, the first policy's cost per 1k was lower in 4 (1 with an interval excluding 0), higher in 2 (2) and equal in 0, and its quality was lower in 3 (0 with an interval excluding 0), higher in 2 (0) and equal in 1.
3. The other cascades against the routers: over 60 cells, 12 with both points at A and so no difference; in the other 48, the first policy's cost per 1k was lower in 9 (7 with an interval excluding 0), higher in 39 (36) and equal in 0, and its quality was lower in 15 (5 with an interval excluding 0), higher in 29 (8) and equal in 4.
4. What the gate adds to a router: over 72 cells, 16 with both points at A and so no difference; in the other 56, the first policy's cost per 1k was lower in 7 (5 with an interval excluding 0), higher in 46 (29) and equal in 3, and its quality was lower in 8 (0 with an interval excluding 0), higher in 45 (7) and equal in 3.
5. What the router adds to a cascade: over 72 cells, 16 with both points at A and so no difference; in the other 56, the first policy's cost per 1k was lower in 31 (22 with an interval excluding 0), higher in 25 (14) and equal in 0, and its quality was lower in 16 (3 with an interval excluding 0), higher in 31 (15) and equal in 9.
6. What a paid gate adds over the free checks: over 60 cells, 14 with both points at A and so no difference; in the other 46, the first policy's cost per 1k was lower in 13 (9 with an interval excluding 0), higher in 33 (29) and equal in 0, and its quality was lower in 18 (3 with an interval excluding 0), higher in 23 (6) and equal in 5.
7. The Jev gate against the Sonnet gate: over 30 cells, 4 with both points at A and so no difference; in the other 26, the first policy's cost per 1k was lower in 23 (21 with an interval excluding 0), higher in 3 (0) and equal in 0, and its quality was lower in 20 (4 with an interval excluding 0), higher in 1 (0) and equal in 5.

The replication summary: for cost per 1k, 184 of the 185 dev intervals that excluded 0 have the same sign on test and 174 of them also exclude 0 there; for quality, 3 of 3 and 3 of 3.

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

The headline is haiku/Jev on the realistic mix at the 99% target, against A. On test A scores 90.9% [86.8%, 94.4%] at $7.51 [$6.99, $8.05] per 1k on the realistic mix and 91.6% [89.4%, 93.7%] at $11.76 [$10.91, $12.66] per 1k on the uniform mix. Test's six points for haiku/Jev follow, each with its quality and cost per 1k (95% intervals), the share of A's cost saved, and the share of test's resamples that meet the target (A19 item 6).

| Mix | Target | Reading | Quality | Cost per 1k | Saved against A | Meets the target in |
| --- | --- | --- | --- | --- | --- | --- |
| realistic | 100% | met | 90.9% [86.8%, 94.4%] | $7.43 [$6.85, $8.02] | 1.1% | 100.0% |
| realistic | 99% | met | 90.6% [86.8%, 94.1%] | $3.96 [$3.43, $4.57] | 47.3% | 77.8% |
| realistic | 97% | met | 89.4% [85.5%, 93.1%] | $2.05 [$1.63, $2.51] | 72.7% | 88.4% |
| uniform | 100% | met | 91.6% [89.4%, 93.7%] | $11.80 [$10.90, $12.74] | -0.4% | 100.0% |
| uniform | 99% | not met | 90.2% [87.7%, 92.4%] | $5.84 [$4.93, $6.79] | 50.3% | 20.2% |
| uniform | 97% | not met | 88.6% [86.0%, 90.9%] | $3.54 [$2.78, $4.35] | 69.9% | 35.0% |

Dev's points for haiku/Jev, beside them, with the share of dev's resamples that meet the target.

| Mix | Target | Quality | Cost per 1k | Saved against A | Meets the target in |
| --- | --- | --- | --- | --- | --- |
| realistic | 100% | 86.7% [81.8%, 91.1%] | $7.95 [$7.37, $8.63] | 1.2% | 100.0% |
| realistic | 99% | 87.2% [82.4%, 91.6%] | $5.02 [$4.41, $5.74] | 37.6% | 89.6% |
| realistic | 97% | 85.8% [81.0%, 90.3%] | $2.57 [$2.12, $3.07] | 68.0% | 89.2% |
| uniform | 100% | 89.5% [87.2%, 91.9%] | $11.96 [$11.07, $12.95] | 0.5% | 100.0% |
| uniform | 99% | 89.8% [87.5%, 92.1%] | $6.34 [$5.49, $7.26] | 47.2% | 95.2% |
| uniform | 97% | 88.1% [85.4%, 90.6%] | $4.02 [$3.26, $4.87] | 66.5% | 90.6% |

### Misses against the expectation

15 of the 72 operating points that route missed their target on test, and 57 met it; the other 42 of the 114 points fall back to A, which is counted as neither.

The expectation above was about 3 misses over the 65 points the tables v3 and v4 cover (the cascades', G's and G+'s), and about 6 allowing for test's own sampling. On test 14 of those 65 missed. E's and F's 7 points that route, which the expectation leaves out, had 1 miss.

8 of the 15 misses sit on a step that another missed point of the same policy also uses, 4 steps in all; the other 7 have a step of their own. Each missed point follows, with test's quality less the target's share of A's quality (in points), the share of test's resamples that meet the target, and the missed points on its step.

| Policy | Mix | Target | Quality less the target | Meets the target in | Same step as |
| --- | --- | --- | --- | --- | --- |
| E Rules | realistic | 97% | 0.00 | 49.2% | none |
| G Jev | realistic | 100% | -0.63 | 12.2% | uniform 100% |
| G Jev | uniform | 100% | -0.34 | 12.2% | realistic 100% |
| G Jev | uniform | 99% | -0.19 | 36.0% | none |
| G+ Bolstered Jev | realistic | 99% | -0.14 | 42.6% | uniform 99% |
| G+ Bolstered Jev | uniform | 100% | -0.18 | 30.9% | none |
| G+ Bolstered Jev | uniform | 99% | -0.43 | 21.8% | realistic 99% |
| haiku/Jev | uniform | 99% | -0.55 | 20.2% | none |
| haiku/Jev | uniform | 97% | -0.31 | 35.0% | none |
| G/free | realistic | 100% | -0.02 | 37.1% | uniform 100% |
| G/free | uniform | 100% | -0.10 | 37.1% | realistic 100% |
| G/Jev | uniform | 100% | -0.10 | 37.1% | none |
| G/Sonnet | realistic | 100% | -0.02 | 37.1% | uniform 100% |
| G/Sonnet | uniform | 100% | -0.10 | 37.1% | realistic 100% |
| G+/free | uniform | 99% | -0.20 | 36.2% | none |

### The declared questions

The same-target comparisons are 51 in all (A17 item 22). They are descriptive, with no multiplicity correction: about
one interval in twenty excludes 0 by chance. Each is read in the six mix-and-target cells, for the difference in cost per
1k and in quality, with its paired interval. The write-up leads with these questions in this order, not with whichever
intervals came out significant. In each table a dagger (†) marks a point where the first policy fell back to A, a double dagger (‡) one where the second did, `both at A` a cell where both did, which holds no difference, and a star (*) an interval that excludes 0. A cell where one policy fell back compares the other with A.

#### haiku/Jev against G

Comparisons: 1. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 6 cells, 0 with both points at A and so no difference; in the other 6, the first policy's cost per 1k was lower in 4 (2 with an interval excluding 0), higher in 2 (2) and equal in 0, and its quality was lower in 3 (0 with an interval excluding 0), higher in 3 (0) and equal in 0.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less G | +$0.79 [+$0.59, +$1.00]* | -$1.43 [-$1.87, -$0.99]* | -$0.22 [-$0.76, +$0.33] | +$0.56 [+$0.45, +$0.67]* | -$1.88 [-$2.79, -$1.00]* | -$0.72 [-$1.51, +$0.09] |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less G | +0.63 [0.00, +1.86] | +0.54 [-1.38, +2.77] | -0.21 [-2.41, +1.76] | +0.34 [0.00, +0.91] | -0.36 [-1.54, +0.84] | -0.59 [-1.95, +0.77] |

#### haiku/Jev against G+

Comparisons: 1. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 6 cells, 0 with both points at A and so no difference; in the other 6, the first policy's cost per 1k was lower in 4 (1 with an interval excluding 0), higher in 2 (2) and equal in 0, and its quality was lower in 3 (0 with an interval excluding 0), higher in 2 (0) and equal in 1.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less G+ | -$0.08 [-$0.20, +$0.02]‡ | -$0.57 [-$1.04, -$0.08]* | $0.00 [-$0.34, +$0.35] | +$0.41 [+$0.14, +$0.75]* | -$0.77 [-$1.64, +$0.17] | +$0.75 [+$0.06, +$1.49]* |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less G+ | 0.00 [0.00, 0.00]‡ | +0.78 [-1.27, +3.01] | -0.20 [-2.09, +1.58] | +0.18 [-0.32, +0.80] | -0.12 [-1.46, +1.21] | -1.35 [-2.76, +0.07] |

#### The other cascades against the routers

Comparisons: 10. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 60 cells, 12 with both points at A and so no difference; in the other 48, the first policy's cost per 1k was lower in 9 (7 with an interval excluding 0), higher in 39 (36) and equal in 0, and its quality was lower in 15 (5 with an interval excluding 0), higher in 29 (8) and equal in 4.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/free less E | both at A | +$1.77 [+$1.55, +$2.00]†* | +$5.59 [+$5.07, +$6.12]†* | both at A | +$2.83 [+$2.42, +$3.27]†* | +$5.36 [+$4.78, +$5.97]†* |
| haiku/free less F | both at A | +$1.23 [+$0.97, +$1.52]†* | +$2.87 [+$2.46, +$3.29]†* | both at A | both at A | +$4.49 [+$3.84, +$5.17]†* |
| haiku/free less G | +$0.87 [+$0.67, +$1.09]†* | +$2.12 [+$1.83, +$2.40]†* | +$5.24 [+$4.78, +$5.73]†* | +$0.51 [+$0.41, +$0.62]†* | +$4.03 [+$3.40, +$4.72]†* | +$7.50 [+$6.69, +$8.36]†* |
| haiku/free less G+ | both at A | +$2.98 [+$2.59, +$3.39]†* | +$5.46 [+$4.98, +$5.97]†* | +$0.36 [+$0.11, +$0.68]†* | +$5.15 [+$4.45, +$5.96]†* | +$8.97 [+$8.17, +$9.84]†* |
| haiku/Jev less E | -$0.08 [-$0.20, +$0.02]‡ | -$1.79 [-$2.21, -$1.36]* | +$0.13 [-$0.31, +$0.61] | +$0.04 [-$0.07, +$0.15]‡ | -$3.08 [-$3.86, -$2.34]* | -$2.86 [-$3.58, -$2.20]* |
| haiku/Jev less F | -$0.08 [-$0.20, +$0.02]‡ | -$2.32 [-$2.79, -$1.85]* | -$2.59 [-$3.10, -$2.09]* | +$0.04 [-$0.07, +$0.15]‡ | -$5.92 [-$6.67, -$5.21]‡* | -$3.73 [-$4.51, -$2.99]* |
| haiku/Sonnet less E | both at A | +$1.77 [+$1.55, +$2.00]†* | +$6.46 [+$3.44, +$10.28]* | both at A | +$2.83 [+$2.42, +$3.27]†* | +$4.72 [+$2.57, +$7.01]* |
| haiku/Sonnet less F | both at A | +$1.23 [+$0.97, +$1.52]†* | +$3.74 [+$0.84, +$7.37]* | both at A | both at A | +$3.86 [+$1.64, +$6.26]* |
| haiku/Sonnet less G | +$0.87 [+$0.67, +$1.09]†* | +$2.12 [+$1.83, +$2.40]†* | +$6.11 [+$3.25, +$9.84]* | +$0.51 [+$0.41, +$0.62]†* | +$4.03 [+$3.40, +$4.72]†* | +$6.86 [+$4.65, +$9.29]* |
| haiku/Sonnet less G+ | both at A | +$2.98 [+$2.59, +$3.39]†* | +$6.32 [+$3.27, +$10.07]* | +$0.36 [+$0.11, +$0.68]†* | +$5.15 [+$4.45, +$5.96]†* | +$8.33 [+$6.07, +$10.69]* |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/free less E | both at A | -0.04 [-0.41, +0.34]† | +2.73 [-0.15, +5.83]† | both at A | +0.90 [-0.15, +2.00]† | +0.26 [-1.17, +1.55]† |
| haiku/free less F | both at A | +0.04 [-0.26, +0.34]† | -0.06 [-1.99, +1.76]† | both at A | both at A | -0.32 [-1.58, +0.89]† |
| haiku/free less G | +0.63 [0.00, +1.86]† | +0.82 [0.00, +2.22]† | +1.22 [-1.31, +3.72]† | +0.34 [0.00, +0.91]† | +1.11 [+0.03, +2.28]†* | +2.47 [+0.88, +4.15]†* |
| haiku/free less G+ | both at A | +1.05 [+0.13, +2.56]†* | +1.22 [-1.51, +3.94]† | +0.18 [-0.32, +0.80]† | +1.35 [+0.31, +2.59]†* | +1.72 [-0.03, +3.41]† |
| haiku/Jev less E | 0.00 [0.00, 0.00]‡ | -0.31 [-2.12, +1.42] | +1.30 [-0.81, +3.49] | 0.00 [0.00, 0.00]‡ | -0.57 [-2.22, +1.08] | -2.81 [-4.56, -1.06]* |
| haiku/Jev less F | 0.00 [0.00, 0.00]‡ | -0.23 [-2.02, +1.49] | -1.49 [-3.78, +0.50] | 0.00 [0.00, 0.00]‡ | -1.47 [-2.78, -0.19]‡* | -3.38 [-5.03, -1.83]* |
| haiku/Sonnet less E | both at A | -0.04 [-0.41, +0.34]† | +2.02 [+0.53, +3.98]* | both at A | +0.90 [-0.15, +2.00]† | -2.20 [-3.82, -0.58]* |
| haiku/Sonnet less F | both at A | +0.04 [-0.26, +0.34]† | -0.78 [-2.48, +0.95] | both at A | both at A | -2.78 [-4.22, -1.40]* |
| haiku/Sonnet less G | +0.63 [0.00, +1.86]† | +0.82 [0.00, +2.22]† | +0.50 [-1.21, +2.28] | +0.34 [0.00, +0.91]† | +1.11 [+0.03, +2.28]†* | +0.01 [-1.20, +1.40] |
| haiku/Sonnet less G+ | both at A | +1.05 [+0.13, +2.56]†* | +0.51 [-0.50, +2.06] | +0.18 [-0.32, +0.80]† | +1.35 [+0.31, +2.59]†* | -0.74 [-2.01, +0.62] |

#### What the gate adds to a router

Comparisons: 12. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 72 cells, 16 with both points at A and so no difference; in the other 56, the first policy's cost per 1k was lower in 7 (5 with an interval excluding 0), higher in 46 (29) and equal in 3, and its quality was lower in 8 (0 with an interval excluding 0), higher in 45 (7) and equal in 3.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| E/free less E | both at A | +$0.05 [$0.00, +$0.13] | +$0.06 [$0.00, +$0.16] | both at A | +$0.11 [$0.00, +$0.34] | +$0.12 [$0.00, +$0.35] |
| E/Jev less E | both at A | -$1.86 [-$2.31, -$1.43]* | +$1.97 [+$1.42, +$2.57]* | both at A | +$0.99 [+$0.27, +$1.68]* | +$1.36 [+$0.60, +$2.09]* |
| E/Sonnet less E | both at A | +$1.77 [+$1.55, +$2.00]†* | +$5.59 [+$5.07, +$6.12]†* | both at A | +$2.83 [+$2.42, +$3.27]†* | +$5.36 [+$4.78, +$5.97]†* |
| F/free less F | both at A | $0.00 [$0.00, $0.00] | $0.00 [$0.00, $0.00] | both at A | both at A | $0.00 [$0.00, $0.00] |
| F/Jev less F | both at A | +$1.24 [+$0.97, +$1.57]* | -$0.75 [-$1.25, -$0.23]* | both at A | -$3.99 [-$4.58, -$3.44]‡* | +$0.50 [-$0.28, +$1.28] |
| F/Sonnet less F | both at A | +$1.23 [+$0.97, +$1.52]†* | +$2.87 [+$2.46, +$3.29]†* | both at A | both at A | +$4.49 [+$3.84, +$5.17]†* |
| G/free less G | +$0.01 [$0.00, +$0.04] | +$0.01 [$0.00, +$0.04] | +$0.01 [$0.00, +$0.04] | +$0.01 [$0.00, +$0.02] | +$0.01 [$0.00, +$0.02] | +$0.01 [$0.00, +$0.02] |
| G/Jev less G | -$2.15 [-$2.55, -$1.77]* | -$1.48 [-$1.92, -$1.04]* | +$1.64 [+$1.14, +$2.17]* | +$0.03 [+$0.01, +$0.07]* | +$0.09 [-$0.70, +$0.89] | +$3.55 [+$2.64, +$4.50]* |
| G/Sonnet less G | +$1.74 [+$0.64, +$3.80]* | +$2.34 [+$0.98, +$4.75]* | +$6.35 [+$4.09, +$9.43]* | +$1.00 [+$0.41, +$1.96]* | +$3.53 [+$2.50, +$4.75]* | +$6.99 [+$5.82, +$8.38]* |
| G+/free less G+ | both at A | +$0.01 [$0.00, +$0.04] | +$0.01 [$0.00, +$0.04] | +$0.01 [$0.00, +$0.02] | +$0.01 [$0.00, +$0.02] | +$0.01 [$0.00, +$0.02] |
| G+/Jev less G+ | both at A | -$0.28 [-$0.75, +$0.20] | +$2.19 [+$1.74, +$2.69]* | +$0.26 [+$0.03, +$0.58]* | +$1.51 [+$0.72, +$2.33]* | +$5.33 [+$4.46, +$6.23]* |
| G+/Sonnet less G+ | -$0.24 [-$0.57, +$0.08]‡ | +$2.74 [+$2.37, +$3.13]* | +$5.21 [+$4.69, +$5.75]* | +$0.36 [+$0.11, +$0.68]†* | +$5.15 [+$4.45, +$5.96]†* | +$8.97 [+$8.17, +$9.84]†* |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| E/free less E | both at A | +0.06 [0.00, +0.19] | +0.67 [0.00, +1.95] | both at A | +0.16 [0.00, +0.48] | +0.39 [0.00, +1.02] |
| E/Jev less E | both at A | -0.31 [-2.12, +1.42] | +2.46 [+0.03, +5.07]* | both at A | +0.46 [-0.91, +1.82] | -0.16 [-1.49, +1.09] |
| E/Sonnet less E | both at A | -0.04 [-0.41, +0.34]† | +2.73 [-0.15, +5.83]† | both at A | +0.90 [-0.15, +2.00]† | +0.26 [-1.17, +1.55]† |
| F/free less F | both at A | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | both at A | both at A | 0.00 [0.00, 0.00] |
| F/Jev less F | both at A | +0.04 [-0.26, +0.34] | -0.33 [-2.22, +1.43] | both at A | -0.42 [-1.22, +0.37]‡ | -0.74 [-2.06, +0.50] |
| F/Sonnet less F | both at A | +0.04 [-0.26, +0.34]† | -0.06 [-1.99, +1.76]† | both at A | both at A | -0.32 [-1.58, +0.89]† |
| G/free less G | +0.61 [0.00, +1.82] | +0.61 [0.00, +1.82] | +0.61 [0.00, +1.82] | +0.23 [0.00, +0.70] | +0.23 [0.00, +0.70] | +0.23 [0.00, +0.70] |
| G/Jev less G | +0.99 [-0.33, +2.91] | +0.54 [-1.38, +2.77] | +0.95 [-0.94, +2.79] | +0.23 [0.00, +0.70] | +0.69 [-0.31, +1.77] | +2.05 [+0.67, +3.49]* |
| G/Sonnet less G | +0.61 [0.00, +1.82] | +1.53 [+0.17, +3.44]* | +1.26 [-0.67, +3.13] | +0.23 [0.00, +0.70] | +1.56 [+0.68, +2.64]* | +2.92 [+1.48, +4.47]* |
| G+/free less G+ | both at A | +0.61 [0.00, +1.82] | +0.61 [0.00, +1.82] | +0.23 [0.00, +0.70] | +0.23 [0.00, +0.70] | +0.23 [0.00, +0.70] |
| G+/Jev less G+ | both at A | +0.78 [-1.27, +3.01] | +0.95 [-1.23, +3.19] | +0.18 [-0.32, +0.80] | +0.93 [-0.33, +2.33] | +1.30 [-0.27, +2.90] |
| G+/Sonnet less G+ | +0.11 [-0.04, +0.30]‡ | +1.16 [+0.28, +2.63]* | +1.33 [-1.38, +4.03] | +0.18 [-0.32, +0.80]† | +1.35 [+0.31, +2.59]†* | +1.72 [-0.03, +3.41]† |

#### What the router adds to a cascade

Comparisons: 12. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 72 cells, 16 with both points at A and so no difference; in the other 56, the first policy's cost per 1k was lower in 31 (22 with an interval excluding 0), higher in 25 (14) and equal in 0, and its quality was lower in 16 (3 with an interval excluding 0), higher in 31 (15) and equal in 9.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| E/free less haiku/free | both at A | -$1.72 [-$1.94, -$1.51]‡* | -$5.53 [-$6.06, -$5.02]‡* | both at A | -$2.72 [-$3.13, -$2.34]‡* | -$5.24 [-$5.83, -$4.67]‡* |
| E/Jev less haiku/Jev | +$0.08 [-$0.02, +$0.20]† | -$0.08 [-$0.17, -$0.02]* | +$1.83 [+$1.42, +$2.32]* | -$0.04 [-$0.15, +$0.07]† | +$4.07 [+$3.41, +$4.80]* | +$4.22 [+$3.46, +$5.06]* |
| E/Sonnet less haiku/Sonnet | both at A | both at A | -$0.87 [-$4.54, +$2.11]† | both at A | both at A | +$0.63 [-$1.70, +$2.86]† |
| F/free less haiku/free | both at A | -$1.23 [-$1.52, -$0.97]‡* | -$2.87 [-$3.29, -$2.46]‡* | both at A | both at A | -$4.49 [-$5.17, -$3.84]‡* |
| F/Jev less haiku/Jev | +$0.08 [-$0.02, +$0.20]† | +$3.56 [+$3.15, +$3.99]* | +$1.84 [+$1.43, +$2.33]* | -$0.04 [-$0.15, +$0.07]† | +$1.93 [+$1.42, +$2.50]* | +$4.23 [+$3.47, +$5.07]* |
| F/Sonnet less haiku/Sonnet | both at A | both at A | -$0.87 [-$4.54, +$2.11]† | both at A | both at A | +$0.63 [-$1.70, +$2.86]† |
| G/free less haiku/free | -$0.86 [-$1.06, -$0.66]‡* | -$2.10 [-$2.39, -$1.82]‡* | -$5.22 [-$5.72, -$4.76]‡* | -$0.51 [-$0.62, -$0.40]‡* | -$4.03 [-$4.72, -$3.39]‡* | -$7.49 [-$8.35, -$6.69]‡* |
| G/Jev less haiku/Jev | -$2.94 [-$3.31, -$2.57]* | -$0.04 [-$0.15, +$0.01] | +$1.86 [+$1.46, +$2.35]* | -$0.52 [-$0.63, -$0.42]* | +$1.98 [+$1.47, +$2.56]* | +$4.28 [+$3.52, +$5.11]* |
| G/Sonnet less haiku/Sonnet | +$0.87 [-$0.23, +$2.93]‡ | +$0.22 [-$1.10, +$2.63]‡ | +$0.24 [-$2.03, +$1.70] | +$0.49 [-$0.09, +$1.46]‡ | -$0.50 [-$1.42, +$0.65]‡ | +$0.13 [-$1.99, +$2.10] |
| G+/free less haiku/free | both at A | -$2.97 [-$3.38, -$2.57]‡* | -$5.44 [-$5.95, -$4.96]‡* | -$0.36 [-$0.68, -$0.10]‡* | -$5.15 [-$5.96, -$4.45]‡* | -$8.96 [-$9.84, -$8.16]‡* |
| G+/Jev less haiku/Jev | +$0.08 [-$0.02, +$0.20]† | +$0.28 [+$0.18, +$0.34]* | +$2.19 [+$1.78, +$2.67]* | -$0.15 [-$0.29, -$0.03]* | +$2.28 [+$1.77, +$2.86]* | +$4.58 [+$3.83, +$5.42]* |
| G+/Sonnet less haiku/Sonnet | -$0.24 [-$0.57, +$0.08]‡ | -$0.24 [-$0.57, +$0.08]‡ | -$1.11 [-$4.68, +$1.84] | both at A | both at A | +$0.63 [-$1.70, +$2.86]† |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| E/free less haiku/free | both at A | +0.11 [-0.23, +0.46]‡ | -2.06 [-4.97, +0.71]‡ | both at A | -0.74 [-1.79, +0.26]‡ | +0.14 [-1.06, +1.45]‡ |
| E/Jev less haiku/Jev | 0.00 [0.00, 0.00]† | 0.00 [0.00, 0.00] | +1.15 [+0.17, +2.54]* | 0.00 [0.00, 0.00]† | +1.02 [-0.16, +2.17] | +2.65 [+1.34, +4.12]* |
| E/Sonnet less haiku/Sonnet | both at A | both at A | +0.71 [-1.87, +3.10]† | both at A | both at A | +2.46 [+0.81, +4.16]†* |
| F/free less haiku/free | both at A | -0.04 [-0.34, +0.26]‡ | +0.06 [-1.76, +1.99]‡ | both at A | both at A | +0.32 [-0.89, +1.58]‡ |
| F/Jev less haiku/Jev | 0.00 [0.00, 0.00]† | +0.27 [-1.45, +2.07] | +1.15 [+0.17, +2.54]* | 0.00 [0.00, 0.00]† | +1.05 [+0.15, +2.06]* | +2.65 [+1.34, +4.12]* |
| F/Sonnet less haiku/Sonnet | both at A | both at A | +0.71 [-1.87, +3.10]† | both at A | both at A | +2.46 [+0.81, +4.16]†* |
| G/free less haiku/free | -0.02 [-0.06, 0.00]‡ | -0.21 [-0.53, +0.06]‡ | -0.61 [-2.77, +1.70]‡ | -0.10 [-0.32, 0.00]‡ | -0.87 [-1.90, +0.11]‡ | -2.23 [-3.81, -0.74]‡* |
| G/Jev less haiku/Jev | +0.36 [-0.48, +1.76] | 0.00 [0.00, 0.00] | +1.15 [+0.17, +2.54]* | -0.10 [-0.32, 0.00] | +1.05 [+0.15, +2.06]* | +2.65 [+1.34, +4.12]* |
| G/Sonnet less haiku/Sonnet | -0.02 [-0.06, 0.00]‡ | +0.71 [-0.02, +2.08]‡ | +0.76 [-1.06, +2.46] | -0.10 [-0.32, 0.00]‡ | +0.45 [-0.15, +1.20]‡ | +2.91 [+1.42, +4.50]* |
| G+/free less haiku/free | both at A | -0.45 [-0.89, -0.06]‡* | -0.61 [-3.04, +1.86]‡ | +0.06 [-0.31, +0.48]‡ | -1.11 [-2.19, -0.10]‡* | -1.48 [-3.13, +0.19]‡ |
| G+/Jev less haiku/Jev | 0.00 [0.00, 0.00]† | 0.00 [0.00, 0.00] | +1.15 [+0.17, +2.54]* | 0.00 [0.00, 0.00] | +1.05 [+0.15, +2.06]* | +2.65 [+1.34, +4.12]* |
| G+/Sonnet less haiku/Sonnet | +0.11 [-0.04, +0.30]‡ | +0.11 [-0.04, +0.30]‡ | +0.82 [-1.76, +3.29] | both at A | both at A | +2.46 [+0.81, +4.16]†* |

#### What a paid gate adds over the free checks

Comparisons: 10. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 60 cells, 14 with both points at A and so no difference; in the other 46, the first policy's cost per 1k was lower in 13 (9 with an interval excluding 0), higher in 33 (29) and equal in 0, and its quality was lower in 18 (3 with an interval excluding 0), higher in 23 (6) and equal in 5.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less haiku/free | -$0.08 [-$0.20, +$0.02]‡ | -$3.55 [-$3.96, -$3.15]‡* | -$5.46 [-$5.97, -$4.98]‡* | +$0.04 [-$0.07, +$0.15]‡ | -$5.92 [-$6.67, -$5.21]‡* | -$8.22 [-$9.10, -$7.44]‡* |
| haiku/Sonnet less haiku/free | both at A | both at A | +$0.87 [-$2.11, +$4.54]‡ | both at A | both at A | -$0.63 [-$2.86, +$1.70]‡ |
| E/Jev less E/free | both at A | -$1.91 [-$2.36, -$1.46]* | +$1.91 [+$1.35, +$2.51]* | both at A | +$0.87 [+$0.17, +$1.54]* | +$1.24 [+$0.45, +$2.02]* |
| E/Sonnet less E/free | both at A | +$1.72 [+$1.51, +$1.94]†* | +$5.53 [+$5.02, +$6.06]†* | both at A | +$2.72 [+$2.34, +$3.13]†* | +$5.24 [+$4.67, +$5.83]†* |
| F/Jev less F/free | both at A | +$1.24 [+$0.97, +$1.57]* | -$0.75 [-$1.25, -$0.23]* | both at A | -$3.99 [-$4.58, -$3.44]‡* | +$0.50 [-$0.28, +$1.28] |
| F/Sonnet less F/free | both at A | +$1.23 [+$0.97, +$1.52]†* | +$2.87 [+$2.46, +$3.29]†* | both at A | both at A | +$4.49 [+$3.84, +$5.17]†* |
| G/Jev less G/free | -$2.17 [-$2.56, -$1.79]* | -$1.49 [-$1.94, -$1.05]* | +$1.63 [+$1.13, +$2.16]* | +$0.03 [$0.00, +$0.06]* | +$0.09 [-$0.71, +$0.89] | +$3.55 [+$2.62, +$4.50]* |
| G/Sonnet less G/free | +$1.72 [+$0.63, +$3.78]* | +$2.32 [+$0.97, +$4.73]* | +$6.33 [+$4.07, +$9.42]* | +$0.99 [+$0.41, +$1.96]* | +$3.53 [+$2.49, +$4.74]* | +$6.99 [+$5.81, +$8.38]* |
| G+/Jev less G+/free | both at A | -$0.30 [-$0.77, +$0.19] | +$2.18 [+$1.72, +$2.66]* | +$0.25 [+$0.02, +$0.57]* | +$1.51 [+$0.71, +$2.33]* | +$5.32 [+$4.46, +$6.22]* |
| G+/Sonnet less G+/free | -$0.24 [-$0.57, +$0.08]‡ | +$2.72 [+$2.36, +$3.12]* | +$5.20 [+$4.68, +$5.73]* | +$0.36 [+$0.10, +$0.68]†* | +$5.15 [+$4.45, +$5.96]†* | +$8.96 [+$8.16, +$9.84]†* |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less haiku/free | 0.00 [0.00, 0.00]‡ | -0.27 [-2.07, +1.45]‡ | -1.42 [-3.62, +0.60]‡ | 0.00 [0.00, 0.00]‡ | -1.47 [-2.78, -0.19]‡* | -3.06 [-4.72, -1.49]‡* |
| haiku/Sonnet less haiku/free | both at A | both at A | -0.71 [-3.10, +1.87]‡ | both at A | both at A | -2.46 [-4.16, -0.81]‡* |
| E/Jev less E/free | both at A | -0.38 [-2.18, +1.34] | +1.79 [-0.35, +3.97] | both at A | +0.30 [-1.01, +1.59] | -0.55 [-1.77, +0.55] |
| E/Sonnet less E/free | both at A | -0.11 [-0.46, +0.23]† | +2.06 [-0.71, +4.97]† | both at A | +0.74 [-0.26, +1.79]† | -0.14 [-1.45, +1.06]† |
| F/Jev less F/free | both at A | +0.04 [-0.26, +0.34] | -0.33 [-2.22, +1.43] | both at A | -0.42 [-1.22, +0.37]‡ | -0.74 [-2.06, +0.50] |
| F/Sonnet less F/free | both at A | +0.04 [-0.26, +0.34]† | -0.06 [-1.99, +1.76]† | both at A | both at A | -0.32 [-1.58, +0.89]† |
| G/Jev less G/free | +0.38 [-0.46, +1.78] | -0.06 [-1.87, +1.70] | +0.34 [-1.26, +1.44] | 0.00 [0.00, 0.00] | +0.46 [-0.45, +1.38] | +1.82 [+0.52, +3.14]* |
| G/Sonnet less G/free | 0.00 [0.00, 0.00] | +0.92 [+0.13, +2.35]* | +0.65 [-0.92, +1.80] | 0.00 [0.00, 0.00] | +1.32 [+0.53, +2.32]* | +2.68 [+1.36, +4.10]* |
| G+/Jev less G+/free | both at A | +0.17 [-1.65, +1.89] | +0.34 [-1.54, +2.01] | -0.06 [-0.48, +0.31] | +0.70 [-0.51, +2.01] | +1.07 [-0.40, +2.53] |
| G+/Sonnet less G+/free | +0.11 [-0.04, +0.30]‡ | +0.55 [+0.21, +0.96]* | +0.72 [-1.76, +3.15] | -0.06 [-0.48, +0.31]† | +1.11 [+0.10, +2.19]†* | +1.48 [-0.19, +3.13]† |

#### The Jev gate against the Sonnet gate

Comparisons: 5. Each pair is the first policy's point less the second's at one mix and one target, paired over the same resamples, in each of the six mix-and-target cells. Over 30 cells, 4 with both points at A and so no difference; in the other 26, the first policy's cost per 1k was lower in 23 (21 with an interval excluding 0), higher in 3 (0) and equal in 0, and its quality was lower in 20 (4 with an interval excluding 0), higher in 1 (0) and equal in 5.

Difference in cost per 1k, in dollars:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less haiku/Sonnet | -$0.08 [-$0.20, +$0.02]‡ | -$3.55 [-$3.96, -$3.15]‡* | -$6.33 [-$10.00, -$3.27]* | +$0.04 [-$0.07, +$0.15]‡ | -$5.92 [-$6.67, -$5.21]‡* | -$7.58 [-$9.90, -$5.39]* |
| E/Jev less E/Sonnet | both at A | -$3.63 [-$4.02, -$3.21]‡* | -$3.63 [-$4.02, -$3.21]‡* | both at A | -$1.85 [-$2.42, -$1.33]‡* | -$4.00 [-$4.59, -$3.45]‡* |
| F/Jev less F/Sonnet | both at A | +$0.01 [-$0.22, +$0.23]‡ | -$3.62 [-$4.01, -$3.20]‡* | both at A | -$3.99 [-$4.58, -$3.44]‡* | -$3.99 [-$4.58, -$3.44]‡* |
| G/Jev less G/Sonnet | -$3.89 [-$6.13, -$2.66]* | -$3.81 [-$6.14, -$2.47]* | -$4.70 [-$7.74, -$2.41]* | -$0.97 [-$1.94, -$0.38]* | -$3.44 [-$4.61, -$2.44]* | -$3.44 [-$4.61, -$2.44]* |
| G+/Jev less G+/Sonnet | +$0.24 [-$0.08, +$0.57]† | -$3.02 [-$3.49, -$2.58]* | -$3.02 [-$3.49, -$2.58]* | -$0.11 [-$0.24, +$0.02]‡ | -$3.64 [-$4.23, -$3.09]‡* | -$3.64 [-$4.23, -$3.09]‡* |

Difference in quality, in percentage points:

| First less second | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| haiku/Jev less haiku/Sonnet | 0.00 [0.00, 0.00]‡ | -0.27 [-2.07, +1.45]‡ | -0.71 [-2.30, +0.31] | 0.00 [0.00, 0.00]‡ | -1.47 [-2.78, -0.19]‡* | -0.60 [-1.90, +0.58] |
| E/Jev less E/Sonnet | both at A | -0.27 [-2.07, +1.45]‡ | -0.27 [-2.07, +1.45]‡ | both at A | -0.44 [-1.35, +0.42]‡ | -0.42 [-1.22, +0.37]‡ |
| F/Jev less F/Sonnet | both at A | 0.00 [0.00, 0.00]‡ | -0.27 [-2.07, +1.45]‡ | both at A | -0.42 [-1.22, +0.37]‡ | -0.42 [-1.22, +0.37]‡ |
| G/Jev less G/Sonnet | +0.38 [-0.46, +1.78] | -0.99 [-2.41, -0.15]* | -0.31 [-0.71, +0.02] | 0.00 [0.00, 0.00] | -0.87 [-1.70, -0.16]* | -0.87 [-1.70, -0.16]* |
| G+/Jev less G+/Sonnet | -0.11 [-0.30, +0.04]† | -0.38 [-2.16, +1.38] | -0.38 [-2.16, +1.38] | 0.00 [0.00, 0.00]‡ | -0.42 [-1.22, +0.37]‡ | -0.42 [-1.22, +0.37]‡ |

### The replication summary

For each dev comparison whose interval excluded 0, whether test's interval has the same sign and also excludes 0. Cost
and quality are counted apart, and a comparison counts once for each mix-and-target cell where its dev interval
excluded 0.

| | Dev intervals excluding 0 | Same sign on test | Same sign, and excluding 0 on test |
|---|---|---|---|
| Difference in cost per 1k | 185 | 184 | 174 |
| Difference in quality | 3 | 3 | 3 |

### Every policy at its operating points

Every frozen policy at its dev-selected operating points, under both mixes and per task, in the tables below. IFEval's and MBPP+'s
per-task results carry their warning wherever they appear, and the write-up draws no per-task conclusion from them (A7
item 2). IFEval: Only 156 test items, below the plan's minimum of a few hundred: indicative. MBPP+: Only 107 test items,
below the plan's minimum of a few hundred: indicative.

#### Under both mixes

Each row is a policy at one mix and one target: whether the target is met (A19 item 6), quality and cost per 1k with 95% intervals, the share of A's cost saved, and the fraction of the hindsight oracle's savings captured at that target (none where the tables hold none).

| Policy | Mix | Target | Reading | Quality | Cost per 1k | Saved against A | Captured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E Rules | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E Rules | realistic | 99% | met | 90.9% [86.9%, 94.4%] | $5.74 [$5.33, $6.21] | 23.5% | 24.5% |
| E Rules | realistic | 97% | not met | 88.1% [83.8%, 91.8%] | $1.92 [$1.73, $2.11] | 74.5% | none |
| E Rules | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| E Rules | uniform | 99% | met | 90.7% [88.4%, 92.8%] | $8.93 [$8.32, $9.54] | 24.1% | 25.4% |
| E Rules | uniform | 97% | met | 91.4% [89.1%, 93.4%] | $6.40 [$5.98, $6.87] | 45.6% | 47.6% |
| F Trained | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F Trained | realistic | 99% | met | 90.8% [86.8%, 94.3%] | $6.28 [$5.77, $6.81] | 16.4% | 17.1% |
| F Trained | realistic | 97% | met | 90.9% [86.9%, 94.4%] | $4.64 [$4.19, $5.12] | 38.2% | 39.6% |
| F Trained | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F Trained | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F Trained | uniform | 97% | met | 92.0% [89.7%, 94.0%] | $7.27 [$6.65, $7.90] | 38.2% | 39.9% |
| G Jev | realistic | 100% | not met | 90.2% [86.1%, 94.0%] | $6.64 [$6.03, $7.24] | 11.6% | none |
| G Jev | realistic | 99% | met | 90.1% [85.8%, 93.8%] | $5.39 [$4.77, $6.05] | 28.2% | 29.4% |
| G Jev | realistic | 97% | met | 89.6% [85.6%, 93.1%] | $2.27 [$1.83, $2.76] | 69.7% | 72.3% |
| G Jev | uniform | 100% | not met | 91.3% [89.0%, 93.4%] | $11.25 [$10.37, $12.19] | 4.4% | none |
| G Jev | uniform | 99% | not met | 90.5% [88.0%, 92.7%] | $7.73 [$6.82, $8.65] | 34.3% | none |
| G Jev | uniform | 97% | met | 89.2% [86.6%, 91.5%] | $4.26 [$3.57, $5.00] | 63.7% | 66.6% |
| G+ Bolstered Jev | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| G+ Bolstered Jev | realistic | 99% | not met | 89.8% [85.6%, 93.5%] | $4.53 [$3.98, $5.11] | 39.7% | none |
| G+ Bolstered Jev | realistic | 97% | met | 89.6% [85.5%, 93.2%] | $2.05 [$1.77, $2.35] | 72.7% | 75.3% |
| G+ Bolstered Jev | uniform | 100% | not met | 91.5% [89.1%, 93.5%] | $11.40 [$10.55, $12.33] | 3.1% | none |
| G+ Bolstered Jev | uniform | 99% | not met | 90.3% [87.7%, 92.5%] | $6.61 [$5.88, $7.37] | 43.8% | none |
| G+ Bolstered Jev | uniform | 97% | met | 89.9% [87.5%, 92.2%] | $2.79 [$2.48, $3.16] | 76.2% | 79.6% |
| haiku/free | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| haiku/free | realistic | 99% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| haiku/free | realistic | 97% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| haiku/free | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| haiku/free | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| haiku/free | uniform | 97% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| haiku/Jev | realistic | 100% | met | 90.9% [86.8%, 94.4%] | $7.43 [$6.85, $8.02] | 1.1% | 1.1% |
| haiku/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%] | $3.96 [$3.43, $4.57] | 47.3% | 49.3% |
| haiku/Jev | realistic | 97% | met | 89.4% [85.5%, 93.1%] | $2.05 [$1.63, $2.51] | 72.7% | 75.3% |
| haiku/Jev | uniform | 100% | met | 91.6% [89.4%, 93.7%] | $11.80 [$10.90, $12.74] | -0.4% | -0.4% |
| haiku/Jev | uniform | 99% | not met | 90.2% [87.7%, 92.4%] | $5.84 [$4.93, $6.79] | 50.3% | none |
| haiku/Jev | uniform | 97% | not met | 88.6% [86.0%, 90.9%] | $3.54 [$2.78, $4.35] | 69.9% | none |
| haiku/Sonnet | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| haiku/Sonnet | realistic | 99% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| haiku/Sonnet | realistic | 97% | met | 90.2% [86.2%, 93.5%] | $8.38 [$5.32, $12.25] | -11.6% | -12.0% |
| haiku/Sonnet | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| haiku/Sonnet | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| haiku/Sonnet | uniform | 97% | met | 89.2% [86.7%, 91.4%] | $11.13 [$8.86, $13.54] | 5.4% | 5.6% |
| E/free | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E/free | realistic | 99% | met | 91.0% [87.0%, 94.5%] | $5.79 [$5.35, $6.26] | 22.9% | 23.9% |
| E/free | realistic | 97% | met | 88.8% [84.9%, 92.4%] | $1.98 [$1.77, $2.21] | 73.7% | 76.4% |
| E/free | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| E/free | uniform | 99% | met | 90.9% [88.5%, 93.0%] | $9.04 [$8.39, $9.72] | 23.1% | 24.4% |
| E/free | uniform | 97% | met | 91.8% [89.5%, 93.8%] | $6.52 [$6.05, $7.04] | 44.6% | 46.5% |
| E/Jev | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%] | $3.88 [$3.35, $4.49] | 48.3% | 50.4% |
| E/Jev | realistic | 97% | met | 90.6% [86.8%, 94.1%] | $3.88 [$3.35, $4.49] | 48.3% | 50.0% |
| E/Jev | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| E/Jev | uniform | 99% | met | 91.2% [88.9%, 93.2%] | $9.91 [$8.88, $10.94] | 15.7% | 16.6% |
| E/Jev | uniform | 97% | met | 91.2% [88.9%, 93.3%] | $7.76 [$6.79, $8.75] | 34.0% | 35.6% |
| E/Sonnet | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E/Sonnet | realistic | 99% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E/Sonnet | realistic | 97% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| E/Sonnet | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| E/Sonnet | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| E/Sonnet | uniform | 97% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/free | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F/free | realistic | 99% | met | 90.8% [86.8%, 94.3%] | $6.28 [$5.77, $6.81] | 16.4% | 17.1% |
| F/free | realistic | 97% | met | 90.9% [86.9%, 94.4%] | $4.64 [$4.19, $5.12] | 38.2% | 39.6% |
| F/free | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/free | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/free | uniform | 97% | met | 92.0% [89.7%, 94.0%] | $7.27 [$6.65, $7.90] | 38.2% | 39.9% |
| F/Jev | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F/Jev | realistic | 99% | met | 90.9% [86.8%, 94.4%] | $7.52 [$6.94, $8.12] | -0.1% | -0.1% |
| F/Jev | realistic | 97% | met | 90.6% [86.8%, 94.1%] | $3.89 [$3.36, $4.50] | 48.2% | 49.9% |
| F/Jev | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/Jev | uniform | 99% | met | 91.2% [88.9%, 93.3%] | $7.77 [$6.80, $8.77] | 33.9% | 35.8% |
| F/Jev | uniform | 97% | met | 91.2% [88.9%, 93.3%] | $7.77 [$6.80, $8.77] | 33.9% | 35.4% |
| F/Sonnet | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F/Sonnet | realistic | 99% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F/Sonnet | realistic | 97% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| F/Sonnet | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/Sonnet | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| F/Sonnet | uniform | 97% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| G/free | realistic | 100% | not met | 90.8% [86.8%, 94.4%] | $6.65 [$6.04, $7.26] | 11.4% | none |
| G/free | realistic | 99% | met | 90.7% [86.6%, 94.2%] | $5.41 [$4.79, $6.05] | 28.0% | 29.2% |
| G/free | realistic | 97% | met | 90.3% [86.4%, 93.6%] | $2.29 [$1.85, $2.77] | 69.6% | 72.1% |
| G/free | uniform | 100% | not met | 91.5% [89.3%, 93.6%] | $11.25 [$10.38, $12.19] | 4.3% | none |
| G/free | uniform | 99% | met | 90.8% [88.3%, 92.9%] | $7.73 [$6.83, $8.66] | 34.2% | 36.1% |
| G/free | uniform | 97% | met | 89.4% [86.8%, 91.7%] | $4.27 [$3.57, $5.01] | 63.7% | 66.5% |
| G/Jev | realistic | 100% | met | 91.2% [87.6%, 94.5%] | $4.49 [$3.92, $5.12] | 40.3% | 42.3% |
| G/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%] | $3.92 [$3.38, $4.52] | 47.9% | 49.9% |
| G/Jev | realistic | 97% | met | 90.6% [86.8%, 94.1%] | $3.92 [$3.38, $4.52] | 47.9% | 49.6% |
| G/Jev | uniform | 100% | not met | 91.5% [89.3%, 93.6%] | $11.28 [$10.41, $12.21] | 4.1% | none |
| G/Jev | uniform | 99% | met | 91.2% [88.9%, 93.3%] | $7.82 [$6.85, $8.82] | 33.5% | 35.3% |
| G/Jev | uniform | 97% | met | 91.2% [88.9%, 93.3%] | $7.82 [$6.85, $8.82] | 33.5% | 35.0% |
| G/Sonnet | realistic | 100% | not met | 90.8% [86.8%, 94.4%] | $8.38 [$6.99, $10.74] | -11.6% | none |
| G/Sonnet | realistic | 99% | met | 91.6% [88.0%, 94.9%] | $7.73 [$6.26, $10.22] | -2.9% | -3.1% |
| G/Sonnet | realistic | 97% | met | 90.9% [87.1%, 94.4%] | $8.62 [$6.29, $11.76] | -14.8% | -15.3% |
| G/Sonnet | uniform | 100% | not met | 91.5% [89.3%, 93.6%] | $12.24 [$11.12, $13.56] | -4.1% | none |
| G/Sonnet | uniform | 99% | met | 92.1% [89.9%, 94.1%] | $11.26 [$10.05, $12.75] | 4.3% | 4.5% |
| G/Sonnet | uniform | 97% | met | 92.1% [89.9%, 94.1%] | $11.26 [$10.05, $12.75] | 4.3% | 4.5% |
| G+/free | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| G+/free | realistic | 99% | met | 90.4% [86.3%, 94.0%] | $4.54 [$4.00, $5.12] | 39.5% | 41.2% |
| G+/free | realistic | 97% | met | 90.3% [86.4%, 93.6%] | $2.07 [$1.79, $2.37] | 72.5% | 75.1% |
| G+/free | uniform | 100% | met | 91.7% [89.4%, 93.7%] | $11.40 [$10.56, $12.33] | 3.0% | 3.2% |
| G+/free | uniform | 99% | not met | 90.5% [88.1%, 92.7%] | $6.61 [$5.88, $7.38] | 43.8% | none |
| G+/free | uniform | 97% | met | 90.1% [87.7%, 92.4%] | $2.80 [$2.48, $3.16] | 76.2% | 79.6% |
| G+/Jev | realistic | 100% | falls back to A | 90.9% [86.8%, 94.4%] | $7.51 [$6.99, $8.05] | 0.0% | 0.0% |
| G+/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%] | $4.24 [$3.71, $4.85] | 43.5% | 45.4% |
| G+/Jev | realistic | 97% | met | 90.6% [86.8%, 94.1%] | $4.24 [$3.71, $4.85] | 43.5% | 45.1% |
| G+/Jev | uniform | 100% | met | 91.6% [89.4%, 93.7%] | $11.65 [$10.79, $12.59] | 0.9% | 1.0% |
| G+/Jev | uniform | 99% | met | 91.2% [88.9%, 93.3%] | $8.12 [$7.15, $9.12] | 30.9% | 32.6% |
| G+/Jev | uniform | 97% | met | 91.2% [88.9%, 93.3%] | $8.12 [$7.15, $9.12] | 30.9% | 32.3% |
| G+/Sonnet | realistic | 100% | met | 91.0% [86.9%, 94.5%] | $7.27 [$6.70, $7.86] | 3.2% | 3.4% |
| G+/Sonnet | realistic | 99% | met | 91.0% [86.9%, 94.5%] | $7.27 [$6.70, $7.86] | 3.2% | 3.4% |
| G+/Sonnet | realistic | 97% | met | 91.0% [86.9%, 94.5%] | $7.27 [$6.70, $7.86] | 3.2% | 3.4% |
| G+/Sonnet | uniform | 100% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| G+/Sonnet | uniform | 99% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |
| G+/Sonnet | uniform | 97% | falls back to A | 91.6% [89.4%, 93.7%] | $11.76 [$10.91, $12.66] | 0.0% | 0.0% |

#### IFEval, per task

IFEval has 156 test items. Only 156 test items, below the plan's minimum of a few hundred: indicative. Each cell is the policy's quality and cost per 1k on this task alone, at the operating point the mix and target froze; intervals are in figures/tables_test.json.

| Policy | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| Frontier only | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 |
| E Rules | 92.9%, $27.00 | 93.6%, $11.33 | 87.2%, $2.34 | 92.9%, $27.00 | 91.0%, $19.17 | 93.6%, $11.33 |
| F Trained | 92.9%, $27.00 | 92.9%, $19.57 | 93.6%, $12.75 | 92.9%, $27.00 | 92.9%, $27.00 | 94.2%, $14.34 |
| G Jev | 92.9%, $26.92 | 92.9%, $23.25 | 89.7%, $9.82 | 92.9%, $26.92 | 91.0%, $19.11 | 89.7%, $9.82 |
| G+ Bolstered Jev | 92.9%, $27.00 | 89.7%, $12.53 | 89.1%, $3.40 | 93.6%, $26.19 | 89.7%, $12.53 | 89.1%, $3.40 |
| haiku/free | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 |
| haiku/Jev | 92.9%, $27.97 | 92.9%, $22.84 | 91.0%, $12.60 | 92.9%, $27.97 | 91.7%, $17.37 | 90.4%, $11.18 |
| haiku/Sonnet | 92.9%, $27.00 | 92.9%, $27.00 | 91.0%, $17.08 | 92.9%, $27.00 | 92.9%, $27.00 | 91.0%, $17.01 |
| E/free | 92.9%, $27.00 | 94.2%, $11.78 | 87.8%, $2.78 | 92.9%, $27.00 | 91.7%, $19.62 | 94.2%, $11.78 |
| E/Jev | 92.9%, $27.00 | 92.9%, $22.37 | 92.9%, $22.37 | 92.9%, $27.00 | 92.9%, $24.45 | 92.9%, $22.37 |
| E/Sonnet | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 |
| F/free | 92.9%, $27.00 | 92.9%, $19.57 | 93.6%, $12.75 | 92.9%, $27.00 | 92.9%, $27.00 | 94.2%, $14.34 |
| F/Jev | 92.9%, $27.00 | 92.9%, $28.72 | 92.9%, $22.37 | 92.9%, $27.00 | 92.9%, $22.37 | 92.9%, $22.37 |
| F/Sonnet | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 |
| G/free | 92.9%, $26.92 | 92.9%, $23.25 | 89.7%, $9.82 | 92.9%, $26.92 | 91.0%, $19.11 | 89.7%, $9.82 |
| G/Jev | 92.9%, $23.72 | 92.9%, $22.38 | 92.9%, $22.38 | 92.9%, $27.00 | 92.9%, $22.38 | 92.9%, $22.38 |
| G/Sonnet | 92.9%, $26.96 | 94.2%, $24.04 | 93.6%, $23.46 | 92.9%, $26.96 | 94.2%, $24.04 | 94.2%, $24.04 |
| G+/free | 92.9%, $27.00 | 89.7%, $12.53 | 89.1%, $3.40 | 93.6%, $26.19 | 89.7%, $12.53 | 89.1%, $3.40 |
| G+/Jev | 92.9%, $27.00 | 92.9%, $22.72 | 92.9%, $22.72 | 92.9%, $27.10 | 92.9%, $22.72 | 92.9%, $22.72 |
| G+/Sonnet | 94.2%, $24.55 | 94.2%, $24.55 | 94.2%, $24.55 | 92.9%, $27.00 | 92.9%, $27.00 | 92.9%, $27.00 |

#### Banking77, per task

Banking77 has 240 test items. Each cell is the policy's quality and cost per 1k on this task alone, at the operating point the mix and target froze; intervals are in figures/tables_test.json.

| Policy | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| Frontier only | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 |
| E Rules | 90.0%, $8.54 | 90.4%, $8.39 | 85.8%, $4.40 | 90.0%, $8.54 | 90.4%, $8.39 | 90.4%, $8.39 |
| F Trained | 90.0%, $8.54 | 90.0%, $8.22 | 90.0%, $7.13 | 90.0%, $8.54 | 90.0%, $8.54 | 90.4%, $7.25 |
| G Jev | 90.0%, $7.77 | 89.2%, $5.28 | 86.2%, $1.64 | 90.0%, $7.77 | 87.9%, $3.66 | 86.2%, $1.64 |
| G+ Bolstered Jev | 90.0%, $8.54 | 89.6%, $6.32 | 89.2%, $4.38 | 90.0%, $7.97 | 89.6%, $6.32 | 89.2%, $4.38 |
| haiku/free | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 |
| haiku/Jev | 90.0%, $7.56 | 88.8%, $3.71 | 87.9%, $1.66 | 90.0%, $7.56 | 88.8%, $2.55 | 85.8%, $1.12 |
| haiku/Sonnet | 90.0%, $8.54 | 90.0%, $8.54 | 87.9%, $6.91 | 90.0%, $8.54 | 90.0%, $8.54 | 86.7%, $6.41 |
| E/free | 90.0%, $8.54 | 90.4%, $8.39 | 85.8%, $4.40 | 90.0%, $8.54 | 90.4%, $8.39 | 90.4%, $8.39 |
| E/Jev | 90.0%, $8.54 | 88.8%, $3.61 | 88.8%, $3.61 | 90.0%, $8.54 | 89.2%, $6.22 | 88.8%, $3.61 |
| E/Sonnet | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 |
| F/free | 90.0%, $8.54 | 90.0%, $8.22 | 90.0%, $7.13 | 90.0%, $8.54 | 90.0%, $8.54 | 90.4%, $7.25 |
| F/Jev | 90.0%, $8.54 | 90.0%, $8.54 | 88.8%, $3.66 | 90.0%, $8.54 | 88.8%, $3.66 | 88.8%, $3.66 |
| F/Sonnet | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 |
| G/free | 90.0%, $7.77 | 89.2%, $5.28 | 86.2%, $1.64 | 90.0%, $7.77 | 87.9%, $3.66 | 86.2%, $1.64 |
| G/Jev | 88.8%, $3.67 | 88.8%, $3.67 | 88.8%, $3.67 | 90.0%, $7.77 | 88.8%, $3.67 | 88.8%, $3.67 |
| G/Sonnet | 90.0%, $8.51 | 90.0%, $8.19 | 90.0%, $8.73 | 90.0%, $8.51 | 90.0%, $8.19 | 90.0%, $8.19 |
| G+/free | 90.0%, $8.54 | 89.6%, $6.32 | 89.2%, $4.38 | 90.0%, $7.97 | 89.6%, $6.32 | 89.2%, $4.38 |
| G+/Jev | 90.0%, $8.54 | 88.8%, $4.01 | 88.8%, $4.01 | 90.0%, $7.98 | 88.8%, $4.01 | 88.8%, $4.01 |
| G+/Sonnet | 90.0%, $8.22 | 90.0%, $8.22 | 90.0%, $8.22 | 90.0%, $8.54 | 90.0%, $8.54 | 90.0%, $8.54 |

#### MMLU-Pro, per task

MMLU-Pro has 240 test items. Each cell is the policy's quality and cost per 1k on this task alone, at the operating point the mix and target froze; intervals are in figures/tables_test.json.

| Policy | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| Frontier only | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 |
| E Rules | 92.9%, $7.29 | 90.8%, $3.93 | 90.8%, $3.93 | 92.9%, $7.29 | 90.8%, $3.93 | 90.8%, $3.93 |
| F Trained | 92.9%, $7.29 | 92.1%, $5.82 | 92.9%, $4.14 | 92.9%, $7.29 | 92.9%, $7.29 | 92.5%, $4.47 |
| G Jev | 92.5%, $7.17 | 92.1%, $6.50 | 90.0%, $4.46 | 92.5%, $7.17 | 92.5%, $6.03 | 90.0%, $4.46 |
| G+ Bolstered Jev | 92.9%, $7.29 | 92.1%, $4.86 | 91.7%, $2.29 | 92.5%, $7.03 | 92.1%, $4.86 | 91.7%, $2.29 |
| haiku/free | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 |
| haiku/Jev | 92.9%, $7.46 | 92.5%, $4.10 | 88.8%, $2.14 | 92.9%, $7.46 | 89.6%, $2.80 | 88.3%, $1.37 |
| haiku/Sonnet | 92.9%, $7.29 | 92.9%, $7.29 | 90.8%, $14.76 | 92.9%, $7.29 | 92.9%, $7.29 | 88.3%, $14.08 |
| E/free | 92.9%, $7.29 | 90.8%, $3.93 | 90.8%, $3.93 | 92.9%, $7.29 | 90.8%, $3.93 | 90.8%, $3.93 |
| E/Jev | 92.9%, $7.29 | 92.5%, $3.93 | 92.5%, $3.93 | 92.9%, $7.29 | 92.9%, $7.26 | 92.5%, $3.93 |
| E/Sonnet | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 |
| F/free | 92.9%, $7.29 | 92.1%, $5.82 | 92.9%, $4.14 | 92.9%, $7.29 | 92.9%, $7.29 | 92.5%, $4.47 |
| F/Jev | 92.9%, $7.29 | 92.9%, $7.11 | 92.5%, $3.93 | 92.9%, $7.29 | 92.5%, $3.93 | 92.5%, $3.93 |
| F/Sonnet | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 |
| G/free | 92.5%, $7.17 | 92.1%, $6.50 | 90.0%, $4.46 | 92.5%, $7.17 | 92.5%, $6.03 | 90.0%, $4.46 |
| G/Jev | 92.9%, $5.47 | 92.5%, $4.08 | 92.5%, $4.08 | 92.5%, $7.17 | 92.5%, $4.08 | 92.5%, $4.08 |
| G/Sonnet | 92.5%, $8.00 | 92.5%, $7.72 | 92.5%, $10.38 | 92.5%, $8.00 | 92.5%, $7.72 | 92.5%, $7.72 |
| G+/free | 92.9%, $7.29 | 92.1%, $4.86 | 91.7%, $2.29 | 92.5%, $7.03 | 92.1%, $4.86 | 91.7%, $2.29 |
| G+/Jev | 92.9%, $7.29 | 92.5%, $4.29 | 92.5%, $4.29 | 92.9%, $7.12 | 92.5%, $4.29 | 92.5%, $4.29 |
| G+/Sonnet | 92.5%, $11.48 | 92.5%, $11.48 | 92.5%, $11.48 | 92.9%, $7.29 | 92.9%, $7.29 | 92.9%, $7.29 |

#### MBPP+, per task

MBPP+ has 107 test items. Only 107 test items, below the plan's minimum of a few hundred: indicative. Each cell is the policy's quality and cost per 1k on this task alone, at the operating point the mix and target froze; intervals are in figures/tables_test.json.

| Policy | realistic 100% | realistic 99% | realistic 97% | uniform 100% | uniform 99% | uniform 97% |
| --- | --- | --- | --- | --- | --- | --- |
| Frontier only | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 |
| E Rules | 90.7%, $4.21 | 90.7%, $4.21 | 88.8%, $0.93 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $1.96 |
| F Trained | 90.7%, $4.21 | 90.7%, $3.68 | 90.7%, $2.67 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $3.03 |
| G Jev | 89.7%, $3.13 | 89.7%, $2.60 | 90.7%, $1.14 | 89.7%, $3.13 | 90.7%, $2.10 | 90.7%, $1.14 |
| G+ Bolstered Jev | 90.7%, $4.21 | 89.7%, $2.72 | 89.7%, $1.11 | 89.7%, $4.39 | 89.7%, $2.72 | 89.7%, $1.11 |
| haiku/free | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 |
| haiku/Jev | 90.7%, $4.23 | 90.7%, $1.12 | 89.7%, $0.54 | 90.7%, $4.23 | 90.7%, $0.65 | 89.7%, $0.49 |
| haiku/Sonnet | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $7.00 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $7.00 |
| E/free | 90.7%, $4.21 | 90.7%, $4.21 | 89.7%, $0.96 | 90.7%, $4.21 | 90.7%, $4.21 | 91.6%, $1.98 |
| E/Jev | 90.7%, $4.21 | 90.7%, $1.12 | 90.7%, $1.12 | 90.7%, $4.21 | 89.7%, $1.72 | 90.7%, $1.12 |
| E/Sonnet | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 |
| F/free | 90.7%, $4.21 | 90.7%, $3.68 | 90.7%, $2.67 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $3.03 |
| F/Jev | 90.7%, $4.21 | 90.7%, $3.98 | 90.7%, $1.12 | 90.7%, $4.21 | 90.7%, $1.12 | 90.7%, $1.12 |
| F/Sonnet | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 |
| G/free | 90.7%, $3.15 | 90.7%, $2.62 | 91.6%, $1.16 | 90.7%, $3.15 | 91.6%, $2.12 | 91.6%, $1.16 |
| G/Jev | 91.6%, $1.70 | 90.7%, $1.14 | 90.7%, $1.14 | 90.7%, $3.17 | 90.7%, $1.14 | 90.7%, $1.14 |
| G/Sonnet | 90.7%, $5.51 | 91.6%, $5.08 | 90.7%, $6.17 | 90.7%, $5.51 | 91.6%, $5.08 | 91.6%, $5.08 |
| G+/free | 90.7%, $4.21 | 90.7%, $2.74 | 90.7%, $1.14 | 90.7%, $4.42 | 90.7%, $2.74 | 90.7%, $1.14 |
| G+/Jev | 90.7%, $4.21 | 90.7%, $1.47 | 90.7%, $1.47 | 90.7%, $4.42 | 90.7%, $1.47 | 90.7%, $1.47 |
| G+/Sonnet | 90.7%, $3.99 | 90.7%, $3.99 | 90.7%, $3.99 | 90.7%, $4.21 | 90.7%, $4.21 | 90.7%, $4.21 |

### The hindsight oracle and the random-mixture envelope

On the realistic mix the envelope runs from B (haiku) at 87.8% and $0.26 per 1k to A at 90.9% and $7.51 per 1k, and the oracle reaches at most 92.8%; 46 items fail on every config. On the uniform mix the envelope runs from B (haiku) at 85.8% and $0.42 per 1k to A at 91.6% and $11.76 per 1k, and the oracle reaches at most 93.8%; 46 items fail on every config.

At each quality target the oracle's cost per 1k and the share of A's cost it saves, and the cost per 1k of the random mixture that reaches the target with the envelope's fraction of the oracle's savings it captures (95% interval). Each policy's captured fraction at its operating points is in the last column of the table of every policy above.

| Mix | Target | Target quality | Oracle cost per 1k | Oracle saved against A | Envelope cost per 1k | Envelope captured |
| --- | --- | --- | --- | --- | --- | --- |
| realistic | 100% | 90.9% | $0.36 | 95.2% | $7.51 | 0.0% [0.0%, 82.4%] |
| realistic | 99% | 90.0% | $0.31 | 95.8% | $5.34 | 30.1% [18.2%, 100.0%] |
| realistic | 97% | 88.1% | $0.26 | 96.5% | $1.01 | 89.7% [53.0%, 100.0%] |
| uniform | 100% | 91.6% | $0.68 | 94.2% | $11.76 | 0.0% [0.0%, 35.1%] |
| uniform | 99% | 90.7% | $0.61 | 94.8% | $8.44 | 29.7% [16.0%, 58.6%] |
| uniform | 97% | 88.9% | $0.50 | 95.7% | $4.82 | 61.6% [45.0%, 82.8%] |

Built from the test matrix as on dev (A8 item 4). Test has no repeat runs, so dev's flip rates carry over, and the oracle curve is an optimistic upper
bound (A6 item 2).

### The live replay

The replay ran A, haiku/Jev realistic 99% and haiku/Jev realistic 97% end to end through the standard API on 100 items (25 Banking77, 25 IFEval, 25 MBPP+, 25 MMLU-Pro), drawn at the unlock. The machine was Apple M5 Pro, Darwin 27.0.0 arm64, Python 3.13.13; the network was Wi-Fi (en0) behind a private gateway, no network name or address recorded. It started at 2026-11-20T05:39:21+00:00 and ended at 2026-11-20T06:10:18+00:00.

Quality is the share of the replay's answers graded correct, and the offline estimate is the same policy on the same items from test's matrix. Cost per request is at list price.

| Policy | n | Quality, live | Quality, offline | Cost per request, live | Cost per request, offline | Escalation, live | Escalation, offline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 100 | 82% | 81% | $0.01093 | $0.01133 | 0% | 0% |
| haiku/Jev realistic 99% | 100 | 83% | 83% | $0.00690 | $0.00786 | 46% | 48% |
| haiku/Jev realistic 97% | 100 | 82% | 79% | $0.00484 | $0.00457 | 22% | 21% |

Latency is from the request to the answer returned. A policy's first call and any request a crash cut are set apart and not timed, so the requests timed are fewer than n where there is one; the offline estimate is from dev's committed medians (A17 item 16).

| Policy | Requests timed | Median, live (s) | Median, offline (s) | p90, live (s) | p90, offline (s) |
| --- | --- | --- | --- | --- | --- |
| A | 99 | 2.46 | 3.95 | 14.90 | 14.00 |
| haiku/Jev realistic 99% | 99 | 3.49 | 4.26 | 21.53 | 20.96 |
| haiku/Jev realistic 97% | 100 | 2.57 | 4.06 | 14.80 | 20.96 |

The first call, the reconnects and the requests a crash cut, reported apart: A: first call 2.49 s, reconnects none, cut by a crash 0; haiku/Jev realistic 99%: first call 2.11 s, reconnects none, cut by a crash 0; haiku/Jev realistic 97%: first call none, reconnects none, cut by a crash 0.

Quality by task, with each side's n.

| Policy | Task | n, live | Quality, live | n, offline | Quality, offline |
| --- | --- | --- | --- | --- | --- |
| A | Banking77 | 25 | 76% | 25 | 72% |
| A | IFEval | 25 | 84% | 25 | 84% |
| A | MBPP+ | 25 | 84% | 25 | 84% |
| A | MMLU-Pro | 25 | 84% | 25 | 84% |
| haiku/Jev realistic 99% | Banking77 | 25 | 76% | 25 | 72% |
| haiku/Jev realistic 99% | IFEval | 25 | 88% | 25 | 84% |
| haiku/Jev realistic 99% | MBPP+ | 25 | 84% | 25 | 88% |
| haiku/Jev realistic 99% | MMLU-Pro | 25 | 84% | 25 | 88% |
| haiku/Jev realistic 97% | Banking77 | 25 | 72% | 25 | 72% |
| haiku/Jev realistic 97% | IFEval | 25 | 80% | 25 | 76% |
| haiku/Jev realistic 97% | MBPP+ | 25 | 88% | 25 | 84% |
| haiku/Jev realistic 97% | MMLU-Pro | 25 | 88% | 25 | 84% |

The answers are shown, not judged (A17 item 26.5), and its first call and any reconnects are reported apart from the rest.

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
