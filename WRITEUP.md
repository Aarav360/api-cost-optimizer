# Write Up

<!-- The final write-up goes here. -->

## Notes

Topics to come back to for the write-up, newest last. Details are in PLAN_v4.md's amendments.

### 2026-10-16 · Measurement validity
- Only objective graders, no LLM judge. Each was checked against every reference answer in its candidate dataset (over 90,000 answers and variants).
- A format failure is counted separately from a wrong answer.
- Known item noise is kept rather than corrected: ambiguous Banking77 labels, some wrong MMLU-Pro answer keys, quirky MBPP+ tests.
- Talking point: accuracy ceilings below 100% partly reflect data noise, which lowers every config alike.

### 2026-10-16 · Free-text answers understate cheap models (Amendment A3)
- Four tasks ran in free text and as JSON (structured outputs) on the same items. JSON was right where free text was wrong 13 times, and the reverse twice. Format failures fell from 10 to 0.
- All six of Haiku's MATH "failures" had the right answer, just without `\boxed{}`, so its score went from 14/20 to 20/20.
- Talking point: format slips can look like a quality gap and push routing toward expensive models.
- Caveat: there is no "reasoning" field in the schema, because one can trigger safety refusals.

### 2026-10-16 · Saturated tasks dropped (Amendment A3)
- Every config scored 20/20 on CRUXEval and on MATH, including 20 Level 5 MATH items tested under a rule fixed before the run.
- Talking point: a saturated task gives a router nothing to decide. But easy traffic is where real savings come from, so treat this as a limitation and note that the headline depends on the task mix.

### 2026-10-16 · Cheap model, long thinking (Amendment A4)
- On hard items, Haiku at high effort wrote about 3× as many output tokens as Opus, sometimes more than 30,000. Its output cap went up to 64,000, because hitting the cap counts as a failure.
- Talking point: a low price per token doesn't settle cost. Haiku was still about 13× cheaper per item here, but cost has to be measured per task.

### 2026-10-16 · "Newer" benchmarks weren't uncontaminated (negative result, Amendment A4)
- IFBench and LiveCodeBench v5–v6 were tested as extra items for the two small tasks, and not added.
- LiveCodeBench's 2024–25 contest problems were saturated: 20/20 for Sonnet and both Opus configs.
- IFBench didn't separate the configs: all scored 35–50%, with Haiku highest.
- Talking point: a benchmark's release date doesn't guarantee it postdates a model's training data.

### 2026-10-16 · Safety-classifier refusals on benign prompts
- 8 of 80 IFBench responses were refused under "bio", for example a question on particle concentrations for biosensors. Opus refused 4 of 20.
- Talking point: refusals are a real cost for routing and cascades in production. The study counts them as failures and turns refusal fallbacks off, so every answer comes from its own config.

### 2026-10-16 · Cross-fitting for the trained router (Stage 1 decision, applied in Stage 5)
- IFEval (521 items) and MBPP+ (357) are small. Shrinking dev or test widens the error bars; shrinking train handicaps the trained router on exactly those tasks.
- Kept the standard 40/30/30 split everywhere. The trained router learns from train plus all but one fold of dev, predicts that held-out fold, and rotates through the folds. Thresholds are set on those out-of-fold predictions, and test stays sealed.
- Considered: 20/40/40 and 30/35/35. Their error bars differed by only about ±1 point.
- Talking point: a learned router gets enough training data without loosening the sealed test.

### 2026-10-16 · Choosing the "realistic" task mix (Amendment A5)
- Headline savings depend heavily on the task mix, so both mixes were fixed before any results: uniform 25% each, and realistic MBPP+ 65 / Banking77 20 / IFEval 10 / MMLU-Pro 5, taken from Anthropic Economic Index API traffic shares.
- Cost combines as weighted dollars, then savings. Averaging per-task savings percentages would distort them, because tasks cost different amounts per request and cheap tasks are a small part of the bill.
- Caveat: the realistic mix is two-thirds coding, but MBPP+ is easier than real coding traffic, so that headline likely overstates savings.
- Talking point: predeclared weights from a published source rule out choosing them after seeing the numbers.

### 2026-10-18 · Run-to-run noise and the oracle (Amendment A6)
- These models reject `temperature`, so each stored answer is one draw. 75 dev items run three times through every config to measure how often an item's right/wrong result flips.
- The oracle is the most exposed: it sends each item to the cheapest config that happened to succeed, so a lucky draw counts as savings no router could rely on.
- Rule fixed before any data: flip rates are reported next to the CIs, and if any config flips on 5% or more of items, the oracle curve is called an optimistic upper bound.
- Talking point: a hindsight ceiling built from single samples overstates what's achievable when outputs are nondeterministic.

### 2026-10-18 · The collection run (Amendment A6)
- 9,521 requests through the Batches API in about 10 minutes, all succeeded first time, for $29.37 (projected $32.03). Test answers sealed with their hash recorded.
- Haiku and Opus-low flip on 5% or more of the noise items (6.7%, 5.3%), so the oracle is reported as an optimistic upper bound, as fixed beforehand.
- Both Opus configs refused a benign virology question ("where do Lassa and Ebola come from?") on every run, while Haiku and Sonnet answered it correctly.
- Talking point: here the cheapest model can answer what the frontier model won't, and a refusal costs a router the same as a wrong answer.

### 2026-10-19 · Opus at low effort matches Opus at medium on dev (Amendment A8)
- On dev, `opus_low` scored the same as `opus` (89.7% vs 89.5% uniform, 87.8% vs 86.7% realistic) for about 80% of the cost. The gap is likely noise.
- Kept `opus` as the frontier anyway: swapping it after seeing dev would be a post-hoc choice. The random-mixture envelope already credits the savings, so any router has to beat always-`opus_low`.
- Talking point: the effort setting is a cost lever in its own right. Medium effort bought no measurable quality on these tasks.

### 2026-10-19 · Which "standard error" sets the safety margin (Amendment A8)
- Thresholds must clear their quality target by one standard error, but the plan didn't say which. On the realistic mix a score's own standard error is about 2.4 points, while its gap from Opus on the same items has 1.2–1.8.
- The unpaired one would have put the realistic targets almost out of reach (above every model's dev quality at 100% and 99%). The paired one is right, because targets are relative to Opus on the same items.
- Caught in the Stage 3 design review, before any router existed.
- Talking point: in paired comparisons the luck of the item draw cancels out, and ignoring that throws away savings for no protection.

### 2026-10-19 · Sonnet loses to a weighted coin flip between Haiku and Opus-low (Amendment A8)
- On dev the random-mixture envelope is one straight line from `haiku` to `opus_low` under both mixes. Picking between those two at fixed odds, without reading the prompt, beats `sonnet` at its own cost: 85.8% vs 84.5% realistic, 87.6% vs 86.1% uniform.
- So `sonnet` adds nothing as a fixed choice, and any router that uses it has to earn its place on routing alone.
- Talking point: a mid-priced model can sit below a blend of the cheapest and a top config, so the reference to beat isn't always-cheapest but this envelope.

### 2026-10-20 · The hindsight oracle meets every target near Haiku's cost (Amendment A8)
- On dev the oracle starts at always-`haiku` and climbs to 91.5% realistic and 93.6% uniform, above `opus`, because 28 items are ones `opus` gets wrong and a cheaper config gets right.
- The 100/99/97% quality targets cost $0.40/$0.37/$0.31 per 1k realistic and $0.65/$0.59/$0.51 uniform, against $0.29 and $0.43 for always-`haiku`. A perfect router would need little more than Haiku's bill.
- Decision: an item's weight scales its cost as well as its score, because policies are scored on mix-weighted cost. Weighting only the score would reorder upgrades between tasks, and a brute-force check on random matrices shows that order misses the LP's optimum.
- Talking point: the oracle is an optimistic ceiling (single samples, with lucky draws counted as savings), so the gap between it and any real router is the price of not knowing in advance which items are easy.

### 2026-10-20 · Paired intervals on dev (Amendment A8, items 8 and 9)
- Each config's own 95% interval on the realistic mix spans about ±4.5 points, so every cross overlaps. Paired against `opus` on the same resamples, the picture sharpens: `haiku` is 3.4 points behind (−7.0 to −0.1), and `opus_low` is 1.0 ahead (−1.3 to +3.6), an interval that contains 0.
- Safety margins (one paired standard error) are 1.24–1.79 points realistic and 0.71–1.26 uniform, about half of each config's own standard error (2.2–2.5 realistic).
- The bootstrap's seed (2026) was fixed before any interval was seen, and not changed afterwards. A8 item 9's unpaired bars came out within 0.08 points of the design review's.
- Talking point: overlapping error bars don't mean "no difference". The paired interval is the one that answers "is this config worse than the frontier?".

### 2026-10-20 · Keeping the one-standard-error safety margin (Amendment A8, item 9)
- A router tuned to clear its quality target by one paired standard error still misses the target in about one resample in six. Its captured-fraction interval therefore usually reads "n/a to X", because a lower bound needs fewer than 2.5% of resamples to miss.
- Considered moving the margin to 1.5 or 2 standard errors. At 1.5 about one resample in 15 still misses, so the "n/a" stays, yet the 99% target already asks for more than any single config scores on dev. Only about 2 gives a numeric lower bound, at the cost of a lower operating point's savings and of operating points altogether at the 99% and 100% targets for weaker routers.
- Decision: keep one standard error, and report "met the target in X% of resamples" in a column of its own beside every captured fraction. Recorded under A8 item 9 before any router exists.
- Talking point: a wider safety margin doesn't make a router better. It hides its chance of missing behind a more cautious, more expensive operating point. Reporting the chance is more honest than buying it away.

### 2026-10-20 · Chart v0's per-task picture (Amendment A8, items 10 and 12)
- IFEval separates the configs most: `haiku` 83.4% against `opus` 93.6% on dev, and its own oracle reaches 98.1%. That's where a router has the most to do, though IFEval's 157 dev items make it indicative only (A7).
- On MMLU-Pro `haiku` is within 2.5 points of `opus` and above `sonnet` (86.7% vs 85.0%). On Banking77 it trails `opus` by 5.4 points.
- On MBPP+, 65% of the realistic mix, `opus_low` scores above `opus` (86.0% vs 84.1%). That's the source of the headline `opus_low` finding, on 107 items.
- Talking point: a mix's average hides where routing matters, which is why per-task results are always shown, even though claims rest on the mixes and on the two larger tasks.

### 2026-10-20 · Rules settled in the Stage 3 branch review (Amendment A8, items 4 and 7)
- Matched cost: the oracle's curve ends at its costliest corner, little more than Haiku's bill on dev, so any real router would have had no oracle quality to compare with at its cost. A cost past a reference's top corner now reaches its maximum quality, since a bigger budget needn't be spent.
- Traces: a policy must climb the config ladder (skipping is fine). Stepping down would pay for a dearer config and return a cheaper one's answer. A negative or non-finite overhead is refused too, since it would make a policy look cheaper.
- The checks on A8's numbers now read the plan itself, not copies typed into the check script, and they catch the old $6.56 → $6.57 slip.
- Talking point: rules the plan only described became refusals before any router existed, so no Stage 4–7 result can lean on a malformed trace.

### 2026-10-20 · Picking a router's threshold without fooling ourselves (Amendment A9, items 9 and 10)
- Every threshold is its own policy. Taking the cheapest one that clears its target on dev picks the luckiest of many, so on test it would miss more often than the one in six the safety margin allows.
- Decision: walk down from always-`opus` in a fixed order and stop at the first threshold that doesn't clear by the margin (fixed-sequence testing, as in Learn-then-Test). The steps are the points where some item's config changes, fixed from router scores before any outcome is looked up. Where nothing cheaper clears, the router falls back to always-`opus` and is reported as saving nothing.
- Rejected a fixed 0.05 grid. Dev's per-task success rates bunch between 0.82 and 0.94, so the rules router would get four policies, and four tasks would change config at once at t = 0.85.
- Talking point: trying many thresholds costs nothing only if their order is fixed in advance. The walk keeps the safety margin's promise however many steps it takes.

### 2026-10-20 · The rules router learns from train, not dev (Amendment A9, items 4, 5 and 13)
- The plan's sketch had hand-set numbers. Instead, each bucket scores each config by its success rate on train. Hand-set numbers come from looking somewhere, and if that's dev, dev sets both the numbers and the thresholds, and test misses more often.
- The rules router and the trained router now learn from the same train labels, so comparing them tests hand-written cues against embeddings.
- The rules freeze by hash, in a commit made before any rules-router number on dev exists, so anyone can check the order in the git log.
- Talking point: on this benchmark the rules will spot the task almost every time, because the prompts are templated. That says nothing about real traffic, and the write-up says so.

### 2026-10-20 · The rules router on dev: real savings, but no better than mixing configs at the same spend (Amendment A9, item 18)
- At the 100% target the walk certifies nothing cheaper than always-`opus`, so E falls back to A under both mixes. At 99% it saves 21% (realistic) and 23% (uniform). At 97% it saves 73% realistic, at 85.7% quality for $2.16 per 1k against A's $8.04.
- At matched cost, E's paired difference from the random-mixture envelope has an interval that includes 0 in every cell. The best is +1.0 points at 97% realistic (−1.4 to +3.0). Reading the prompt hasn't yet beaten ignoring it at the same spend.
- The one-standard-error margin behaves as designed. At 97% realistic only 84% of resamples meet the target, which is roughly the one-in-six miss rate A8 item 9 accepts.
- Talking point: a router that saves money can still fail to beat a coin-flip mix of configs at the same cost. The envelope comparison is what tells them apart, and it sets the bar for the trained and Jev routers.
