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
