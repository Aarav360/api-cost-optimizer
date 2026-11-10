# API Cost Optimizer

How much can you cut LLM inference cost without materially reducing answer quality, and which
decision strategy gets closest to the best possible tradeoff?

This project measures that on Anthropic's Claude 5.5 models. It compares routers (which pick a
model per request) and cascades (which try a cheap model first and escalate when a quality gate
fails) against two references: a hindsight oracle that shows the most any strategy could save,
and random mixtures that show what you get with no routing intelligence at all.

**Status:** Stages 0–6 are done. The pilot chose four tasks ([Amendment A3](PLAN_v4.md)), the
benchmark of 2,478 items is split and frozen ([Amendment A5](PLAN_v4.md)), and the collection run
has answered every item through every config for $29.37: train and dev are scored, and test is
sealed until Stage 8 ([Amendment A6](PLAN_v4.md)). Stage 3 built the offline simulator, the four
references and a paired bootstrap, and drew Pareto chart v0 on dev ([Amendment A8](PLAN_v4.md),
whose item 12 records its headline numbers). Stage 4 built the routers' shared parts (the request a
router sees, router-score tables, the walk that picks thresholds, McNemar's test) and the first
router, E, a set of hand-written rules whose router scores are train success rates. E was frozen
before any dev number was computed, then walked on dev: chart and table v1 show its steps and its
six operating points, and [Amendment A9](PLAN_v4.md) item 18 records them with the hash of E's
section of the operating-points file. Amendment A10, from the branch review, adds the decision rule
to the freeze and records how E's frozen cues read A9. Stage 5 built F, the trained router: a local,
pinned embedding model (`gte-modernbert-base`) reads each request with its template lines removed, and
one logistic regression per config below the frontier turns the embedding into router scores. F is
cross-fitted: five fold models, each learning from train and four of dev's five folds, give every dev
item an out-of-fold router score, and test will be scored by the fold model a hash of the request
picks ([Amendment A11](PLAN_v4.md)). It was frozen in two commits, the settings before any dev label
was read and then the fold weights (items 19 and 20), and walked on dev like E. F-train, the task-rates
diagnostic, within-task concordance and a learning curve explain F's result and are diagnostics, not
policies. Chart and table v2 put F beside E, and item 21 records F's operating points, the
comparisons and the diagnostics' headline numbers. At matched cost F's paired difference from the
random-mixture envelope includes 0 at each of its six operating points, as E's does. Stage 6 built
two routers on Jev, TypeSafe's decision model (`jev-1.14.0`, one call per request each): G, the Jev
router, asks it zero-shot one success statement per config below the frontier (that a small, a
mid-sized or a large config will answer the request correctly, with no names, efforts or rates) and
uses its answers as router scores, and G+, the bolstered Jev router, shows it 16 past
train requests with their outcomes, asks 18 questions and turns the answers into router scores with
one logistic regression per config, fitted on train alone ([Amendment A15](PLAN_v4.md)). The settings
were frozen in one commit before any Jev call on a benchmark request and G+'s weights in another
before any dev call (items 13 and 19), and the dev collection came after both. Chart and table v3 put
G and G+ beside E and F, with their costs including Jev's overhead on every request, and item 20
records their operating points, the hashes of their sections and the first dev call's server `Date`.
At the 97% target on the realistic mix G costs $3.62 per 1,000 requests and G+ $2.81, against A's $8.04,
with Jev's overhead in both. Within-task concordance on `haiku` is 0.696 for G and 0.757 for G+, against
F's 0.571 (diagnostics, dev only). At matched cost G's paired difference from the random-mixture
envelope excludes 0 at 0 of its six operating points and G+'s at 1 (uniform mix, 97% target: +2.0
points, interval +0.7 to +3.3), a descriptive reading with no multiplicity correction. Stage 7, gates
and cascades, is next. Every result so far is on dev, with the test split still sealed.

![Pareto chart v3 on dev: the four fixed configs with their intervals, the random-mixture envelope, the hindsight oracle, the rules router E, the trained router F, the Jev router G and the bolstered Jev router G+, each with its steps as a line and a cross at each of its operating points, under the realistic and uniform task mixes](figures/pareto.png)

For the same numbers as an interactive page, with routers and references to toggle, zoom and a readout for
every point, open [`figures/pareto_explorer.html`](figures/pareto_explorer.html) in a browser (it loads Plotly
from jsDelivr).

## Approach

- **One results matrix, then offline evaluation.** Every benchmark item runs once through every
  model config. Routers and cascades are then scored by lookup, with no new API calls, so every
  strategy is compared on exactly the same answers.
- **References, not just baselines.** Besides always-frontier and always-cheapest, the matrix
  gives a hindsight oracle (a multiple-choice knapsack relaxed to a linear program) and the
  envelope of random mixtures. Systems report the fraction of the oracle's savings they capture
  at the same quality target.
- **Routers, simplest first:** a rules baseline, then a classifier trained on prompt embeddings,
  then Jev (TypeSafe's decision model, zero-shot and bolstered). Routers are compared at matched cost or matched quality,
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

The pilot ran 20 items from each candidate through every config, plus 20 MATH Level 5 items,
and kept four task types. MATH and CRUXEval were dropped because every config answered every
item correctly, which leaves no routing decision to study. IFBench and LiveCodeBench were then
tested as ways to add items to the two smallest tasks and left out ([Amendment A4](PLAN_v4.md)):
LiveCodeBench was saturated too, and IFBench scored every config about the same while drawing
refusals.

| Dataset | Task type | Scoring | License | Pilot outcome |
|---|---|---|---|---|
| [IFEval](https://huggingface.co/datasets/google/IFEval) | instruction following | strict instruction checks | Apache 2.0 | **kept** |
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) ([copy used](https://huggingface.co/datasets/mteb/banking77)) | classification | exact label | CC BY 4.0 | **kept**, JSON answer |
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | multiple choice | answer letter | MIT | **kept**, JSON answer |
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | coding | run the tests | Apache 2.0 | **kept** |
| [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | math | answer equivalence | MIT | dropped: saturated, Level 5 too |
| [CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval) | code reasoning (output prediction) | Python literal | MIT | dropped: saturated |
| [IFBench](https://huggingface.co/datasets/allenai/IFBench_test) | instruction following | strict instruction checks | ODC-BY | not added: no separation, refusals |
| [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite) v5–v6 | coding | run the tests | see card | not added: saturated |

Dataset revisions are pinned in [harness/loaders.py](harness/loaders.py), so an item ID always
means the same item.

## Running it

Requires [uv](https://docs.astral.sh/uv/), Docker (for the code sandbox), and an Anthropic API
key in a `.env` file at the project root (`ANTHROPIC_API_KEY=...`; it's gitignored).

```bash
uv sync
git config core.hooksPath .githooks        # once per clone: run the fast checks before every commit
.githooks/pre-commit                       # the fast checks: ruff, mypy, check_sealed and the analysis checks below
uv run python -m harness.check_sealed      # no code but harness/collect.py names the sealed test store, and only allowlisted files name the benchmark
uv run python -m harness.smoke_test        # one Haiku call: checks the key and the response store
uv run python -m harness.sandbox --build   # build the pinned sandbox image
uv run python -m harness.check_sandbox     # escape checks + every MBPP+ reference solution
uv run python -m harness.build_pilot       # regenerate the pilot files (same seed, same items)
uv run python -m harness.check_evaluators  # every evaluator against the reference answers (needs the sandbox)
uv run python -m harness.run_pilot --configs haiku --per-source 3   # 18-item trial
uv run python -m harness.run_pilot         # full pilot: 120 items x 4 configs, then the Stage 0 summary
uv run python -m harness.run_pilot --pilot-file data/pilot_math_level5.jsonl --formats structured
uv run python -m harness.run_pilot --pilot-file data/pilot_added_sources.jsonl --formats free
uv run python -m harness.build_benchmark   # rebuild the frozen benchmark (seeded; byte-identical)
uv run python -m harness.check_benchmark   # verify it against its manifest hashes
uv run python -m harness.check_collect     # the collection runner end to end against a fake Batches API (no calls)
uv run python -m harness.collect plan      # Stage 2 request counts and projected cost; then submit, wait, score, seal
uv run python -m analysis.fit_rules_router # fit E on train and write routers/rules_rates.json (then frozen, A9 item 17)
uv run python -m analysis.rules_table      # E's dev router-score table: score(request) on every dev request
uv run python -m analysis.dev_diagnostic   # the dev task-inference diagnostic, on this benchmark's templates
uv run python -m analysis.rules_walk       # the walk on dev: writes E's section of routers/operating_points.json
uv run python -m analysis.pareto           # chart v3 on dev: E's, F's, G's and G+'s steps and operating points, tables v0 and v1, the figures and their JSONs
uv run python -m analysis.check_simulator  # the analysis package on hand-worked cases, then on dev against the figures and A8's record
uv run python -m analysis.check_routers    # the router foundation, the benchmark reader and the router-score table
uv run python -m analysis.check_comparisons  # McNemar's test and the matched-cost difference
uv run python -m analysis.check_walk       # breakpoints, steps, the walk and the operating-points file, on synthetic tables
uv run python -m analysis.check_metrics    # within-task concordance, log loss, the same-target comparison and the operating-points allowlist
uv run python -m analysis.check_rules_router  # E's cues, fit and freeze hashes (A9 item 17), and A10's record against the frozen code
uv run python -m analysis.check_chart_routers  # the chart script's router list: the four routers' entries and styles, two routers drawn
uv run python -m analysis.check_jev_names  # G's and G+'s names, the allowlist accepting their sections, and their two chart styles
uv run python -m analysis.fetch_model      # once: the five embedding-model files at A11's pinned commit, into .cache/ (the router never fetches)
uv run python -m analysis.check_trained_router  # the trained router's foundation, on a fake backend; with the snapshot, also the real tokenizer's window and that the snapshot directory holds only the five pinned files
uv run python -m analysis.fit_trained_router caches    # F's embedding caches of train and dev requests, into results/trained_embeddings/ (no outcome read)
uv run python -m analysis.fit_trained_router settings  # F's settings and F-train's weights, from train alone (A11 items 5, 7, 12)
uv run python -m analysis.measure_trained_router drift | latency | environment | tokens | record [--write]  # the drift limit, latency and item 19 (run latency alone)
uv run python -m analysis.fit_trained_router folds     # the fit commit only: the five fold models (reads dev outcomes, prints nothing about them)
uv run python -m analysis.check_trained_freeze  # F's train-side fit on synthetic data, and A11 item 19 against its sources; the hook adds the snapshot and train matrix
uv run python -m analysis.check_trained_fit  # A11 item 20 against the fold weights; the hook adds the history and the refit to 1e-9
uv run python -m analysis.trained_table   # F's dev router-score table (through score_with: embeds the 744 dev requests, about 20 s) and F-train's, and each fold model's quantiles (no outcome read)
uv run python -m analysis.trained_walk    # F's walks on dev, and F's section of routers/operating_points.json (E's untouched)
uv run python -m analysis.check_trained_dev  # F and F-train on dev, F's table against the cache's vectors; the hook only, it needs the snapshot, dev cache and matrix
uv run python -m analysis.learning_curve  # the learning curve's numbers on dev: F refit at 25, 50, 75 and 100% of its rows, F-train beside, the 75% to 100% difference (no walk)
uv run python -m analysis.check_learning_curve  # the learning curve's subsamples and refits on synthetic rows; the hook adds dev: the 100% point equals F's router scores exactly
uv run python -m analysis.trained_diagnostics  # F-train and the task rates walked, within-task concordance, log loss, same-target comparisons (nothing written)
uv run python -m analysis.check_trained_diagnostics  # the diagnostics against hand-worked and independent counts; the hook only
uv run python -m analysis.check_stage4     # E on dev, tables v0 and v1, A9's record (item 18) and the Stage 3 numbers A9 quotes, against the JSONs
uv run python -m analysis.table_v2         # table v2's additions (same-target comparisons, diagnostics, per-fold quantiles) and the learning-curve figure: writes figures/table_v2.json, learning_curve.json and .png
uv run python -m analysis.check_stage5     # chart and table v2, the learning-curve figure and A11's record of F (item 21) against the committed files; the hook adds the rebuild from the dev matrix
uv run python -m analysis.build_jev_settings  # the Jev settings from train alone: past requests, rates, noise sample, question texts -> routers/jev_settings.json (committed with the call-building code before any benchmark call, A15 item 13)
uv run python -m analysis.smoke_jev        # before the settings commit: made-up requests only through G and G+, and the measured token overheads and byte budgets
uv run python -m analysis.collect_jev collect --router both --split train  # one Jev call per request per router into results/jev/ (resumable; also --split dev after the freeze commit; refuses test); also noise, report, budget; needs TYPESAFE_API_KEY in .env
uv run python -m analysis.fit_jev_router   # G+'s fit on train's stored answers: C by cross-validation, the signal check and D6, then routers/jev_weights.json (the freeze commit, A15 item 19)
uv run python -m analysis.jev_train_report # what the train collection left behind: counts, tokens, latency, the first call's Date, body hashes
uv run python -m analysis.jev_table        # G's and G+'s dev router-score tables, through the router's own parser on the stored dev answers, and their quantiles (no outcome read)
uv run python -m analysis.jev_walk         # G's and G+'s walks on dev and their sections jev and jev_bolstered of routers/operating_points.json
uv run python -m analysis.jev_comparisons  # G's and G+'s same-target comparisons, matched-cost rows, McNemar rows, within-task concordance and dev latency (nothing written)
uv run python -m analysis.table_v3         # table v3: G's and G+'s operating points, the comparisons, the diagnostics and dev latency, into figures/table_v3.json
uv run python -m analysis.explorer         # the interactive explorer: chart v3's two JSONs into figures/pareto_explorer.html (rerun after pareto)
uv run python -m analysis.check_jev_settings  # the settings builder on a synthetic benchmark; the hook compares the committed file with a rebuild
uv run python -m analysis.check_jev_call   # call bodies equal hand-written JSON, the byte budget's cut, malformed settings refused
uv run python -m analysis.check_jev_client # the client's retries and G's scoring against a fake Jev server on a loopback port (no network)
uv run python -m analysis.check_jev_plus   # G+'s features, clipping and numpy heads against scikit-learn's predict_proba
uv run python -m analysis.check_jev_collect  # G+ end to end and the collection script, against the fake server and a temporary results folder
uv run python -m analysis.check_jev_fit    # G+'s fit, the train signal check and the noise measure, on synthetic data
uv run python -m analysis.check_jev_settings_commit  # the settings commit's promises: budgets set, the file rebuilt from its seed, the commit's files unchanged
uv run python -m analysis.check_jev_freeze # A15 item 19 against the committed files and git history; the hook adds the stored train responses and a refit to 1e-9
uv run python -m analysis.check_jev_dev    # G's and G+'s dev tables, quantiles and walks; the hook adds the stored dev responses and the freeze commit's order
uv run python -m analysis.check_jev_comparisons  # the comparisons on a hand-worked synthetic matrix; the hook adds dev and the stored records
uv run python -m analysis.check_stage6     # chart and table v3 and A15's record of G and G+ (item 20) against the committed files; the hook adds the rebuild from the dev matrix and the stored records
uv run python -m analysis.check_explorer   # the explorer against chart v3's JSONs: the committed page is their build, and its numbers and words are theirs
```

The matrix-reading commands need the dev results matrix in `results/`, so they need the collection
run's stores; so do `check_simulator`, `check_stage4` and the F checks marked above as hook-only, which the
hook runs where the matrix is present, and the refit half of `check_rules_router`. The Jev commands
that read stored answers also need `results/jev/`, which holds each call's whole response and is never
committed. No check calls Jev: the checks use a fake Jev server (`analysis/fake_jev.py`) on a loopback
port, and only `collect_jev` (and `smoke_jev` before the settings commit) sends a request, with the key
from `.env`. Chart v3 takes about half a minute, and `table_v3` and `check_stage6` (with its rebuild)
a minute or two each. Chart v2 took about 15 seconds, and `table_v2` about 30 more for the learning curve's 60 refits: the paired bootstrap
re-solves the hindsight oracle on every resample. Run the three E steps in the order `rules_table`,
`dev_diagnostic`, `rules_walk` (A9 item 13), which is also the order of their commits. F's steps follow
A11 item 15: `fetch_model`, `fit_trained_router caches` and `settings`, the freeze commit, `fit_trained_router
folds` (the fit commit), then `trained_table`, `trained_walk`, the diagnostics, `pareto` and `table_v2`.
The diagnostics' walks stay in memory and never enter the operating-points file. G's and G+'s steps
follow A15 item 13: `build_jev_settings` and `smoke_jev` (made-up requests only), the settings commit,
`collect_jev` on train, the noise pass, `fit_jev_router`, the freeze commit, `collect_jev` on dev,
`jev_table`, `jev_walk`, `jev_comparisons`, then `pareto`, `table_v3` and the record (item 20).

The first LiveCodeBench load downloads about 690 MB and writes its decoded tests (about 500 MB)
to `.cache/`.

Every answer is scored right or wrong (`quality` 1 or 0), and IFEval counts as right only when
every instruction is followed. An answer missing the format its prompt asks for (`Answer: X`,
`\boxed{}`, a code block) scores as a format failure, which is counted separately from a wrong
answer.

The pilot ran Banking77, MMLU-Pro, MATH and CRUXEval twice on the same items: once in free text,
and once with the answer requested as JSON through the API's structured outputs. JSON removed
every format failure without costing accuracy, so Banking77 and MMLU-Pro answer as JSON from
here on. IFEval and MBPP+ stay free text, since IFEval's instructions are about the text's shape
and code belongs in a code block.

Every API response is cached on disk under `results/` (not committed), keyed by a hash of the
full request, so re-running analysis never pays for the same call twice.

Model-written code runs in a fresh Docker container per execution: no network, a read-only file
system, none of your files, an unprivileged user, and caps on memory, processes, time and output.

## Layout

```text
PLAN_v4.md           study design (frozen before Stage 0, with amendments A1–A15)
GLOSSARY.md          the study's vocabulary from Stage 3 on: policy, trace, config ladder, reference, request,
                     router score, threshold, walk, operating point, ...
WRITEUP.md           the final write-up (to come) and running notes for it
CODING_STANDARDS.md  the judgement calls code review applies; ruff and mypy (pyproject.toml) cover the rest
.githooks/pre-commit the fast checks, run before every commit; CI runs them too (.github/workflows/)
configs/models.yaml  pinned model configs, output caps and price snapshot
configs/task_mix.yaml  the uniform and realistic task mixes
harness/             API client, response store, dataset loaders, evaluators, code sandbox, pilot
                     builder and runner, benchmark builder and checker, collection runner, check
                     and review scripts, LiveCodeBench test runner (unused; Amendment A4)
harness/ifeval/      Google's IFEval instruction checkers, vendored at a pinned commit
harness/ifbench/     AI2's IFBench instruction checkers, vendored at a pinned commit
analysis/            the offline simulator (traces to outcomes and totals), the results-matrix loader,
                     references C and D, the paired bootstrap, the benchmark reader, the router-score table,
                     the walk and operating points, McNemar's test, E's fit on train and its freeze hashes,
                     E's table, diagnostic and walk on dev, the trained router's fold builder, fit script and
                     freeze measurements, F's and F-train's tables and F's walk, the diagnostics and the
                     learning curve, G's and G+'s collection, fit, tables, walks and comparisons, chart v3,
                     tables v2 and v3, the interactive explorer's build and template (explorer.py,
                     explorer_template.html) and the check scripts (jev_*.py and collect_jev.py are Stage 6's;
                     fake_jev.py is the fake server the checks use)
routers/             the deployable routers (imports nothing from analysis or harness): the config ladder,
                     the request, the decision rule (frozen, A10 item 1), the rules router E (frozen, A9
                     item 17) and its rates file rules_rates.json, the trained router F in trained.py
                     (frozen, A11 item 19) with its settings trained_settings.json, F-train's weights
                     trained_f_train_weights.json and the five fold models' weights
                     trained_fold_weights.json (the fit, A11 item 20), the Jev routers G and G+ in jev.py
                     (the client, the parser, G's scores, G+'s features and numpy heads; frozen, A15 item 19)
                     with jev_call.py (builds each call's body; frozen at the settings commit),
                     jev_settings.json (question texts, past requests with outcomes, rates, noise sample)
                     and jev_weights.json (G+'s C, standardizer and three heads; the freeze commit), and
                     operating_points.json, E's, F's, G's and G+'s thresholds from the walk (never
                     F-train's or a diagnostic's)
figures/             chart v3 on dev: the headline and per-task figures, each a PNG with a JSON of every
                     number it plots, tables v0 and v1 included, E's, F's, G's and G+'s rows alike;
                     table_v2.json, the same-target comparisons and the diagnostics; learning_curve.png and
                     .json; table_v3.json, G's and G+'s operating points, comparisons and diagnostics;
                     pareto_explorer.html, chart v3's numbers as an interactive page
sandbox/             Dockerfile for the code sandbox image
data/benchmark.jsonl the frozen benchmark: 2,478 items split 40/30/30, plus its hash manifest
data/dev_folds.json  dev's five folds for cross-fitting F, and train_folds.json, train's five folds for choosing C
                     (item IDs only, built without any outcome; hashes in A11 item 19)
data/pilot*.jsonl    Stage 0 pilot items: 20 per candidate dataset, 20 MATH Level 5 items, and
                     20 each from IFBench and LiveCodeBench
data/collection_manifest.json  batch IDs, counts and the sealed test store's hash (Amendment A6)
results/             response store, dev matrix, Jev's stored responses (results/jev/) and sealed test store (not
                     committed)
.cache/              downloaded tool data and LiveCodeBench's decoded tests (not committed)
docs/history/        earlier versions of the plan
```

## Progress

- [x] Study plan frozen (v4, amendments A1–A15)
- [x] Harness: API client, on-disk response store, pinned model configs
- [x] Dataset loaders at pinned revisions, and 120 pilot items
- [x] Docker code sandbox (18/18 escape checks; 377/377 MBPP+ reference solutions pass under the
      grader's limits, with 1 excluded)
- [x] Evaluators for the six candidate datasets, checked against every reference answer in the
      candidate pools
- [x] Pilot runner, tested on an 18-item Haiku trial
- [x] Pilot run ($4.72): four tasks chosen, JSON answers for Banking77 and MMLU-Pro, output caps
      set, collection run projected at about $26–30 against the $150 ceiling
- [x] IFBench and LiveCodeBench tested as extra items for the two smallest tasks ($3.89); not added
- [x] Stage 1: benchmark frozen. IFEval 521, Banking77 800, MMLU-Pro 800 and MBPP+ 357 items, split
      40/30/30 and stratified, with a noise subset, the two task mixes and hashes
- [x] Stage 2: collection run ($29.37); train and dev scored, test sealed
- [x] Stage 3: offline simulator, references A–D (the random-mixture envelope and the hindsight
      oracle as an LP), the paired bootstrap and safety margin, and Pareto chart v0 on dev with the
      per-task figure and table v0
- [x] Stage 4: the shared router parts, the rules router E fitted on train and frozen, then on dev
      its router-score table, task-inference diagnostic and walk; chart and table v1 with E's six
      operating points, and A9's record of them (item 18); A10 adds the decision rule to the freeze
- [x] Stage 5: the trained router F (a pinned local embedding model and logistic regressions),
      cross-fitted on dev in five folds and frozen in two commits (A11 items 19 and 20); its walk and
      operating points on dev; the diagnostics (F-train, the task-rates diagnostic, within-task
      concordance, a learning curve); chart and table v2 and the learning-curve figure; A11's record
      of F (item 21)
- [x] Stage 6: Jev's routers G (zero-shot) and G+ (bolstered, calibrated on train), one call per request
      each; the settings frozen before any Jev call on a benchmark request and G+'s weights before any
      dev call (A15 items 13 and 19); their dev tables, walks and operating points; the comparisons and
      diagnostics; chart and table v3; A15's record of G and G+ (item 20)
- [ ] Stages 7–8: gates and cascades, final evaluation on the sealed test split

## License

The code and documentation are released under the [MIT License](LICENSE), except
`harness/ifeval/` (Google's code) and `harness/ifbench/` (AI2's code), which stay under the Apache
License 2.0 (a copy is in each folder). Dataset items in
`data/`, and anything the loaders download, remain under their original licenses, listed in the
[pilot datasets table](#pilot-candidate-datasets) above.
