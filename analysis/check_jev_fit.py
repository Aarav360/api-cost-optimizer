"""Check G+'s fit, the train signal check and the noise measure (PLAN_v4.md §8 3c+; Amendment A15, items 7, 10 and 19;
Amendment A11, items 7 and 8; Amendment A12, item 1; issue #37).

Run from the project root:  uv run python -m analysis.check_jev_fit

Synthetic data throughout: a synthetic benchmark, results store and stored records in #30's formats, in a temporary
folder. No Jev call, no test item and no real train response is read here (Stage 6 · 9 runs the fit on those).

- the pieces, hand-worked: within-task concordance, the train signal check (an intercept-only model, the task-rates
  model, G's raw router scores clipped for log loss, G+ cross-validated) and the noise measure (D6 and the shift);
- the model: a fit that doesn't converge is refused, the C choice is reproducible, and a C at either end of the grid is
  reported;
- the fit rows: train items outside the past requests' source groups, less the items Jev refused, both counted, with
  run-0 labels (refusals and `max_tokens` stops are failures in the store already);
- the weights file in #30's format: read by `routers/jev.py`'s own reader, and its numpy heads match scikit-learn's
  `predict_proba` to 1e-9;
- train's cells alone: a dev row that the whole-store loader refuses never reaches the fit, and a source check flags a
  script that loads the whole matrix (a known positive).
"""

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit, cross_val_predict

from analysis import check_jev_settings as synth
from analysis import fit_jev_router as fit
from analysis import matrix
from analysis.build_jev_settings import build_from_files, settings_text
from analysis.folds import build_folds
from routers import jev, jev_call
from routers.request import from_row

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object], error: type[Exception] = fit.JevFitError) -> bool:
    try:
        call()
    except error:
        return True
    return False


def near(got: float | None, want: float, tolerance: float = 1e-6) -> bool:
    return got is not None and abs(got - want) <= tolerance


OPTIONS = ("follow written instructions to produce text", "choose a label from a given list",
           "answer a multiple-choice question", "write code")  # fmt: skip
TASK_OPTION = {"a": 0, "b": 1, "c": 2, "d": 3}  # the synthetic Jev's request-type answer follows the task


def synthetic_answers(row: dict, passes: dict[str, bool], wobble: dict[str, float] | None = None) -> dict:
    """What a Jev response's `answers` hold for one item: its haiku and sonnet success answers follow whether each
    passed (a noisy but informative Jev), the difficulty answers vary with the item's number, the request type
    follows the task. `wobble` adds a per-question shift, for a second pass."""
    n = int(row["item_id"].split(":")[1])
    shift = wobble or {}
    noul: dict[str, float] = {}
    for config in ("haiku", "sonnet", "opus_low"):
        a = 0.8 - 0.02 * (n % 3) if passes[config] else 0.2 + 0.02 * (n % 3)
        noul |= {f"success_{config}_a": a, f"success_{config}_b": a - 0.05, f"success_{config}_c": 1 - a + 0.05}
        noul[f"success_{config}"] = a  # G's question
    for k in range(1, 9):
        noul[f"difficulty_{k}"] = 0.1 + 0.1 * ((n * k + len(row["item_id"])) % 7)
    noul = {key: round(min(max(v + shift.get(key, 0.0), 0.0), 1.0), 2) for key, v in noul.items()}
    probabilities = {option: 0.05 for option in OPTIONS}
    probabilities[OPTIONS[TASK_OPTION[row["source"]]]] = 0.85
    answers: dict = {key: {"type": "noul", "noul": v} for key, v in noul.items()}
    answers["request_type"] = {"type": "choice", "choice": max(probabilities, key=lambda o: probabilities[o]),
                               "confidence": 0.85, "probabilities": probabilities}  # fmt: skip
    return answers


def synthetic_record(router: str, row: dict, passes: dict[str, bool], pass_number: int = 1, refused: bool = False,
                     wobble: dict[str, float] | None = None, split: str = "train") -> dict:  # fmt: skip
    """One stored record in #30's format. A refused one is a 200 whose answers lack every asked question."""
    answers = {} if refused else synthetic_answers(row, passes, wobble)
    if router == "jev":  # G asks its three statements only, under their own IDs
        keep = {f"success_{c}": answers[f"success_{c}"] for c in synth.CONFIGS} if answers else {}
    else:  # G+ asks 18 questions: nine success statements, eight difficulty statements, the request type
        keep = {key: v for key, v in answers.items() if key not in {f"success_{c}" for c in synth.CONFIGS}}
    return {"router": router, "split": split, "item_id": row["item_id"], "pass": pass_number,
            "body_sha256": "0" * 64, "response": {"model": "jev-1.14.0", "answers": keep,
                                                  "usage": {"input_tokens": 900, "output_tokens": 60}},
            "model": "jev-1.14.0", "usage": {"input_tokens": 900, "output_tokens": 60}, "request_id": "req_" + "0" * 32,
            "server_date": "Wed, 28 Oct 2026 00:01:57 GMT", "seconds": 0.2, "attempts": 1, "failed_attempts": [],
            "new_connection": False, "refused": refused, "collected_at": "2026-10-28T00:01:57"}  # fmt: skip


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def pipeline_checks() -> None:
    rows = synth.synthetic_rows()
    train = [r for r in rows if r["split"] == "train"]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bench = synth.write_benchmark(directory, rows)
        store = synth.write_store(directory, rows, bad_dev_price=True)  # a dev row the whole-store loader refuses
        built = build_from_files(bench, synth.write_store(directory, rows, name="good.sqlite"))
        settings_path = directory / "settings.json"
        settings_path.write_text(settings_text(built.settings), encoding="utf-8")
        fold_of = build_folds("train", bench)
        eligible = [r for r in train if r["source_group"] not in built.excluded_groups]
        noise_ids = built.settings["noise_sample"]["item_ids"]
        outside_noise = [r["item_id"] for r in eligible if r["item_id"] not in noise_ids]
        plus_refused = set(outside_noise[:3])  # Jev refused these on G+'s call (none is in the noise sample)
        g_refused = outside_noise[3]  # and this one on G's alone

        def passes_of(row: dict) -> dict[str, bool]:
            number = int(row["item_id"].split(":")[1])
            return {c: not synth.fails(c, number) for c in synth.CONFIGS}

        plus = [synthetic_record("jev_bolstered", r, passes_of(r), refused=r["item_id"] in plus_refused) for r in train]
        g = [synthetic_record("jev", r, passes_of(r), refused=r["item_id"] == g_refused) for r in train]
        by_id = {r["item_id"]: r for r in train}
        wobble = {"success_haiku": 0.02, "success_sonnet": -0.04}
        plus2 = [synthetic_record("jev_bolstered", by_id[i], passes_of(by_id[i]), 2) for i in noise_ids]
        g2 = [synthetic_record("jev", by_id[i], passes_of(by_id[i]), 2, wobble=wobble) for i in noise_ids]
        write_records(directory / "jev_bolstered_train.jsonl", plus + plus2)
        write_records(directory / "jev_train.jsonl", g + g2)

        def run(**changes: object) -> fit.FitResult:
            args: dict = {"settings_path": settings_path, "records_dir": directory, "benchmark_file": bench,
                          "store": store, "fold_of": fold_of}  # fmt: skip
            return fit.fit_from_files(**{**args, **changes})

        print("\n-- the fit rows --")
        result = run()
        expected_ids = sorted(r["item_id"] for r in eligible if r["item_id"] not in plus_refused)
        check(result.counts.train == 240 and result.counts.excluded == 48 and result.counts.refused == 3,
              "240 train items, 48 of them in the past requests' groups, 3 refused: counted")  # fmt: skip
        check(list(result.item_ids) == expected_ids and len(expected_ids) == 189,
              "the fit rows are the 189 train items outside the groups that Jev didn't refuse")  # fmt: skip
        check(result.counts.g_refused == 1, "the one item Jev refused on G's call alone is counted, and kept as a row")
        check(g_refused in result.item_ids, "a G refusal doesn't drop a row: G scores 0 there")
        labels_ok = all(
            result.labels[c].tolist() == [int(not synth.fails(c, int(i.split(":")[1]))) for i in result.item_ids]
            for c in synth.CONFIGS
        )
        check(labels_ok, "the labels are run 0's verdicts, by the hand-set patterns, not run 1's or 2's")

        print("\n-- the weights file --")
        data = result.weights
        check(list(data) == ["features", "standardizer", "heads"] and list(data["standardizer"]) == ["mean", "scale"]
              and list(data["heads"]) == list(synth.CONFIGS)
              and all(list(h) == ["c", "coef", "intercept"] and len(h["coef"]) == 16 for h in data["heads"].values()),
              "the weights file has #30's sections: features, standardizer, then a head per config with c, coef, intercept")  # fmt: skip
        settings = jev_call.parse_settings(built.settings)
        weights = jev.parse_weights(json.loads(fit.weights_text(data)), settings)
        check(
            weights.features == jev.feature_names(settings), "routers/jev.py's reader accepts the file, with its names"
        )
        check(all(weights.heads[c].c == result.chosen[c].c for c in synth.CONFIGS), "each head carries its chosen C")
        # The deployable path, item by item: stored answers -> features -> numpy heads, against scikit-learn's own
        # predict_proba on the standardized features of the same rows.
        answered = {r["item_id"]: fit.unwrap_answers(r["response"]) for r in plus}
        features = np.array([jev.g_plus_features(answered[i], from_row(by_id[i]), settings, weights)
                             for i in result.item_ids])  # fmt: skip
        z = (features - weights.mean) / weights.scale
        worst = 0.0
        for config in synth.CONFIGS:
            numpy_p = np.array([jev.g_plus_heads(row, weights)[config] for row in features])
            worst = max(worst, float(np.abs(numpy_p - result.models[config].predict_proba(z)[:, 1]).max()))
        check(
            worst < 1e-9, f"the numpy heads reading the file match scikit-learn's predict_proba to 1e-9 ({worst:.1e})"
        )
        check(bool(np.abs(weights.mean).max() > 0) and bool((weights.scale > 0).all()),
              "the standardizer was fitted on the rows: means move and scales are positive")  # fmt: skip

        print("\n-- the signal check and D6 on the files --")
        check(set(result.signal) == set(synth.CONFIGS)
              and all(list(result.signal[c]) == list(fit.SIGNAL_MODELS) for c in synth.CONFIGS),
              "a signal check per config, over the four models")  # fmt: skip
        check(
            result.signal["haiku"]["G+"]["log loss"] < result.signal["haiku"]["intercept only"]["log loss"],
            "on this informative synthetic Jev, G+ beats the intercept-only model's log loss",
        )
        g_noise = result.noise[fit.G]
        check(near(g_noise["haiku"].d6, 0.02) and near(g_noise["haiku"].shift, 0.02)
              and near(g_noise["sonnet"].d6, 0.04) and near(g_noise["sonnet"].shift, -0.04)
              and near(g_noise["opus_low"].d6, 0.0) and near(g_noise["opus_low"].shift, 0.0),
              "G's D6 and shift on the noise sample: the second pass moved haiku +0.02 and sonnet -0.04 (hand-set)")  # fmt: skip
        check(all(near(n.d6, 0.0) and near(n.shift, 0.0) for n in result.noise[fit.G_PLUS].values()),
              "G+'s D6 and shift are zero when its second pass repeats the first")  # fmt: skip

        # G+'s weights through the fitted file, by the deployable path, for the noise check's expected value: the one item
        # whose second pass is refused scores 0 for every config, so D6 is that item's pass-1 score over 100 items.
        refused_pass2 = noise_ids[0]
        write_records(directory / "jev_bolstered_train.jsonl", plus
                      + [synthetic_record("jev_bolstered", by_id[i], passes_of(by_id[i]), 2, refused=i == refused_pass2)
                         for i in noise_ids])  # fmt: skip
        flipped = run()
        before = jev.g_plus_scores(answered[refused_pass2], False, from_row(by_id[refused_pass2]), settings,
                                   jev.parse_weights(flipped.weights, settings))  # fmt: skip
        check(all(near(flipped.noise[fit.G_PLUS][c].d6, before[c] / 100, 1e-9)
                  and near(flipped.noise[fit.G_PLUS][c].shift, -before[c] / 100, 1e-9) for c in synth.CONFIGS),
              "a request refused in pass 2 scores 0 there: G+'s D6 is its pass-1 score over the 100 items, shift negative")  # fmt: skip
        write_records(directory / "jev_bolstered_train.jsonl", plus + plus2)

        text = fit.report(result)
        check("fit rows 189" in text and "refused on G+'s call 3" in text and "in the past requests' groups 48" in text,
              "the printout carries the counts: 240 train items, 48 excluded, 3 refused, 189 fit rows")  # fmt: skip
        check(all(f"{c}: C = {result.chosen[c].c:g}" in text for c in synth.CONFIGS)
              and ("of the grid" in text) == any(result.chosen[c].at_end for c in synth.CONFIGS)
              and "D6" in text and "G+" in text and "task rates" in text,
              "the printout carries each C, flags one at an end of the grid, and holds the signal check and D6")  # fmt: skip

        print("\n-- refusals, each whole --")
        path = directory / "jev_bolstered_train.jsonl"
        write_records(path, plus[1:] + plus2)
        check(refused(lambda: run()), "a fit row with no stored G+ answer is refused")
        write_records(path, [*plus, plus[0], *plus2])
        check(refused(lambda: run()), "a request stored twice for one pass is refused")
        write_records(path, [{k: v for k, v in plus[0].items() if k != "seconds"}, *plus[1:], *plus2])
        check(refused(lambda: run()), "a record missing a field of #30's format is refused")
        first = by_id[plus[0]["item_id"]]
        write_records(
            path, [synthetic_record("jev_bolstered", first, passes_of(first), split="dev"), *plus[1:], *plus2]
        )
        check(refused(lambda: run()), "a record of another split is refused: only train's answers are read")
        write_records(path, [dict(plus[0], router="jev"), *plus[1:], *plus2])
        check(refused(lambda: run()), "a record of the other router is refused")
        write_records(path, plus + plus2[1:])
        check(refused(lambda: run()), "a noise-sample item with no second-pass answer is refused")
        write_records(path, plus + plus2)
        check(not refused(lambda: run()), "known positive: the same inputs, restored, are accepted")
        target = directory / "weights.json"
        fit.write_weights(fit.weights_text(result.weights), target, overwrite=False)
        check(target.read_text(encoding="utf-8") == fit.weights_text(result.weights), "the weights file is written")
        check(refused(lambda: fit.write_weights("{}", target, overwrite=False)),
              "an existing weights file is not overwritten without the flag")  # fmt: skip

        print("\n-- train's cells alone --")
        check(not synth.loads_whole_matrix(Path(fit.__file__).read_text(encoding="utf-8")),
              "the fit script never loads the whole matrix: no load, from_store or unfiltered read_store")  # fmt: skip
        check(synth.loads_whole_matrix("from analysis import matrix\nmatrix.load(path)")
              and synth.loads_whole_matrix("read_store(path)") and not synth.loads_whole_matrix("read_store(p, split='train')"),
              "known positives: the source check flags a whole-matrix load and passes a train-only one")  # fmt: skip
        check(refused(lambda: matrix.load(store), matrix.MatrixError),
              "known positive: the store's bad dev row refuses the whole-store loader")  # fmt: skip
        check(result.counts.train == 240, "the fit ran on that same store, since only train's rows were queried")


def main() -> int:
    print("-- within-task concordance --")
    # Cell (t, 0): successes .9 and .5, failures .2 and .5: wins .9>.2, .9>.5, .5>.2, a tie .5=.5 -> 3.5 of 4 pairs.
    # Cell (t, 1): success .1, failure .8: 0 of 1. Cell (u, 0): failure .3, success .6: 1 of 1.
    # A pair across cells never counts, so t holds 3.5 of 5 and the pool 4.5 of 6.
    scores = np.array([0.9, 0.2, 0.5, 0.5, 0.1, 0.8, 0.3, 0.6])
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    cells = [("t", 0)] * 4 + [("t", 1)] * 2 + [("u", 0)] * 2
    got = fit.concordance(scores, y, cells, ["t"] * 6 + ["u"] * 2)
    check(near(got[None], 0.75) and near(got["t"], 0.7) and near(got["u"], 1.0),
          "concordance is 0.75 pooled, 0.7 for t and 1.0 for u, counting pairs within a task and fold only")  # fmt: skip
    check(fit.concordance(np.array([0.5, 0.6]), np.array([1, 1]), [("t", 0)] * 2, ["t", "t"]) == {None: None, "t": None},
          "a cell with no success-failure pair gives no concordance, not zero")  # fmt: skip

    print("\n-- the train signal check --")
    # Five folds of four rows: in each fold, task p's two rows then task q's two, y = [1, 0, 1, 0], except that fold 4's
    # p rows are both successes (y = [1, 1, 1, 0]).
    #   intercept only: the other folds' success rate is 9/16 for folds 0-3 and 8/16 for fold 4; the folds' log losses
    #   are 0.701021 (4 times) and ln 2, mean 0.699447. Every pair within a cell is a tie, so concordance is 0.5.
    #   task rates: p's rate is 5/8 in folds 0-3 and 4/8 in fold 4, q's 4/8 throughout; mean log loss 0.706055; every
    #   cell's two rows share a rate, so again 0.5.
    #   G (raw scores, clipped to [0.005, 0.995] for log loss alone): mean log loss 0.651735, with q's fold-4 failure at
    #   1.00 costing -ln 0.005; concordance: p wins 2.5 of 4 pairs, q 4 of 5, pooled 6.5 of 9.
    folds = np.repeat(np.arange(5), 4)
    tasks = np.array(["p", "p", "q", "q"] * 5)
    y = np.array([1, 0, 1, 0] * 4 + [1, 1, 1, 0])
    g_p = [(0.9, 0.0), (0.9, 0.3), (0.4, 0.6), (0.5, 0.5), (0.6, 0.6)]
    g_q = [(0.8, 0.1)] * 4 + [(0.2, 1.0)]
    g_raw = np.array([s for k in range(5) for s in (*g_p[k], *g_q[k])])
    z = np.random.default_rng(3).normal(size=(20, 2))
    result = fit.signal_check(z, y, folds, tasks, 1.0, g_raw)
    check(list(result) == ["intercept only", "task rates", "G", "G+"], "the four models, in the record's order")
    check(near(result["intercept only"]["log loss"], 0.699447) and near(result["task rates"]["log loss"], 0.706055),
          "intercept-only and task-rates log loss match the hand-worked 0.699447 and 0.706055")  # fmt: skip
    check(near(result["G"]["log loss"], 0.651735),
          "G's log loss clips a raw 1.00 on a failure to 0.995 and a 0.00 to 0.005, giving the hand-worked 0.651735")  # fmt: skip
    check(near(result["intercept only"]["concordance"][None], 0.5) and near(result["task rates"]["concordance"][None], 0.5),
          "a model that gives a task and fold one score has concordance 0.5, all ties")  # fmt: skip
    check(near(result["G"]["concordance"][None], 6.5 / 9) and near(result["G"]["concordance"]["p"], 0.625)
          and near(result["G"]["concordance"]["q"], 0.8),
          "G's concordance is the hand-worked 6.5 of 9 pairs: 0.625 for p and 0.8 for q")  # fmt: skip
    # G+ by another route: scikit-learn's own cross-validation machinery with the same settings, one predefined split.
    expected_p = cross_val_predict(
        LogisticRegression(C=1.0, solver="newton-cholesky", tol=1e-10, max_iter=100), z, y,
        cv=PredefinedSplit(folds), method="predict_proba",
    )[:, 1]  # fmt: skip
    expected_loss = float(np.mean([log_loss(y[folds == k], expected_p[folds == k], labels=[0, 1]) for k in range(5)]))
    check(near(result["G+"]["log loss"], expected_loss, 1e-9),
          "G+'s cross-validated log loss equals scikit-learn's cross_val_predict over the same folds")  # fmt: skip

    print("\n-- the model and the choice of C --")
    check(
        refused(lambda: fit.fit_model(z, y, 1.0, max_iter=1)), "known positive: a fit that doesn't converge is refused"
    )
    check(not refused(lambda: fit.fit_model(z, y, 1.0)), "the same fit with the real settings converges")
    check(refused(lambda: fit.fit_model(z, np.zeros(20, dtype=int), 1.0)), "a head with only one outcome is refused")
    check(fit.GRID[0] == 1e-4 and fit.GRID[-1] == 1e2 and len(fit.GRID) == 13
          and all(abs(fit.GRID[i + 1] / fit.GRID[i] - 10 ** 0.5) < 1e-9 for i in range(12)),
          "the grid is 13 log-spaced values from 1e-4 to 1e2")  # fmt: skip
    rng = np.random.default_rng(11)
    big = rng.normal(size=(200, 3))
    big_folds = np.arange(200) % 5
    separable = (big[:, 0] > 0).astype(int)  # a clean signal: the loss keeps falling as the penalty weakens
    middling = (big[:, 0] + rng.normal(size=200) > 0).astype(int)  # a noisy one: an interior C wins
    loud, mid = fit.choose_c(big, separable, big_folds), fit.choose_c(big, middling, big_folds)
    noise_rng = np.random.default_rng(0)  # 8 pure-noise features on 100 rows: nothing to learn, the most penalty wins
    noise_x = noise_rng.normal(size=(100, 8))
    noise_y = noise_rng.permutation(np.tile([0, 1], 50))
    quiet = fit.choose_c(noise_x, noise_y, np.arange(100) % 5)
    check(loud == fit.choose_c(big, separable, big_folds), "the choice of C is reproducible: same data, same answer")
    check(loud.at_end == "upper end" and loud.c == 1e2, "a clean signal lands at the upper end of the grid, reported")
    check(quiet.at_end == "lower end" and quiet.c == 1e-4, "no signal lands at the lower end, reported")
    check(mid.at_end == "" and mid.c == 1.0, "a noisy signal lands inside the grid, reported as no end")
    check(len(loud.losses) == 13 and loud.losses[-1] == min(loud.losses), "every grid value's mean log loss is kept")

    print("\n-- D6 and the shift --")
    # Pass 2 less pass 1, by hand:
    #   i1: haiku .8 -> .7, sonnet .6 -> .6, opus_low .4 -> .5
    #   i2: haiku .5 -> .6, sonnet .5 -> .4, opus_low .5 -> .5
    #   i3 (refused in pass 1, so 0 for every config): haiku 0 -> .2, sonnet 0 -> .1, opus_low 0 -> 0
    #   haiku: -.1, +.1, +.2: D6 = .4/3 = 0.133333, shift = +.2/3 = 0.066667
    #   sonnet: 0, -.1, +.1: D6 = .2/3 = 0.066667, shift 0
    #   opus_low: +.1, 0, 0: D6 = shift = 0.033333
    one = {"i1": dict(haiku=0.8, sonnet=0.6, opus_low=0.4), "i2": dict(haiku=0.5, sonnet=0.5, opus_low=0.5),
           "i3": dict(haiku=0.0, sonnet=0.0, opus_low=0.0)}  # fmt: skip
    two = {"i1": dict(haiku=0.7, sonnet=0.6, opus_low=0.5), "i2": dict(haiku=0.6, sonnet=0.4, opus_low=0.5),
           "i3": dict(haiku=0.2, sonnet=0.1, opus_low=0.0)}  # fmt: skip
    noise = fit.noise_difference(one, two, ("haiku", "sonnet", "opus_low"))
    check(
        near(noise["haiku"].d6, 0.4 / 3) and near(noise["haiku"].shift, 0.2 / 3), "haiku: D6 0.133333, shift 0.066667"
    )
    check(near(noise["sonnet"].d6, 0.2 / 3) and near(noise["sonnet"].shift, 0.0), "sonnet: D6 0.066667, shift 0")
    check(
        near(noise["opus_low"].d6, 0.1 / 3) and near(noise["opus_low"].shift, 0.1 / 3),
        "opus_low: D6 and shift 0.033333",
    )
    check(refused(lambda: fit.noise_difference(one, {"i1": two["i1"]}, ("haiku",))),
          "passes that score different items are refused")  # fmt: skip

    pipeline_checks()

    print("\n" + ("all checks passed" if not problems else f"{len(problems)} checks failed"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
