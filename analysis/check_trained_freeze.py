"""Check the trained router's freeze and its train-side fit (PLAN_v4.md §8 3b; Amendment A11, items 5-8, 10, 12, 15, 19).

Run from the project root:  uv run python -m analysis.check_trained_freeze

On synthetic data worked out by hand, then the committed files against Amendment A11 item 19. Never a test item, and
no dev outcome: the fold fit is checked on synthetic rows only.

- template lines at the 2% cut, the standardizer, the fit (the numpy scorer against scikit-learn's `predict_proba` to
  1e-9, a fit that doesn't converge refused), the grid and C, within-task concordance and the signal check, the fold
  fit (a fold model never sees its own fold), the weights files, the cache files;
- the freeze (item 19): the recorded modules are the import closure worked out from the router's imports, every
  recorded hash is its file's, a changed byte changes a hash, every entry equals its source (the settings, the
  installed library versions, the train prompts' counts), and the drift limit is 100 times the overall gap;
- the hook only (it needs the model snapshot, the embedding caches and the train matrix, which CI lacks): the five model
  files' hashes, the caches' hashes, token counts, the template lines, standardizer, each C, F-train's weights and the
  signal check reproduced from train, and a sample of requests embedded again against the caches within the drift
  limit, with a known positive.
"""

import ast
import importlib.metadata
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from analysis import fit_trained_router as fit
from analysis import measure_trained_router as measure
from analysis.benchmark import read_rows
from analysis.fake_backend import FakeBackend
from analysis.freeze import closure, sha256
from analysis.plan import PLAN, amendment_item
from harness.client import PROJECT_ROOT
from routers import trained
from routers.ladder import SCORED
from routers.request import Request, from_row
from routers.trained import TrainedRouter

HASHED = re.compile(r"`([\w./-]+)`: `([0-9a-f]{64})`")  # a path and its sha256, as item 19 writes them

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except fit.FitError:
        return True
    return False


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def printed_names(source: str, function: str) -> set[str]:
    """Every name that a `print` call inside the function reads, from its source."""
    tree = ast.parse(source)
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return {
        name.id
        for call in ast.walk(body)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print"
        for name in ast.walk(call)
        if isinstance(name, ast.Name) and name.id != "print"
    }


def synthetic_settings(lines: list[str], mean: np.ndarray, scale: np.ndarray) -> trained.Settings:
    files = dict.fromkeys(trained.MODEL_FILES, "0" * 64)
    return trained.parse_settings(fit.settings_data(files, lines, mean, scale, dict.fromkeys(SCORED, 1.0)))


def main() -> int:
    print("-- template lines (item 5) --")
    # 100 prompts: "keep" is in exactly 2 (once with spaces around it), "drop" in 1, "dup" three times in one prompt
    # and in no other, "twice" in 2 prompts and twice in one of them.
    prompts = [f"u{i}" for i in range(100)]
    prompts[0] = "keep\nu0"
    prompts[1] = "  keep  \nu1"  # whitespace-trimmed, so it is the same line
    prompts[2] = "drop\nu2"
    prompts[3] = "twice\ntwice\nu3"
    prompts[4] = "twice\nu4"
    prompts[5] = "dup\ndup\ndup\nu5"
    prompts[6] = "\n\n"
    check(fit.learn_template_lines(prompts) == ["keep", "twice"],
          "a line in exactly 2% of prompts is a template line; one in 1%, even repeated within its prompt, is not")  # fmt: skip
    check(fit.learn_template_lines(prompts, percent=3) == [], "at a 3% cut the lines in two prompts are out")
    check(fit.learn_template_lines(["", "\n  \n"] * 50) == [], "blank lines are never template lines")
    check(fit.learn_template_lines([]) == [], "no prompts, no template lines")
    check(trained.strip_template_lines("keep\nu0\n\nu0b", ["keep"]) == "u0\n\nu0b",
          "the router removes a listed line and keeps the blank one")  # fmt: skip

    print("\n-- the standardizer (item 7) --")
    mean, scale = fit.standardizer(np.array([[3.0, 4.0], [0.0, 5.0]]))
    # Rows normalize to (0.6, 0.8) and (0, 1): means 0.3 and 0.9, standard deviations 0.3 and 0.1.
    check(np.allclose(mean, [0.3, 0.9]) and np.allclose(scale, [0.3, 0.1]),
          "the mean and standard deviation are of the length-normalized embeddings (hand-worked)")  # fmt: skip
    check(refused(lambda: fit.standardizer(np.array([[1.0, 0.0], [2.0, 0.0]]))), "a constant dimension is refused")
    check(refused(lambda: fit.standardizer(np.array([[0.0, 0.0], [1.0, 2.0]]))), "a zero embedding is refused")
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(60, 5)) + 2.0
    mean5, scale5 = fit.standardizer(embeddings)
    requests = [Request(f"r{i}", {"s": 1} if i % 2 else None) for i in range(60)]
    x = fit.feature_matrix(synthetic_settings([], mean5, scale5), embeddings, requests)
    check(np.allclose(x[:, :5].mean(axis=0), 0, atol=1e-12) and np.allclose(x[:, :5].std(axis=0), 1),
          "the router's features on the fitted embeddings have mean 0 and standard deviation 1")  # fmt: skip
    check(x.shape == (60, 6) and list(x[:4, 5]) == [0.0, 1.0, 0.0, 1.0],
          "the schema flag is the last column, unstandardized")  # fmt: skip

    print("\n-- the fit (items 7 and 10) --")
    y = (x[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    model = fit.fit_model(x, y, 1.0)
    head = fit.head_of(model)
    router = TrainedRouter(synthetic_settings([], mean5, scale5), (dict.fromkeys(SCORED, head),) * 5,
                           backend=FakeBackend({}, default=[1.0] * 5))  # fmt: skip
    worst = max(abs(router.score_features(0, row)["haiku"] - p) for row, p in zip(x, model.predict_proba(x)[:, 1]))
    check(worst < 1e-9, f"the numpy scorer matches scikit-learn's predict_proba to 1e-9 (largest gap {worst:.1e})")
    check(model.n_iter_[0] < fit.SOLVER["max_iter"], "newton-cholesky converges well inside 100 iterations here")
    saved = dict(fit.SOLVER)
    fit.SOLVER["max_iter"] = 1
    try:
        check(refused(lambda: fit.fit_model(x, y, 1.0)), "known positive: a fit that doesn't converge is refused")
    finally:
        fit.SOLVER.update(saved)
    check(not refused(lambda: fit.fit_model(x, y, 1.0)), "the same fit with the real settings is not refused")
    check(refused(lambda: fit.fit_model(x, np.zeros(60, dtype=int), 1.0)), "a config with only one outcome is refused")
    check(fit.GRID[0] == 1e-4 and fit.GRID[-1] == 1e2 and len(fit.GRID) == 13
          and all(abs(fit.GRID[i + 1] / fit.GRID[i] - 10 ** 0.5) < 1e-9 for i in range(12)),
          "the grid is 13 log-spaced values from 1e-4 to 1e2")  # fmt: skip
    folds = np.arange(60) % 5
    noise = np.random.default_rng(0).integers(0, 2, size=60)
    pure_noise = fit.choose_c(x, noise, folds)
    check(pure_noise.at_end == "lower end" and pure_noise.c == 1e-4,
          "known positive: labels that depend on nothing choose the smallest C and say it is at the grid's end")  # fmt: skip
    signal = fit.choose_c(x, y, folds)
    check(signal.c > 1e-4 and len(signal.losses) == 13 and signal.losses[list(fit.GRID).index(signal.c)] == min(signal.losses),
          "labels that depend on the embedding choose a larger C, the one with the lowest mean log loss")  # fmt: skip

    print("\n-- within-task concordance and the signal check (items 7 and 12) --")
    # Cell (t1, fold 0): successes have router scores 0.9 and 0.5, failures 0.2 and 0.5: 3 pairs right, one tie = 3.5 of 4.
    # Cell (t2, fold 0): the success's router score is below the failure's: 0 of 1. A cell with no failure holds no pair.
    scores = np.array([0.9, 0.2, 0.5, 0.5, 0.3, 0.7, 0.4, 0.6])
    outcome = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    task_of = ["t1"] * 4 + ["t2"] * 2 + ["t3"] * 2
    cell_of = [("t1", 0)] * 4 + [("t2", 0)] * 2 + [("t3", 0)] * 2
    got = fit.concordance(scores, outcome, cell_of, task_of)
    check(got["t1"] == 0.875 and got["t2"] == 0.0 and got["t3"] is None and got[None] == 0.7, "pooled 3.5 of 5 pairs, t1 0.875, t2 0, and t3 with no pair is None")  # fmt: skip
    # Five folds of two items in one task; y per fold: (1,0) (1,1) (0,0) (1,0) (1,1).
    y10 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    f10 = np.repeat(np.arange(5), 2)
    t10 = np.array(["a"] * 10)
    x10 = rng.normal(size=(10, 3))
    signal10 = fit.signal_check(x10, y10, f10, t10, 1.0)
    others = [(y10.sum() - y10[f10 == k].sum()) / 8 for k in range(5)]  # the other folds' mean, hand-worked
    expected = np.mean([-(y10[f10 == k].mean() * np.log(m) + (1 - y10[f10 == k].mean()) * np.log(1 - m))
                        for k, m in enumerate(others)])  # fmt: skip
    check(close(signal10["intercept only"]["log loss"], float(expected))
          and close(signal10["task rates"]["log loss"], float(expected)),
          "the intercept-only and task-rates log losses are the other folds' mean rate's (hand-worked)")  # fmt: skip
    check(signal10["task rates"]["concordance"][None] == 0.5 and signal10["intercept only"]["concordance"][None] == 0.5,
          "the task-rates model scores exactly 0.5 within each task and fold")  # fmt: skip
    # Fold k has 1, 3, 5, 7, 9 successes of 10, so its task rates (from the other folds) fall as k rises.
    y50 = np.concatenate([[1] * s + [0] * (10 - s) for s in (1, 3, 5, 7, 9)])
    f50 = np.repeat(np.arange(5), 10)
    t50 = np.array(["a"] * 50)
    signal50 = fit.signal_check(rng.normal(size=(50, 3)), y50, f50, t50, 1.0)
    rates = np.zeros(50)
    for k in range(5):
        rates[f50 == k] = y50[f50 != k].mean()
    pooled = fit.concordance(rates, y50, [("a",)] * 50, t50.tolist())[None]
    check(signal50["task rates"]["concordance"][None] == 0.5 and pooled is not None and pooled < 0.5,
          "known positive: the same rates counted across folds come out below 0.5, within folds exactly 0.5")  # fmt: skip

    print("\n-- the fold fit (item 15) --")
    train_x, dev_x = rng.normal(size=(40, 4)), rng.normal(size=(10, 4))
    train_y = {c: (rng.random(40) > 0.4).astype(int) for c in SCORED}
    dev_y = {c: (rng.random(10) > 0.4).astype(int) for c in SCORED}
    dev_folds = np.repeat(np.arange(5), 2)
    cs = {"haiku": 0.1, "sonnet": 1.0, "opus_low": 0.01}
    models, rows = fit.fit_fold_models(train_x, train_y, dev_x, dev_y, dev_folds, cs)
    check(
        rows == [48] * 5 and len(models) == 5, "each fold model learns from train and the 8 dev rows outside its fold"
    )
    flipped = {c: v.copy() for c, v in dev_y.items()}
    flipped["haiku"][dev_folds == 2] = 1 - flipped["haiku"][dev_folds == 2]
    again, _ = fit.fit_fold_models(train_x, train_y, dev_x, flipped, dev_folds, cs)
    same = [np.array_equal(models[k]["haiku"].coef, again[k]["haiku"].coef) for k in range(5)]
    check(same == [False, False, True, False, False],
          "known positive: flipping fold 2's labels changes every fold model except fold model 2's")  # fmt: skip
    one = fit.fit_head(
        np.vstack([train_x, dev_x[dev_folds != 0]]),
        np.concatenate([train_y["sonnet"], dev_y["sonnet"][dev_folds != 0]]),
        1.0,
    )
    check(np.array_equal(one.coef, models[0]["sonnet"].coef) and one.intercept == models[0]["sonnet"].intercept,
          "a head is fitted at its own config's C on its own labels")  # fmt: skip

    source = (PROJECT_ROOT / "analysis" / "fit_trained_router.py").read_text()
    printed = printed_names(source, "fit_folds")
    check(printed == {"k", "n", "hashlib", "text", "trained", "PROJECT_ROOT"},
          f"the fold fit prints only each fold model's row count and the weights file's hash (it names {sorted(printed)})")  # fmt: skip
    check(printed_names("def fit_folds():\n    y = 1\n    print(f'{y}')\n", "fit_folds") == {"y"},
          "known positive: the guard finds a print of a label")  # fmt: skip

    print("\n-- the files --")
    text = fit.fold_weights_text([{c: fit.Head(np.array([0.1, -2.5e-9, 1 / 3]), 1 / 7) for c in SCORED}] * 5)
    parsed = trained.parse_fold_weights(__import__("json").loads(text))
    check(len(parsed) == 5 and parsed[3]["sonnet"].coef[2] == 1 / 3 and parsed[3]["sonnet"].intercept == 1 / 7,
          "the fold-weights file reads back through the router's reader with every float exact")  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "f_train.json"
        path.write_text(fit.f_train_weights_text(models[1]))
        five = fit.read_f_train_weights(path)
        check(len(five) == 5 and all(np.array_equal(five[k]["opus_low"].coef, models[1]["opus_low"].coef) for k in range(5)),
              "F-train's one model reads back in all five fold slots")  # fmt: skip
        ids = [row["item_id"] for row in read_rows(split="dev")]
        directory = Path(tmp)
        fit.write_cache(
            "dev", fit.Cache(ids[::-1], np.zeros((len(ids), 2)), [1] * len(ids), [False] * len(ids)), directory
        )
        check(
            refused(lambda: fit.read_cache("dev", directory)),
            "a cache whose items aren't the reader's rows, in order, is refused",
        )
        fit.write_cache("dev", fit.Cache(ids, np.ones((len(ids), 2)), [1] * len(ids), [False] * len(ids)), directory)
        back = fit.read_cache("dev", directory)
        check(
            back.item_ids == ids and back.embeddings.shape == (len(ids), 2), "a cache of the reader's rows reads back"
        )
        check(refused(lambda: fit.read_cache("train", directory)), "a missing cache is refused with what to run")

    print("\n-- the freeze (Amendment A11 item 19) --")
    record = amendment_item(PLAN.read_text(), "A11", 19)
    recorded = dict(HASHED.findall(record))
    frozen = measure.frozen_files()
    entry = "routers/trained.py"
    modules = closure(entry)
    check(
        modules == ["routers/__init__.py", "routers/ladder.py", "routers/request.py", entry],
        "F's import closure, worked out from its imports, is the package, the ladder, the request and the router",
    )
    check("routers/decision.py" not in modules, "the decision rule is outside F's closure, frozen by A10 item 1")
    check(all(p in recorded for p in frozen) and len(frozen) == len(set(frozen)),
          f"item 19 records the closure and the {len(measure.FROZEN_OTHERS)} other frozen files, once each")  # fmt: skip
    for relative in frozen:
        check(sha256(PROJECT_ROOT / relative) == recorded.get(relative), f"{relative} hashes as recorded")
    check(set(trained.MODEL_FILES) <= set(recorded), "the five model files' sha256 are recorded")
    with tempfile.TemporaryDirectory() as tmp:
        for relative in frozen:
            copy = Path(tmp) / Path(relative).name
            copy.write_bytes((PROJECT_ROOT / relative).read_bytes() + b" ")
            check(
                sha256(copy) != recorded.get(relative),
                f"a byte added to a copy of {relative} changes its hash",
            )
    from analysis.freeze import closure as closure_of

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "routers").mkdir()
        (root / "routers" / "__init__.py").write_text("")
        (root / "routers" / "trained.py").write_text("import json\nfrom routers import helper\n")
        (root / "routers" / "helper.py").write_text("")
        check(closure_of(entry, root) == ["routers/__init__.py", "routers/helper.py", entry],
              "known positive: a module the router imports joins the closure, so a record without it differs")  # fmt: skip
    check(
        not any(".cache" in p or p.startswith("results") for p in frozen),
        "no model file or cache is in the commit's files",
    )

    print("\n-- item 19 against its sources --")
    settings = trained.read_settings()
    check(bool(settings.window == 8192 and settings.compute_precision == "float32" and settings.reference_compile is False
          and settings.solver == fit.SOLVER and settings.threads == fit.THREADS and settings.attention == fit.ATTENTION),
          "the committed settings are A11's: window 8192, float32, sdpa, newton-cholesky at 1e-10 and 100, reference_compile off")  # fmt: skip
    check(len(settings.mean) == 768 and bool((settings.scale > 0).all()) and all(c in fit.GRID for c in settings.c.values()),
          "the standardizer is 768 wide with positive scale, and each C is a grid value")  # fmt: skip
    check(re.search(rf"`{settings.repository}` at commit `{settings.commit}`", record) is not None
          and all(recorded.get(n) == h for n, h in settings.files.items()),
          "item 19 records the model's repository, commit and five sha256, equal to the settings'")  # fmt: skip
    check(f"window {settings.window} tokens, compute precision {settings.compute_precision}, attention implementation "
          f"{settings.attention}, {settings.threads} threads, `reference_compile` false. Solver: {settings.solver['name']}, "
          f"tolerance {settings.solver['tol']:g}, at most {settings.solver['max_iter']} iterations" in record,
          "item 19 records the window, precision, attention, threads, reference_compile and the solver, equal to the settings'")  # fmt: skip
    versions = re.search(r"\*\*Library versions\*\* \(item 4\): (.*?)\.\n", record + "\n")
    pairs = dict(p.split(" ") for p in versions[1].split(", ")) if versions else {}
    installed = {name: importlib.metadata.version(name).split("+")[0] for name in measure.LIBRARIES}
    check(pairs == installed, f"item 19's library versions are the installed ones {installed}")
    pins = (PROJECT_ROOT / "pyproject.toml").read_text()
    check(all(f'"{name}=={version}"' in pins for name, version in pairs.items()), "and they are pyproject's exact pins")
    check(
        all(f"{c} {settings.c[c]:.6g} (" in record for c in SCORED),
        "item 19 records each config's C, equal to the settings'",
    )
    check(re.search(r"\*\*Environment\*\* \(recorded, not pinned; item 4\): macOS [\d.]+ \(build \w+\), .+, PyTorch's math library: BLAS_INFO=\w+", record)
          is not None, "item 19 records the macOS build, the chip and PyTorch's math-library line")  # fmt: skip
    gap = re.search(r"overall ([0-9.e+-]+)\.\n", record)
    limit = re.search(r"\*\*Drift limit\*\*: `([0-9.e+-]+)`, 100 times the overall gap", record)
    check(gap is not None and limit is not None and abs(float(limit[1]) - 100 * float(gap[1])) <= 0.011 * float(limit[1]),
          "the drift limit is 100 times the overall gap (to the digits printed)")  # fmt: skip
    check(measure.drift_limit() == float(limit[1] if limit else 0) > 0, "`drift_limit()` reads it back from the plan")
    check("Rerun at one thread" in record and ("matched the float32 router's bit for bit" in record or "did not match" in record),
          "item 19 records whether the rerun at one thread matched bit for bit")  # fmt: skip
    check(re.search(r"median [\d.]+ ms, p90 [\d.]+ ms", record) is not None and "F-train's heads in all five slots" in record
          and "Load time" in record, "item 19 records the latency, the load time and that F-train's heads stood in")  # fmt: skip
    train = [row["prompt"] for row in read_rows(split="train")]
    check(fit.learn_template_lines(train) == list(settings.template_lines),
          "the template lines in the settings are the ones learned from the train prompts")  # fmt: skip
    check(all(fit.learn_template_lines(train, percent=p) == list(settings.template_lines) for p in (1, 5)),
          "any cut from 1% to 5% gives the same list")  # fmt: skip
    counts = measure.prompt_counts()
    check(f"the {counts['n']} train prompts: {counts['template lines']} in all. By task: "
          + ", ".join(f"{t} {v['template lines']}" for t, v in counts["by task"].items()) in record,
          "item 19's template-line counts by task are the train prompts'")  # fmt: skip
    below = counts["most common line below the cut"]
    check(f"is in {below['prompts']} prompts ({100 * below['share']:.2f}%), under the 2% cut" in record,
          "item 19's share for the most common line below the cut is the train prompts'")  # fmt: skip
    check("Stripped text length by task**, in characters (shortest, median, longest), train: "
          + "; ".join(f"{t} {v['characters'][0]:.0f}, {v['characters'][1]:.0f}, {v['characters'][2]:.0f}"
                      for t, v in counts["by task"].items()) in record,
          "item 19's stripped text lengths by task are the train prompts'")  # fmt: skip

    snapshot = trained.SNAPSHOT / "model.safetensors"
    cache = fit.CACHE_DIR / "train.npy"
    matrix = PROJECT_ROOT / "results" / "dev_matrix.sqlite"
    if snapshot.exists() and cache.exists() and fit.cache_paths("dev")[0].exists() and matrix.exists():
        hook_only(record, settings)
    else:
        print("\n(the model snapshot, the caches or the train matrix is missing: the hook-only section is skipped)")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def hook_only(record: str, settings: trained.Settings) -> None:
    print("\n-- the model files and caches (the hook only) --")
    recorded = dict(HASHED.findall(record))
    check(not _refuses_snapshot(settings), "the five model files' sha256 match the settings")
    check(fit.model_file_hashes() == settings.files == {n: recorded.get(n) for n in trained.MODEL_FILES},
          "and the hashes the snapshot gives are the settings' and item 19's")  # fmt: skip
    for name in measure.CACHE_FILES:
        check(
            sha256(fit.CACHE_DIR / name) == recorded.get(name),
            f"the cache file {name} hashes as item 19 records",
        )
    caches = {split: fit.read_cache(split) for split in ("train", "dev")}
    check(all(len(c.embeddings[0]) == 768 and c.embeddings.dtype == np.float64 for c in caches.values()),
          "the caches hold float64 vectors 768 wide")  # fmt: skip
    check(all(np.array_equal(c.embeddings, c.embeddings.astype(np.float32).astype(np.float64)) for c in caches.values()),
          "every cached value is exactly a float32 value, upcast")  # fmt: skip
    tokens = measure.cache_counts(settings)
    check(f"{tokens['train']['truncated']} truncated and {tokens['train']['emptied']} left empty; dev "
          f"{tokens['dev']['truncated']} truncated and {tokens['dev']['emptied']} left empty" in record,
          "item 19's truncation and empty-text counts are the caches'")  # fmt: skip
    check(f"longest stripped text is {tokens['train']['longest stripped']} tokens on train and "
          f"{tokens['dev']['longest stripped']} on dev" in record, "item 19's longest stripped texts are the caches' token counts")  # fmt: skip

    print("\n-- the train fit reproduced (the hook only) --")
    side = fit.fit_train_side()
    mean = np.array(side.settings["standardizer"]["mean"])
    scale = np.array(side.settings["standardizer"]["scale"])
    check(np.allclose(mean, settings.mean, rtol=0, atol=1e-12) and np.allclose(scale, settings.scale, rtol=0, atol=1e-12),
          "the standardizer reproduces from the train caches")  # fmt: skip
    check(side.settings["c"] == settings.c, f"each config's C reproduces from train: {settings.c}")
    check(side.settings["template_lines"] == list(settings.template_lines), "the template lines reproduce from train")
    check(side.settings["model"]["files"] == settings.files, "the model file hashes reproduce from the snapshot")
    stored_f_train = fit.read_f_train_weights()[0]
    gap = max(float(np.abs(side.f_train[c].coef - stored_f_train[c].coef).max()) for c in SCORED)
    check(gap < 1e-9 and all(abs(side.f_train[c].intercept - stored_f_train[c].intercept) < 1e-9 for c in SCORED),
          f"F-train's weights refit from train to 1e-9 (largest gap {gap:.1e})")  # fmt: skip
    check(all(f"{name} {r['log loss']:.4f}, {r['concordance'][None]:.4f}" in record for c in SCORED
              for name, r in side.signal[c].items()), "item 19's train signal check is the recomputed one")  # fmt: skip
    check(", ".join(f"{t} {n} of {total}" for t, (n, total) in side.every_config_fails.items()) in record,
          "item 19's train items that every scored config fails are the train matrix's")  # fmt: skip
    check(all(f"{c} {side.chosen[c].c:.6g} (" + ("at the " + side.chosen[c].at_end + " of the grid)" if side.chosen[c].at_end
                                                    else "inside the grid)") in record for c in SCORED),
          "item 19 says whether each C sits at an end of the grid")  # fmt: skip

    print("\n-- a sample of requests embedded again (the hook only) --")
    # Per split and task, four requests evenly spaced by stripped length from the shortest to the longest: 32
    # embeddings, a few seconds. The full 1,735 take over a minute, too long for every commit.
    limit = measure.drift_limit()
    router = TrainedRouter(settings, fit.read_f_train_weights())
    sample: list[tuple[str, int, Request]] = []
    for split, cache in (("train", fit.read_cache("train")), ("dev", fit.read_cache("dev"))):
        rows = read_rows(split=split)
        tokens_of = np.array(cache.tokens)
        for task in dict.fromkeys(row["source"] for row in rows):
            in_task = [i for i, row in enumerate(rows) if row["source"] == task]
            ranked = sorted(in_task, key=lambda i: (tokens_of[i], i))
            for rank in np.linspace(0, len(ranked) - 1, 4).round().astype(int):
                sample.append((split, ranked[rank], from_row(rows[ranked[rank]])))
    worst, gaps, swapped = 0.0, [], []
    for split, i, request in sample:
        cached = {"train": caches["train"], "dev": caches["dev"]}[split].embeddings
        again = router.score(request)
        from_cache = router.score_features(trained.fold(request), router.features(cached[i], request))
        worst = max(worst, max(abs(again[c] - from_cache[c]) for c in SCORED))
        gaps.append(float(np.abs(router.embed(request) - cached[i]).max()))
        other = router.features(cached[(i + 1) % len(cached)], request)  # another request's cached vector
        swapped.append(max(abs(again[c] - router.score_features(trained.fold(request), other)[c]) for c in SCORED))
    check(len(sample) == 32 and worst <= limit,
          f"{len(sample)} requests embedded again give router scores within the drift limit {limit:.3g} of their cached embedding "
          f"(largest gap {worst:.2e}, {sum(g == 0 for g in gaps)} of {len(gaps)} embeddings identical)")  # fmt: skip
    check(max(swapped) > limit, "known positive: another request's cached embedding is further than the drift limit")


def _refuses_snapshot(settings: trained.Settings) -> bool:
    try:
        trained.verify_snapshot(settings)
    except trained.SnapshotError:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
