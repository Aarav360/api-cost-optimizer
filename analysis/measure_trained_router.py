"""Measure the trained router before the freeze and write Amendment A11 item 19 (PLAN_v4.md §8 3b; A11 items 4, 11, 15, 19).

Run from the project root:  uv run python -m analysis.measure_trained_router <command>

- `drift`: every train request, one at a time, through the float32 router and through the real model upcast to float64
  (passed in through the backend seam from the same verified snapshot), with F-train's heads in all five slots. The
  largest gap in router scores, per config and overall; the drift limit, 100 times the overall gap; then a rerun at one
  thread, expected to match the float32 router scores bit for bit (item 4).
- `latency`: `score(request)` over the 744 dev requests, one at a time, through the hash path, the model loaded first,
  50 warm-up calls excluded, the router scores thrown away (item 11). Run it alone: other work on the machine slows it.
- `record`: prints item 19 (`--write` puts it into PLAN_v4.md) from the committed files, the caches, the train
  outcomes and the measurements the two commands above stored under `results/trained_embeddings/`.

Reads prompts, output schemas and train outcomes only: no dev outcome (item 15). Dev requests are embedded for timing
and for the cache; their router scores are never stored, printed or summarised.

`drift_limit()` is what every later check that embeds a request again holds the result to.
"""

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from analysis import fit_trained_router as fit
from analysis.benchmark import read_rows
from analysis.freeze import sha256
from analysis.plan import PLAN, amendment_item
from routers import trained
from routers.ladder import SCORED
from routers.request import Request, from_row
from routers.trained import TrainedRouter

MEASUREMENTS = fit.CACHE_DIR / "measurements.json"
WARM_UP = 50  # calls made before the timing starts, so the first call's setup isn't timed (item 11)
DRIFT_FACTOR = 100  # the drift limit is this many times the largest gap (item 4)
LIBRARIES = ("torch", "transformers", "tokenizers", "numpy", "scikit-learn")

# What is hashed: F's import closure (worked out by a check), the fit script, the fold builder, the settings, F-train's
# weights and both fold files (item 15). The measurement script is not part of the freeze.
FROZEN_OTHERS = (
    "analysis/fit_trained_router.py",
    "analysis/folds.py",
    "routers/trained_settings.json",
    "routers/trained_f_train_weights.json",
    "data/dev_folds.json",
    "data/train_folds.json",
)
CACHE_FILES = ("train.npy", "train_items.json", "dev.npy", "dev_items.json")


# ---------------------------------------------------------------------------------------------------------------
# Environment


def command_output(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()


def library_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in LIBRARIES}


def math_library_line() -> str:
    """PyTorch's BLAS and LAPACK build settings, as `torch.__config__` states them."""
    import torch

    found = re.findall(r"\b(BLAS_INFO|LAPACK_INFO)=(\w+)", torch.__config__.show())
    return ", ".join(f"{key}={value}" for key, value in dict(found).items())


def environment() -> dict[str, str]:
    """The macOS build, the chip and PyTorch's math-library line: recorded, not pinned (item 4)."""
    return {
        "os": f"macOS {command_output('sw_vers', '-productVersion')} (build {command_output('sw_vers', '-buildVersion')})"
        if platform.system() == "Darwin" else f"{platform.system()} {platform.release()}",
        "chip": command_output("sysctl", "-n", "machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.machine(),
        "math library": math_library_line(),
        "python": platform.python_version(),
    }  # fmt: skip


def stored() -> dict[str, Any]:
    return json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}


def store(key: str, value: Any) -> None:
    data = stored()
    data[key] = value
    MEASUREMENTS.write_text(fit.json_text(data))


# ---------------------------------------------------------------------------------------------------------------
# The drift measurement (item 4)


def f_train_router(settings: trained.Settings, backend: Any) -> TrainedRouter:
    return TrainedRouter(settings, fit.read_f_train_weights(), backend=backend)


def score_all(router: TrainedRouter, requests: list[Request]) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings (float64) and router scores (one row each, in SCORED order), one request at a time."""
    vectors, scores = [], []
    for request in requests:
        vector = router.embed(request)
        vectors.append(vector)
        # `score(request)` without embedding twice: the hash picks the fold model, the heads read the same features.
        by_config = router.score_features(trained.fold(request), router.features(vector, request))
        scores.append([by_config[config] for config in SCORED])
    return np.array(vectors), np.array(scores)


def drift() -> dict[str, Any]:
    settings = trained.read_settings()
    requests = [from_row(row) for row in read_rows(split="train")]
    base = trained.load_backend(settings)
    _, float32_scores = score_all(f_train_router(settings, base), requests)
    _, float64_scores = score_all(f_train_router(settings, base.upcast()), requests)
    gap = np.abs(float32_scores - float64_scores).max(axis=0)
    overall = float(gap.max())
    # The rerun at one thread: loading sets PyTorch's thread count, so it is the last thing done.
    one = trained.load_backend(replace(settings, threads=1))
    _, one_thread_scores = score_all(f_train_router(settings, one), requests)
    result = {
        "gap": {config: float(largest) for config, largest in zip(SCORED, gap, strict=True)},
        "overall": overall,
        "limit": DRIFT_FACTOR * overall,
        "one thread matches bit for bit": bool(np.array_equal(float32_scores, one_thread_scores)),
        "one thread largest gap": float(np.abs(float32_scores - one_thread_scores).max()),
        "requests": len(requests),
    }
    store("drift", result)
    return result


# ---------------------------------------------------------------------------------------------------------------
# Latency (item 11)


def machine() -> dict[str, str]:
    env = environment()
    return {"machine": f"{env['chip']}, {platform.system()} {platform.release()} {platform.machine()}",
            "python": platform.python_version(), "torch": library_versions()["torch"]}  # fmt: skip


def latency() -> dict[str, Any]:
    settings = trained.read_settings()
    rows = read_rows(split="dev")
    requests = [from_row(row) for row in rows]
    start = time.perf_counter()
    trained.verify_snapshot(settings)
    verify = time.perf_counter() - start
    start = time.perf_counter()
    backend = trained.load_backend(settings)
    load = time.perf_counter() - start  # includes the file check
    router = f_train_router(settings, backend)  # F-train's heads stand in for all five fold models
    for request in requests[:WARM_UP]:
        router.score(request)
    times = []
    for request in requests:
        tick = time.perf_counter_ns()
        router.score(request)
        times.append(time.perf_counter_ns() - tick)
    ms = np.array(times) / 1e6
    tasks = np.array([row["source"] for row in rows])

    def summary(sample: np.ndarray) -> dict[str, float]:
        return {"median ms": float(np.median(sample)), "p90 ms": float(np.percentile(sample, 90)), "n": len(sample)}

    result = {"overall": summary(ms), "per task": {t: summary(ms[tasks == t]) for t in dict.fromkeys(tasks.tolist())},
              "load seconds": load, "file check seconds": verify, "threads": settings.threads,
              "heads": "F-train's heads in all five slots", **machine()}  # fmt: skip
    store("latency", result)
    return result


# ---------------------------------------------------------------------------------------------------------------
# Counts that need only prompts (and, for tokens, the caches and the tokenizer)


def median_range(values: list[int] | list[float]) -> tuple[float, float, float]:
    return float(min(values)), float(np.median(values)), float(max(values))


def prompt_counts() -> dict[str, Any]:
    """From train prompts: template lines by task, the share of prompts holding the most common line below the cut,
    stripped text length in characters by task. Computed from the committed benchmark file alone."""
    rows = read_rows(split="train")
    lines = trained.read_settings().template_lines
    shares = fit.line_counts([row["prompt"] for row in rows])
    below = max((n, line) for line, n in shares.items() if line not in set(lines))
    by_task: dict[str, dict[str, Any]] = {}
    for task in dict.fromkeys(row["source"] for row in rows):
        prompts = [row["prompt"] for row in rows if row["source"] == task]
        stripped = [len(trained.strip_template_lines(p, lines)) for p in prompts]
        present = {line for p in prompts for line in {x.strip() for x in p.split("\n")} if line in set(lines)}
        by_task[task] = {"prompts": len(prompts), "template lines": len(present), "characters": median_range(stripped)}
    return {"n": len(rows), "template lines": len(lines), "by task": by_task,
            "most common line below the cut": {"prompts": below[0], "share": below[0] / len(rows)}}  # fmt: skip


def cache_counts(settings: trained.Settings) -> dict[str, Any]:
    """From the caches' token counts: the longest stripped text, the truncation and empty-text counts, and stripped
    length by task, for train and for dev. No tokenizer needed."""
    out: dict[str, Any] = {}
    for split in ("train", "dev"):
        rows = read_rows(split=split)
        cache = fit.read_cache(split)
        tokens = np.array(cache.tokens)
        tasks = np.array([row["source"] for row in rows])
        out[split] = {"longest stripped": int(tokens.max()), "truncated": int((tokens > settings.window).sum()),
                      "emptied": int(sum(cache.emptied)),
                      "by task": {t: median_range(tokens[tasks == t].tolist()) for t in dict.fromkeys(tasks.tolist())}}  # fmt: skip
    return out


def token_counts() -> dict[str, Any]:
    """In the model's tokens: where Banking77's query starts in the whole prompt, plus `cache_counts`. Needs the
    snapshot's tokenizer and the caches."""
    settings = trained.read_settings()
    tokenizer = trained.load_backend(settings).tokenizer
    out = cache_counts(settings)
    starts, totals = [], []
    for row in read_rows(split="train"):
        if row["source"] != "banking77":
            continue
        prompt = row["prompt"]
        at = prompt.rindex("Query: ") + len("Query: ")
        encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        starts.append(next(i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > a and a >= at))
        totals.append(len(encoded["input_ids"]))
    out["banking77 query start"] = median_range(starts)
    out["banking77 whole prompt"] = median_range(totals)
    store("tokens", out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Item 19


def drift_limit(plan: str | None = None) -> float:
    """The drift limit recorded in A11 item 19: what a check that embeds a request again holds router scores to."""
    record = amendment_item(plan if plan is not None else PLAN.read_text(), "A11", 19)
    match = re.search(r"\*\*Drift limit\*\*: `([0-9.e+-]+)`", record)
    if match is None:
        raise fit.FitError("A11 item 19 records no drift limit")
    return float(match.group(1))


def frozen_files() -> list[str]:
    from analysis.freeze import closure

    return [*closure("routers/trained.py"), *FROZEN_OTHERS]


def three_digits(value: float) -> str:
    return f"{value:.3g}"


def record_text(side: fit.TrainSide, measured: dict[str, Any]) -> str:
    """Item 19, as it is written into the plan. Every number is from a committed file, a cache or a stored
    measurement; `analysis/check_trained_freeze.py` compares what it can with its source."""
    settings = trained.read_settings()
    drift_m, lat, tok = measured["drift"], measured["latency"], measured["tokens"]
    counts = prompt_counts()
    env = measured["environment"]
    root = fit.PROJECT_ROOT
    out = [
        "19. **The freeze record** (item 15), written in the freeze commit, before any F number on dev. Nothing here "
        "reads a dev outcome. `analysis/check_trained_freeze.py` compares each entry with its source, and each hash "
        "with its file. sha256, as in A7 item 3:",
        f"    - **The model** (item 3): `{settings.repository}` at commit `{settings.commit}`. The five files the "
        "router reads (item 4), as `analysis/fetch_model.py` fetched them:",
        *(f"      - `{name}`: `{digest}`" for name, digest in settings.files.items()),
        f"    - **Settings** (items 4, 5 and 7): window {settings.window} tokens, compute precision "
        f"{settings.compute_precision}, attention implementation {settings.attention}, {settings.threads} threads, "
        f"`reference_compile` false. Solver: {settings.solver['name']}, tolerance {settings.solver['tol']:g}, at most "
        f"{settings.solver['max_iter']} iterations. The template lines' cut is {fit.TEMPLATE_PERCENT}% of train "
        "prompts. The C grid is 13 log-spaced values from 1e-4 to 1e2.",
        "    - **Library versions** (item 4): " + ", ".join(f"{k} {v}" for k, v in measured["versions"].items()) + ".",
        f"    - **Environment** (recorded, not pinned; item 4): {env['os']}, {env['chip']}, PyTorch's math library: "
        f"{env['math library']}, Python {env['python']}.",
        f"    - **Drift measurement** (item 4), over the {drift_m['requests']} train requests, one at a time, "
        "float32 router against the real model upcast to float64, F-train's heads standing in for the fold models "
        "in both: largest gap in router scores "
        + ", ".join(f"{c} {three_digits(v)}" for c, v in drift_m["gap"].items())
        + f"; overall {three_digits(drift_m['overall'])}.",
        f"    - **Drift limit**: `{three_digits(drift_m['limit'])}`, {DRIFT_FACTOR} times the overall gap. Any check that embeds a "
        "request again holds router scores to it.",
        "    - **Rerun at one thread**: the router scores "
        + (
            "matched the float32 router's bit for bit."
            if drift_m["one thread matches bit for bit"]
            else f"did not match bit for bit; the largest gap was {three_digits(drift_m['one thread largest gap'])}."
        ),
        "    - **Frozen files**, F's import closure in `routers/`, the fit script, the fold builder, the settings, "
        "F-train's weights and both fold files (item 15):",
        *(f"      - `{relative}`: `{sha256(root / relative)}`" for relative in frozen_files()),
        "    - **Embedding caches**, kept under `results/trained_embeddings/`, not committed:",
        *(f"      - `{name}`: `{sha256(fit.CACHE_DIR / name)}`" for name in CACHE_FILES),
        f"    - **Template lines** (item 5), from the {counts['n']} train prompts: {counts['template lines']} in all. "
        "By task: "
        + ", ".join(f"{t} {v['template lines']}" for t, v in counts["by task"].items())
        + f". The most common line below the cut is in {counts['most common line below the cut']['prompts']} prompts "
        f"({100 * counts['most common line below the cut']['share']:.2f}%), under the {fit.TEMPLATE_PERCENT}% cut; the "
        "lowest template line is in "
        f"{100 * min(n for line, n in fit.line_counts([r['prompt'] for r in read_rows(split='train')]).items() if line in set(settings.template_lines)) / counts['n']:.1f}%, "
        "so any cut from 1% to 5% gives the same list.",
        "    - **Stripped text length by task**, in characters (shortest, median, longest), train: "
        + "; ".join(
            f"{t} {v['characters'][0]:.0f}, {v['characters'][1]:.0f}, {v['characters'][2]:.0f}"
            for t, v in counts["by task"].items()
        )
        + ".",
        "    - **In the model's tokens**: Banking77's query starts at token "
        f"{tok['banking77 query start'][1]:.0f} of the whole prompt (earliest {tok['banking77 query start'][0]:.0f}, "
        f"latest {tok['banking77 query start'][2]:.0f}), and the prompt runs {tok['banking77 whole prompt'][0]:.0f} "
        f"to {tok['banking77 whole prompt'][2]:.0f} tokens, so a 256 or 512-token window would cut the query off. The "
        f"longest stripped text is {tok['train']['longest stripped']} tokens on train and "
        f"{tok['dev']['longest stripped']} on dev, against the window of {settings.window}.",
        f"    - **Truncation and empty-text counts**: train {tok['train']['truncated']} truncated and "
        f"{tok['train']['emptied']} left empty; dev {tok['dev']['truncated']} truncated and {tok['dev']['emptied']} "
        "left empty.",
        "    - **C per config** (item 7), by five-fold cross-validation within train, mean log loss: "
        + ", ".join(
            f"{c} {side.chosen[c].c:.6g}"
            f"{' (at the ' + side.chosen[c].at_end + ' of the grid)' if side.chosen[c].at_end else ' (inside the grid)'}"
            for c in SCORED
        )
        + ".",
        "    - **The train signal check** (item 7), cross-validated, for the intercept-only model, the task-rates model "
        "and F: log loss, then within-task concordance (pairs within the same task and train fold):",
        *(
            f"      - {c}: "
            + "; ".join(
                f"{name} {r['log loss']:.4f}, {r['concordance'][None]:.4f}" for name, r in side.signal[c].items()
            )
            + "."
            for c in SCORED
        ),
        "    - **Train items that every scored config fails**, per task: "
        + ", ".join(f"{t} {n} of {total}" for t, (n, total) in side.every_config_fails.items())
        + ".",
        f"    - **Latency** (item 11): `score(request)` over the {lat['overall']['n']} dev requests, one at a time, "
        f"through the hash path, with {lat['heads']}, the model loaded first, {WARM_UP} warm-up calls excluded, the "
        f"router scores thrown away: median {lat['overall']['median ms']:.1f} ms, p90 {lat['overall']['p90 ms']:.1f} ms. "
        "Per task (median, p90 in ms): "
        + "; ".join(f"{t} {v['median ms']:.1f}, {v['p90 ms']:.1f}" for t, v in lat["per task"].items())
        + f". Load time {lat['load seconds']:.2f} s, of which the file check took {lat['file check seconds']:.2f} s. "
        f"On {lat['machine']}, Python {lat['python']}, PyTorch {lat['torch']}, {lat['threads']} threads. It depends on "
        "the machine, so it is quoted as measured and isn't re-checked.",
    ]
    return "\n".join(out) + "\n"


def write_record(text: str, path: Path = PLAN) -> None:
    plan = path.read_text()
    start = plan.index("\n19. **The freeze record**") + 1
    end = plan.index("\n20. **The fit record**") + 1
    path.write_text(plan[:start] + text + plan[end:])


def main(argv: list[str]) -> int:
    if argv == ["drift"]:
        print(json.dumps(drift(), indent=1))
    elif argv == ["latency"]:
        print(json.dumps(latency(), indent=1))
    elif argv == ["environment"]:
        store("environment", environment())
        store("versions", library_versions())
        print(json.dumps(stored()["environment"], indent=1))
    elif argv == ["tokens"]:
        print(json.dumps(token_counts(), indent=1))
    elif argv in (["record"], ["record", "--write"]):
        text = record_text(fit.fit_train_side(), stored())
        if len(argv) == 2:
            write_record(text)
        print(text)
    else:
        print(
            "usage: python -m analysis.measure_trained_router drift | latency | environment | tokens | record [--write]"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
