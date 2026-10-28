"""Check the Jev freeze commit's promises (PLAN_v4.md §8 3c; Amendment A15, items 10, 13 and 19; issue #39).

Run from the project root:  uv run python -m analysis.check_jev_freeze

Calls no Jev, reads no dev or test request or answer. It first runs the train report's measures on synthetic records
worked out by hand (each with a known positive), then compares A15 item 19 with its sources:

- the measures (synthetic): counts, tokens, latency with the first call and reconnects set apart, the model ID, the
  server `Date`, the body hashes and the truncation count;
- item 19 against the committed files (also in CI): each recorded hash is its file's, the closure list is the one worked
  out from `routers/jev.py`'s imports, C per config is the weights file's, the noise sample is the settings file's,
  Jev's identity is the settings', the library versions are the installed ones;
- the order in the git history (where the commits are present; CI's shallow clone lacks them): the settings commit is
  an ancestor of the freeze commit, which is the commit that added the weights file, and the files the settings commit
  froze are unchanged since it;
- the stored train responses (the hook only, since they live in `results/jev/`): their hashes, every model ID, every body
  hash against the frozen call building, the counts, tokens and latency, the first call's `Date` after the settings
  commit's commit time, and item 19's fit entries (C, the signal check, D6) against a refit from the stored answers;
- the key: no stored byte holds the real key (where a key and results are present).
"""

import email.utils
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from analysis import collect_jev, fit_jev_router
from analysis import jev_train_report as report
from analysis.benchmark import read_rows
from analysis.check_jev_settings_commit import COMMIT_FILES, changed_since, settings_commit
from analysis.freeze import closure
from analysis.plan import PLAN, amendment_item
from harness.client import PROJECT_ROOT
from routers import jev, jev_call
from routers.ladder import SCORED
from routers.request import Request, from_row

HASHED = re.compile(r"`([\w./-]+)`: `([0-9a-f]{64})`")  # a path and its sha256, as items 19 and 20 write them
SETTINGS_COMMIT = re.compile(r"\*\*The settings commit\*\*: `([0-9a-f]{40})`, committed (\S+?)\.")
FIRST_CALL = re.compile(r"\*\*The first train call\*\*: server Date `([^`]+)`")
WEIGHTS = "routers/jev_weights.json"
FIT_SCRIPT = "analysis/fit_jev_router.py"
SETTINGS_BUILDER = "analysis/build_jev_settings.py"
SETTINGS = "routers/jev_settings.json"
STORED = {router: f"results/jev/{router}_train.jsonl" for router in collect_jev.ROUTERS}

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except report.ReportError:
        return True
    return False


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def git_bytes(commit: str, relative: str) -> bytes | None:
    done = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=PROJECT_ROOT, capture_output=True, check=False)
    return done.stdout if done.returncode == 0 else None


def missing_lines(item: str, expected: Sequence[str]) -> list[str]:
    """The expected entries that item 19 does not hold, word for word."""
    return [line for line in expected if line not in item]


def frozen_paths() -> list[str]:
    """Item 19's committed files (A15 item 19): `routers/jev.py`'s import closure, the settings builder, the fit script,
    the settings file and the weights file."""
    return sorted({*closure("routers/jev.py"), SETTINGS_BUILDER, FIT_SCRIPT, SETTINGS, WEIGHTS})


def key_in(root: Path, key: str) -> list[str]:
    """Files under `root` holding the key's bytes (an empty key matches nothing)."""
    if not key:
        return []
    needle = key.encode()
    return [str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file() and needle in p.read_bytes()]


def record(
    item: str, seconds: float, *, tokens: int = 100, number: int = 1, attempts: int = 1, failed: int = 0,
    new: bool = False, refusal: bool = False, model: str = "jev-1.14.0", date: str = "Wed, 28 Oct 2026 03:04:01 GMT",
    sha: str = "0" * 64,
) -> dict:  # fmt: skip
    """A stored record with only the fields the measures read, in values easy to work by hand."""
    return {
        "item_id": item, "pass": number, "body_sha256": sha, "model": model, "server_date": date, "seconds": seconds,
        "usage": {"input_tokens": tokens, "output_tokens": 5}, "attempts": attempts,
        "failed_attempts": [{"status": 500}] * failed, "new_connection": new, "refused": refusal,
    }  # fmt: skip


def fit_lines(result: fit_jev_router.FitResult) -> list[str]:
    """The fit entries of item 19, from the fit script's own result: the counts, C per config and whether it sits at an
    end of the grid, the signal check (log loss, then pooled within-task concordance) and D6 with the shift."""
    fmt = fit_jev_router.fmt
    lines = [fit_jev_router.report(result).splitlines()[0]]
    for config in SCORED:
        end = result.chosen[config].at_end
        lines.append(f"{config} {result.chosen[config].c:g} ({end + ' of the grid' if end else 'inside the grid'})")
        lines.append(f"{config}: " + "; ".join(
            f"{name} {fmt(row['log loss'])}, {fmt(row['concordance'][None])}" for name, row in result.signal[config].items()
        ))  # fmt: skip
    for router, by_config in result.noise.items():
        lines.append(f"{router}: " + ", ".join(f"{c} D6 {v.d6:.4f} shift {v.shift:+.4f}" for c, v in by_config.items()))
    return lines


def largest_gap(first: Mapping, second: Mapping) -> float:
    """The largest absolute difference between two weights files' numbers (C, standardizer, coefficients, intercepts)."""
    gaps = [
        abs(a - b)
        for part in ("mean", "scale")
        for a, b in zip(first["standardizer"][part], second["standardizer"][part], strict=True)
    ]
    for config in SCORED:
        one, two = first["heads"][config], second["heads"][config]
        gaps += [abs(one["c"] - two["c"]), abs(one["intercept"] - two["intercept"])]
        gaps += [abs(a - b) for a, b in zip(one["coef"], two["coef"], strict=True)]
    return max(gaps)


def guards() -> None:
    expected = ["alpha 0.4063, 0.5000"]
    check(
        missing_lines("... alpha 0.4063, 0.5000; beta ...", expected) == [], "an entry that is in the record is found"
    )
    check(missing_lines("... alpha 0.4064, 0.5000; beta ...", expected) == expected,
          "known positive: an entry with one digit changed is named")  # fmt: skip
    weights = (
        json.loads(Path(PROJECT_ROOT / WEIGHTS).read_text(encoding="utf-8"))
        if (PROJECT_ROOT / WEIGHTS).exists()
        else None
    )
    if weights is not None:
        moved = json.loads(json.dumps(weights))
        moved["heads"]["haiku"]["coef"][0] += 1e-6
        check(largest_gap(weights, weights) == 0 and 1e-7 < largest_gap(weights, moved) < 1e-5,
              "known positive: a coefficient moved by 1e-6 is a gap of 1e-6, so the 1e-9 refit tolerance can fail")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "a.txt").write_text("nothing here\n")
        check(key_in(root, "dummy-key-0123") == [] and key_in(root, "") == [], "a folder without the key names nothing")
        (root / "b.txt").write_text("x dummy-key-0123 y\n")
        check(key_in(root, "dummy-key-0123") == ["b.txt"], "known positive: a file holding the key is named")


def committed_files(item: str) -> None:
    recorded = dict(HASHED.findall(item))
    files = {path: digest for path, digest in recorded.items() if not path.startswith("results/")}
    expected = frozen_paths()
    check(sorted(files) == expected,
          f"item 19 hashes the {len(expected)} files of routers/jev.py's closure, the builder, the fit script, the settings "
          "and the weights, and no other committed file")  # fmt: skip
    for path in expected:
        target = PROJECT_ROOT / path
        check(target.exists() and files.get(path) == report.file_sha256(target), f"{path} hashes as item 19 records it")
    check(sorted(path for path in recorded if path.startswith("results/")) == sorted(STORED.values()),
          "item 19 hashes train's stored responses for each router")  # fmt: skip

    settings = json.loads((PROJECT_ROOT / SETTINGS).read_text(encoding="utf-8"))
    identity = settings["jev"]
    wanted = [
        f"endpoint `{identity['endpoint']}`", f"model `{identity['model']}`",
        f"${identity['price_per_million_input_tokens']} per million input tokens, checked {identity['price_checked']}",
    ]  # fmt: skip
    check(
        missing_lines(item, wanted) == [], "Jev's identity is the settings file's: endpoint, model, price and its date"
    )
    versions = [f"numpy {importlib.metadata.version('numpy')}", f"scikit-learn {importlib.metadata.version('scikit-learn')}",
                f"Python {platform.python_version()}"]  # fmt: skip
    check(missing_lines(item, versions) == [], f"the versions are the installed ones: {', '.join(versions)}")
    sample = settings["noise_sample"]
    ids = ", ".join(f"`{i}`" for i in sample["item_ids"])
    check(len(sample["item_ids"]) == 100 and missing_lines(item, [f"seed {sample['seed']}, 100 item IDs: {ids}"]) == [],
          "the noise sample's 100 item IDs and seed are the settings file's")  # fmt: skip

    weights = json.loads((PROJECT_ROOT / WEIGHTS).read_text(encoding="utf-8"))
    check(set(weights["heads"]) == set(SCORED) and all(len(h["coef"]) == jev.FEATURE_COUNT for h in weights["heads"].values())
          and len(weights["standardizer"]["mean"]) == jev.FEATURE_COUNT,
          f"the weights file holds {len(SCORED)} heads of {jev.FEATURE_COUNT} coefficients and a standardizer to match")  # fmt: skip
    for config in SCORED:
        c = weights["heads"][config]["c"]
        end = (
            "lower end of the grid"
            if c == fit_jev_router.GRID[0]
            else "upper end of the grid"
            if c == fit_jev_router.GRID[-1]
            else "inside the grid"
        )
        check(
            missing_lines(item, [f"{config} {c:g} ({end})"]) == [], f"{config}'s C is the weights file's: {c:g}, {end}"
        )


def history(item: str) -> None:
    found = SETTINGS_COMMIT.search(item)
    check(found is not None, "item 19 names the settings commit by its full hash and commit time")
    if not found or git("cat-file", "-e", f"{found[1]}^{{commit}}").returncode != 0:
        print("(the settings commit is not in this clone: the history section is skipped)")
        return
    commit, when = found[1], found[2]
    check(
        settings_commit(PROJECT_ROOT, SETTINGS) == commit,
        "the settings commit is the one that first added the settings file",
    )
    check(
        git("show", "-s", "--format=%cI", commit).stdout.strip() == when,
        "item 19's commit time is the settings commit's",
    )
    check(changed_since(PROJECT_ROOT, commit, COMMIT_FILES) == [],
          f"the files the settings commit froze are unchanged since {commit[:10]}")  # fmt: skip
    check(
        git("cat-file", "-e", f"{'0' * 40}^{{commit}}").returncode != 0,
        "known positive: a hash that names no commit is found missing",
    )
    check(git("cat-file", "-e", f"{commit}:{WEIGHTS}").returncode != 0, "the settings commit held no weights file")
    dev = sorted((PROJECT_ROOT / "results" / "jev").glob("*_dev.jsonl"))
    freeze = settings_commit(PROJECT_ROOT, WEIGHTS)
    if freeze is None:
        print("      (the weights file is not committed yet: this is the freeze commit)")
        check(dev == [], "no dev response is stored before the freeze commit")
        return
    check(
        git("merge-base", "--is-ancestor", commit, freeze).returncode == 0,
        "the settings commit is an ancestor of the freeze commit",
    )
    check(
        git("merge-base", "--is-ancestor", freeze, commit).returncode != 0,
        "known positive: the freeze commit is not an ancestor of the settings commit",
    )
    recorded = dict(HASHED.findall(item))
    held = {p: git_bytes(freeze, p) for p in frozen_paths()}
    check(all(b is not None for b in held.values())
          and all(hashlib.sha256(b).hexdigest() == recorded[p] for p, b in held.items() if b is not None),
          "the freeze commit held each frozen file as item 19 records it")  # fmt: skip
    when_freeze = datetime.fromisoformat(git("show", "-s", "--format=%cI", freeze).stdout.strip())
    for path in dev:
        records = collect_jev.read_records_file(path, path.name.split("_dev")[0], "dev")
        check(
            bool(records) and report.first_date(records) > when_freeze,
            f"{path.name}'s first call comes after the freeze commit",
        )


def stored(item: str) -> None:
    recorded = dict(HASHED.findall(item))
    loaded = collect_jev.load_settings()
    train = read_rows(split="train")
    task_of = {row["item_id"]: row["source"] for row in train}
    requests = {row["item_id"]: from_row(row) for row in train}
    expected: list[str] = []
    by_router = {}
    for router, relative in STORED.items():
        check(
            report.file_sha256(PROJECT_ROOT / relative) == recorded.get(relative),
            f"{relative} hashes as item 19 records it",
        )
        records = collect_jev.read_records(router, "train")
        by_router[router] = records
        counts, rebuilt = report.call_counts(records), report.rebuild(router, records, requests, loaded.call)
        check(counts.models == ("jev-1.14.0",), f"{router}: every returned model ID is jev-1.14.0")
        check(rebuilt.mismatched == [], f"{router}: every stored body hashes to what the frozen call building gives")
        pass_one = {r["item_id"] for r in records if r["pass"] == 1}
        pass_two = {r["item_id"] for r in records if r["pass"] == 2}
        check(pass_one == set(requests) and pass_two == set(loaded.noise_item_ids),
              f"{router}: every train request has its pass 1, and the noise sample alone has a pass 2")  # fmt: skip
        expected += list(report.record_lines(router, counts, rebuilt, report.latency(records, task_of)))
    check(missing_lines(item, expected) == [], "item 19's calls and latency entries are the stored records' measures")

    first = min(report.first_date(records) for records in by_router.values())
    named = FIRST_CALL.search(item)
    check(named is not None and email.utils.parsedate_to_datetime(named[1]) == first,
          f"item 19's first train call Date is the first stored call's: {first:%a, %d %b %Y %H:%M:%S} GMT")  # fmt: skip
    found = SETTINGS_COMMIT.search(item)
    check(
        found is not None and first > datetime.fromisoformat(found[2]),
        "the first train call came after the settings commit's time",
    )
    stamps = sorted(datetime.fromisoformat(r["collected_at"]) for rs in by_router.values() for r in rs)
    check(f"collected from {stamps[0].isoformat()} to {stamps[-1].isoformat()}" in item,
          "the dates and times of the train calls are the stored records'")  # fmt: skip

    result = fit_jev_router.fit_from_files()
    weights = json.loads((PROJECT_ROOT / WEIGHTS).read_text(encoding="utf-8"))
    gap = largest_gap(json.loads(fit_jev_router.weights_text(result.weights)), weights)
    check(gap < 1e-9, f"G+'s refit from the stored answers reproduces the weights file to 1e-9 (largest gap {gap:.1e})")
    check(
        missing_lines(item, fit_lines(result)) == [],
        "item 19's fit entries are the refit's: counts, C, signal check and D6",
    )


def key_not_stored() -> None:
    key = dotenv_values(PROJECT_ROOT / ".env").get(jev.KEY_VARIABLE) or ""
    if not key or not (PROJECT_ROOT / "results" / "jev").exists():
        print("(no key or no results/jev here: this section is skipped)")
        return
    check(key_in(PROJECT_ROOT / "results" / "jev", key) == [], "results/jev holds no copy of the real key")
    check(all(key_in(PROJECT_ROOT / folder, key) == [] for folder in ("routers", "analysis", "docs", "figures")),
          "nor do the committed folders")  # fmt: skip


def main() -> int:
    print("-- the train report's measures (synthetic records, worked by hand) --")
    records = [
        record("a:1", 0.5, tokens=100, new=True),  # the first call, which opened the connection
        record("a:2", 0.1, tokens=200),
        record("b:1", 0.2, tokens=300, attempts=3, failed=2),
        record("b:2", 0.3, tokens=400, new=True, refusal=True),  # a reconnect
        record("a:3", 0.4, tokens=500, attempts=2, failed=1),
        record("a:1", 0.5, tokens=600, number=2),  # the noise pass
    ]
    counts = report.call_counts(records)
    check(counts.calls == 6 and counts.pass_two == 1, "six calls, one of them in the noise pass")
    check(counts.refused == 1 and counts.retries == 3 and counts.failed_attempts == 3,
          "one refusal; attempts beyond the first 0 + 0 + 2 + 0 + 1 + 0 = 3; failed attempts 2 + 1 = 3")  # fmt: skip
    check(counts.reconnects == 1, "one reconnect besides the first call")
    check(counts.input_tokens == 2100 and counts.median_tokens == 350 and counts.max_tokens == 600,
          "tokens: total 2100, median of 100..600 is 350, largest 600")  # fmt: skip
    check(counts.models == ("jev-1.14.0",), "the model IDs seen are listed")
    check(report.call_counts([*records, record("c:1", 0.1, model="jev-1.13.0")]).models == ("jev-1.13.0", "jev-1.14.0"),
          "known positive: a call under another model ID is listed beside the right one")  # fmt: skip

    timing = report.latency(records, {"a:1": "alpha", "a:2": "alpha", "a:3": "alpha", "b:1": "beta", "b:2": "beta"})
    # Left after the first call and the reconnect: 0.1, 0.2, 0.4, 0.5. Median (0.2 + 0.4) / 2 = 0.3; p90 sits at
    # position 0.9 * 3 = 2.7 of the sorted list, 0.4 + 0.7 * (0.5 - 0.4) = 0.47.
    check(abs(timing.median - 0.3) < 1e-12 and abs(timing.p90 - 0.47) < 1e-12 and timing.count == 4,
          "median 0.3 and p90 0.47 of the four calls left after the first and the reconnect")  # fmt: skip
    check(timing.first == 0.5 and timing.reconnects == [0.3], "the first call's time and the reconnects' are set apart")
    # Alpha keeps 0.1, 0.4 and 0.5 (the first call is out): median 0.4, p90 at position 1.8, 0.4 + 0.8 * 0.1 = 0.48.
    alpha = timing.by_task["alpha"]
    check(abs(alpha[0] - 0.4) < 1e-12 and abs(alpha[1] - 0.48) < 1e-12 and alpha[2] == 3
          and timing.by_task["beta"] == (0.2, 0.2, 1),
          "per task: alpha's median 0.4 and p90 0.48 over three calls, beta's one call 0.2 (its reconnect is out)")  # fmt: skip
    check(refused(lambda: report.latency([record("a:1", 0.5, new=True)], {"a:1": "alpha"})),
          "known negative: a file of one call, with nothing left after the first, is refused")  # fmt: skip
    check(refused(lambda: report.latency(records, {"a:1": "alpha"})), "a call whose task is unknown is refused")

    first = report.first_date(records)
    check(first == email.utils.parsedate_to_datetime("Wed, 28 Oct 2026 03:04:01 GMT"), "the first call's Date is read")
    check(
        refused(lambda: report.first_date([record("a:1", 0.1, date="yesterday")])), "a Date that isn't one is refused"
    )
    check(refused(lambda: report.first_date([])), "no calls, no first Date")

    print("\n-- body hashes and truncation (made-up requests, the committed settings) --")
    settings = jev_call.read_settings()
    request = Request("Say hi.", None)
    good = jev_call.g_call(request, settings).sha256
    items = {"made-up:1": request}
    rebuilt = report.rebuild("jev", [record("made-up:1", 0.1, sha=good)], items, settings)
    check(rebuilt.mismatched == [] and rebuilt.cut == 0, "a record whose hash the frozen code builds matches, uncut")
    rebuilt = report.rebuild("jev", [record("made-up:1", 0.1, sha="f" * 64)], items, settings)
    check(rebuilt.mismatched == ["made-up:1"], "known positive: a record with another hash is named")
    long = {"made-up:2": Request("x" * 40_000, None)}
    cut = jev_call.g_call(long["made-up:2"], settings)
    rebuilt = report.rebuild("jev", [record("made-up:2", 0.1, sha=cut.sha256)], long, settings)
    check(cut.cut_bytes > 0 and rebuilt.cut == 1 and rebuilt.mismatched == [],
          "a prompt over the byte budget is counted as cut, and its cut body still hashes as recorded")  # fmt: skip
    plus = jev_call.g_plus_call(request, settings).sha256
    check(report.rebuild("jev_bolstered", [record("made-up:1", 0.1, sha=plus)], items, settings).mismatched == [],
          "G+'s body is rebuilt by G+'s call building")  # fmt: skip
    check(report.rebuild("jev", [record("made-up:1", 0.1, sha=plus)], items, settings).mismatched == ["made-up:1"],
          "known positive: G+'s body is not G's")  # fmt: skip
    check(refused(lambda: report.rebuild("jev", [record("nope", 0.1)], items, settings)),
          "a record of an item with no request is refused")  # fmt: skip

    item = amendment_item(PLAN.read_text(encoding="utf-8"), "A15", 19)
    print("\n-- item 19's guards (synthetic) --")
    guards()

    print("\n-- item 19 against the committed files --")
    committed_files(item)

    print("\n-- the order in the git history (where the commits are present) --")
    history(item)

    print("\n-- the stored train responses and the fit (the hook only) --")
    if all((PROJECT_ROOT / path).exists() for path in STORED.values()):
        stored(item)
    else:
        print("(results/jev holds no train responses here: this section is skipped)")

    print("\n-- the key --")
    key_not_stored()

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
