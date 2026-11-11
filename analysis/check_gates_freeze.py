"""Check Stage 7's freeze commit (PLAN_v4.md §9; Amendment A17, items 12, 16, 17, 20.3 to 20.6 and 28; issue #69).

Run from the project root:  uv run python -m analysis.check_gates_freeze

Calls no Jev and no Anthropic API, and reads no dev or test answer, outcome or gate verdict. It first runs the train
measures (`analysis.gate_train`) on synthetic records worked out by hand, each with a known positive, then compares A17
item 28 with its sources:

- item 28 against the committed files (also in CI): each recorded hash is its file's, the identities are the settings',
  the cutoffs file is the one item 28 names;
- the order in the git history (where the commits are present): the settings commit is an ancestor of the freeze commit,
  the commit that first added the cutoffs file, and the files the settings commit froze are unchanged since it;
- the stored train records (the hook only, since they live in `results/gates/`): their hashes; every record names a train
  or dev item; every returned model ID is the pinned one; every Jev body and Sonnet request hashes to what the frozen code
  builds; the gate-score tables rebuild from the records; both cutoffs recompute from them; the noise lists rebuild from
  the drawing rule; item 28's text is the one `analysis.gate_train_report` writes from the records; the first train Jev
  call's Date and the train batch's creation time come after the settings commit's time;
- the keys: no stored byte holds a real key (where a key and results are present).

Until the freeze commit writes item 28, the sections that read it say so and pass.
"""

import email.utils
import hashlib
import json
import re
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

from dotenv import dotenv_values

from analysis import collect_sonnet_gate, noise_samples
from analysis import gate_train as train
from analysis import gate_train_report as report
from analysis.benchmark import read_rows
from analysis.check_gates_settings_commit import COMMIT_FILES, SETTINGS, SMOKE_RECORD
from analysis.check_jev_settings_commit import changed_since, settings_commit
from analysis.checking import git, git_bytes, refused
from analysis.collect_gates import read_free_records
from analysis.collect_jev_gate import read_records as read_jev_records
from analysis.gate_answers import read_answers
from analysis.gate_cutoff import youden_cutoff
from analysis.gate_spend import GateSpendError
from analysis.gate_table import PAID_GATES
from analysis.plan import PLAN, amendment_item
from harness.client import PROJECT_ROOT
from routers import jev, jev_gate, sonnet_gate
from routers.request import from_row

HASHED = re.compile(r"`([\w./-]+)`: `([0-9a-f]{64})`")  # a path and its sha256, as item 28 writes them
SETTINGS_COMMIT = re.compile(r"\*\*The settings commit\*\*: `([0-9a-f]{40})`, committed (\S+?)\.")
FIRST_CALL = re.compile(r"the first train Jev call's server Date `([^`]+)`; the train batch's `created_at` (\S+) ")
CONDITIONS = re.compile(r"^ *- Conditions: .*$", re.MULTILINE)
PINNED_JEV, PINNED_SONNET = "jev-1.14.0", "claude-sonnet-5-5"

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def jev_record(item: str, config: str, number: int, correct: float, follows: float, mistake: float) -> dict:
    """A stored Jev gate record with only the fields the noise measure reads."""
    answers = {k: {"type": "noul", "noul": v} for k, v in
               (("jev_gate_correct", correct), ("jev_gate_follows", follows), ("jev_gate_mistake", mistake))}  # fmt: skip
    return {"item_id": item, "config": config, "pass": number, "response": {"model": "jev-1.14.0", "answers": answers}}


def sonnet_record(item: str, config: str, score: int | None) -> dict:
    """A stored Sonnet record (batch or latency) with only the fields the drift measure reads."""
    text = "{}" if score is None else f'{{"probability_correct": {score}}}'
    message = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
               "usage": {"input_tokens": 10, "output_tokens": 2}}  # fmt: skip
    return {"item_id": item, "config": config, "result": {"type": "succeeded", "message": message}}


def difference_checks() -> None:
    first = {"a": 0.50, "b": 0.80, "c": 0.20}
    second = {"a": 0.52, "b": 0.77, "c": 0.20}
    got = train.difference(first, second)
    # |0.02| + |-0.03| + |0| = 0.05 over 3; signed 0.02 - 0.03 + 0 = -0.01 over 3.
    check(got.pairs == 3 and abs(got.mean_abs - 0.05 / 3) < 1e-12 and abs(got.shift + 0.01 / 3) < 1e-12,
          "D6 is the mean absolute difference 0.05/3 and the shift the mean signed one, second less first")  # fmt: skip
    check(got.d6 == got.mean_abs, "with no floor D6 is the mean absolute difference")
    flat = train.difference({"a": 0.5, "b": 0.7}, {"a": 0.5, "b": 0.7}, floor=0.01)
    check(
        flat.mean_abs == 0 and flat.d6 == 0.01 and flat.shift == 0, "known positive: identical scores floor D6 at 0.01"
    )
    big = train.difference({"a": 0.1}, {"a": 0.3}, floor=0.01)
    check(abs(big.d6 - 0.2) < 1e-12, "a difference above the floor is D6 itself")
    gap = train.difference({"a": 0.5, "b": None, "c": 0.4}, {"a": 0.6, "b": 0.5, "c": None})
    check(gap.pairs == 1 and gap.unscored == 2 and abs(gap.d6 - 0.1) < 1e-12,
          "a pair with a refused call (no score) in either pass is counted apart and left out of D6")  # fmt: skip
    check(refused(train.GateTrainError, lambda: train.difference({"a": 0.5}, {"b": 0.5})),
          "known negative: two passes over different items are refused")  # fmt: skip
    check(refused(train.GateTrainError, lambda: train.difference({"a": None}, {"a": 0.5})),
          "known negative: no scored pair is refused")  # fmt: skip


def noise_checks() -> None:
    settings = jev_gate.read_settings()
    one = [
        jev_record("t:1", "haiku", 1, 0.9, 0.9, 0.1),
        jev_record("t:2", "haiku", 1, 0.5, 0.5, 0.5),
        jev_record("t:3", "haiku", 1, 0.5, 0.5, 0.5),  # not in the noise pass
        jev_record("t:1", "haiku", 2, 0.9, 0.9, 0.1),
        jev_record("t:2", "haiku", 2, 0.9, 0.9, 0.1),
    ]
    got = train.jev_noise(one, settings)
    # t:1 is the same in both passes; t:2 goes from 0.5 to 0.9 on every statement (mistake 0.5 -> 0.1 inverts to 0.9),
    # so its score goes from 0.5 to the log-odds mean of three 0.9s, which is 0.9.
    check(
        set(got) == {"haiku"} and got["haiku"].pairs == 2, "the noise pass is read per config over the items it re-sent"
    )
    check(abs(got["haiku"].shift - 0.2) < 1e-9 and abs(got["haiku"].d6 - 0.2) < 1e-9,
          "pass 2 less pass 1: t:1 differs by 0 and t:2 by 0.4, so D6 and the shift are 0.2")  # fmt: skip
    lonely = [jev_record("t:1", "haiku", 2, 0.9, 0.9, 0.1)]
    check(refused(train.GateTrainError, lambda: train.jev_noise(lonely, settings)),
          "known negative: a pass-2 record with no pass 1 is refused")  # fmt: skip


def drift_checks() -> None:
    batch = [sonnet_record("t:1", "haiku", 90), sonnet_record("t:2", "haiku", 40), sonnet_record("t:3", "haiku", 70)]
    standard = [sonnet_record("t:1", "haiku", 90), sonnet_record("t:2", "haiku", 50), sonnet_record("t:3", "haiku", 70)]
    got = train.sonnet_drift(batch, standard)
    # Only t:2 differs, by 0.10; over 3 answers D6 is 0.10/3 = 0.0333, above the 0.01 floor, and the shift is +0.0333.
    check(abs(got["haiku"].d6 - 0.1 / 3) < 1e-12 and abs(got["haiku"].shift - 0.1 / 3) < 1e-12,
          "the Sonnet drift is the standard call's score/100 less the batch's: D6 and the shift 0.1/3")  # fmt: skip
    same = train.sonnet_drift(batch, batch)
    check(same["haiku"].mean_abs == 0 and same["haiku"].d6 == 0.01,
          "known positive: calls that repeat exactly give D6 the 0.01 floor, not 0")  # fmt: skip
    check(refused(train.GateTrainError, lambda: train.sonnet_drift(batch[:2], standard)),
          "known negative: a latency call with no batch score is refused")  # fmt: skip


def free_record(item: str, config: str, *, stop: bool = True, schema: bool = True, applied: bool = False,
                code: bool = True, code_applied: bool = False, first: str | None = None) -> dict:  # fmt: skip
    """A stored free-check record with only the fields the counts read."""
    return {"item_id": item, "config": config, "stop_passed": stop, "schema_passed": schema, "schema_applied": applied,
            "code_passed": code, "code_applied": code_applied, "passed": first is None, "failed_first": first}  # fmt: skip


def count_checks() -> None:
    free = [
        free_record("t:1", "haiku"),
        free_record("t:2", "haiku", code=False, code_applied=True, first="code"),
        free_record("t:3", "haiku", stop=False, code=False, code_applied=True, first="stop"),
        free_record("t:4", "haiku", schema=False, applied=True, first="schema"),
        free_record("t:1", "sonnet"),
    ]
    jev = [
        {"item_id": "t:1", "config": "haiku", "pass": 1, "refused": False, "attempts": 1, "failed_attempts": [],
         "answer_cut_bytes": 0, "prompt_cut_bytes": 0},
        {"item_id": "t:5", "config": "haiku", "pass": 1, "refused": True, "attempts": 3, "failed_attempts": [{}, {}],
         "answer_cut_bytes": 40, "prompt_cut_bytes": 0},
        {"item_id": "t:1", "config": "haiku", "pass": 2, "refused": True, "attempts": 2, "failed_attempts": [{}],
         "answer_cut_bytes": 0, "prompt_cut_bytes": 7},
    ]  # fmt: skip
    sonnet = [
        {"item_id": "t:1", "config": "haiku", "refused": False, "stop_reason": "end_turn"},
        {"item_id": "t:5", "config": "haiku", "refused": True, "stop_reason": "max_tokens"},
        {"item_id": "t:6", "config": "haiku", "refused": True, "stop_reason": "refusal"},
    ]  # fmt: skip
    got = train.count_configs(free, jev, sonnet, ("haiku", "sonnet"))
    haiku = got["haiku"]
    check(haiku.answers == 4 and haiku.rejected == 3, "haiku: four answers checked, three rejected by a free check")
    check(
        haiku.failed_first == {"stop": 1, "schema": 1, "code": 1},
        "each rejection is counted once, under the check that failed first",
    )
    check(haiku.failing == {"stop": 1, "schema": 1, "code": 2},
          "and the answers failing each check are counted apart: the stop-failing answer also fails the code check")  # fmt: skip
    check(haiku.jev_calls == 2 and haiku.jev_refused == 1 and haiku.jev_cut == 1 and haiku.jev_retries == 2,
          "Jev: two pass-1 calls (the noise pass is apart), one refused, one cut, 0 + 2 retries")  # fmt: skip
    check(haiku.jev_failed_attempts == 2, "and its failed attempts are counted")
    check(haiku.sonnet_calls == 3 and haiku.sonnet_refused == 2 and haiku.sonnet_cap_stops == 1,
          "Sonnet: three verdicts, two refused (a cap stop and a refusal), one of them a cap stop")  # fmt: skip
    check(got["sonnet"].answers == 1 and got["sonnet"].rejected == 0 and got["sonnet"].jev_calls == 0,
          "a config with no Jev call has zero, not a missing entry")  # fmt: skip
    check(refused(train.GateTrainError, lambda: train.count_configs(free, jev, sonnet, ("opus_low",))),
          "known negative: a config with no free-check record is refused")  # fmt: skip
    log = {"batches": [{"split": "train", "counts": {"succeeded": 90, "errored": 3, "expired": 1}},
                       {"split": "dev", "counts": {"succeeded": 5, "errored": 9}},
                       {"split": "train", "counts": {"succeeded": 4}}]}  # fmt: skip
    check(
        train.batch_retries(log, "train") == 4,
        "the batches' resent requests are the errored and expired ones: 3 + 1 (train's only)",
    )


def cutoff_checks() -> None:
    answers = [(0.9, True), (0.8, True), (0.7, False), (0.4, True), (0.3, False), (None, False)]
    # Wrong: 0.7, 0.3, refused (3). Right: 0.9, 0.8, 0.4 (3). At 0.8: rejects wrong 0.7, 0.3, refused = 3/3, right 0.4 = 1/3:
    # J = 2/3. At 0.9: wrong 3/3, right 0.8 and 0.4 = 2/3: J = 1/3. At 0.4: wrong 0.3 and refused = 2/3, right 0: J = 2/3.
    # A tie between 0.4 and 0.8 goes to the higher cutoff, 0.8.
    got = train.cutoff_report(answers, 0.8)
    check(abs(got.j - 2 / 3) < 1e-12 and got.wrong == 3 and got.right == 3 and got.wrong_rejected == 3 and got.right_rejected == 1,
          "at 0.8: J = 3/3 - 1/3, with all 3 wrong answers (the refused call included) and 1 of 3 right ones rejected")  # fmt: skip
    check(
        abs(train.cutoff_report(answers, 0.4).j - 2 / 3) < 1e-12,
        "at 0.4 J ties at 2/3, so the tie is real and goes to 0.8",
    )
    check(refused(train.GateTrainError, lambda: train.cutoff_report([(0.5, True)], 0.5)),
          "known negative: answers that are all right have no J")  # fmt: skip


def latency_checks() -> None:
    def call(item: str, seconds: float, first: bool = False) -> dict:
        return {"item_id": item, "seconds": seconds, "first_call": first}

    records = [call("a:1", 9.0, True), call("a:2", 1.0), call("b:1", 2.0), call("a:3", 3.0), call("b:2", 4.0)]
    got = train.sonnet_latency(records, {"a:1": "alpha", "a:2": "alpha", "a:3": "alpha", "b:1": "beta", "b:2": "beta"})
    # The first call (9.0) is out. Left: 1, 2, 3, 4: median 2.5; p90 at position 2.7 of the sorted list, 3.7.
    check(got.count == 4 and abs(got.median - 2.5) < 1e-12 and abs(got.p90 - 3.7) < 1e-12 and got.first == 9.0,
          "median 2.5 and p90 3.7 over the four calls left once the first (9.0 s) is set apart")  # fmt: skip
    alpha, beta = got.by_task["alpha"], got.by_task["beta"]
    check(alpha[2] == beta[2] == 2 and abs(alpha[0] - 2.0) < 1e-12 and abs(alpha[1] - 2.8) < 1e-12
          and abs(beta[0] - 3.0) < 1e-12 and abs(beta[1] - 3.8) < 1e-12,
          "per task (median, p90, calls): alpha 1 and 3 give 2.0 and 2.8; beta 2 and 4 give 3.0 and 3.8")  # fmt: skip
    task_of = {"a:1": "alpha", "a:2": "alpha"}
    check(refused(train.GateTrainError, lambda: train.sonnet_latency([call("a:1", 1.0), call("a:2", 1.0)], task_of)),
          "known negative: a sample whose first call is not flagged is refused")  # fmt: skip
    check(refused(train.GateTrainError, lambda: train.sonnet_latency([call("a:1", 1.0, True)], task_of)),
          "known negative: a sample with nothing left after the first call is refused")  # fmt: skip


def spend_checks() -> None:
    def verdict(i: int, o: int) -> dict:
        return {"usage": {"input_tokens": i, "output_tokens": o}}

    got = train.train_spend(
        jev=[verdict(600_000, 20), verdict(400_000, 20)], batch=[verdict(1000, 100)], latency=[verdict(2000, 200)],
        jev_price=0.042, input_price=2.0, output_price=10.0, batch_discount=0.5,
    )  # fmt: skip
    check(got.jev_input_tokens == 1_000_000 and abs(got.jev - 0.042) < 1e-12,
          "Jev: 1,000,000 input tokens at $0.042 per million is $0.042")  # fmt: skip
    check(abs(got.batch_list - 0.003) < 1e-12 and abs(got.batch - 0.0015) < 1e-12,
          "the batch verdict: 1000 in and 100 out is $0.003 at list price, $0.0015 billed at half")  # fmt: skip
    check(abs(got.latency - 0.006) < 1e-12 and abs(got.total - (0.042 + 0.0015 + 0.006)) < 1e-12,
          "the latency call is billed at list ($0.006), and the total adds what was billed: 0.042 + 0.0015 + 0.006")  # fmt: skip
    # The report reads billed tokens with the $25 cap's reader (A17 item 18), so it refuses what the cap refuses.
    prices = {"jev_price": 0.042, "input_price": 2.0, "output_price": 10.0, "batch_discount": 0.5}
    for what, bad in {"a fractional token count": verdict(1000, 100) | {"usage": {"input_tokens": 1000.5, "output_tokens": 100}},
                      "a count written as text": verdict(1000, 100) | {"usage": {"input_tokens": "1000", "output_tokens": 100}},
                      "a negative count": verdict(-1, 100), "no usage at all": {}}.items():  # fmt: skip
        check(refused(GateSpendError, partial(train.train_spend, jev=[], batch=[bad], latency=[], **prices)),
              f"known positive: a record with {what} is refused, as the cap refuses it")  # fmt: skip


def missing_lines(item: str, expected: list[str]) -> list[str]:
    """The expected entries that item 28 does not hold, word for word."""
    return [line for line in expected if line not in item]


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def held_bytes(commit: str, relative: str) -> bytes | None:
    """The file as `commit` holds it, byte for byte, or None where the commit holds no such file."""
    done = git_bytes("show", f"{commit}:{relative}")
    return done.stdout if done.returncode == 0 else None


def key_in(root: Path, key: str) -> list[str]:
    """Files under `root` holding the key's bytes (an empty key matches nothing)."""
    if not key:
        return []
    needle = key.encode()
    return [str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file() and needle in p.read_bytes()]


def guard_checks() -> None:
    check(missing_lines("... alpha 0.4063 ...", ["alpha 0.4063"]) == [], "an entry that is in the record is found")
    check(missing_lines("... alpha 0.4064 ...", ["alpha 0.4063"]) == ["alpha 0.4063"],
          "known positive: an entry with one digit changed is named")  # fmt: skip
    check(CONDITIONS.search("      - Conditions: Apple M5, Wi-Fi.\n") is not None
          and CONDITIONS.search("      - Collected: from x to y.\n") is None,
          "the conditions line is found by its words, and a collected line is not it")  # fmt: skip
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "a.txt").write_text("nothing here\n")
        check(key_in(root, "dummy-key-0123") == [] and key_in(root, "") == [], "a folder without the key names nothing")
        (root / "b.txt").write_text("x dummy-key-0123 y\n")
        check(key_in(root, "dummy-key-0123") == ["b.txt"], "known positive: a file holding the key is named")
    check(re.fullmatch(r"msgbatch_\w+", "msgbatch_011MfgV77vzfmuexkbFAX8t7") is not None, "a batch ID is a word")


def committed_files(item: str) -> None:
    recorded = dict(HASHED.findall(item))
    files = {path: digest for path, digest in recorded.items() if not path.startswith("results/")}
    expected = sorted({*COMMIT_FILES, report.CUTOFFS_FILE})
    check(sorted(files) == expected,
          f"item 28 hashes the {len(COMMIT_FILES)} files the settings commit froze and the cutoffs file, and no other committed file")  # fmt: skip
    for path in expected:
        target = PROJECT_ROOT / path
        check(
            target.exists() and files.get(path) == sha256_of(target.read_bytes()),
            f"{path} hashes as item 28 records it",
        )
    check(sorted(path for path in recorded if path.startswith("results/")) == sorted(report.STORED),
          "item 28 hashes train's stored records: the free checks, the Jev gate, the Sonnet gate and the latency calls")  # fmt: skip

    settings = json.loads((PROJECT_ROOT / SETTINGS).read_text(encoding="utf-8"))
    client = json.loads((PROJECT_ROOT / "routers/jev_settings.json").read_text(encoding="utf-8"))["jev"]
    sonnet = settings["sonnet"]
    wanted = [
        f"the Jev gate calls `{client['model']}` at `{client['endpoint']}`, ${client['price_per_million_input_tokens']} per "
        f"million input tokens, checked {client['price_checked']}",
        f"the Sonnet gate calls `{sonnet['model']}` at {sonnet['effort']} effort with `max_tokens` {sonnet['max_tokens']}",
        f"`{settings['code_check']['image']}`, digest `{settings['code_check']['image_digest']}`",
    ]
    check(missing_lines(item, wanted) == [], "the gates' identities, the image and its digest are the settings'")
    check(client["model"] == PINNED_JEV and sonnet["model"] == PINNED_SONNET, "the pinned model IDs are A17's")

    cutoffs = json.loads((PROJECT_ROOT / report.CUTOFFS_FILE).read_text(encoding="utf-8"))
    check(set(cutoffs) == set(PAID_GATES) and all(isinstance(v, float) and 0 <= v <= 1 for v in cutoffs.values()),
          "the cutoffs file holds one cutoff in [0, 1] for each paid gate")  # fmt: skip
    check(all(f"{gate}: cutoff {value!r}," in item for gate, value in cutoffs.items()),
          "item 28 records each cutoff as the cutoffs file has it")  # fmt: skip
    found = SETTINGS_COMMIT.search(item)
    check(found is not None, "item 28 names the settings commit by its full hash and commit time")


def history(item: str) -> None:
    found = SETTINGS_COMMIT.search(item)
    if not found or git("cat-file", "-e", f"{found[1]}^{{commit}}").returncode != 0:
        print("(the settings commit is not in this clone: the history section is skipped)")
        return
    commit, when = found[1], found[2]
    check(
        settings_commit(PROJECT_ROOT, SMOKE_RECORD) == commit,
        "the settings commit is the one that first added the smoke record",
    )
    check(
        git("show", "-s", "--format=%cI", commit).stdout.strip() == when,
        "item 28's commit time is the settings commit's",
    )
    check(
        changed_since(PROJECT_ROOT, commit, COMMIT_FILES) == [],
        f"the files the settings commit froze are unchanged since {commit[:10]}",
    )
    check(
        git("cat-file", "-e", f"{'0' * 40}^{{commit}}").returncode != 0,
        "known positive: a hash that names no commit is found missing",
    )
    check(
        git("cat-file", "-e", f"{commit}:{report.CUTOFFS_FILE}").returncode != 0,
        "the settings commit held no cutoffs file",
    )
    freeze = settings_commit(PROJECT_ROOT, report.CUTOFFS_FILE)
    dev = sorted((PROJECT_ROOT / "results" / "gates").glob("*_dev.jsonl"))
    if freeze is None:
        print("      (the cutoffs file is not committed yet: this is the freeze commit)")
        check(dev == [] and not (PROJECT_ROOT / "results" / "gates" / "sonnet_gate_dev.jsonl").exists(),
              "no dev gate record is stored before the freeze commit")  # fmt: skip
        return
    check(
        git("merge-base", "--is-ancestor", commit, freeze).returncode == 0,
        "the settings commit is an ancestor of the freeze commit",
    )
    check(git("merge-base", "--is-ancestor", freeze, commit).returncode != 0,
          "known positive: the freeze commit is not an ancestor of the settings commit")  # fmt: skip
    recorded = dict(HASHED.findall(item))
    held = {p: held_bytes(freeze, p) for p in recorded if not p.startswith("results/")}
    check(all(b is not None and sha256_of(b) == recorded[p] for p, b in held.items()),
          "the freeze commit held each committed file as item 28 records it")  # fmt: skip
    when_freeze = datetime.fromisoformat(git("show", "-s", "--format=%cI", freeze).stdout.strip())
    for path in dev:
        records = read_jev_records("dev") if path.name.startswith("jev_gate") else []
        if records:
            first = min(email.utils.parsedate_to_datetime(r["server_date"]) for r in records)
            check(first > when_freeze, f"{path.name}'s first call comes after the freeze commit")
    log = collect_sonnet_gate.load_log(collect_sonnet_gate.batch_log_path())
    for batch in (b for b in log["batches"] if b["split"] == "dev"):
        check(
            datetime.fromisoformat(batch["created_at"]) > when_freeze,
            f"dev batch {batch['id']} was created after the freeze commit",
        )


def items_in(path: Path) -> set[str]:
    return {json.loads(line)["item_id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def stored(item: str) -> None:
    recorded = dict(HASHED.findall(item))
    folder = PROJECT_ROOT / "results" / "gates"
    for relative in report.STORED:
        check(
            sha256_of((PROJECT_ROOT / relative).read_bytes()) == recorded.get(relative),
            f"{relative} hashes as item 28 records it",
        )

    sample = (PROJECT_ROOT / next(iter(report.STORED))).read_bytes()
    check(sha256_of(sample + b" ") != recorded.get(next(iter(report.STORED))),
          "known positive: a stored file with one byte added no longer hashes as recorded")  # fmt: skip

    # Every gate record, smoke included, names a train or dev item; no test file exists before Stage 8's unlock.
    allowed = {r["item_id"] for split in ("train", "dev") for r in read_rows(split=split)}
    check(all(items_in(path) <= allowed for path in sorted(folder.glob("*.jsonl"))),
          "every gate record names an item of the benchmark reader's train or dev split")  # fmt: skip
    check(not list(folder.glob("*_test.jsonl")), "no test gate record exists")

    free = read_free_records("train")
    jev_records = read_jev_records("train")
    batch = collect_sonnet_gate.read_sonnet_records("train")
    latency = collect_sonnet_gate.read_latency_records("train")
    train_items = {r["item_id"] for r in read_rows(split="train")}
    check({r["item_id"] for r in free} == train_items and len(free) == 3 * len(train_items),
          "the free checks cover every train answer of haiku, sonnet and opus_low once")  # fmt: skip

    # The model IDs are the pinned ones, and every body and request hashes to what the frozen code builds.
    check({r["model"] for r in jev_records} == {PINNED_JEV}
          and all(r["response"]["model"] == PINNED_JEV for r in jev_records),
          f"every Jev gate record returned model ID {PINNED_JEV}")  # fmt: skip
    check(
        {r["model"] for r in batch + latency} == {PINNED_SONNET},
        f"every Sonnet gate record returned model ID {PINNED_SONNET}",
    )
    requests = {r["item_id"]: from_row(r) for r in read_rows(split="train")}
    answers = {(a.item_id, a.config): a.answer for a in read_answers("train")}
    jev_settings = jev_gate.read_settings()
    sonnet_settings = sonnet_gate.load_settings()
    bad_jev = [
        r["item_id"] for r in jev_records
        if jev_gate.gate_call(requests[r["item_id"]], answers[(r["item_id"], r["config"])], jev_settings,
                              budget_bytes=jev_settings.budget_bytes).sha256 != r["body_sha256"]
    ]  # fmt: skip
    check(bad_jev == [], f"every stored Jev body hashes to what the frozen code builds ({len(bad_jev)} differ)")
    bad_sonnet = [
        r["custom_id"] for r in batch + latency
        if sonnet_gate.request_sha256(sonnet_gate.build_request(sonnet_settings, requests[r["item_id"]],
                                                                answers[(r["item_id"], r["config"])])) != r["request_sha256"]
    ]  # fmt: skip
    check(
        bad_sonnet == [],
        f"every stored Sonnet request hashes to what the frozen code builds ({len(bad_sonnet)} differ)",
    )

    first = min(email.utils.parsedate_to_datetime(r["server_date"]) for r in jev_records if r["pass"] == 1)
    named = FIRST_CALL.search(item)
    check(named is not None and email.utils.parsedate_to_datetime(named[1]) == first,
          f"item 28's first train Jev call Date is the first stored call's: {first:%a, %d %b %Y %H:%M:%S} GMT")  # fmt: skip
    log = collect_sonnet_gate.load_log(collect_sonnet_gate.batch_log_path())
    created = min(datetime.fromisoformat(b["created_at"]) for b in log["batches"] if b["split"] == "train")
    check(
        named is not None and datetime.fromisoformat(named[2]) == created, "and its train batch created_at is the log's"
    )
    found = SETTINGS_COMMIT.search(item)
    if found:
        settings_time = datetime.fromisoformat(found[2])
        check(first > settings_time and created > settings_time,
              "the first train Jev call and the train batch both came after the settings commit's time")  # fmt: skip
        check(min(datetime.fromisoformat(r["code_checked_at"]) for r in free if r["code_applied"]) > settings_time,
              "so did the first code-check record")  # fmt: skip

    # The tables rebuild, the cutoffs recompute, the noise lists rebuild, and item 28 is what the records write.
    built = report.load()
    for gate in PAID_GATES:
        recomputed = youden_cutoff(report.train_answers(built.table, built.matrix, gate))
        cutoffs = json.loads((PROJECT_ROOT / report.CUTOFFS_FILE).read_text(encoding="utf-8"))
        check(recomputed == cutoffs[gate] == built.cutoffs[gate],
              f"the {gate} cutoff recomputes from the stored train records: {recomputed!r}")  # fmt: skip
    check(len(built.table.rows) == len(free) > 0,
          "train's gate-score table rebuilds from the records, one row per answer")  # fmt: skip
    lists = noise_samples.noise_lists(free, noise_samples.read_noise_settings())
    check(built.lists == lists and len(lists["jev"]) == 100 and len(lists["sonnet"]) == 80,
          "the noise lists rebuild from the drawing rule: 100 for the Jev gate, 80 for the Sonnet gate")  # fmt: skip
    check(set(lists["sonnet"]) <= set(lists["jev"]), "the Sonnet gate's list is within the Jev gate's")
    pass_two = {r["item_id"] for r in jev_records if r["pass"] == 2}
    check(pass_two == {i for i, _ in lists["jev"]}, "the Jev noise pass sent the list and nothing else")
    check(
        {r["item_id"] for r in latency} == {i for i, _ in lists["sonnet"]},
        "the Sonnet latency calls are the list and nothing else",
    )
    check(len(batch) == sum(r["passed"] for r in free if r["config"] in sonnet_settings.gated_configs)
          and sum(r["pass"] == 1 for r in jev_records) == sum(r["passed"] for r in free),
          "every answer that passed the free checks has its Jev call (and its Sonnet verdict, bar opus)")  # fmt: skip
    item_lists = [
        re.findall(r"`([\w.-]+:[\w.-]+)`", line)
        for line in item.splitlines()
        if "noise list" in line or "latency sample" in line
    ]
    check([[i for i, _ in lists["jev"]], [i for i, _ in lists["sonnet"]]] == [lst for lst in item_lists if lst],
          "item 28 holds both noise lists, as the rule builds them")  # fmt: skip
    condition = CONDITIONS.search(item)
    check(condition is not None, "item 28 records the machine and network conditions")
    if condition:
        expected = report.render(built, condition[0].strip()[2:])
        check(expected.strip() == item.strip(),
              "item 28's text is what analysis.gate_train_report writes from the stored records")  # fmt: skip
        check(expected.replace("cutoff", "cutof", 1).strip() != item.strip() and expected.strip() != item.replace("0", "1", 1).strip(),
              "known positive: item 28 with one word or one digit changed is not the rebuilt text")  # fmt: skip
        if expected.strip() != item.strip():
            for number, (a, b) in enumerate(zip(expected.splitlines(), item.splitlines(), strict=False)):
                if a != b:
                    print(f"      first difference at line {number}:\n      rebuilt {a[:200]}\n      item 28 {b[:200]}")
                    break


def keys_not_stored() -> None:
    values = dotenv_values(PROJECT_ROOT / ".env")
    keys = [v for k in (jev.KEY_VARIABLE, "ANTHROPIC_API_KEY") if (v := values.get(k))]
    folder = PROJECT_ROOT / "results" / "gates"
    if not keys or not folder.exists():
        print("(no key or no results/gates here: this section is skipped)")
        return
    check(all(key_in(folder, key) == [] for key in keys), "results/gates holds no copy of a real key")
    check(all(key_in(PROJECT_ROOT / name, key) == [] for key in keys for name in ("routers", "analysis", "docs", "figures")),
          "nor do the committed folders")  # fmt: skip


def main() -> int:
    print("-- D6 and the shift (synthetic, worked by hand) --")
    difference_checks()
    noise_checks()
    drift_checks()
    print("\n-- counts per config and the cutoffs' J (synthetic, worked by hand) --")
    count_checks()
    cutoff_checks()
    print("\n-- the Sonnet latency sample and the train spend (synthetic, worked by hand) --")
    latency_checks()
    spend_checks()
    print("\n-- the check's own guards --")
    guard_checks()
    item = amendment_item(PLAN.read_text(encoding="utf-8"), "A17", 28)
    if "It will hold:" in item:
        print("\n(item 28 is not written yet: this is before the freeze commit, and the rest is skipped)")
    else:
        print("\n-- item 28 against the committed files --")
        committed_files(item)
        print("\n-- the order in the git history (where the commits are present) --")
        history(item)
        print("\n-- the stored train records (the hook only) --")
        if (PROJECT_ROOT / "results" / "gates" / "free_checks_train.jsonl").exists():
            stored(item)
        else:
            print("(results/gates holds no train records here: this section is skipped)")
        print("\n-- the keys --")
        keys_not_stored()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
