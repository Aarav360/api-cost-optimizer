"""Check the final evaluation's complete record and the holds that tie test's tables to it (PLAN_v4.md Amendment A19, items 10
and 13; issue #77, #92).

Run from the project root:  uv run python -m analysis.check_final_record

The rehearsal's record, from a whole go on the made-up world (`analysis/rehearsal_world.py`), goes through
`analysis/final_record_fields.py`'s `verify`, the function #95 runs on the committed record after the go. No results/, no
Docker, no Jev or Anthropic call, and nothing near the real store. Expected values come from sources apart from the code under
test: item 10's bullets as written in the plan, the standard SHA-256 test vector for "abc", hashes taken here with `hashlib`
over the files on disk, and the made-up world's own counts (8 items, 4 configs).

- **the fields.** The rehearsal's record holds every field of item 10 (the fields cover its eight bullets, none more), and a
  record with any one field removed is refused naming that field. Finer breaks are found too: a check of the unlock left out, a
  drift measure without its D8, a replay policy without its first call, a step without its end or request counts, a go whose
  hook failed. A resume's record holds each go's unlock, so a field gone from the first go's unlock is named by its go.
- **nothing shown early.** The record's steps before `tables` hold no key that carries an outcome (the stops, refusals, cuts,
  truncations, fold shares); the check shows a planted one is found. The tables step holds them.
- **the files.** The record holds the sha256 of every test record file: 16 here, the ones `STAGE_8_FILES` names, E's and F's
  scores, test's matrix and the final evaluation's own files. Each equals a hash taken here, the hash function gives the standard
  vector for a known file, and a file changed by one byte, one removed, one the record doesn't name, an entry for a file that isn't
  one and an entry with a wrong size each refuse. The closing step refuses a go whose file is missing.
- **the holds.** The tables, the stored files and the record agree on the made-up world, and each hold fails when the number it
  holds moves: an item count (tables, steps, the F and G counts), a config's graded answers, the request counts of the free checks,
  the Jev gate, the Sonnet gate and G and G+ against the stored files and the tables, the refusals and truncations (the record's
  stops against the tables', the gates' refusal counts against the stored calls), the spend (against the stored records, the caps and
  the steps' own), and the times (order, the unlock's, the replay's).
"""

import copy
import dataclasses
import hashlib
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from analysis import final_record_fields as fields
from analysis.check_final_run import Rehearsal, rehearsed
from analysis.final_record_fields import FilesError, field_problems, file_hashes, file_problems, held_problems, verify
from analysis.final_run import StepRefusal, record_step
from analysis.router_names import G_PLUS

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# Item 10's bullets, each as the prefixes of the fields that hold it (written from the plan's text, not from FIELDS)
BULLETS = {
    "the unlock's commit, time and hook result, each check's result, the environment": {"unlock"},
    "the liveness calls' returned model IDs": {"liveness"},
    "the drift checks' numbers: D6, D8, the shifts, the flips, F's largest difference": {"drift"},
    "each step's start and end, request counts, refusals and truncations, spend against each cap": {
        "steps",
        "stops",
        "truncations",
        "refusals",
        "spend",
    },
    "the first test Jev call's Date, the Sonnet test batch's ID and creation time": {"order"},
    "F's fold shares on test, and the live-replay items drawn at the unlock": {"folds", "replay"},
    "the live replay's conditions, with the first call and any reconnects apart": {"replay"},
    "the sha256 of every test record file": {"files"},
}
# Taking out a field takes out what is read through it
ALSO_LOST = {"unlock.checks": {"replay.items"}}
# A named test file for each of the other kinds of record file, as the closing step must name them
EXPECTED_FILES = 16  # 9 Stage 8 gate and Jev files, 2 results files (scores, matrix), 5 of the final evaluation's own
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # the standard test vector for b"abc"
OUTCOME_KEYS = {"stops", "refused", "rejected", "cut", "truncated", "emptied", "fold_shares", "fold_counts", "jev_refused", "jev_cut",
                "sonnet_refused", "sonnet_cap_stops", "quality", "correct", "wrong"}  # fmt: skip


def keys_in(node: Any) -> Iterator[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from keys_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from keys_in(value)


def sha256_here(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(go: dict, name: str) -> dict:
    return next(s for s in go["steps"] if s["name"] == name)


def deleter(path: fields.FieldPath) -> Callable[[dict], None]:
    """A damage that removes the leaf `path` names."""
    return lambda go: fields.delete_at(go, path)


def broken(record: dict, damage: Callable[[dict], None]) -> list[str]:
    """What `field_problems` reports for a copy of the record after `damage` is applied to its last go."""
    changed = copy.deepcopy(record)
    damage(changed["gos"][-1])
    return field_problems(changed)


# ---- the fields -------------------------------------------------------------------------------------------------------


def the_fields(record: dict) -> None:
    print("-- the record holds every field of item 10, and a record without any one is refused --")
    check(field_problems(record) == [], f"the rehearsal's record holds every field {field_problems(record)}")
    prefixes = {f.name.split(".")[0] for f in fields.FIELDS}
    wanted = set().union(*BULLETS.values())
    check(
        prefixes == wanted, f"the fields cover item 10's eight bullets and no more: {sorted(prefixes ^ wanted)} differ"
    )
    check(len({f.name for f in fields.FIELDS}) == len(fields.FIELDS), "no two fields share a name")
    last = len(record["gos"])
    for field in fields.FIELDS:
        found = broken(record, field.damage or deleter(field.path))
        expected = {f"go {last}: {name}" for name in {field.name, *ALSO_LOST.get(field.name, ())}}
        check(set(found) == expected, f"with {field.name} removed the record is refused naming it {found}")
    print("\n-- finer breaks --")

    def drop_check(go: dict) -> None:
        go["unlock"]["checks"].pop(3)

    def drop_d8(go: dict) -> None:
        del named(go, "drift_jev_router")["detail"]["routers"][G_PLUS]["measures"]["sonnet"]["d8"]

    def drop_first_call(go: dict) -> None:
        del next(iter(named(go, "replay")["detail"]["results"].values()))["live"]["first_call_s"]

    def drop_requests(go: dict) -> None:
        del named(go, "calls_rules_trained")["requests"]["sent"]

    def fail_hook(go: dict) -> None:
        go["unlock"]["hook"]["passed"] = False

    def short_hash(go: dict) -> None:
        next(iter(go["steps"][-1]["detail"]["files"].values()))["sha256"] = "abc"

    def unfinished(go: dict) -> None:
        named(go, "drift_jev_gate")["status"] = "skipped"

    def drop_step(go: dict) -> None:
        go["steps"].remove(named(go, "grading"))

    def no_batch_id(go: dict) -> None:
        del named(go, "calls_sonnet_gate")["detail"]["batches"][0]["id"]

    def shares_off(go: dict) -> None:
        named(go, "tables")["detail"]["extras"]["trained_facts"]["fold_shares"]["0"] += 0.2

    for what, damage, field_name in [
        ("a check of the unlock left out", drop_check, "unlock.checks"),
        ("a drift measure without its D8", drop_d8, "drift.G+.measures"),
        ("a replay policy without its first call", drop_first_call, "replay.calls"),
        ("a step without its request counts", drop_requests, "steps"),
        ("a hook that did not pass", fail_hook, "unlock.hook"),
        ("a file's sha256 that isn't one", short_hash, "files"),
        ("a step left skipped", unfinished, "steps"),
        ("a step left out", drop_step, "steps"),
        ("a Sonnet batch without its ID", no_batch_id, "order.sonnet_batches"),
        ("fold shares that don't sum to one", shares_off, "folds.F"),
    ]:
        found = broken(record, damage)
        check(found == [f"go {last}: {field_name}"], f"{what} is refused as {field_name} {found}")
    bad: dict
    for what, bad in [("an empty record", {"gos": []}), ("a record with no gos", {})]:
        check(field_problems(bad) == ["gos"], f"{what} is refused")

    print("\n-- a resume's record keeps each go's unlock --")
    twice = copy.deepcopy(record)
    twice["gos"].insert(0, copy.deepcopy(record["gos"][0]))
    check(field_problems(twice) == [], "a record of two gos holding both unlocks is whole")
    del twice["gos"][0]["unlock"]["commit"]
    check(
        field_problems(twice) == ["go 1: unlock.commit"],
        f"a commit gone from the first go's unlock is named by its go {field_problems(twice)}",
    )
    twice["gos"][0]["steps"] = []  # an earlier go stopped early: only its unlock is held
    check(field_problems(twice) == ["go 1: unlock.commit"], "the earlier go's steps are not held, only its unlock")


def nothing_early(record: dict) -> None:
    print("\n-- the steps before the tables hold no outcome; the tables step holds them --")
    steps = record["gos"][-1]["steps"]
    cut = [s["name"] for s in steps].index("tables")
    early = [k for s in steps[:cut] for k in keys_in(s["detail"])]
    check(
        not (OUTCOME_KEYS & set(early)),
        f"no step before tables holds an outcome's key {sorted(OUTCOME_KEYS & set(early))}",
    )
    check({"stops", "refused", "truncated", "fold_shares", "jev_refused"} <= set(keys_in(steps[cut]["detail"])),
          "the tables step holds the stops, the refusals, the truncations and the fold shares")  # fmt: skip
    planted = {"detail": {"calls": {"refused": 1}}}
    check(
        OUTCOME_KEYS & set(keys_in(planted)) == {"refused"},
        "a planted outcome count in an early step's detail is found (the known positive)",
    )


# ---- the files --------------------------------------------------------------------------------------------------------


def the_files(r: Rehearsal, record: dict) -> None:
    print("\n-- the sha256 of every test record file --")
    results, output = r.world.results, r.world.output
    kept = record["gos"][-1]["steps"][-1]["detail"]["files"]
    check(len(kept) == EXPECTED_FILES, f"the record names {EXPECTED_FILES} files: {len(kept)}")
    named = fields.named_files(results, output)
    check(
        set(kept) == set(named) and all(path.is_file() for path in named.values()),
        "they are the files the final evaluation writes, all on disk",
    )
    check(all(kept[name]["sha256"] == sha256_here(path) and kept[name]["bytes"] == path.stat().st_size for name, path in named.items()),
          "each sha256 and size in the record equals one taken here from the file")  # fmt: skip
    expected_names = {"gates/free_checks_test.jsonl", "gates/jev_gate_test.jsonl", "gates/jev_gate_drift.jsonl", "gates/sonnet_gate_test.jsonl",
                      "gates/sonnet_gate_drift.jsonl", "jev/jev_test.jsonl", "jev/jev_bolstered_test.jsonl", "jev/jev_drift.jsonl",
                      "jev/jev_bolstered_drift.jsonl", "test_router_scores.json", "test_matrix.sqlite", "final/liveness.jsonl", "final/replay.jsonl",
                      "final/replay_requests.jsonl", "final/sonnet_batches_test.json", "final/tables_test.json"}  # fmt: skip
    check(expected_names <= set(kept), f"the record names each of test's files {sorted(expected_names - set(kept))}")
    check(
        "final_record.json" not in " ".join(kept) and "final_run.log" not in " ".join(kept),
        "the record does not hash itself or its log",
    )
    check(file_problems(kept, results, output) == [], "the files on disk match the record")

    with tempfile.TemporaryDirectory() as folder:
        small = Path(folder) / "abc.txt"
        small.write_bytes(b"abc")
        check(fields.sha256_of(small) == ABC_SHA256, "the hash function gives the standard SHA-256 of b'abc'")

        copy_of = Path(folder) / "results"
        shutil.copytree(results, copy_of)
        out_copy = Path(folder) / "output"
        shutil.copytree(output, out_copy)
        victim = copy_of / "gates" / "jev_gate_test.jsonl"
        victim.write_bytes(victim.read_bytes() + b" ")
        found = file_problems(kept, copy_of, out_copy)
        check(
            found == ["gates/jev_gate_test.jsonl differs from its sha256 in the record"],
            f"a file with one byte added is found {found}",
        )
        victim.write_bytes(victim.read_bytes()[:-1])
        check(file_problems(kept, copy_of, out_copy) == [], "and the same file restored is not")
        (out_copy / "tables_test.json").unlink()
        found = file_problems(kept, copy_of, out_copy)
        check(
            found == ["the record names final/tables_test.json, which is not on disk"],
            f"a file removed from disk is found {found}",
        )
        try:
            file_hashes(copy_of, out_copy)
            check(False, "hashing refuses a missing file")
        except FilesError as error:
            check("final/tables_test.json" in str(error), f"hashing refuses a missing file, naming it: {error}")
        (copy_of / "gates" / "stray_test.jsonl").write_bytes(b"{}\n")
        found = file_problems({n: e for n, e in kept.items()}, copy_of, out_copy)
        check(
            "gates/stray_test.jsonl is a test record file with no entry in the record" in found,
            f"a test record file the record doesn't name is found {found}",
        )
        check(file_problems({**kept, "notes.txt": kept["test_matrix.sqlite"]}, results, output)
              == ["the record names notes.txt, which is not a test record file"], "an entry for a file that is not a test record file is found")  # fmt: skip
        wrong_size = {**kept, "test_matrix.sqlite": {**kept["test_matrix.sqlite"], "bytes": 1}}
        check(
            file_problems(wrong_size, results, output) == ["test_matrix.sqlite differs from its sha256 in the record"],
            "an entry with a wrong size is found",
        )

        # The closing step names the files itself: with one missing the step refuses, naming it, with no record kept
        bare = dataclasses.replace(r.world, results=copy_of, output=out_copy)

        class Closing:
            world = bare
            said: list[str] = []

            def say(self, text: str) -> None:
                self.said.append(text)

        try:
            record_step(Closing())  # type: ignore[arg-type]
            check(False, "the closing step refuses a missing file")
        except StepRefusal as error:
            check(
                "final/tables_test.json" in str(error), "the closing step refuses a go whose file is missing, naming it"
            )


# ---- the holds --------------------------------------------------------------------------------------------------------


def the_holds(r: Rehearsal, record: dict) -> None:
    print("\n-- the tables, the stored files and the record agree, and each hold fails when its number moves --")
    results, output, tables = r.world.results, r.world.output, r.tables
    check(
        verify(record, tables, results, output) == [],
        f"the rehearsal's record, tables and files agree {verify(record, tables, results, output)}",
    )
    check(
        verify({"gos": []}, tables, results, output) == ["the record lacks gos"],
        "a record with a field missing is refused before the holds",
    )

    def held(
        change_record: Callable[[dict], None] = lambda go: None, change_tables: Callable[[dict], None] = lambda t: None
    ) -> list[str]:
        record_, tables_ = copy.deepcopy(record), copy.deepcopy(tables)
        change_record(record_["gos"][-1])
        change_tables(tables_)
        return held_problems(record_, tables_, results, output)

    def detail(go: dict, name: str) -> dict:
        return next(s for s in go["steps"] if s["name"] == name)["detail"]

    def step(go: dict, name: str) -> dict:
        return next(s for s in go["steps"] if s["name"] == name)

    def has(found: list[str], word: str) -> bool:
        return any(word in line for line in found)

    check(held() == [], "the holds pass on the untouched copies")
    cases: list[tuple[str, list[str], str]] = [
        ("the tables' item count moved", held(change_tables=lambda t: t.update(n=t["n"] + 1)), "items"),
        (
            "the tables step's item count moved",
            held(lambda go: detail(go, "tables").update(items=99)),
            "the tables step records",
        ),
        (
            "F's item count moved",
            held(lambda go: detail(go, "tables")["extras"]["trained_facts"].update(items=9)),
            "F's scores",
        ),
        (
            "G's call count moved",
            held(lambda go: detail(go, "tables")["extras"]["jev_facts"]["G"].update(calls=7)),
            "G's and G+'s calls",
        ),
        (
            "a config's graded answers moved",
            held(change_tables=lambda t: t["parts"]["stops"]["haiku"].update(correct=0)),
            "graded answers",
        ),
        ("the grading count moved", held(lambda go: detail(go, "grading").update(stored=31)), "grading stored"),
        (
            "a config's free-checked answers moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["opus"].update(answers=7)),
            "free-checked",
        ),
        (
            "the free checks' request count moved",
            held(lambda go: step(go, "calls_free_checks")["requests"].update(sent=31)),
            "the free checks",
        ),
        (
            "the Jev gate's request count moved",
            held(lambda go: step(go, "calls_jev_gate")["requests"].update(sent=29)),
            "the Jev gate",
        ),
        (
            "the Jev gate's calls in the tables moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["haiku"].update(jev_calls=6)),
            "the Jev gate",
        ),
        (
            "the Sonnet gate's verdicts in the tables moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["haiku"].update(sonnet_calls=1)),
            "the Sonnet gate",
        ),
        (
            "G and G+'s request count moved",
            held(lambda go: step(go, "calls_jev_router")["requests"].update(sent=15)),
            "G and G+",
        ),
        ("the replay's item count moved", held(lambda go: detail(go, "replay").update(items=7)), "the replay used"),
        (
            "the record's stops moved",
            held(lambda go: detail(go, "tables")["stops"]["opus"].update(max_tokens=3)),
            "refusals and truncations",
        ),
        (
            "the tables' stops moved",
            held(change_tables=lambda t: t["parts"]["stops"]["sonnet"].update(refusal=2)),
            "refusals and truncations",
        ),
        (
            "the Jev gate's refusals moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["opus"].update(jev_refused=2)),
            "Jev gate's refusals",
        ),
        (
            "the Sonnet gate's refusals moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["opus"].update(sonnet_refused=1)),
            "Sonnet gate's refusals",
        ),
        (
            "the Sonnet gate's cap stops moved",
            held(lambda go: detail(go, "tables")["extras"]["gate_table"]["opus"].update(sonnet_cap_stops=1)),
            "cap stops",
        ),
        (
            "the recorded spend moved",
            held(lambda go: detail(go, "record")["spend"]["calls"].update(recorded=1.5)),
            "recorded spend",
        ),
        (
            "the live pool's spend passed its cap",
            held(lambda go: detail(go, "record")["spend"]["live"].update(recorded=13.0)),
            "over a cap",
        ),
        (
            "an earlier step's spend rose above the closing one",
            held(lambda go: detail(go, "calls_jev_router")["spend"]["calls"].update(recorded=0.9)),
            "less than an earlier",
        ),
        (
            "two steps' times swapped",
            held(lambda go: step(go, "grading").update(started=step(go, "tables")["started"])),
            "not in order",
        ),
        ("the go ended before its last step", held(lambda go: go.update(ended=go["started"])), "not in order"),
        (
            "the unlock's time moved out of its step",
            held(lambda go: go["unlock"].update(time=go["steps"][-1]["ended"])),
            "unlock's time",
        ),
        (
            "the replay's own times moved out of its step",
            held(lambda go: detail(go, "replay")["conditions"].update(go_ended=go["ended"])),
            "replay's own times",
        ),
        ("the go stopped", held(lambda go: go.update(stopped={"step": "tables"})), "stopped at tables"),
    ]
    for what, found, word in cases:
        check(has(found, word), f"{what}, and the hold names it {found}")
    # (the item count is every hold's n, so moving it fails each hold that counts items)
    wide = [what for what, found, _ in cases if len(found) > 2 and "item count moved" not in what]
    check(not wide, f"each moved number fails its own hold, not the whole set {wide}")

    # The same through `verify`, with a file's lines changed in a copy (the hash differs too, and is named first)
    with tempfile.TemporaryDirectory() as folder:
        copy_of = Path(folder) / "results"
        shutil.copytree(results, copy_of)
        gate_file = copy_of / "gates" / "free_checks_test.jsonl"
        lines = gate_file.read_bytes().split(b"\n")
        gate_file.write_bytes(b"\n".join(lines[:-2]) + b"\n")
        found = verify(record, tables, copy_of, output)
        check(has(found, "gates/free_checks_test.jsonl differs") and has(found, "the free checks"),
              f"a stored file with a line gone is found by its hash and by the free checks' hold {found}")  # fmt: skip


def main() -> int:
    with rehearsed() as r:
        outcome, _, _ = r.go()
        check(outcome.stopped_at is None, f"a go on the made-up world runs every step {outcome}")
        record = r.record
        the_fields(record)
        nothing_early(record)
        the_files(r, record)
        the_holds(r, record)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
