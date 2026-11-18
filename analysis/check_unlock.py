"""Check the unlock's first check and the handle's three reads (PLAN_v4.md Amendment A19, items 3 and 4; issue #78).

Run from the project root:  uv run python -m analysis.check_unlock

On a made-up world only (`analysis/made_up_world.py`): a made-up benchmark with test rows, a made-up sealed store written
by the collection module's own writer, and the manifest its own `seal` wrote. No results/, no Docker, no Jev or
Anthropic call, and nothing near the real sealed store. Expected values are written out by hand below.

- the unlock returns a handle whose requests, answers and graded items are exactly the made-up items'; the answers read
  holds five fields, run 0 only, and a marker in the expected answers and one in the stored thinking blocks never
  reach it (the graded read, which holds both, is the known positive);
- the unlock refuses a world with no store path, a missing store, a manifest that records no store or another store, a
  store the manifest doesn't record, and a store with one byte changed: each its own check, each raising before the
  store is opened as a database (`sqlite3.connect` is made to fail, and shown to bite);
- a handle built any other way (by hand, copied, subclassed, never initialised, or no handle at all) is refused by
  each read, and by `require_issued`;
- a stored item that is no test item of the benchmark refuses a read whole, naming the item;
- every existing refusal of a plain "test" split still holds, on the made-up benchmark.
"""

import copy
import dataclasses
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from analysis import unlock as unlock_module
from analysis.benchmark import SealedItemError, read_rows
from analysis.checking import refused
from analysis.gate_answers import Answer, read_answers, requests_of
from analysis.made_up_world import EXPECTED_MARKER, THINKING_MARKER, MadeUp, made_up_world
from analysis.unlock import Handle, HandleError, UnlockRefusal, require_issued, unlock
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refusal_check(world) -> str | None:
    """The check an unlock of `world` refuses at, or None if it unlocks. The store is never opened as a database."""
    with mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened")):
        try:
            unlock(world)
        except UnlockRefusal as refusal:
            return refusal.check
    return None


BANKING = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
LETTER = {"type": "object", "properties": {"letter": {"type": "string"}}, "required": ["letter"]}

# By hand, in the benchmark's file order.
EXPECTED_REQUESTS = {
    "banking77:made-test-0": Request("Which kind of card problem is this: where is my card?", BANKING),
    "mmlu_pro:made-test-1": Request("What is 2 + 2? A) 3 B) 4", LETTER),
    "ifeval:made-test-2": Request("Write one sentence about rain.", None),
}
# By hand, in item then config order, run 0 only.
EXPECTED_ANSWERS = [
    Answer("banking77:made-test-0", "haiku", 0, '{"label": "card_arrival"}', "end_turn"),
    Answer("banking77:made-test-0", "opus", 0, '{"label": "card_arrival"}', "end_turn"),
    Answer("ifeval:made-test-2", "haiku", 0, "Rain fell all night.", "end_turn"),
    Answer("ifeval:made-test-2", "opus", 0, "Rain softly fell on the roof.", "end_turn"),
    Answer("mmlu_pro:made-test-1", "haiku", 0, '{"letter": "', "max_tokens"),
    Answer("mmlu_pro:made-test-1", "opus", 0, '{"letter": "B"}', "end_turn"),
]
# By hand, in item, config and run order: custom ID, run, stop reason, the item's expected answer.
EXPECTED_GRADED = [
    ("banking77-made-test-0_haiku_r0", 0, "end_turn", f"card_arrival {EXPECTED_MARKER}-0"),
    ("banking77-made-test-0_haiku_r1", 1, "end_turn", f"card_arrival {EXPECTED_MARKER}-0"),
    ("banking77-made-test-0_opus_r0", 0, "end_turn", f"card_arrival {EXPECTED_MARKER}-0"),
    ("ifeval-made-test-2_haiku_r0", 0, "end_turn", f"one sentence {EXPECTED_MARKER}-2"),
    ("ifeval-made-test-2_opus_r0", 0, "end_turn", f"one sentence {EXPECTED_MARKER}-2"),
    ("mmlu_pro-made-test-1_haiku_r0", 0, "max_tokens", f"B {EXPECTED_MARKER}-1"),
    ("mmlu_pro-made-test-1_opus_r0", 0, "end_turn", f"B {EXPECTED_MARKER}-1"),
]


def reads(made: MadeUp) -> None:
    print("-- the unlock and the handle's three reads, on a made-up world --")
    handle = unlock(made.world)
    check(
        isinstance(handle, Handle) and handle.world is made.world, "the unlock returns a handle that carries the world"
    )
    names = [result.name for result in handle.results]
    check(
        names == ["hook", "git_facts", "replay_draw", "sealed_store_path", "manifest_record", "sealed_store_sha256"],
        f"its results name the checks {names}",
    )
    sha = handle.results[-1].detail["sha256"]
    check(sha == json.loads((made.world.root / "data" / "collection_manifest.json").read_text())["test_sealed"]["sha256"],
          "the sha256 it records is the manifest's")  # fmt: skip

    requests = handle.requests()
    check(requests == EXPECTED_REQUESTS and list(requests) == list(EXPECTED_REQUESTS),
          "requests: exactly the three test items' prompts and output schemas, in file order")  # fmt: skip
    answers = handle.answers()
    check(answers == EXPECTED_ANSWERS, f"answers: exactly the six run 0 answers, hand-written {len(answers)}")
    check(not any(a.answer == '{"label": "lost_card"}' for a in answers), "answers: the run 1 result is left out")
    graded = handle.graded_items()
    check([(g.custom_id, g.run, g.response["stop_reason"], g.item["expected_answer"]) for g in graded] == EXPECTED_GRADED,
          "graded items: every stored result with its item's expected answer and its response")  # fmt: skip
    check(graded[0].model == "made-up-haiku" and graded[0].effort == "high" and graded[0].batch_id == "msgbatch_made_up"
          and graded[0].request_hash == "hash-0-haiku", "graded items: the request's model, effort, hash and batch")  # fmt: skip

    print("\n-- the answers read gives no expected answer and no grade --")
    check([f.name for f in dataclasses.fields(Answer)] == ["item_id", "config", "run", "answer", "stop_reason"],
          "an answer has the five fields, by hand")  # fmt: skip
    check(unlock_module.ANSWERS_QUERY == "SELECT item_id, config, run, answer, stop_reason FROM answers WHERE run = 0 "
          "ORDER BY item_id, config", "the answers read's SQL names those five columns and run 0 only")  # fmt: skip
    check(EXPECTED_MARKER not in repr(answers) and THINKING_MARKER not in repr(answers),
          "no marker from the expected answers or the stored thinking reaches the answers read")  # fmt: skip
    check(EXPECTED_MARKER in repr(graded) and THINKING_MARKER in repr(graded),
          "the markers are in the graded read, so the search above can find them")  # fmt: skip


def flip_a_byte(made: MadeUp) -> None:
    os.chmod(made.paths.sealed, 0o644)
    data = bytearray(made.paths.sealed.read_bytes())
    data[len(data) // 2] ^= 0x01
    made.paths.sealed.write_bytes(bytes(data))


def rewrite_manifest(made: MadeUp, change: Callable[[dict], object]) -> None:
    manifest = json.loads(made.paths.manifest.read_text())
    change(manifest)
    made.paths.manifest.write_text(json.dumps(manifest))


def refusals() -> None:
    print("\n-- the unlock refuses, each at its own check, and opens nothing --")
    with made_up_world() as made:
        check(
            refusal_check(made.world) is None, "the untouched made-up world unlocks (the cases below change one thing)"
        )
        with mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened")):
            check(refused(AssertionError, lambda: unlock(made.world).answers()),
                  "the guard bites: a read of the store would have failed it")  # fmt: skip

    with made_up_world() as made:
        no_path = dataclasses.replace(made.world, sealed_store=None)
        check(refusal_check(no_path) == "sealed_store_path", "a world with no store path refuses at sealed_store_path")

    with made_up_world() as made:
        gone = dataclasses.replace(made.world, sealed_store=made.world.root / "results" / "nothing_here.sqlite")
        check(refusal_check(gone) == "sealed_store_exists", "a path that is no file refuses at sealed_store_exists")

    with made_up_world() as made:
        before = made.paths.sealed.read_bytes()
        flip_a_byte(made)
        check(made.paths.sealed.read_bytes() != before, "the tampered store differs from the sealed one by a byte")
        check(
            refusal_check(made.world) == "sealed_store_sha256",
            "a store with one byte changed refuses at sealed_store_sha256",
        )

    with made_up_world() as made:
        other = made.paths.sealed.with_name("another_store.sqlite")
        other.write_bytes(
            made.paths.sealed.read_bytes()
        )  # the same bytes, so the same hash, under a name no manifest records
        check(refusal_check(dataclasses.replace(made.world, sealed_store=other)) == "manifest_record",
              "a store the manifest doesn't record (same bytes, another name) refuses at manifest_record")  # fmt: skip

    with made_up_world() as made:
        rewrite_manifest(made, lambda manifest: manifest.pop("test_sealed"))
        check(
            refusal_check(made.world) == "manifest_record", "a manifest with no store record refuses at manifest_record"
        )

    with made_up_world() as made:
        made.paths.manifest.write_text("not json")
        check(refusal_check(made.world) == "manifest_record", "an unreadable manifest refuses at manifest_record")

    with made_up_world() as made:
        made.paths.manifest.unlink()
        check(refusal_check(made.world) == "manifest_record", "a world with no manifest refuses at manifest_record")

    with made_up_world() as made:
        rewrite_manifest(made, lambda manifest: manifest["test_sealed"].update(sha256="0" * 64))
        check(
            refusal_check(made.world) == "sealed_store_sha256",
            "a manifest recording another hash refuses at sealed_store_sha256",
        )


class SubHandle(Handle):
    __slots__ = ()


def forged(handle: Handle) -> dict[str, object]:
    """Handles the unlock didn't issue, built each way there is, with the good handle's own world."""
    return {
        "built by hand": Handle(handle.world, handle.results),
        "a copy": copy.copy(handle),
        "a subclass": SubHandle(handle.world, handle.results),
        "never initialised": object.__new__(Handle),
        "no handle": None,
        "a string": "handle",
    }


def handle_refusals() -> None:
    print("\n-- a handle the unlock didn't issue is refused by every read --")
    with made_up_world() as made:
        handle = unlock(made.world)
        check(
            require_issued(handle) is handle,
            "the issued handle passes require_issued, so the refusals below are the handle's",
        )
        for how, fake in forged(handle).items():
            check(refused(HandleError, lambda f=fake: require_issued(f)), f"require_issued refuses a handle {how}")  # type: ignore[misc]
            if isinstance(fake, Handle):
                for read in ("requests", "answers", "graded_items"):
                    check(refused(HandleError, getattr(fake, read)), f"{read} refuses a handle {how}")

    print("\n-- a stored item that is no test item refuses the read whole --")
    with made_up_world() as made:
        rows = [json.loads(line) for line in made.world.benchmark.read_text().splitlines()]
        kept = [r for r in rows if r["item_id"] != "ifeval:made-test-2"]
        made.world.benchmark.write_text("".join(json.dumps(r) + "\n" for r in kept))
        handle = unlock(made.world)
        for read in ("answers", "graded_items"):
            try:
                getattr(handle, read)()
                check(False, f"{read} refuses a store holding an item the benchmark's test rows lack")
            except HandleError as error:
                check(
                    "ifeval:made-test-2" in str(error),
                    f"{read} refuses a store holding an item the benchmark lacks, naming it",
                )
        check(
            list(handle.requests()) == ["banking77:made-test-0", "mmlu_pro:made-test-1"],
            "requests follow the benchmark's test rows",
        )


def plain_test_refusals() -> None:
    print('\n-- every existing refusal of a plain "test" split still holds --')
    with made_up_world() as made:
        file = made.world.benchmark
        check(
            refused(SealedItemError, lambda: read_rows(file, split="test")), "the benchmark reader refuses split='test'"
        )
        check(refused(SealedItemError, lambda: read_rows(file, item_ids=["mmlu_pro:made-test-1"])),
              "the benchmark reader refuses a named test item")  # fmt: skip
        check(refused(SealedItemError, lambda: requests_of("test", file)), "the gates' request reader refuses 'test'")
        check(refused(SealedItemError, lambda: read_answers("test", Path("unused.sqlite"), file)),
              "the gates' answer reader refuses 'test' before it opens a store")  # fmt: skip
        check([r["item_id"] for r in read_rows(file)] == ["banking77:made-train", "mmlu_pro:made-dev"],
              "the benchmark reader still yields train and dev rows only")  # fmt: skip


def main() -> int:
    with made_up_world() as made:
        reads(made)
    refusals()
    handle_refusals()
    plain_test_refusals()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
