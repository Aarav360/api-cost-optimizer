"""A made-up world for the final evaluation's checks and its rehearsal (PLAN_v4.md Amendment A19, items 3 and 13).

Not a check: a module `analysis/check_unlock.py` and the rehearsal (#79) import. It builds, in a folder of the caller's,
a project-shaped root (`data/` and `results/` as the real one has them) holding

- a made-up benchmark: three test items (a classification and a multiple-choice item with an output schema, and a free
  one), each with an expected answer carrying a marker that must reach grading only, and a train and a dev item;
- a made-up sealed store written by the collection module's own writer (`open_sealed`), seven stored results: two
  configs for each test item, and one run-1 result that the answers read must leave out; each response holds a thinking
  block with a marker of its own;
- its collection manifest, written by the collection module's own `seal`, so the hash the unlock checks comes from the
  writer and not from this module.

Nothing here touches the real store, the real benchmark or `results/`: the store sits under the caller's folder, and
`seal` makes it read-only, so `made_up_world()` makes it writable again when its folder goes.
"""

import contextlib
import dataclasses
import io
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from analysis.unlock import MANIFEST
from analysis.world import World
from harness.collect import SEALED_COLUMNS, Job, Paths, make_custom_id, open_sealed, paths_under, seal

CONFIGS = ("haiku", "opus")
EXPECTED_MARKER = "EXPECTED-MARKER"  # in every test item's expected answer: grading's alone
THINKING_MARKER = "THINKING-MARKER"  # in every stored response's thinking block: no answers read gives it

BANKING = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
LETTER = {"type": "object", "properties": {"letter": {"type": "string"}}, "required": ["letter"]}


def row(item_id: str, split: str, prompt: str, expected: str, schema: dict | None = None) -> dict:
    made = {"item_id": item_id, "split": split, "source": item_id.split(":")[0], "source_group": item_id,
            "task_type": "made up", "evaluation_method": "made up", "meta": {}, "prompt": prompt,
            "expected_answer": expected}  # fmt: skip
    return made if schema is None else {**made, "output_schema": schema}


BENCHMARK_ROWS = [
    row("banking77:made-train", "train", "Which kind of card problem is this: lost?", "lost_card", BANKING),
    row(
        "banking77:made-test-0",
        "test",
        "Which kind of card problem is this: where is my card?",
        f"card_arrival {EXPECTED_MARKER}-0",
        BANKING,
    ),  # fmt: skip
    row("mmlu_pro:made-dev", "dev", "What is 3 + 3? A) 5 B) 6", "B", LETTER),
    row("mmlu_pro:made-test-1", "test", "What is 2 + 2? A) 3 B) 4", f"B {EXPECTED_MARKER}-1", LETTER),
    row("ifeval:made-test-2", "test", "Write one sentence about rain.", f"one sentence {EXPECTED_MARKER}-2"),
]
TEST_IDS = ["banking77:made-test-0", "mmlu_pro:made-test-1", "ifeval:made-test-2"]

# (test item's position, config, run, stop reason, answer text): the stored results, seven of them.
STORED = [
    (0, "haiku", 0, "end_turn", '{"label": "card_arrival"}'),
    (0, "opus", 0, "end_turn", '{"label": "card_arrival"}'),
    (1, "haiku", 0, "max_tokens", '{"letter": "'),
    (1, "opus", 0, "end_turn", '{"letter": "B"}'),
    (2, "haiku", 0, "end_turn", "Rain fell all night."),
    (2, "opus", 0, "end_turn", "Rain softly fell on the roof."),
    (0, "haiku", 1, "end_turn", '{"label": "lost_card"}'),
]


def stored_row(position: int, config: str, run: int, stop_reason: str, answer: str) -> tuple:
    item_id = TEST_IDS[position]
    response = {
        "model": f"made-up-{config}",
        "stop_reason": stop_reason,
        "content": [
            {"type": "thinking", "thinking": f"{THINKING_MARKER}-{position}"},
            {"type": "text", "text": answer},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    return (make_custom_id(item_id, config, run), item_id, config, f"made-up-{config}", "high", run, "msgbatch_made_up",
            f"hash-{position}-{config}", f"made-up-{config}", stop_reason, 11, 7, None, answer, json.dumps(response),
            "2026-01-01T00:00:00+00:00")  # fmt: skip


def stored_row_values(item_id: str, config: str, run: int, settings: dict, request_hash: str, response: dict) -> tuple:
    """A sealed row for a made-up result under the real config's `settings` (model and effort) and the hash of the
    request it answers, which the rehearsal's grading rebuilds. The answer is the response's text blocks."""
    usage = response["usage"]
    answer = "".join(block["text"] for block in response["content"] if block["type"] == "text")
    return (make_custom_id(item_id, config, run), item_id, config, settings["model"], settings["effort"], run,
            "msgbatch_made_up", request_hash, response["model"], response["stop_reason"], usage["input_tokens"],
            usage["output_tokens"], None, answer, json.dumps(response), "2026-01-01T00:00:00+00:00")  # fmt: skip


@dataclass(frozen=True)
class MadeUp:
    world: World
    paths: Paths  # the collection module's view of the made-up folder, for a check that tampers with the store


@dataclass(frozen=True)
class Content:
    """What a made-up world holds: its benchmark rows, and its sealed rows (`SEALED_COLUMNS`' values) each with the
    item ID of the test item it answers. The rehearsal (`analysis/rehearsal_world.py`) brings its own."""

    benchmark: list[dict]
    sealed: list[tuple[tuple, str]]


DEFAULT = Content(BENCHMARK_ROWS, [(stored_row(*made), TEST_IDS[made[0]]) for made in STORED])


def build(folder: Path, content: Content = DEFAULT) -> MadeUp:
    """The made-up world under `folder`, laid out as the project's own root is."""
    # The collection module's throwaway layout under results/, as its own checks use it, with the manifest where a
    # world's root keeps it: the real store's path is never read here.
    paths = dataclasses.replace(paths_under(folder / "results"), manifest=folder / MANIFEST)
    benchmark = folder / "data" / "made_up_items.jsonl"
    for path in (benchmark, paths.sealed, paths.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text("".join(json.dumps(made) + "\n" for made in content.benchmark), encoding="utf-8")

    conn = open_sealed(paths)
    jobs = []
    for values, item_id in content.sealed:
        conn.execute(f"INSERT INTO answers VALUES ({', '.join('?' * len(SEALED_COLUMNS))})", values)
        test_item = next(made for made in content.benchmark if made["item_id"] == item_id)
        jobs.append(Job(values[0], "test", test_item, values[2], values[5], {}))
    conn.commit()
    conn.close()
    with contextlib.redirect_stdout(io.StringIO()):  # seal reports its hash; a check's output stays its own
        sealed = seal(paths, jobs, None)
    if sealed != 0:
        raise RuntimeError("the made-up store did not seal")
    world = World(root=folder, sealed_store=paths.sealed, benchmark=benchmark, results=folder / "results" / "test",
                  output=folder / "output")  # fmt: skip
    return MadeUp(world, paths)


@contextmanager
def made_up_world(content: Content = DEFAULT) -> Iterator[MadeUp]:
    """A made-up world in a temporary folder, removed afterwards (the sealed store made writable first)."""
    with tempfile.TemporaryDirectory() as folder:
        made = build(Path(folder), content)
        try:
            yield made
        finally:
            if made.paths.sealed.exists():
                os.chmod(made.paths.sealed, 0o644)
