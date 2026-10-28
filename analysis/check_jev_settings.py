"""Check the Jev settings builder (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 4 to 6, 10 and 12; issue #35).

Run from the project root:  uv run python -m analysis.check_jev_settings

A synthetic benchmark file and a synthetic results store (four tasks of 60 train items in groups of three, dev items,
test rows, three runs, outcomes that follow hand-set patterns), then the real train data where results/ is present.
No Jev call is made and no test answer is read: the synthetic test rows exist only to show they never reach a file.

- the past requests: 2 haiku failures and 2 passes per task, the outcomes as the hand-set patterns give them, whole
  requests with null for a missing schema and a schema's key order kept, one shuffled order (golden IDs), run 0 only;
- the exclusion: the past requests' groups are reported, `excluded_groups` returns them from a settings file, and
  they are absent from the fit items, the rates and the noise sample, which a hand count of the rates confirms;
- the rates: whole percents, with the half-up rule worked by hand;
- the noise sample: 100 items, 25 per task, train only, no repeats;
- the texts: G's three statements, G+'s nine with rates filled in, the eight difficulty statements and the
  request-type question, against hand-written text;
- the file: #30's format, `budget_bytes` null, no expected answer or test item in it, non-ASCII kept;
- a rebuild from the seed equals the first build, in any row order, and another seed gives another selection;
- refusals, each whole: a stratum too small, a task too small for the noise sample, a missing outcome, a dev row;
- train's cells alone: `load_train` holds no dev item and never examines a dev row (a bad dev row refuses the
  whole-store path and not this one), and a source check flags a module that loads the whole matrix (known positives);
- where results/ is present (the hook): the real train data builds 16 past requests and 100 noise items, 25 per task,
  a rebuild is equal, and a committed `routers/jev_settings.json` equals the build but for `budget_bytes`.
"""

import ast
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from analysis import build_jev_settings as builder
from analysis import matrix
from analysis.benchmark import read_rows
from analysis.build_jev_settings import SettingsBuildError, build, build_from_files, settings_text
from harness.client import load_models
from harness.collect import MAIN

TASKS = ("a", "b", "c", "d")
SECRET = "SECRET-EXPECTED-ANSWER"
CONFIGS = ("haiku", "sonnet", "opus_low")
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus_low": "claude-opus-5-5"}
# The hand-set outcome patterns, over an item's number within its task: each config fails on every n-th item.
FAILS_EVERY = {"haiku": 5, "sonnet": 7, "opus_low": 11}
SCHEMA = {"type": "object", "zeta": 1, "alpha": {"mid": 2, "beta": 3}}  # not in sorted order, on purpose

# The selection the seed 2026 gives on the synthetic benchmark, in state order, worked out once and pinned here so a
# change in the draw is a deliberate edit of this list.
GOLDEN_PAST = [
    "a:015", "a:029", "b:025", "d:053", "a:040", "d:038", "d:040", "b:057", "b:030", "c:000", "c:004", "c:035",
    "a:044", "b:004", "d:045", "c:007",
]  # fmt: skip

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object], error: type[Exception] = SettingsBuildError) -> bool:
    try:
        call()
    except error:
        return True
    return False


def fails(config: str, number: int) -> bool:
    return number % FAILS_EVERY[config] == 0


def benchmark_row(item_id: str, split: str, task: str, group: str, number: int) -> dict:
    row = {"item_id": item_id, "split": split, "source": task, "task_type": task, "source_group": group,
           "prompt": f"Prompt {item_id}, with a pröbe of non-ASCII text.", "evaluation_method": "exact",
           "expected_answer": SECRET, "meta": {}}  # fmt: skip
    if task in ("b", "d"):
        row["output_schema"] = json.loads(json.dumps(SCHEMA))
    return row


def synthetic_rows(train_per_task: int = 60) -> list[dict]:
    rows = []
    for task in TASKS:
        for n in range(train_per_task):
            rows.append(benchmark_row(f"{task}:{n:03d}", "train", task, f"{task}:g{n // 3}", n))
        for n in range(10):
            rows.append(benchmark_row(f"{task}:dev{n}", "dev", task, f"{task}:dg{n}", n))
        for n in range(5):
            rows.append(benchmark_row(f"{task}:test{n}", "test", task, f"{task}:tg{n}", n))
    return rows


def write_benchmark(directory: Path, rows: list[dict], name: str = "bench.jsonl") -> Path:
    path = directory / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def store_rows(rows: list[dict], bad_dev_price: bool = False) -> list[tuple]:
    """Stored rows for train and dev items: runs 0, 1 and 2, where runs 1 and 2 flip every outcome (so a loader that
    reads past run 0 gives other rates). The cost is the batch cost the loader's check expects."""
    models = load_models()
    out = []
    for row in rows:
        if row["split"] == "test":
            continue
        number = int(row["item_id"].split(":")[1].removeprefix("dev").removeprefix("test"))
        for config in CONFIGS:
            for run in (0, 1, 2):
                passed = (not fails(config, number)) if row["split"] == "train" else False
                if run:
                    passed = not passed
                snapshot = "1999-01-01" if bad_dev_price and row["item_id"] == "a:dev0" else models["price_snapshot"]
                record = {"model": MODELS[config], "input_tokens": 1000, "output_tokens": 500}
                cost = matrix.list_price(record, models["prices"]) * models["batch_discount"]
                out.append((row["item_id"], row["split"], row["source"], config, MODELS[config], run, 1000, 500, cost,
                            1 if passed else 0, snapshot))  # fmt: skip
    return out


def write_store(directory: Path, rows: list[dict], name: str = "store.sqlite", bad_dev_price: bool = False) -> Path:
    path = directory / name
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"CREATE TABLE rows ({', '.join(matrix.STORE_COLUMNS)})")
        conn.executemany(f"INSERT INTO rows VALUES ({', '.join('?' * len(matrix.STORE_COLUMNS))})",
                         store_rows(rows, bad_dev_price))  # fmt: skip
        conn.commit()
    return path


def loads_whole_matrix(source: str) -> bool:
    """Whether a module's source loads more than train's cells: a call to `load` or `from_store`, or to `read_store`
    without a split, whether through the module or by bare name."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in ("load", "from_store"):
                return True
            if name == "read_store" and not any(k.arg == "split" for k in node.keywords):
                return True
    return False


def main() -> int:
    rows = synthetic_rows()
    train = [r for r in rows if r["split"] == "train"]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bench = write_benchmark(directory, rows)
        store = write_store(directory, rows)
        built = build_from_files(bench, store)
        settings = built.settings
        plus = settings["g_plus"]
        past = plus["past_requests"]
        row_of = {r["item_id"]: r for r in train}
        task_of = {r["item_id"]: r["source"] for r in train}

        print("-- the past requests --")
        check(len(past) == 16, "sixteen past requests")
        for task in TASKS:
            mine = [p for p in past if task_of[p["item_id"]] == task]
            failed = [p for p in mine if not p["outcomes"]["haiku"]]
            check(len(mine) == 4 and len(failed) == 2, f"task {task}: 4 past requests, 2 that haiku failed")
        check(all(p["outcomes"]["haiku"] == (not fails("haiku", int(p["item_id"][2:]))) for p in past),
              "haiku's outcomes are the hand-set pattern's run-0 verdicts")  # fmt: skip
        check(all(p["outcomes"][c] == (not fails(c, int(p["item_id"][2:]))) for p in past for c in CONFIGS),
              "every config's outcomes are the pattern's run-0 verdicts, not run 1's or 2's")  # fmt: skip
        check([p["item_id"] for p in past] == GOLDEN_PAST, "the order is the golden shuffled order of seed 2026")
        check([task_of[p["item_id"]] for p in past] != sorted(task_of[p["item_id"]] for p in past),
              "the order is not grouped by task")  # fmt: skip
        check(all(set(p) == {"item_id", "prompt", "output_schema", "outcomes"} for p in past),
              "each past request is its item ID, prompt, output schema and outcomes, nothing else")  # fmt: skip
        check(all(p["prompt"] == row_of[p["item_id"]]["prompt"] for p in past), "prompts are stored whole")
        check(all(p["output_schema"] == row_of[p["item_id"]].get("output_schema") for p in past),
              "schemas are stored whole, null where the item has none")  # fmt: skip
        check(any(p["output_schema"] is None for p in past) and any(p["output_schema"] for p in past),
              "both a null and an object schema are among them")  # fmt: skip

        print("\n-- the exclusion --")
        groups = {row_of[p["item_id"]]["source_group"] for p in past}
        check(built.excluded_groups == groups and len(groups) == 16, "the 16 past requests' groups are reported")
        check(builder.excluded_groups(settings, train) == groups, "excluded_groups reads them from a settings file")
        check(not (set(built.fit_item_ids) & {r["item_id"] for r in train if r["source_group"] in groups}),
              "no item of an excluded group is among the fit items")  # fmt: skip
        check(len(built.fit_item_ids) == len(train) - 48, "the fit items are train less the 16 groups of three")
        check(any(i not in {p["item_id"] for p in past} and row_of[i]["source_group"] in groups for i in row_of),
              "a group-mate of a past request exists, so the exclusion is more than the 16 items")  # fmt: skip

        print("\n-- the rates --")
        left = [int(r["item_id"][2:]) for r in train if r["source_group"] not in groups]
        check(len(left) == 192, "192 train items lie outside the groups")
        for config in CONFIGS:
            passes = sum(not fails(config, n) for n in left)
            expected = (200 * passes + len(left)) // (2 * len(left))  # independent count over the hand-set pattern
            check(
                plus["rates"][config] == expected, f"{config}'s rate is {expected}%, over the items outside the groups"
            )
        check(all(type(v) is int and 0 <= v <= 100 for v in plus["rates"].values()), "rates are whole percents")
        check([builder.whole_percent(*a) for a in ((2, 3), (1, 8), (1, 3), (5, 8), (0, 5), (5, 5))]
              == [67, 13, 33, 63, 0, 100], "whole percents: 2/3 is 67, 1/8 (12.5) is 13, 1/3 is 33, 5/8 is 63")  # fmt: skip
        check(refused(lambda: builder.whole_percent(0, 0)), "a rate over no items is refused")

        print("\n-- the noise sample --")
        noise = settings["noise_sample"]
        ids = noise["item_ids"]
        check(noise["seed"] == 2026 and len(ids) == 100 and len(set(ids)) == 100, "100 distinct items, seed 2026")
        check(Counter(task_of[i] for i in ids) == {t: 25 for t in TASKS}, "25 per task")
        check(all(i in task_of for i in ids), "every item is a train item")
        check(not (set(ids) & {r["item_id"] for r in train if r["source_group"] in groups}),
              "no item is in an excluded group")  # fmt: skip

        print("\n-- the texts --")
        g = settings["g"]["success_statements"]
        check(g == [{"id": "success_haiku", "config": "haiku", "text": "A small, inexpensive model will answer this "
                                                                        "request correctly."},
                    {"id": "success_sonnet", "config": "sonnet", "text": "A mid-sized model will answer this request "
                                                                          "correctly."},
                    {"id": "success_opus_low", "config": "opus_low", "text": "A large model will answer this request "
                                                                              "correctly."}],
              "G's three statements are item 4's, with their IDs")  # fmt: skip
        filled = builder.g_plus_success_statements({"haiku": 74, "sonnet": 88, "opus_low": 91})
        haiku = "Claude Haiku 5.5, a small, fast model running at high effort, which answered 74% of comparable requests correctly,"
        sonnet = "Claude Sonnet 5.5, a mid-sized model running at high effort, which answered 88% of comparable requests correctly,"
        opus = (
            "Claude Opus 5.5, a large model running at low effort, which answered 91% of comparable requests correctly,"
        )
        a, b, c = ("will answer this request correctly.",
                   "will give a fully correct answer that follows every instruction in the request.",
                   "will make at least one mistake on this request.")  # fmt: skip
        expected_texts = [f"{d} {x}" for d in (haiku, sonnet, opus) for x in (a, b, c)]
        check([s["text"] for s in filled] == expected_texts, "G+'s nine success statements read as item 6 words them")
        check([s["id"] for s in filled] == [f"success_{k}_{p}" for k in CONFIGS for p in "abc"]
              and [s["phrasing"] for s in filled] == list("abc") * 3, "their IDs and phrasings")  # fmt: skip
        check([s["config"] for s in filled] == [k for k in CONFIGS for _ in "abc"], "their configs")
        built_texts = [s["text"] for s in plus["success_statements"]]
        check(all(f"answered {plus['rates'][s['config']]}% of comparable" in s["text"] for s in plus["success_statements"])
              and not any("{" in t for t in built_texts), "the built file has the real rates filled in, no braces")  # fmt: skip
        check(settings["display_names"] == {"haiku": "Claude Haiku 5.5 (high effort)",
                                            "sonnet": "Claude Sonnet 5.5 (high effort)",
                                            "opus_low": "Claude Opus 5.5 (low effort)"},
              "the display names")  # fmt: skip
        check([d["text"] for d in plus["difficulty_statements"]] == [
            "Answering this request correctly requires several steps of reasoning or calculation.",
            "The request sets strict, checkable rules for the answer's format, length or wording.",
            "The request has exactly one correct answer.",
            "The request asks for working code.",
            "The request is ambiguous: careful experts could reasonably give different answers.",
            "Answering correctly needs specialised expert knowledge.",
            "The request contains several separate instructions, all of which must be met.",
            "The request contains a detail that is easy to overlook and that changes the right answer.",
        ], "the eight difficulty statements, in item 6's order")  # fmt: skip
        check([d["id"] for d in plus["difficulty_statements"]] == [f"difficulty_{n}" for n in range(1, 9)],
              "their IDs are difficulty_1 to difficulty_8")  # fmt: skip
        check(plus["request_type"] == {"id": "request_type", "text": "What kind of request is this?",
                                       "options": ["follow written instructions to produce text",
                                                   "choose a label from a given list",
                                                   "answer a multiple-choice question", "write code"]},
              "the request-type question and its four options")  # fmt: skip

        print("\n-- the file --")
        text = settings_text(settings)
        data = json.loads(text)
        check(list(data) == ["jev", "display_names", "g", "g_plus", "noise_sample", "budget_bytes", "retry"],
              "the top-level keys, in #30's order")  # fmt: skip
        check(data["jev"] == {"endpoint": "https://api.typesafe.ai/v1/systemone", "model": "jev-1.14.0",
                              "price_per_million_input_tokens": 0.042, "price_checked": "2026-10-25"},
              "Jev's endpoint, model, price and date")  # fmt: skip
        check(data["budget_bytes"] == {"jev": None, "jev_bolstered": None}, "budget_bytes is null for both routers")
        check(data["retry"] == {"attempts": 8, "backoff_start_s": 1, "backoff_cap_s": 60, "retry_after_cap_s": 600,
                                "timeout_s": 30}, "the retry settings")  # fmt: skip
        check(data["g_plus"]["clip"] == [0.005, 0.995], "the clip")
        check(list(data["g_plus"]) == ["success_statements", "difficulty_statements", "request_type", "rates",
                                       "past_requests", "clip"], "G+'s keys, in #30's order")  # fmt: skip
        with_schema = next(p for p in data["g_plus"]["past_requests"] if p["output_schema"])
        check(list(with_schema["output_schema"]) == ["type", "zeta", "alpha"]
              and list(with_schema["output_schema"]["alpha"]) == ["mid", "beta"],
              "a schema keeps the benchmark's key order, not a sorted one")  # fmt: skip
        check("pröbe" in text and "\\u00f6" not in text, "non-ASCII text is kept, not escaped")
        check(SECRET not in text and "test" not in "".join(ids_of(data)), "no expected answer and no test item in it")
        check(all(not any(t in text for t in (f'"{x}"',)) for x in ("meta", "source", "task_type", "source_group")),
              "no task or group field in it")  # fmt: skip

        print("\n-- a rebuild --")
        check(
            settings_text(build_from_files(bench, store).settings) == text, "a rebuild from the seed equals the first"
        )
        shuffled = write_benchmark(directory, list(reversed(rows)), "bench_reversed.jsonl")
        check(settings_text(build_from_files(shuffled, store).settings) == text,
              "a rebuild from the rows in another order equals the first")  # fmt: skip
        other = build_from_files(bench, store, seed=7).settings
        check(settings_text(other) != text and [p["item_id"] for p in other["g_plus"]["past_requests"]] != GOLDEN_PAST,
              "another seed gives another selection")  # fmt: skip

        print("\n-- refusals --")
        success = {(r["item_id"], c): not fails(c, int(r["item_id"][2:])) for r in train for c in CONFIGS}
        none_fail = {**success, **{(f"a:{n:03d}", "haiku"): True for n in range(0, 60, 5)}}
        check(refused(lambda: build(train, none_fail)), "a task where haiku failed nothing is refused")
        one_fail = {**success, **{(f"a:{n:03d}", "haiku"): True for n in range(5, 60, 5)}}
        check(refused(lambda: build(train, one_fail)), "a task where haiku failed once, not twice, is refused")
        small = [r for r in train if r["source"] != "a" or int(r["item_id"][2:]) < 15]
        check(refused(lambda: build(small, success)), "a task with under 25 items outside the groups is refused")
        missing = {k: v for k, v in success.items() if k != ("b:010", "sonnet")}
        check(refused(lambda: build(train, missing)), "a missing outcome is refused")
        dev_row = next(r for r in rows if r["split"] == "dev")
        check(refused(lambda: build([*train, dev_row], success)), "a row list with a dev row among train's is refused")
        check(refused(lambda: build([r for r in rows if r["split"] == "dev"], success)), "dev rows are refused")
        check(refused(lambda: builder.excluded_groups(settings, [r for r in train if r["source"] != "a"])),
              "a past request outside the rows given is refused")  # fmt: skip

        print("\n-- train's cells alone --")
        train_matrix = matrix.load_train(store, bench)
        check(
            {i.split for i in train_matrix.items.values()} == {"train"}, "the train-only matrix holds only train items"
        )
        check(not any(i.startswith(f"{t}:dev") for (i, _c) in train_matrix.cells for t in TASKS),
              "no dev item has a cell in it")  # fmt: skip
        check(len(train_matrix.cells) == 720, "it holds the 240 train items' 3 cells each, run 0 only")
        check({r["split"] for r in matrix.read_store(store, split="train")} == {"train"},
              "the store query itself keeps train's rows alone")  # fmt: skip
        whole = matrix.from_store(matrix.read_store(store), bench)
        check({i.split for i in whole.items.values()} == {"train", "dev"},
              "known positive: the whole-matrix path holds dev items as well")  # fmt: skip
        bad = write_store(directory, rows, "bad.sqlite", bad_dev_price=True)
        check(refused(lambda: matrix.from_store(matrix.read_store(bad), bench), matrix.MatrixError),
              "known positive: a bad dev row refuses the whole-matrix path")  # fmt: skip
        check(len(matrix.load_train(bad, bench).cells) == 720 and build_from_files(bench, bad).settings == settings,
              "the same store builds the same settings through train's cells, never examining the dev row")  # fmt: skip
        positives = ["from analysis import matrix\nmatrix.load()", "from analysis.matrix import from_store, read_store\n"
                     "from_store(read_store(p))", "from analysis.matrix import load\nload(p)",
                     "from analysis import matrix\nmatrix.read_store(p)"]  # fmt: skip
        check(all(loads_whole_matrix(s) for s in positives), "known positives: source that loads the whole matrix")
        check(not loads_whole_matrix("from analysis import matrix\nmatrix.load_train(p)\nmatrix.read_store(p, split='train')"),
              "source that loads train's cells alone is not flagged")  # fmt: skip
        source = Path(builder.__file__).read_text()
        check(not loads_whole_matrix(source) and "load_train(" in source,
              "the builder's source calls load_train and loads no whole matrix")  # fmt: skip

    if MAIN.matrix.exists():
        real_section()
    else:
        print("\n(results/ is absent: the real train data section is skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def ids_of(data: dict) -> list[str]:
    return data["noise_sample"]["item_ids"] + [p["item_id"] for p in data["g_plus"]["past_requests"]]


def real_section(store: Path | None = None) -> None:
    """The real train data: the selection's shape, a rebuild, and the committed file where there is one."""
    print("\n-- the real train data --")
    built = build_from_files(store=store)
    settings = built.settings
    train = read_rows(split="train")
    task_of = {r["item_id"]: r["source"] for r in train}
    past = settings["g_plus"]["past_requests"]
    check(len(past) == 16 and Counter(task_of[p["item_id"]] for p in past) == {t: 4 for t in set(task_of.values())},
          "16 past requests, 4 per task")  # fmt: skip
    ids = settings["noise_sample"]["item_ids"]
    check(
        len(set(ids)) == 100 and set(Counter(task_of[i] for i in ids).values()) == {25}, "100 noise items, 25 per task"
    )
    check(not (set(ids) & {r["item_id"] for r in train if r["source_group"] in built.excluded_groups}),
          "no noise item is in an excluded group")  # fmt: skip
    check(settings_text(build_from_files(store=store).settings) == settings_text(settings), "a rebuild is equal")
    print(f"      rates {settings['g_plus']['rates']}, {len(built.excluded_groups)} excluded groups")
    if builder.SETTINGS_FILE.exists():
        committed = json.loads(builder.SETTINGS_FILE.read_text(encoding="utf-8"))
        committed["budget_bytes"] = settings["budget_bytes"]
        check(settings_text(committed) == settings_text(settings),
              "the committed settings file equals the build, apart from budget_bytes")  # fmt: skip
    else:
        print("      (no committed routers/jev_settings.json yet: Stage 6 · 8 commits it)")


if __name__ == "__main__":
    sys.exit(main())
