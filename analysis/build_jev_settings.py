"""The Jev settings builder: past requests, rates, noise sample and question texts (PLAN_v4.md §8 3c and 3c+;
Amendment A15, items 4 to 6, 10 and 12; GitHub issue #30's format of the settings file).

It builds `routers/jev_settings.json` from train alone and reads no Jev answer. Its inputs are train's item IDs,
tasks, source groups, prompts and output schemas, through the benchmark reader, and train's run-0 outcomes, loaded
through `matrix.load_train` so dev's cells never enter memory (Amendment A12, item 1). The test split is not
reachable from here.

- **Past requests (item 5):** within each task, 2 items `haiku` failed and 2 it passed, drawn within each stratum,
  then placed in one order shuffled across tasks. Each is stored whole (item ID, prompt, output schema or null) with
  every config's run-0 outcome, refusals and `max_tokens` stops already false in the matrix (A11 item 6).
- **Excluded groups (items 5, 6, 10):** the past requests' source groups. `excluded_groups` gives them back from a
  settings file, so the fit, the cross-validation, the signal check, the rates and the noise sample leave them out.
- **Rates (item 6):** per config, run-0 success over train items outside those groups, as a whole percent
  (a half rounds up). Overall only, never per task.
- **Noise sample (item 10):** 100 train items, 25 per task, outside those groups, for the second pass.
- **Texts (items 4 and 6):** G's three statements, G+'s nine success statements with the rates filled in, the eight
  difficulty statements and the request-type question, worded here before any Jev output exists.

**Seeded draws:** a draw orders candidates by sha256 of "seed:purpose:item_id", as `analysis/folds.py` orders its
groups, so a draw depends on neither Python's random generator nor the order the rows arrive in.

**The file** is JSON with the code's key order (the sections and fields as the issue lists them) and each schema's
own key order from the benchmark; keys are not sorted and non-ASCII text is kept (A15 item 8's state keeps a schema's
keys as the benchmark has them). `budget_bytes` is null until the smoke test fills it, before the settings commit.

Write the file from the project root:  uv run python -m analysis.build_jev_settings
(refuses to overwrite an existing file; pass --overwrite, which resets `budget_bytes` to null).
"""

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analysis import matrix
from analysis.benchmark import read_rows
from routers.ladder import SCORED
from routers.request import from_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "routers" / "jev_settings.json"

SEED = 2026
PER_STRATUM = 2  # past requests per task and haiku outcome (item 5)
NOISE_PER_TASK = 25  # item 10: 100 items over the four tasks
PHRASINGS = ("a", "b", "c")  # c is the negated one (item 7)

JEV = {
    "endpoint": "https://api.typesafe.ai/v1/systemone",
    "model": "jev-1.14.0",
    "price_per_million_input_tokens": 0.042,
    "price_checked": "2026-10-25",
}
CLIP = [0.005, 0.995]
RETRY = {"attempts": 8, "backoff_start_s": 1, "backoff_cap_s": 60, "retry_after_cap_s": 600, "timeout_s": 30}

# A config as G+ describes it: the model's name, how it is sized and its effort (item 6). The display name that the
# outcomes and the statements both use is built from the first and last, so it is defined once.
MODEL_NAMES = {"haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5", "opus_low": "Claude Opus 5.5"}
SIZES = {"haiku": "a small, fast model", "sonnet": "a mid-sized model", "opus_low": "a large model"}
EFFORTS = {"haiku": "high", "sonnet": "high", "opus_low": "low"}
DISPLAY_NAMES = {config: f"{MODEL_NAMES[config]} ({EFFORTS[config]} effort)" for config in SCORED}

# G's statements name no model, effort or rate (item 4).
G_STATEMENTS = {
    "haiku": "A small, inexpensive model will answer this request correctly.",
    "sonnet": "A mid-sized model will answer this request correctly.",
    "opus_low": "A large model will answer this request correctly.",
}

# Item 6's eight difficulty statements, in its order.
DIFFICULTY_STATEMENTS = (
    "Answering this request correctly requires several steps of reasoning or calculation.",
    "The request sets strict, checkable rules for the answer's format, length or wording.",
    "The request has exactly one correct answer.",
    "The request asks for working code.",
    "The request is ambiguous: careful experts could reasonably give different answers.",
    "Answering correctly needs specialised expert knowledge.",
    "The request contains several separate instructions, all of which must be met.",
    "The request contains a detail that is easy to overlook and that changes the right answer.",
)

# Item 6's request-type question: four generic options, none naming a task.
REQUEST_TYPE_TEXT = "What kind of request is this?"
REQUEST_TYPE_OPTIONS = (
    "follow written instructions to produce text",
    "choose a label from a given list",
    "answer a multiple-choice question",
    "write code",
)

# What each phrasing says after the config's description (item 6). The description carries the rate in all three.
CLAUSES = {
    "a": "will answer this request correctly.",
    "b": "will give a fully correct answer that follows every instruction in the request.",
    "c": "will make at least one mistake on this request.",
}


class SettingsBuildError(ValueError):
    """Inputs the builder refuses. Nothing is built."""


@dataclass(frozen=True)
class Built:
    settings: dict
    excluded_groups: frozenset[str]  # the past requests' source groups
    fit_item_ids: tuple[str, ...]  # train items outside those groups, in ID order (before refusals are dropped)


def whole_percent(successes: int, total: int) -> int:
    """100 * successes / total as a whole percent, a half rounding up, in integer arithmetic."""
    if total <= 0:
        raise SettingsBuildError("a rate needs at least one item")
    return (200 * successes + total) // (2 * total)


def g_statements() -> list[dict]:
    return [{"id": f"success_{config}", "config": config, "text": G_STATEMENTS[config]} for config in SCORED]


def g_plus_success_statements(rates: dict[str, int]) -> list[dict]:
    """Nine statements, three phrasings per config, with each config's whole-percent rate filled in."""
    statements = []
    for config in SCORED:
        description = f"{MODEL_NAMES[config]}, {SIZES[config]} running at {EFFORTS[config]} effort, which answered "
        description += f"{rates[config]}% of comparable requests correctly,"
        for phrasing in PHRASINGS:
            statements.append({"id": f"success_{config}_{phrasing}", "config": config, "phrasing": phrasing,
                               "text": f"{description} {CLAUSES[phrasing]}"})  # fmt: skip
    return statements


def difficulty_statements() -> list[dict]:
    return [{"id": f"difficulty_{n}", "text": text} for n, text in enumerate(DIFFICULTY_STATEMENTS, start=1)]


def request_type_question() -> dict:
    return {"id": "request_type", "text": REQUEST_TYPE_TEXT, "options": list(REQUEST_TYPE_OPTIONS)}


def draw_order(item_ids: list[str], purpose: str, seed: int) -> list[str]:
    """The items in a seeded order that depends on their IDs alone."""
    return sorted(item_ids, key=lambda item_id: hashlib.sha256(f"{seed}:{purpose}:{item_id}".encode()).hexdigest())


def select_past_requests(rows: list[dict], success: dict[tuple[str, str], bool], seed: int = SEED) -> list[dict]:
    """Item 5: per task, PER_STRATUM items haiku failed and as many it passed, then one shuffled order.

    Each is `{item_id, prompt, output_schema, outcomes}`, the outcomes keyed by config.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["source"]].append(row)
    chosen = []
    for task in sorted(by_task):
        for passed in (False, True):
            stratum = [r["item_id"] for r in by_task[task] if success[(r["item_id"], "haiku")] is passed]
            if len(stratum) < PER_STRATUM:
                raise SettingsBuildError(f"task {task} has {len(stratum)} items haiku {'passed' if passed else 'failed'}"
                                         f", fewer than the {PER_STRATUM} to draw")  # fmt: skip
            chosen += draw_order(stratum, f"past:{task}:{passed}", seed)[:PER_STRATUM]
    row_of = {r["item_id"]: r for r in rows}
    past = []
    for item_id in draw_order(chosen, "order", seed):
        request = from_row(row_of[item_id])
        outcomes = {config: success[(item_id, config)] for config in SCORED}
        past.append({"item_id": item_id, "prompt": request.prompt, "output_schema": request.output_schema,
                     "outcomes": outcomes})  # fmt: skip
    return past


def build(rows: list[dict], success: dict[tuple[str, str], bool], seed: int = SEED) -> Built:
    """The settings from train's rows (benchmark rows, as the reader gives them) and run-0 successes by
    (item ID, config). Refuses a row outside train, a missing outcome, a stratum or a task too small to draw from."""
    if bad := sorted({r["split"] for r in rows} - {"train"}):
        raise SettingsBuildError(f"only train rows are built from, got {bad}")
    for row in rows:
        for config in SCORED:
            if not isinstance(success.get((row["item_id"], config)), bool):
                raise SettingsBuildError(f"no run-0 outcome for {row['item_id']} on {config}")
    past = select_past_requests(rows, success, seed)
    group_of = {r["item_id"]: r["source_group"] for r in rows}
    excluded = frozenset(group_of[p["item_id"]] for p in past)
    eligible = [r for r in rows if r["source_group"] not in excluded]
    rates = {config: whole_percent(sum(success[(r["item_id"], config)] for r in eligible), len(eligible))
             for config in SCORED}  # fmt: skip
    eligible_by_task: dict[str, list[str]] = defaultdict(list)
    for row in eligible:
        eligible_by_task[row["source"]].append(row["item_id"])
    noise = []
    for task in sorted({r["source"] for r in rows}):
        if len(eligible_by_task[task]) < NOISE_PER_TASK:
            raise SettingsBuildError(f"task {task} has {len(eligible_by_task[task])} items outside the past requests'"
                                     f" groups, fewer than the {NOISE_PER_TASK} to sample")  # fmt: skip
        noise += draw_order(eligible_by_task[task], f"noise:{task}", seed)[:NOISE_PER_TASK]
    settings = {
        "jev": dict(JEV),
        "display_names": dict(DISPLAY_NAMES),
        "g": {"success_statements": g_statements()},
        "g_plus": {
            "success_statements": g_plus_success_statements(rates),
            "difficulty_statements": difficulty_statements(),
            "request_type": request_type_question(),
            "rates": rates,
            "past_requests": past,
            "clip": list(CLIP),
        },
        "noise_sample": {"seed": seed, "item_ids": noise},
        "budget_bytes": {"jev": None, "jev_bolstered": None},
        "retry": dict(RETRY),
    }
    fit_ids = tuple(sorted(r["item_id"] for r in eligible))
    return Built(settings, excluded, fit_ids)


def excluded_groups(settings: dict, rows: list[dict]) -> frozenset[str]:
    """The source groups of a settings file's past requests, for the fit, the cross-validation, the signal check, the
    rates and the noise sample to leave out. `rows` are train's benchmark rows (the reader's)."""
    group_of = {r["item_id"]: r["source_group"] for r in rows}
    try:
        return frozenset(group_of[p["item_id"]] for p in settings["g_plus"]["past_requests"])
    except KeyError as missing:
        raise SettingsBuildError(f"a past request is not among the train rows: {missing}") from None


def read_train(benchmark_file: Path | None = None, store: Path | None = None) -> tuple[list[dict], dict]:
    """Train's benchmark rows and run-0 successes by (item ID, config), from train's cells alone.

    `benchmark_file` and `store` are for a check to feed in; the frozen benchmark and the dev results matrix by
    default. Success is the matrix's verdict, in which refusals and `max_tokens` stops are failures (A11 item 6).
    """
    rows = read_rows(split="train") if benchmark_file is None else read_rows(benchmark_file, split="train")
    cells = matrix.load_train() if store is None else matrix.load_train(store, benchmark_file)
    success = {key: cell.success == 1 for key, cell in cells.cells.items()}
    return rows, success


def build_from_files(benchmark_file: Path | None = None, store: Path | None = None, seed: int = SEED) -> Built:
    rows, success = read_train(benchmark_file, store)
    return build(rows, success, seed)


def settings_text(settings: dict) -> str:
    return json.dumps(settings, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if SETTINGS_FILE.exists() and "--overwrite" not in argv:
        print(f"{SETTINGS_FILE.name} exists; pass --overwrite to rebuild it (its budget_bytes would reset to null)")
        return 1
    built = build_from_files()
    SETTINGS_FILE.write_text(settings_text(built.settings), encoding="utf-8")
    print(f"wrote {SETTINGS_FILE.relative_to(PROJECT_ROOT)}: {len(built.settings['g_plus']['past_requests'])} past "
          f"requests, {len(built.excluded_groups)} excluded groups, {len(built.fit_item_ids)} items outside them, "
          f"rates {built.settings['g_plus']['rates']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
