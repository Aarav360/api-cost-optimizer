"""The noise samples' drawing rule (PLAN_v4.md §9; Amendment A17, items 17 and 20.2).

Two samples of `haiku`'s train answers are sent a second time or through the standard API: the Jev gate's noise pass
(25 per task) and the Sonnet gate's latency sample (20 per task, which also gives its drift check). The lists can't be
fixed at the settings commit, since which MBPP+ answers pass the code check is not known until it runs on train. The
rule is, and it is here, in code, so that it is frozen with the gates and the freeze commit's hook check can rebuild
the lists from it:

- the seed and the counts are the `noise` section of `routers/gate_settings.json`;
- for each task, in the section's order, `haiku`'s run-0 train answers of that task, in item-ID order, are put in a
  seeded order: `random.Random(f"{seed}/{task}").shuffle`, so that each task's order depends on the seed and the task
  alone, not on how many items another task has;
- the sample is the first `count` of that order whose answer passes every free check;
- the Sonnet gate's list is the first 20 of each task's Jev list, since both take the first that pass of one order.

No outcome, score or verdict of a paid gate enters the draw: only whether an answer passed the free checks.
"""

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "routers" / "gate_settings.json"


class NoiseSampleError(ValueError):
    """A noise section that is refused, or a draw that can't be made. Nothing is returned."""


@dataclass(frozen=True)
class NoiseSettings:
    seed: int
    config: str
    tasks: tuple[str, ...]
    jev_per_task: int
    sonnet_per_task: int


def read_noise_settings(path: Path = SETTINGS_FILE) -> NoiseSettings:
    """The `noise` section of the gate settings, or a refusal (NoiseSampleError)."""
    try:
        section = json.loads(path.read_text(encoding="utf-8"))["noise"]
        settings = NoiseSettings(section["seed"], section["config"], tuple(section["tasks"]),
                                 section["jev_per_task"], section["sonnet_per_task"])  # fmt: skip
        rule = section["rule"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NoiseSampleError(f"{path} has no readable noise section: {error!r}") from error
    whole = (settings.seed, settings.jev_per_task, settings.sonnet_per_task)
    if not all(type(n) is int for n in whole) or not 1 <= settings.sonnet_per_task <= settings.jev_per_task:
        raise NoiseSampleError("the seed and the counts are whole numbers, with 1 <= Sonnet's count <= Jev's")
    if not settings.tasks or len(set(settings.tasks)) != len(settings.tasks) or not settings.config:
        raise NoiseSampleError("the noise section names a config and distinct tasks")
    if not isinstance(rule, str) or not rule:
        raise NoiseSampleError("the noise section states its drawing rule in words")
    return settings


def task_of(item_id: str) -> str:
    return item_id.split(":", 1)[0]


def draw(passed: Mapping[str, bool], *, seed: int, task: str, count: int) -> list[str]:
    """The first `count` items that passed the free checks, in the task's seeded order. `passed` maps each item ID of
    the task to whether its answer passed. Refuses (NoiseSampleError) a task with fewer than `count` that did."""
    order = sorted(passed)
    random.Random(f"{seed}/{task}").shuffle(order)
    chosen = [item for item in order if passed[item]][:count]
    if len(chosen) < count:
        raise NoiseSampleError(f"{task}: only {len(chosen)} answers pass the free checks, {count} are needed")
    return chosen


def noise_lists(
    free_records: Sequence[Mapping[str, object]], settings: NoiseSettings
) -> dict[str, list[tuple[str, str]]]:
    """The two lists as (item ID, config) pairs, tasks in the section's order: `jev` (the noise pass) and `sonnet`
    (the latency sample). `free_records` are train's stored free-check records; only the config's are read."""
    by_task: dict[str, dict[str, bool]] = {task: {} for task in settings.tasks}
    for record in free_records:
        if record["config"] != settings.config:
            continue
        item = str(record["item_id"])
        if task_of(item) not in by_task:
            raise NoiseSampleError(f"{item} is of a task the noise section doesn't name")
        by_task[task_of(item)][item] = bool(record["passed"])
    jev: list[tuple[str, str]] = []
    sonnet: list[tuple[str, str]] = []
    for task in settings.tasks:
        if not by_task[task]:
            raise NoiseSampleError(f"{task} has no {settings.config} free-check record")
        chosen = draw(by_task[task], seed=settings.seed, task=task, count=settings.jev_per_task)
        jev += [(item, settings.config) for item in chosen]
        sonnet += [(item, settings.config) for item in chosen[: settings.sonnet_per_task]]
    return {"jev": jev, "sonnet": sonnet}
