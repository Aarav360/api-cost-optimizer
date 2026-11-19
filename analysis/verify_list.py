"""The unlock's verify list: every freeze, hash and pin the final evaluation relies on (PLAN_v4.md Amendment A19 items 3
and 13; A9 item 7, A11 item 17, A15 item 17, A17 item 26.1; issue #81).

No freeze record is machine-readable: each amendment records its hashes as text in `PLAN_v4.md`. `real_list()` reads
those records and builds one `Entry` per thing the unlock verifies, so the go can't verify less than the amendments
promise (`analysis/check_verify_list.py` holds the list against each amendment's records, and against the files):

- the benchmark's two hashes (A7 item 3) and a run of `check_benchmark`;
- the rules router's freeze (A9 item 17, A10 item 1);
- F's five model files, settings, F-train's weights, the fold weights (A11 item 20) and the fold files, and the installed
  library versions (A11 item 19), whose macOS build and chip are reported, not refused;
- G's and G+'s hashes and Jev's model ID (A15 item 19);
- the gates' hashes, the fixed cutoffs, the whole operating-points file, the sandbox Dockerfile and the image digest
  (A17 items 28 and 29).

A path that several records name (`routers/ladder.py` is in four) is one entry; two records giving a path two hashes
refuse the build. The sealed store's own hash is not here: it is checked inside the unlock alone (A19 item 3).

An entry knows how to observe its own value (`observe(root)`), so the same entry checks a real root and a made-up copy
of it, and `Probes` is the one place a check swaps the machine out (installed versions, `sw_vers`, Docker).
"""

import importlib.metadata
import io
import json
import platform
import re
import subprocess
from collections import defaultdict
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from analysis.freeze import sha256
from analysis.plan import PLAN, amendment_item

RECORD = re.compile(r"`([\w./-]+)`: `([0-9a-f]{64})`")  # a freeze record's bullet: the path, then its sha256


@dataclass(frozen=True)
class Entry:
    """One thing the unlock verifies: its name (what a refusal and the record call it), the value the amendment records,
    how to observe it under a root, and the records that name it. A `reported` entry is shown when it differs, not refused."""

    name: str
    expected: str
    observe: Callable[[Path], str]
    sources: tuple[str, ...]
    reported: bool = False


@dataclass(frozen=True)
class Verdict:
    observed: str
    matches: bool


def verdict(entry: Entry, root: Path) -> Verdict:
    """What the entry observes under `root` and whether it is what was recorded. A value that can't be read (a missing
    file, Docker down, a library not installed) is a difference, shown as what went wrong."""
    try:
        observed = entry.observe(root)
    except FileNotFoundError:
        observed = "missing"
    except Exception as error:  # noqa: BLE001 - whatever a probe raises, the entry fails with its class and message
        observed = f"unreadable: {type(error).__name__}: {error}"
    return Verdict(observed, observed == entry.expected)


# What a check can swap for the machine: the installed versions, the shell's answers, Docker and `check_benchmark`.


def command_output(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()


def image_id(image: str) -> str:
    from routers.gates import image_digest  # light: `routers.request` and `routers.sandbox` only

    return image_digest(image)


def benchmark_check_passes() -> bool:
    """`harness.check_benchmark`'s own run (A5 item 7), its report held back: it prints a table of counts."""
    from harness.check_benchmark import main

    with redirect_stdout(io.StringIO()):
        return main() == 0


@dataclass(frozen=True)
class Probes:
    version: Callable[[str], str] = importlib.metadata.version
    command: Callable[..., str] = command_output
    image_id: Callable[[str], str] = image_id
    benchmark_check: Callable[[], bool] = benchmark_check_passes
    system: Callable[[], str] = platform.system
    release: Callable[[], str] = platform.release
    machine: Callable[[], str] = platform.machine


REAL = Probes()


# The entries


def file_entry(path: str, digest: str, sources: tuple[str, ...]) -> Entry:
    return Entry(f"file:{path}", digest, lambda root: sha256(root / path), sources)


def model_file_entry(directory: str, name: str, digest: str, source: str) -> Entry:
    return Entry(f"model_file:{name}", digest, lambda root: sha256(root / directory / name), (source,))


def library_entry(name: str, version: str, probes: Probes, source: str) -> Entry:
    """An installed library's version, with a local tag (`2.14.1+cpu`) cut off, as `check_trained_freeze` compares it."""
    return Entry(f"library:{name}", version, lambda root: probes.version(name).split("+")[0], (source,))


def environment_entries(os_line: str, chip: str, probes: Probes, source: str) -> list[Entry]:
    """The macOS build and the chip, as `analysis/measure_trained_router.environment()` writes them: reported only."""

    def observed_os(root: Path) -> str:
        if probes.system() != "Darwin":
            return f"{probes.system()} {probes.release()}"
        return (
            f"macOS {probes.command('sw_vers', '-productVersion')} (build {probes.command('sw_vers', '-buildVersion')})"
        )

    def observed_chip(root: Path) -> str:
        if probes.system() != "Darwin":
            return probes.machine()
        return probes.command("sysctl", "-n", "machdep.cpu.brand_string")

    return [
        Entry("environment:os", os_line, observed_os, (source,), reported=True),
        Entry("environment:chip", chip, observed_chip, (source,), reported=True),
    ]


def model_id_entry(path: str, model: str, source: str) -> Entry:
    return Entry("jev_model_id", model, lambda root: json.loads((root / path).read_text())["jev"]["model"], (source,))


def image_entry(image: str, digest: str, probes: Probes, source: str) -> Entry:
    return Entry("image_digest", digest, lambda root: probes.image_id(image), (source,))


def benchmark_check_entry(probes: Probes, source: str) -> Entry:
    return Entry(
        "check_benchmark", "passed", lambda root: "passed" if probes.benchmark_check() else "failed", (source,)
    )


# Reading the records


def top_level(item: str) -> dict[str, str]:
    """An item's top-level bullets that open with a bold title, by that title: a record's sections."""
    found = {}
    for chunk in re.split(r"\n    - ", item)[1:]:
        if title := re.match(r"\*\*(.+?)\*\*", chunk):
            found[title.group(1)] = chunk
    return found


def section(item: str, title: str, where: str) -> str:
    found = top_level(item)
    if title not in found:
        raise ValueError(f"{where} has no {title!r} bullet, so there is nothing to build its entries from")
    return found[title]


def records_in(text: str, where: str, expect: int | None = None) -> list[tuple[str, str]]:
    found = RECORD.findall(text)
    if not found or (expect is not None and len(found) != expect):
        raise ValueError(f"{where} holds {len(found)} hash records, not the {expect or 'some'} expected")
    return found


def single(pattern: str, text: str, where: str) -> re.Match[str]:
    found = re.search(pattern, text)
    if found is None:
        raise ValueError(f"{where} has no record matching {pattern!r}")
    return found


def real_list(plan: str | None = None, probes: Probes = REAL) -> tuple[Entry, ...]:
    """The verify list, built from the plan's freeze records (`plan` defaults to the committed `PLAN_v4.md`).

    A record the reader can't find refuses the build, so a reworded amendment can't quietly shrink the list.
    """
    plan = PLAN.read_text(encoding="utf-8") if plan is None else plan
    sources: dict[str, list[str]] = defaultdict(list)
    digests: dict[str, str] = {}

    def record(path: str, digest: str, source: str) -> None:
        if digests.setdefault(path, digest) != digest:
            raise ValueError(f"{path} has two recorded hashes: {digests[path]} and {digest} (in {source})")
        if source not in sources[path]:
            sources[path].append(source)

    def item_of(amendment: str, number: int) -> tuple[str, str]:
        return amendment_item(plan, amendment, number), f"{amendment} item {number}"

    a7, where = item_of("A7", 3)
    for path, digest in records_in(a7, where, 2):
        record(path, digest, where)
    a9, where = item_of("A9", 17)
    for path, digest in records_in(a9, where, 5):
        record(path, digest, where)
    a10, where = item_of("A10", 1)
    for path, digest in records_in(a10, where, 1):
        record(path, digest, where)

    a11, where = item_of("A11", 19)
    model = section(a11, "The model", where)
    commit = single(r"`([\w-]+)/([\w-]+)` at commit `([0-9a-f]{40})`", model, where)
    model_files = records_in(model, where, 5)
    for path, digest in records_in(section(a11, "Frozen files", where), where, 10):
        record(path, digest, where)
    versions = re.findall(r"([a-z][\w-]*) (\d+(?:\.\d+)+)", section(a11, "Library versions", where).split(":", 1)[1])
    if len(versions) != 5:
        raise ValueError(f"{where} names {len(versions)} library versions, not 5")
    environment = single(r"(macOS [\d.]+ \(build \w+\)), ([^,]+),", section(a11, "Environment", where), where)
    a11_20, where_20 = item_of("A11", 20)
    for path, digest in records_in(single(r"fold weights\*\*.*", a11_20, where_20).group(0), where_20, 1):
        record(path, digest, where_20)

    a15, where = item_of("A15", 19)
    for path, digest in records_in(section(a15, "Frozen files", where), where, 9):
        record(path, digest, where)
    jev_model = single(r"model `([^`]+)`", section(a15, "Jev's identity", where), where).group(1)
    jev_where = where

    a17, where = item_of("A17", 28)
    for path, digest in records_in(section(a17, "Frozen files", where), where, 18):
        record(path, digest, where)
    for path, digest in records_in(section(a17, "The two fixed cutoffs", where), where, 1):
        record(path, digest, where)
    image = single(r"`(api-cost-optimizer-sandbox:[^`]+)`, digest `(sha256:[0-9a-f]{64})`", a17, where)
    a17_29, where_29 = item_of("A17", 29)
    whole = single(r"`(routers/operating_points\.json)` as committed has the 64-digit sha256 `([0-9a-f]{64})`", a17_29,
                   where_29)  # fmt: skip
    record(whole.group(1), whole.group(2), where_29)

    entries = [file_entry(path, digest, tuple(sources[path])) for path, digest in digests.items()]
    directory = f".cache/{commit.group(2)}/{commit.group(3)}"
    entries += [model_file_entry(directory, name, digest, "A11 item 19") for name, digest in model_files]
    entries.append(model_id_entry("routers/jev_settings.json", jev_model, jev_where))
    entries += [library_entry(name, version, probes, "A11 item 19") for name, version in versions]
    entries.append(image_entry(image.group(1), image.group(2), probes, "A17 item 28"))
    entries.append(benchmark_check_entry(probes, "A7 item 3"))
    entries += environment_entries(environment.group(1), environment.group(2), probes, "A11 item 19")
    return tuple(entries)


@dataclass(frozen=True)
class Report:
    """What a pass over the list found: each passed entry's observed value, each reported difference, the first refusal."""

    checked: dict[str, str] = field(default_factory=dict)
    reported: dict[str, dict[str, str]] = field(default_factory=dict)
    refused: tuple[Entry, Verdict] | None = None


def run(entries: tuple[Entry, ...], root: Path) -> Report:
    """Every entry in order, stopping at the first refusal."""
    checked: dict[str, str] = {}
    reported: dict[str, dict[str, str]] = {}
    for entry in entries:
        found = verdict(entry, root)
        if found.matches:
            checked[entry.name] = found.observed
        elif entry.reported:
            reported[entry.name] = {"recorded": entry.expected, "observed": found.observed}
        else:
            return Report(checked, reported, (entry, found))
    return Report(checked, reported)
