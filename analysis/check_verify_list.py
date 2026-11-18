"""Check the unlock's verify list against the amendments' records, and each entry's refusal (PLAN_v4.md Amendment A19
items 3 and 13; A9 item 7, A11 item 17, A15 item 17, A17 item 26.1; issue #81).

Run from the project root:  uv run python -m analysis.check_verify_list

No results/, Docker, model snapshot, Jev or Anthropic call, and nothing near the real sealed store: the list is built from
the plan's text and held against the project's committed files, and the unlock runs on a made-up world whose root holds
copies of them. The model files, the library versions, the image digest, the macOS build and `check_benchmark` are
stood in by a made-up machine here; `analysis/check_verify_real.py` (hook only) checks them on the real one.

- the list holds each amendment's records: the counts worked by hand (51 entries: 36 files, 5 model files, 5 libraries,
  the model ID, the image, `check_benchmark`, the macOS build and the chip), every hash a record gives is on the list, the
  uncommitted records (named by hand) are the only ones that aren't, and each file's own bytes hash to the record;
- a list missing an entry, or holding another hash, fails that comparison, and a plan with a record missing, reworded or
  recorded twice with two hashes refuses the build;
- every file in `routers/` is on the list, and so is each import closure the freezes name;
- in the unlock, each entry alone refuses naming itself when its file or value is tampered with (appended byte, missing
  file, another model ID, library version, image digest, a failing `check_benchmark`), and the same world unlocks
  untouched; a library version differing refuses, a local tag doesn't, a macOS build or chip that differs is reported and
  the unlock goes on; each entry's result is in the record; nothing opens the store's database.
"""

import dataclasses
import inspect
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

from analysis import unlock as unlock_module
from analysis.freeze import closure, sha256
from analysis.made_up_world import MadeUp, made_up_world
from analysis.plan import PLAN, amendment_item
from analysis.unlock import CHECKS, UnlockRefusal, unlock, verified_list
from analysis.verify_list import Entry, Probes, real_list
from harness.client import PROJECT_ROOT
from routers.sandbox import SandboxError

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# By hand from the amendments: how many `path`: `hash` records each item gives (untracked ones among them).
RECORD_COUNTS = {("A7", 3): 2, ("A9", 17): 5, ("A10", 1): 1, ("A11", 19): 19, ("A11", 20): 1, ("A15", 19): 11,
                 ("A17", 28): 23, ("A17", 29): 0}  # fmt: skip
OPERATING_POINTS = ("routers/operating_points.json", "d183dd1c0ef55b12d2487dcff8275fff2a08339849d3ff9cbb42799436e47c09")
# Records of files kept out of the repository (A11 item 19's four caches, A15 item 19's two train files and A17 item
# 28's four train records): the spec's lists don't name them, so no unlock verifies them.
UNTRACKED = {"train.npy", "train_items.json", "dev.npy", "dev_items.json", "results/jev/jev_train.jsonl",
             "results/jev/jev_bolstered_train.jsonl", "results/gates/free_checks_train.jsonl",
             "results/gates/jev_gate_train.jsonl", "results/gates/sonnet_gate_train.jsonl",
             "results/gates/sonnet_gate_latency.jsonl"}  # fmt: skip
VERSIONS = {
    "numpy": "2.5.4",
    "scikit-learn": "1.9.1",
    "tokenizers": "0.23.3",
    "torch": "2.14.1",
    "transformers": "5.20.2",
}
IMAGE_DIGEST = "sha256:5385631f01e4d67765231145bcb066d51679c99a01770916bee391662e064559"
OS_LINE = "macOS 27.0.1 (build 26A434)"
CHIP = "Apple M5 Pro"
SNAPSHOT = ".cache/gte-modernbert-base/e7f32e3c00f91d699e8c43b53106206bcc72bb22"
MODEL_FILES = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
GATE_ENTRY_POINTS = ["gates", "jev_gate", "sonnet_gate", "cascade", "sandbox"]


def recorded_pairs(item: str) -> list[tuple[str, str]]:
    """Every `path`: `hash` the item records, scanned line by line (the list's own reader works on whole sections)."""
    found = []
    for line in item.splitlines():
        found += re.findall(r"`([^`\s]+)`:\s+`([0-9a-f]{64})`", line)
    return found


def missing_from(entries: tuple[Entry, ...], pairs: list[tuple[str, str]]) -> list[str]:
    """The records `pairs` that no entry holds with the same recorded hash (an entry's name is `file:<path>` or
    `model_file:<name>`)."""
    held = {entry.name.split(":", 1)[1]: entry.expected for entry in entries if ":" in entry.name}
    return [path for path, digest in pairs if held.get(path) != digest and path not in UNTRACKED]


def plan_without(plan: str, needle: str) -> str:
    assert needle in plan
    return plan.replace(needle, "", 1)


def list_against_amendments() -> tuple[Entry, ...]:
    plan = PLAN.read_text(encoding="utf-8")
    entries = real_list(plan)
    names = [entry.name for entry in entries]
    kinds = {kind: sum(name.startswith(kind + ":") for name in names) for kind in ("file", "model_file", "library")}
    check(
        len(entries) == 51 and len(set(names)) == 51,
        f"the list has 51 entries, each with its own name ({len(entries)})",
    )
    check(kinds == {"file": 36, "model_file": 5, "library": 5}, f"36 files, 5 model files and 5 libraries {kinds}")
    check({"jev_model_id", "image_digest", "check_benchmark", "environment:os", "environment:chip"} <= set(names),
          "the model ID, the image digest, a run of check_benchmark, the macOS build and the chip are on it")  # fmt: skip
    check(
        not any("seal" in name for name in names), "the sealed store's own hash isn't on it: the unlock checks it alone"
    )
    check([e.name for e in entries if e.reported] == ["environment:os", "environment:chip"],
          "only the macOS build and the chip are reported, not refused")  # fmt: skip
    by_name = {entry.name: entry for entry in entries}
    check({name.removeprefix("library:"): by_name[name].expected for name in names if name.startswith("library:")}
          == VERSIONS, "the library versions are A11 item 19's, as written out by hand")  # fmt: skip
    check(by_name["image_digest"].expected == IMAGE_DIGEST and by_name["jev_model_id"].expected == "jev-1.14.0",
          "the image digest and Jev's model ID are A17 item 28's and A15 item 19's")  # fmt: skip
    check(by_name["environment:os"].expected == OS_LINE and by_name["environment:chip"].expected == CHIP,
          "the macOS build and the chip are A11 item 19's")  # fmt: skip

    print("\n-- every record is on the list --")
    for (amendment, number), count in RECORD_COUNTS.items():
        item = amendment_item(plan, amendment, number)
        pairs = recorded_pairs(item)
        check(len(pairs) == count, f"{amendment} item {number} gives {count} records ({len(pairs)})")
        check(
            missing_from(entries, pairs) == [],
            f"{amendment} item {number}'s records are all on the list, with its hashes",
        )
    check(missing_from(entries, [OPERATING_POINTS]) == [], "A17 item 29's whole-file hash is on the list")
    check(set(UNTRACKED) == {p for key in (("A11", 19), ("A15", 19), ("A17", 28)) for p, _ in recorded_pairs(amendment_item(plan, *key))
                               if p in UNTRACKED}, "the ten untracked records are the ones left off, and all are in the plan")  # fmt: skip
    return entries


def known_positives(entries: tuple[Entry, ...]) -> None:
    plan = PLAN.read_text(encoding="utf-8")
    pairs = recorded_pairs(amendment_item(plan, "A9", 17))
    short = tuple(entry for entry in entries if entry.name != "file:routers/rules.py")
    check(
        missing_from(short, pairs) == ["routers/rules.py"],
        "known positive: a list missing one entry fails the comparison",
    )
    other = tuple(
        dataclasses.replace(e, expected="0" * 64) if e.name == "file:routers/rules.py" else e for e in entries
    )
    check(
        missing_from(other, pairs) == ["routers/rules.py"],
        "known positive: a list holding another hash for an entry fails it",
    )

    def refuses_build(text: str) -> bool:
        try:
            real_list(text)
        except ValueError:
            return True
        return False

    def edited(amendment: str, number: int, needle: str, change) -> str:
        """The plan with the one line of the item that holds `needle` changed (a path's line also stands in other items)."""
        item = amendment_item(plan, amendment, number)
        lines = [line for line in item.splitlines() if needle in line]
        assert len(lines) == 1 and plan.count(item) == 1, needle
        return plan.replace(item, item.replace(lines[0], change(lines[0])))

    check(refuses_build(edited("A15", 19, "`routers/jev_weights.json`", lambda line: "")),
          "a plan with a frozen-file record missing refuses the build")  # fmt: skip
    check(refuses_build(edited("A17", 28, "`routers/ladder.py`", lambda line: re.sub(r"[0-9a-f]{64}", "f" * 64, line))),
          "a path recorded twice with two hashes refuses the build")  # fmt: skip
    check(refuses_build(plan.replace("as committed has the 64-digit sha256", "as committed has the sha256")),
          "a plan whose whole operating-points hash is reworded refuses the build")  # fmt: skip
    renamed = edited(
        "A11", 20, "`routers/trained_fold_weights.json`", lambda line: line.replace("fold weights**", "fold set**")
    )
    check(refuses_build(renamed), "a plan whose fold-weights record can't be found refuses the build")
    check(not refuses_build(plan), "the plan as committed builds")


def files_and_closures(entries: tuple[Entry, ...]) -> None:
    recorded = {entry.name.removeprefix("file:"): entry.expected for entry in entries if entry.name.startswith("file:")}
    wrong = [path for path, digest in recorded.items() if sha256(PROJECT_ROOT / path) != digest]
    check(not wrong, f"each listed file's own bytes hash to its record {wrong}")
    in_routers = {f"routers/{p.name}" for p in (PROJECT_ROOT / "routers").iterdir() if p.suffix in (".py", ".json")}
    check(len(in_routers) == 22 and in_routers <= set(recorded), "all 22 files in routers/ are on the list")
    entry_points = ["routers/rules.py", "routers/trained.py", "routers/jev.py"]
    entry_points += [f"routers/{name}.py" for name in GATE_ENTRY_POINTS]
    for entry_point in entry_points:
        needed = set(closure(entry_point))
        check(
            needed <= set(recorded), f"{entry_point}'s import closure is on the list {sorted(needed - set(recorded))}"
        )
    check("routers/ladder.py" in closure("routers/jev_gate.py"),
          "known positive: the closure follows imports (jev_gate reaches ladder)")  # fmt: skip
    go = PROJECT_ROOT / "analysis" / ("final" + "_go.py")  # spelled in pieces: check_read_path searches for the name
    check("verify=real_list()" in go.read_text(encoding="utf-8"), "the go's world carries the list built from the plan")


class Machine:
    """A made-up machine: what the probes see. Each attribute is what a check turns to tamper with the entry it names."""

    def __init__(self) -> None:
        self.versions = dict(VERSIONS)
        self.system, self.release, self.arch = "Darwin", "27.0.0", "arm64"
        self.product, self.build, self.chip = "27.0.1", "26A434", CHIP
        self.image, self.benchmark_ok = IMAGE_DIGEST, True
        self.image_error: Exception | None = None

    def version(self, name: str) -> str:
        return self.versions[name]

    def command(self, *args: str) -> str:
        table = {("sw_vers", "-productVersion"): self.product, ("sw_vers", "-buildVersion"): self.build,
                 ("sysctl", "-n", "machdep.cpu.brand_string"): self.chip}  # fmt: skip
        return table[args]

    def image_id(self, image: str) -> str:
        if self.image_error is not None:
            raise self.image_error
        return self.image

    def probes(self) -> Probes:
        return Probes(self.version, self.command, self.image_id, lambda: self.benchmark_ok, lambda: self.system,
                      lambda: self.release, lambda: self.arch)  # fmt: skip


def attempt(world):
    """The handle the unlock returns, or its refusal; the store is never opened as a database."""
    with mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened")):
        try:
            return unlock(world)
        except UnlockRefusal as refusal:
            return refusal


def tamper_with(entry: Entry, root: Path, machine: Machine):
    """(tamper, restore) for the entry's file or value."""
    name = entry.name
    if name.startswith(("file:", "model_file:")):
        path = root / (name.removeprefix("file:") if name.startswith("file:") else f"{SNAPSHOT}/{name.split(':')[1]}")
        original = path.read_bytes()
        return lambda: path.write_bytes(original + b" "), lambda: path.write_bytes(original)
    if name == "jev_model_id":
        path = root / "routers" / "jev_settings.json"
        original = path.read_bytes()
        assert b"jev-1.14.0" in original
        return (lambda: path.write_bytes(original.replace(b"jev-1.14.0", b"jev-9.9.9")),
                lambda: path.write_bytes(original))  # fmt: skip
    if name.startswith("library:"):
        library = name.removeprefix("library:")
        return (lambda: machine.versions.update({library: "9.9.9"}),
                lambda: machine.versions.update({library: VERSIONS[library]}))  # fmt: skip
    if name == "image_digest":
        return lambda: setattr(machine, "image", "sha256:" + "0" * 64), lambda: setattr(machine, "image", IMAGE_DIGEST)
    if name == "check_benchmark":
        return lambda: setattr(machine, "benchmark_ok", False), lambda: setattr(machine, "benchmark_ok", True)
    raise AssertionError(f"no tampering known for {name}")


def lay_out(made: MadeUp, entries: tuple[Entry, ...]) -> tuple[Entry, ...]:
    """Copies of every listed file under the made-up root, and a stand-in model snapshot (the model is too large to copy,
    and `.cache/` is absent in CI): those entries record the stand-ins' hashes."""
    root = made.world.root
    for entry in entries:
        if entry.name.startswith("file:"):
            relative = entry.name.removeprefix("file:")
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / relative, root / relative)
    stood_in = []
    for entry in entries:
        if entry.name.startswith("model_file:"):
            path = root / SNAPSHOT / entry.name.split(":")[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"a stand-in for {entry.name}\n")
            entry = dataclasses.replace(entry, expected=sha256(path))
        stood_in.append(entry)
    return tuple(stood_in)


def reported_by(got) -> dict:
    return {} if isinstance(got, UnlockRefusal) else got.results[2].detail["reported"]


def unlocking() -> None:
    machine = Machine()
    with made_up_world() as made:
        entries = lay_out(made, real_list(probes=machine.probes()))
        root = made.world.root
        check(MODEL_FILES == [e.name.split(":")[1] for e in entries if e.name.startswith("model_file:")],
              "the five model files are A11 item 19's, by name")  # fmt: skip
        names = [entry.name for entry in entries]
        full = dataclasses.replace(made.world, verify=entries)

        print("\n-- the unlock, with the whole list --")
        got = attempt(full)
        check(not isinstance(got, UnlockRefusal), f"the made-up world unlocks with every entry untouched {got}")
        ours = [r for r in getattr(got, "results", ()) if r.name == "verify_list"]
        check(len(ours) == 1 and list(ours[0].detail["checked"]) == names and ours[0].detail["reported"] == {},
              "the record's verify_list result holds each of the 51 entries' observed value, and reports nothing")  # fmt: skip
        empty = attempt(made.world)
        check(not isinstance(empty, UnlockRefusal) and empty.results[2].detail == {"checked": {}, "reported": {}},
              "a world with an empty list unlocks and records an empty one (the other checks' made-up worlds)")  # fmt: skip
        order = [function.__name__ for function in CHECKS]
        check(order.index("verified_list") < order.index("sealed_store_sha256"),
              "the list is verified before the store's hash is read")  # fmt: skip
        check("sealed_store" not in inspect.getsource(verified_list), "the list's check never names the store")
        rules = next(e for e in entries if e.name == "file:routers/rules.py")
        tamper, restore = tamper_with(rules, root, machine)
        tamper()
        refusal = attempt(full)
        check(isinstance(refusal, UnlockRefusal) and refusal.check == rules.name,
              "with the whole list, a tampered file refuses at its own entry")  # fmt: skip
        check(rules.expected in str(refusal), "the refusal gives the recorded hash")
        restore()
        check(not isinstance(attempt(full), UnlockRefusal), "restored, it unlocks again")

        print("\n-- each entry alone, tampered with --")
        untouched_refused, wrong_name, count = [], [], 0
        for entry in entries:
            if entry.reported:
                continue
            count += 1
            world = dataclasses.replace(made.world, verify=(entry,))
            if isinstance(attempt(world), UnlockRefusal):
                untouched_refused.append(entry.name)
            tamper, restore = tamper_with(entry, root, machine)
            tamper()
            refusal = attempt(world)
            if not isinstance(refusal, UnlockRefusal) or refusal.check != entry.name:
                wrong_name.append(entry.name)
            restore()
        check(
            count == 49 and not untouched_refused,
            f"each of the 49 refusable entries unlocks when untouched {untouched_refused}",
        )
        check(not wrong_name, f"a tampered copy of each of the 49 entries refuses, naming that entry {wrong_name}")

        print("\n-- the other ways an entry can fail --")
        gates = root / "routers" / "gates.py"
        original = gates.read_bytes()
        gates.unlink()
        refusal = attempt(
            dataclasses.replace(made.world, verify=tuple(e for e in entries if e.name == "file:routers/gates.py"))
        )
        check(isinstance(refusal, UnlockRefusal) and refusal.check == "file:routers/gates.py" and "missing" in str(refusal),
              "a missing file refuses at its entry, saying it is missing")  # fmt: skip
        gates.write_bytes(original)
        machine.image_error = SandboxError("Docker is not running")
        refusal = attempt(full)
        check(isinstance(refusal, UnlockRefusal) and refusal.check == "image_digest" and "SandboxError" in str(refusal),
              "an image the unlock can't read refuses at image_digest")  # fmt: skip
        machine.image_error = None
        machine.versions["torch"] = "2.14.1+cpu"
        check(not isinstance(attempt(full), UnlockRefusal), "a local tag on a version (2.14.1+cpu) isn't a difference")
        del machine.versions["torch"]
        refusal = attempt(full)
        check(isinstance(refusal, UnlockRefusal) and refusal.check == "library:torch",
              "a library that isn't installed refuses")  # fmt: skip
        machine.versions["torch"] = VERSIONS["torch"]

        print("\n-- a macOS build or chip that differs is reported, and the unlock goes on --")
        machine.build = "26A999"
        reported = reported_by(attempt(full))
        check(reported == {"environment:os": {"recorded": OS_LINE, "observed": "macOS 27.0.1 (build 26A999)"}},
              f"another build is reported with both values, and the unlock returns its handle {reported}")  # fmt: skip
        machine.build, machine.chip = "26A434", "Apple M6 Max"
        reported = reported_by(attempt(full))
        check(reported == {"environment:chip": {"recorded": CHIP, "observed": "Apple M6 Max"}},
              f"another chip is reported {reported}")  # fmt: skip
        machine.system, machine.release, machine.arch = "Linux", "6.1.0", "x86_64"
        reported = reported_by(attempt(full))
        check(sorted(reported) == ["environment:chip", "environment:os"],
              f"another system is reported in both entries {sorted(reported)}")  # fmt: skip


def main() -> int:
    print("-- the list against the amendments --")
    entries = list_against_amendments()
    print("\n-- the list against the files and the import closures --")
    files_and_closures(entries)
    print("\n-- the list's own guards can fail --")
    known_positives(entries)
    print("\n-- the unlock --")
    unlocking()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
