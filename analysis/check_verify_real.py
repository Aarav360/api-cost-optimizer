"""Check that every entry of the real verify list holds on this machine (PLAN_v4.md Amendment A19 items 3 and 13; A9 item 7,
A11 item 17, A15 item 17, A17 item 26.1; issue #81).

Run from the project root:  uv run python -m analysis.check_verify_real

Hook only: it needs the model snapshot under `.cache/` and Docker with the sandbox image, as the unlock does at the go.
`analysis/check_verify_list.py` holds the list against the amendments and each entry's refusal on a made-up root; this
check runs the real list, built from the plan's records, against the working tree, the snapshot, the installed
libraries, the image and a run of `check_benchmark`:

- every entry but the macOS build and chip verifies, and nothing is the sealed store: its hash is checked only inside
  the unlock, at the go, and never here;
- the readings agree with the project's own other sources for them: the model files with `routers/trained.verify_snapshot`
  and its directory, the image digest with `routers.gates.image_digest`, the library versions and the macOS build and
  chip with `analysis/measure_trained_router`'s, which wrote the records;
- an entry holding another value fails, for each kind of reading, and a root with none of the files refuses at once;
- a macOS build or chip that differs from A11 item 19's record is printed, not failed (the unlock reports it and goes on).

Nothing here calls Jev or the Anthropic API, or touches the real sealed store.
"""

import dataclasses
import sys
import tempfile
from pathlib import Path

from analysis import measure_trained_router as measure
from analysis.verify_list import real_list, run, verdict
from harness.client import PROJECT_ROOT
from routers import gates, trained

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    print("-- the real list, on this machine --")
    entries = real_list()
    report = run(entries, PROJECT_ROOT)
    refused = None if report.refused is None else f"{report.refused[0].name}: found {report.refused[1].observed!r}"
    check(
        report.refused is None, f"every entry verifies against the working tree, the snapshot and the image {refused}"
    )
    check(len(report.checked) + len(report.reported) == len(entries) == 51, "all 51 entries were read")
    check(not any("seal" in entry.name for entry in entries), "none of them is the sealed store")
    for name, differs in report.reported.items():
        print(f"NOTE  {name} differs from the record (reported, not refused): {differs}")

    print("\n-- the readings against the project's other sources --")
    check(trained.SNAPSHOT == PROJECT_ROOT / ".cache" / "gte-modernbert-base" / "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
          "the snapshot folder the list reads is the model code's own")  # fmt: skip
    check(sorted(trained.MODEL_FILES) == sorted(n.split(":")[1] for n in report.checked if n.startswith("model_file:")),
          "the five model files are the ones the model code reads")  # fmt: skip
    settings = trained.read_settings()
    trained.verify_snapshot(settings)  # raises SnapshotError, naming each file that differs
    check(all(settings.files[n.split(":")[1]] == v for n, v in report.checked.items() if n.startswith("model_file:")),
          "the model code's own check passes on the snapshot, and its settings hold the same five hashes")  # fmt: skip
    check(report.checked["image_digest"] == gates.image_digest(), "the image digest is the one the gates read")
    seen = {n.removeprefix("library:"): v for n, v in report.checked.items() if n.startswith("library:")}
    check(seen == {k: v.split("+")[0] for k, v in measure.library_versions().items()},
          "the library versions are the ones the measurement script reads")  # fmt: skip
    environment = measure.environment()
    now = {name: found.observed for name, found in ((e.name, verdict(e, PROJECT_ROOT)) for e in entries if e.reported)}
    check(now == {"environment:os": environment["os"], "environment:chip": environment["chip"]},
          "the macOS build and chip are read as the measurement script wrote them")  # fmt: skip

    print("\n-- each kind of reading can fail --")
    kinds = ("file:routers/rules.py", "model_file:config.json", "library:numpy", "image_digest", "jev_model_id")
    for name in kinds:
        entry = next(e for e in entries if e.name == name)
        wrong = dataclasses.replace(entry, expected="not what was recorded")
        check(not verdict(wrong, PROJECT_ROOT).matches and run((wrong,), PROJECT_ROOT).refused is not None,
              f"{name} holding another value refuses")  # fmt: skip
    chip = next(e for e in entries if e.name == "environment:chip")
    other = run((dataclasses.replace(chip, expected="another chip"),), PROJECT_ROOT)
    check(other.refused is None and "environment:chip" in other.reported, "another chip is reported and not refused")
    with tempfile.TemporaryDirectory() as empty:
        bare = run(entries, Path(empty))
        check(bare.refused is not None and "missing" in bare.refused[1].observed,
              "a root holding none of the files refuses at the first, as missing")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
