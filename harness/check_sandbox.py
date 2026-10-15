"""Check the sandbox: escape attempts must fail, and MBPP+ reference solutions must pass.

Run from the project root:  uv run python -m harness.check_sandbox

Re-run after OS or Python upgrades, since Seatbelt behavior can change.
"""

import os
import signal
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harness.client  # noqa: F401  (loads .env, so the key is in this process's environment)
from harness.client import PROJECT_ROOT
from harness.loaders import load_mbppplus
from harness.sandbox import MAX_OUTPUT_BYTES, run_python

OUTSIDE_FILE = Path("/tmp") / "sandbox-escape-test.txt"

# (name, source, check on the result)
ESCAPES = [
    ("runs plain code", "print('hello')", lambda r: r.returncode == 0 and r.stdout.strip() == "hello"),
    ("numpy is importable", "import numpy; print(numpy.__name__)", lambda r: r.stdout.strip() == "numpy"),
    (
        "API key not in environment",
        "import os; print('ANTHROPIC_API_KEY' in os.environ, len(os.environ))",
        lambda r: r.stdout.startswith("False"),
    ),
    ("cannot read project .env", f"open({str(PROJECT_ROOT / '.env')!r}).read()", lambda r: "PermissionError" in r.stderr),
    (
        "cannot read other files in home",
        f"open({str(PROJECT_ROOT / 'PLAN_v4.md')!r}).read()",
        lambda r: "PermissionError" in r.stderr,
    ),
    ("cannot list home folder", f"import os; os.listdir({str(Path.home())!r})", lambda r: "PermissionError" in r.stderr),
    (
        "cannot write outside its temp dir",
        f"open({str(OUTSIDE_FILE)!r}, 'w').write('x')",
        lambda r: "PermissionError" in r.stderr and not OUTSIDE_FILE.exists(),
    ),
    (
        "cannot write into the project",
        f"open({str(PROJECT_ROOT / 'escape.txt')!r}, 'w').write('x')",
        lambda r: "PermissionError" in r.stderr and not (PROJECT_ROOT / "escape.txt").exists(),
    ),
    (
        "no network (raw socket)",
        "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)",
        lambda r: r.returncode != 0 and ("PermissionError" in r.stderr or "OSError" in r.stderr),
    ),
    (
        "no network (DNS + HTTP)",
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3)",
        lambda r: r.returncode != 0 and "URLError" in r.stderr,
    ),
    ("cannot fork", "import os; os.fork()", lambda r: r.returncode != 0 and "Error" in r.stderr),
    (
        "cannot start other programs",
        "import subprocess; subprocess.run(['/bin/ls'])",
        lambda r: r.returncode != 0 and "Error" in r.stderr,
    ),
    (
        "infinite loop is killed",
        "while True: pass",
        lambda r: r.timed_out or r.returncode == -signal.SIGXCPU,
    ),
    (
        "output flood is capped",
        "while True: print('x' * 1000)",
        # Python ignores SIGXFSZ, so hitting the cap raises "File too large" instead of killing it.
        lambda r: (r.returncode == -signal.SIGXFSZ or "File too large" in r.stderr)
        and 0 < len(r.stdout) <= MAX_OUTPUT_BYTES,
    ),
    (
        "can write in its own temp dir",
        "open('scratch.txt', 'w').write('ok'); print(open('scratch.txt').read())",
        lambda r: r.stdout.strip() == "ok",
    ),
]


def check_escapes() -> int:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("NOTE  key not loaded in this process, so the environment check proves less")
    failures = 0
    for name, source, ok in ESCAPES:
        result = run_python(source, timeout_s=3)
        # If the sandbox itself failed to start, the code never ran, so a
        # "blocked" outcome proves nothing.
        passed = "sandbox-exec:" not in result.stderr and ok(result)
        failures += not passed
        detail = f"rc={result.returncode} timed_out={result.timed_out} {result.duration_s:.2f}s"
        print(f"{'PASS' if passed else 'FAIL'}  {name:36s} {detail}")
        if not passed:
            print(f"      stdout: {result.stdout[:200]!r}\n      stderr: {result.stderr[-300:]!r}")
    return failures


def check_mbppplus_references() -> int:
    """Every MBPP+ reference solution should pass its own tests inside the sandbox."""
    from datasets import load_dataset

    from harness.loaders import REVISIONS

    rows = load_dataset("evalplus/mbppplus", split="test", revision=REVISIONS["evalplus/mbppplus"])
    reference = {f"mbppplus:{row['task_id']}": row["code"] for row in rows}

    def run(item):
        tests = item["expected_answer"]
        program = "\n".join(tests["test_imports"]) + "\n" + reference[item["item_id"]] + "\n\n" + tests["test"]
        return item["item_id"], run_python(program, timeout_s=20)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, load_mbppplus()))

    failed = [(item_id, r) for item_id, r in results if r.returncode != 0]
    durations = [r.duration_s for _, r in results]
    print(
        f"MBPP+ references: {len(results) - len(failed)}/{len(results)} pass | "
        f"duration median {statistics.median(durations):.2f}s, max {max(durations):.2f}s"
    )
    for item_id, r in failed:
        print(f"  FAIL {item_id} rc={r.returncode} timed_out={r.timed_out} {r.stderr.strip().splitlines()[-1:]}")
    return len(failed)


if __name__ == "__main__":
    escape_failures = check_escapes()
    print()
    reference_failures = check_mbppplus_references() if "--escapes-only" not in sys.argv else 0
    sys.exit(1 if escape_failures or reference_failures else 0)
