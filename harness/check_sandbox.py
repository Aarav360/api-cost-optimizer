"""Check the sandbox: escape attempts must fail, and MBPP+ reference solutions must pass.

Run from the project root:  uv run python -m harness.check_sandbox

Re-run after Docker Desktop upgrades or after rebuilding the sandbox image.
"""

import os
import signal
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harness.client  # noqa: F401  (loads .env, so the key is in this process's environment)
from harness.client import PROJECT_ROOT
from harness.evaluators import CODE_MEMORY, CODE_TIMEOUT_S, mbppplus_program
from harness.loaders import EXCLUDED, load_mbppplus
from harness.sandbox import MAX_OUTPUT_BYTES, MEMORY, PIDS_LIMIT, SandboxError, run_python

OUTSIDE_FILE = Path("/tmp") / "sandbox-escape-test.txt"

FORK_BOMB = """
import os, time
started = 0
try:
    while True:
        if os.fork() == 0:
            time.sleep(30)
            os._exit(0)
        started += 1
except OSError as e:
    print("blocked after", started, type(e).__name__)
"""

CHILD_PROCESS = """
import subprocess
child = subprocess.run(
    ["python", "-c", "import os, socket; print(os.getuid(), os.path.exists('/Users')); socket.create_connection(('1.1.1.1', 443), timeout=3)"],
    capture_output=True, text=True,
)
print(child.stdout.strip(), child.returncode != 0 and "unreachable" in child.stderr)
"""

# (name, source, timeout_s, check on the result)
ESCAPES = [
    ("runs plain code", "print('hello')", 3, lambda r: r.returncode == 0 and r.stdout.strip() == "hello"),
    ("numpy is importable", "import numpy; print(numpy.__name__)", 3, lambda r: r.stdout.strip() == "numpy"),
    (
        "API key not in environment",
        "import os; print('ANTHROPIC_API_KEY' in os.environ, len(os.environ))",
        3,
        lambda r: r.stdout.startswith("False"),
    ),
    # Your files aren't blocked, they're absent: nothing from the host is mounted.
    (
        "cannot read project .env",
        f"open({str(PROJECT_ROOT / '.env')!r}).read()",
        3,
        lambda r: "FileNotFoundError" in r.stderr,
    ),
    (
        "cannot read other files in home",
        f"open({str(PROJECT_ROOT / 'PLAN_v4.md')!r}).read()",
        3,
        lambda r: "FileNotFoundError" in r.stderr,
    ),
    (
        "cannot list home folder",
        f"import os; os.listdir({str(Path.home())!r})",
        3,
        lambda r: "FileNotFoundError" in r.stderr,
    ),
    (
        "no host folders mounted",
        "import os; mounts = open('/proc/mounts').read(); "
        "print(os.path.exists('/Users'), os.path.exists('/host_mnt'), '/Users' in mounts or 'host_mnt' in mounts)",
        3,
        lambda r: r.stdout.strip() == "False False False",
    ),
    (
        "cannot write outside its work dir",
        f"open({str(OUTSIDE_FILE)!r}, 'w').write('x')",
        3,
        lambda r: "Read-only file system" in r.stderr and not OUTSIDE_FILE.exists(),
    ),
    (
        "cannot write into the project",
        f"open({str(PROJECT_ROOT / 'escape.txt')!r}, 'w').write('x')",
        3,
        lambda r: r.returncode != 0 and not (PROJECT_ROOT / "escape.txt").exists(),
    ),
    (
        "no network (raw socket)",
        "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)",
        5,
        lambda r: r.returncode != 0 and "Network is unreachable" in r.stderr,
    ),
    (
        "no network (DNS + HTTP)",
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3)",
        5,
        lambda r: r.returncode != 0 and "URLError" in r.stderr,
    ),
    (
        "runs unprivileged",
        "import os; s = open('/proc/self/status').read(); "
        "print(os.getuid(), 'CapEff:\\t0000000000000000' in s, 'NoNewPrivs:\\t1' in s)",
        3,
        lambda r: r.stdout.strip() == "10001 True True",
    ),
    # Forking is allowed (the container confines children too), so instead of
    # "cannot fork", check that a child process is exactly as confined.
    (
        "child processes stay confined",
        CHILD_PROCESS,
        8,
        lambda r: r.stdout.strip() == "10001 False True",
    ),
    (
        "fork bomb hits the process limit",
        FORK_BOMB,
        10,
        lambda r: r.stdout.startswith("blocked after") and int(r.stdout.split()[2]) < PIDS_LIMIT,
    ),
    (
        "memory bomb is killed",
        "chunks = []\nwhile True: chunks.append(b'x' * 50_000_000)",
        15,
        lambda r: not r.timed_out and (r.returncode == -signal.SIGKILL or "MemoryError" in r.stderr),
    ),
    (
        "infinite loop is killed",
        "while True: pass",
        3,
        lambda r: r.timed_out or r.returncode == -signal.SIGXCPU,
    ),
    (
        "output flood is capped",
        "while True: print('x' * 1000)",
        10,
        # The host stops at the cap and removes the container.
        lambda r: r.returncode != 0 and not r.timed_out and len(r.stdout) == MAX_OUTPUT_BYTES,
    ),
    (
        "can write in its own work dir",
        "open('scratch.txt', 'w').write('ok'); print(open('scratch.txt').read())",
        3,
        lambda r: r.stdout.strip() == "ok",
    ),
]


def check_escapes() -> int:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("NOTE  key not loaded in this process, so the environment check proves less")
    failures = 0
    for name, source, timeout_s, ok in ESCAPES:
        # If Docker itself fails, the code never ran, so a "blocked" outcome
        # would prove nothing; run_python raises instead of returning.
        try:
            result = run_python(source, timeout_s=timeout_s)
        except SandboxError as e:
            failures += 1
            print(f"FAIL  {name:36s} sandbox error: {e}")
            continue
        passed = ok(result)
        failures += not passed
        detail = f"rc={result.returncode} timed_out={result.timed_out} {result.duration_s:.2f}s"
        print(f"{'PASS' if passed else 'FAIL'}  {name:36s} {detail}")
        if not passed:
            print(f"      stdout: {result.stdout[:200]!r}\n      stderr: {result.stderr[-300:]!r}")
    return failures


def check_mbppplus_references() -> int:
    """Every MBPP+ reference solution should pass its own tests under the grader's limits.

    The grader's time and memory limits are used, including the 3 GB it gives
    mbppplus:255 (evaluators.CODE_MEMORY), so a pass here means code_tests would
    score the reference correct. Items in loaders.EXCLUDED are skipped.
    """
    from datasets import load_dataset

    from harness.loaders import REVISIONS

    rows = load_dataset("evalplus/mbppplus", split="test", revision=REVISIONS["evalplus/mbppplus"])
    reference = {f"mbppplus:{row['task_id']}": row["code"] for row in rows}

    def run(item):
        program = mbppplus_program(reference[item["item_id"]], item["expected_answer"])
        memory = CODE_MEMORY.get(item["item_id"], MEMORY)
        return item["item_id"], run_python(program, timeout_s=CODE_TIMEOUT_S, memory=memory)

    items = load_mbppplus()
    kept = [item for item in items if item["item_id"] not in EXCLUDED]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, kept))

    # The grader's pass rule (evaluators.code_tests).
    failed = [(item_id, r) for item_id, r in results if r.returncode != 0 or r.timed_out]
    durations = [r.duration_s for _, r in results]
    print(
        f"MBPP+ references: {len(results) - len(failed)}/{len(results)} pass, {len(items) - len(kept)} excluded | "
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
