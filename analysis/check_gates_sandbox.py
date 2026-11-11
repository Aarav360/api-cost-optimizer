"""Check the code check in the real sandbox (PLAN_v4.md §9; Amendment A17, item 4). Needs Docker: hook only.

Run from the project root:  uv run python -m analysis.check_gates_sandbox

Made-up code and a made-up request, never a benchmark answer. Each case is a verdict the code check must give through
the grader's sandbox at 20 s and 512 MB: code that passes, fails its `assert`, crashes, never ends (the 20 s limit, so
this check takes about half a minute), calls `math.isclose` without importing `math` (the import rule supplies it), is
an empty block, or has no block. The image digest it records is the image Docker reports. If Docker is down the check
fails: it is not skipped.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from analysis import collect_gates
from routers import gates
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def block(source: str) -> str:
    return f"Here it is.\n\n```python\n{source}\n```"


PROMPT = "Write f that doubles its argument.\n\nassert f(2)==4\n\nTask: double."
CLOSE_PROMPT = "Write f that halves its argument.\n\nassert math.isclose(f(3), 1.5)\n\nTask: halve."
SETTINGS = gates.read_gate_settings().code_check

CASES = [
    # (what, prompt, answer, whether the code check passes it)
    ("code that passes", PROMPT, block("def f(x):\n    return x * 2"), True),
    ("code that fails the assert", PROMPT, block("def f(x):\n    return x + 3"), False),
    ("code that crashes", PROMPT, block("def f(x):\n    raise RuntimeError('boom')"), False),
    ("code that never ends", PROMPT, block("def f(x):\n    while True:\n        pass"), False),
    ("a correct answer that calls math.isclose without importing math", CLOSE_PROMPT,
     block("def f(x):\n    return x / 2"), True),
    ("an empty block", PROMPT, "```python\n\n```", False),
    ("no code block", PROMPT, "The answer is f(x) = 2x.", False),
]  # fmt: skip


def main() -> int:
    print("-- the code check in the real sandbox --")
    for what, prompt, answer, passes in CASES:
        verdict = gates.code_check(Request(prompt), answer, SETTINGS)
        check(verdict.passed is passes and verdict.applied, f"{what}: {'passes' if passes else 'rejects'}")
        if what == "code that never ends":
            check(verdict.timed_out is True and verdict.returncode is None and 20 <= (verdict.duration_s or 0) < 40,
                  "the loop was stopped at the 20 s limit")  # fmt: skip
        if what == "code that crashes":
            check(verdict.returncode == 1 and verdict.stderr_tail == "RuntimeError: boom",
                  "the crash's return code and the last line of stderr are recorded")  # fmt: skip
        if what == "an empty block" or what == "no code block":
            check(verdict.returncode is None, "and the sandbox was not run")

    print("\n-- the image digest --")
    shown = subprocess.run(["docker", "image", "ls", "--no-trunc", "--format", "{{.ID}}", SETTINGS.image],
                           capture_output=True, text=True).stdout.split()  # fmt: skip
    digest = gates.image_digest(SETTINGS.image)
    check(digest.startswith("sha256:") and len(digest) == 71 and shown == [digest],
          "the digest is the 64-hex ID Docker reports for the sandbox image")  # fmt: skip

    print("\n-- end to end: made-up answers in, stored records out --")
    inputs = [collect_gates.GateInput("s1", "haiku", Request(PROMPT), block("def f(x):\n    return x * 2"), "end_turn"),
              collect_gates.GateInput("s2", "haiku", Request(PROMPT), block("def f(x):\n    return 0"), "end_turn")]  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        collect_gates.collect_free(inputs, split="smoke", settings=SETTINGS, digest=digest, results_dir=Path(tmp))
        records = collect_gates.read_free_records("smoke", Path(tmp))
    check([(r["passed"], r["failed_first"], r["code_image_digest"]) for r in records]
          == [(True, None, digest), (False, "code", digest)],
          "one passes and one fails its assert, each record naming the image digest")  # fmt: skip

    if problems:
        print(f"\n{len(problems)} checks failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
