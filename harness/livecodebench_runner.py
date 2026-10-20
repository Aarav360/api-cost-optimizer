"""Grade one LiveCodeBench solution against its tests. Runs inside the sandbox, never on the host.

harness/evaluators.py prepends the inputs as globals: KIND ("stdin" or
"functional"), FN_NAME, SOLUTION (the model's code), TESTS (base64 of the
zlib-compressed JSON list of {"input", "output"}) and TIMEOUT (seconds per
test). The script prints one JSON line: {"passed": n, "total": n, "failure": ...}.

The rules follow LiveCodeBench's lcb_runner/evaluation/testing_util.py
(github.com/LiveCodeBench/LiveCodeBench, MIT License): grading stops at the
first failing test; a stdin program passes a test when every output line,
stripped, matches the expected line exactly or as a list of Decimals; a
functional solution passes when its return value (a tuple counted as a list)
equals the expected JSON value; each test gets TIMEOUT seconds. Two deliberate
differences: a stdin program runs as a real process reading real standard input,
not in-process with a mocked stdin, and the Docker container is the security
boundary instead of LiveCodeBench's reliability_guard.
"""

import base64
import io
import json
import signal
import subprocess
import sys
import zlib
from decimal import Decimal, InvalidOperation

# LiveCodeBench's import_string, prepended to every solution, so code that uses
# a common module without importing it behaves as it does there.
IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
    "import json\nsys.setrecursionlimit(50000)\n"
)

REAL_STDOUT = sys.stdout


def report(passed, total, failure=None):
    REAL_STDOUT.write(json.dumps({"passed": passed, "total": total, "failure": failure}) + "\n")


def truncate(value, length=300):
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= length else text[: length // 2] + " ... " + text[-length // 2 :]


def stripped_lines(text):
    return [line.strip() for line in text.strip().split("\n")]


def as_decimals(line):
    try:
        return [Decimal(token) for token in line.split()]
    except (InvalidOperation, ValueError):
        return None


def output_matches(got, expected):
    got_lines, expected_lines = stripped_lines(got), stripped_lines(expected)
    if len(got_lines) != len(expected_lines):
        return False
    for got_line, expected_line in zip(got_lines, expected_lines):
        if got_line == expected_line:
            continue
        got_numbers, expected_numbers = as_decimals(got_line), as_decimals(expected_line)
        if got_numbers is None or expected_numbers is None or got_numbers != expected_numbers:
            return False
    return True


def run_stdin(tests):
    program = IMPORT_STRING + "\n" + SOLUTION  # noqa: F821
    for i, test in enumerate(tests):
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", program],
                input=test["input"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,  # noqa: F821
            )
        except subprocess.TimeoutExpired:
            return report(i, len(tests), {"test": i, "error": "Time Limit Exceeded"})
        # LiveCodeBench ignores SystemExit, so a nonzero exit counts as a crash
        # only when Python printed a traceback or a signal killed the process.
        crashed = result.returncode < 0 or (
            result.returncode != 0 and "Traceback (most recent call last)" in result.stderr
        )
        if crashed:
            last = result.stderr.strip().splitlines()[-1:] or [f"exit {result.returncode}"]
            return report(i, len(tests), {"test": i, "error": "Runtime Error", "detail": truncate(last[0])})
        if not output_matches(result.stdout, test["output"]):
            return report(
                i,
                len(tests),
                {
                    "test": i,
                    "error": "Wrong Answer",
                    "got": truncate(result.stdout),
                    "expected": truncate(test["output"]),
                },
            )
    report(len(tests), len(tests))


class TestTimeout(Exception):
    pass


def on_alarm(signum, frame):
    raise TestTimeout()


def run_functional(tests):
    signal.signal(signal.SIGALRM, on_alarm)
    sys.stdout = io.StringIO()  # the solution's own prints must not mix with the report
    namespace = {"__name__": "solution"}
    signal.alarm(TIMEOUT)  # noqa: F821
    try:
        exec(IMPORT_STRING + "\n" + SOLUTION, namespace)  # noqa: F821
        method = getattr(namespace["Solution"](), FN_NAME)  # noqa: F821
    except BaseException as e:
        return report(0, len(tests), {"test": None, "error": "Compile Error", "detail": truncate(repr(e))})
    finally:
        signal.alarm(0)
    for i, test in enumerate(tests):
        args = [json.loads(line) for line in test["input"].split("\n")]
        expected = json.loads(test["output"])
        signal.alarm(TIMEOUT)  # noqa: F821
        try:
            got = method(*args)
        except TestTimeout:
            return report(i, len(tests), {"test": i, "error": "Time Limit Exceeded"})
        except BaseException as e:
            return report(i, len(tests), {"test": i, "error": "Runtime Error", "detail": truncate(repr(e))})
        finally:
            signal.alarm(0)
        sys.stdout = io.StringIO()
        if isinstance(got, tuple):
            got = list(got)
        if got != expected:
            return report(
                i,
                len(tests),
                {"test": i, "error": "Wrong Answer", "got": truncate(got), "expected": truncate(expected)},
            )
    report(len(tests), len(tests))


_tests = json.loads(zlib.decompress(base64.b64decode(TESTS)))  # type: ignore[name-defined]  # noqa: F821
if KIND == "functional":  # type: ignore[name-defined]  # noqa: F821
    run_functional(_tests)
else:
    run_stdin(_tests)
