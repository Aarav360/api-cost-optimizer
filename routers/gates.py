"""The free checks (PLAN_v4.md §9; Amendment A17, items 3, 4 and 19): the stop check, the schema check and the code
check. They cost nothing per call and run first in every cascade.

This module is deployable: it imports the standard library and `routers.sandbox` only, and so keeps its own copy of
the grader's code extraction (`harness/evaluators.py`'s `extract_code`; a check proves the two agree).

- The stop check rejects any stop other than `end_turn`.
- The schema check rejects an answer that isn't JSON matching the request's output schema, and doesn't apply when the
  request has none. It can't fire on this benchmark and is kept and reported anyway. There is no JSON Schema library
  here, so `matches_schema` reads the few keywords the benchmark's schemas use and refuses (UnsupportedSchema) any
  other, since ignoring one would let a wrong answer pass.
- The code check applies to a request with a line starting `assert`. It runs the grader's extraction of the answer's
  code and then that line in the sandbox at 20 s and 512 MB for every request, with each standard-library module the
  line names imported first. It passes on a clean exit within the limits. No code block, or an empty one, rejects. A
  sandbox failure is retried and, if it keeps failing, raised (SandboxError); it never becomes a verdict.

`free_checks` runs all three and returns the stored record's fields (see `FREE_FIELDS`). `read_gate_settings` reads
`routers/gate_settings.json`, which holds the checks' limits, the sandbox image and the fifteen cascades' shapes.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routers.request import Request
from routers.sandbox import IMAGE, ExecResult, SandboxError, run_python

SETTINGS_FILE = Path(__file__).resolve().parent / "gate_settings.json"

STOP, SCHEMA, CODE = "stop", "schema", "code"  # the checks, in the order they are tried: `failed_first` names one
END_TURN = "end_turn"

# The fields of a free-check record after the keys every gate record has (`item_id`, `config`, `run`, `split`, `gate`,
# `pass`), in the order stored (A17 item 14; #59's "Stored records").
FREE_FIELDS = (
    "stop_passed",
    "schema_passed", "schema_applied",
    "code_passed", "code_applied", "code_returncode", "code_timed_out", "code_duration_s", "code_stderr_tail",
    "code_image_digest", "code_checked_at",
    "passed", "failed_first",
)  # fmt: skip


# ---- the settings ---------------------------------------------------------------------------------------------


class SettingsError(ValueError):
    """A gate settings file that is malformed. Nothing is read from it."""


@dataclass(frozen=True)
class CodeCheckSettings:
    timeout_s: float
    memory: str
    attempts: int  # tries of one sandbox run before a SandboxError is raised
    wait_s: float  # between tries
    image: str
    image_digest: str | None  # filled at the settings commit
    dockerfile_sha256: str | None  # filled at the settings commit


@dataclass(frozen=True)
class CascadeShape:
    key: str  # the section's key in the operating-points file
    label: str
    start: str  # `haiku`, or a router's section name
    gate: str  # `free`, `jev` or `sonnet`


@dataclass(frozen=True)
class GateSettings:
    code_check: CodeCheckSettings
    cascades: tuple[CascadeShape, ...]
    raw: dict[str, Any]  # the whole file, for the sections the paid gates read (`jev`, `sonnet`)


def read_gate_settings(path: Path = SETTINGS_FILE) -> GateSettings:
    """The code check's settings and the cascade shapes, or a refusal (SettingsError). The other sections are kept
    in `raw` for the paid gates, which check their own."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        code = raw["code_check"]
        cascades = tuple(CascadeShape(c["key"], c["label"], c["start"], c["gate"]) for c in raw["cascades"])
        settings = CodeCheckSettings(
            float(code["timeout_s"]), str(code["memory"]), int(code["attempts"]), float(code["wait_s"]),
            str(code["image"]), code["image_digest"], code["dockerfile_sha256"],
        )  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SettingsError(f"{path} is not a gate settings file: {error!r}") from error
    if settings.timeout_s <= 0 or settings.attempts < 1 or settings.wait_s < 0:
        raise SettingsError("the code check needs a positive time limit, at least one attempt and a wait of 0 or more")
    return GateSettings(settings, cascades, raw)


# ---- the stop and schema checks -------------------------------------------------------------------------------


def stop_check(stop_reason: str | None) -> bool:
    """Whether the answer ended its turn. Always right when it rejects: the grader fails every other stop."""
    return stop_reason == END_TURN


class UnsupportedSchema(ValueError):
    """An output schema with a keyword `matches_schema` doesn't read."""


_KEYWORDS = {"type", "properties", "enum", "required", "additionalProperties"}
_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
}


def matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Whether a parsed JSON value matches the schema's `type`, `enum`, `required`, `properties` and
    `additionalProperties: false`. Raises UnsupportedSchema for any other keyword."""
    if unread := sorted(set(schema) - _KEYWORDS):
        raise UnsupportedSchema(f"the schema check doesn't read the keywords {unread}")
    kind = schema.get("type")
    if kind is not None:
        if kind not in _TYPES:
            raise UnsupportedSchema(f"the schema check doesn't read the type {kind!r}")
        if not _TYPES[kind](value):
            return False
    if "enum" in schema and not any(value == option and type(value) is type(option) for option in schema["enum"]):
        return False
    if isinstance(value, dict):
        if any(name not in value for name in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(matches_schema(value[name], sub) for name, sub in properties.items() if name in value)
    return True


def schema_check(request: Request, answer: str) -> tuple[bool, bool]:
    """(passed, applied). It applies when the request has an output schema, and then passes an answer that is JSON
    matching it. Without a schema it passes and doesn't apply."""
    if request.output_schema is None:
        return True, False
    try:
        value = json.loads(answer)
    except ValueError:
        return False, True
    return matches_schema(value, request.output_schema), True


# ---- the code check -------------------------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?P<lang>[\w+-]*)[ \t]*\n(?P<code>.*?)```", re.DOTALL)
_ASSERT_LINE = re.compile(r"^assert\b.*$", re.MULTILINE)


def extract_code(text: str) -> str | None:
    """The last ```python block, else the last unlabeled ``` block, else None: the grader's rule, copied (the
    harness can't be imported here)."""
    blocks = [(m["lang"].lower(), m["code"]) for m in _CODE_BLOCK.finditer(text)]
    python = [code for lang, code in blocks if lang in ("python", "py", "python3")]
    plain = [code for lang, code in blocks if not lang]
    return (python or plain or [None])[-1]


def assert_line(prompt: str) -> str | None:
    """The request's first line starting `assert`, or None: the code check applies to a request that has one."""
    found = _ASSERT_LINE.search(prompt)
    return found.group(0) if found else None


def assert_modules(line: str) -> list[str]:
    """The standard-library modules an `assert` line names, in order of appearance: the names it uses that are
    standard-library module names (`math` in `math.isclose(...)`). The grader adds its hidden tests' imports, which a
    gate can't see, so a correct answer that never imports `math` would otherwise crash here."""
    try:
        names = [node.id for node in ast.walk(ast.parse(line)) if isinstance(node, ast.Name)]
    except SyntaxError:
        names = re.findall(r"[A-Za-z_]\w*", line)
    return list(dict.fromkeys(name for name in names if name in sys.stdlib_module_names))


def code_program(code: str, line: str) -> str:
    """The program the sandbox runs: the line's imports, the answer's code, then the line."""
    imports = "".join(f"import {name}\n" for name in assert_modules(line))
    return imports + code + "\n\n" + line


Runner = Callable[[str, float, str], ExecResult]


def sandbox_runner(source: str, timeout_s: float, memory: str) -> ExecResult:
    return run_python(source, timeout_s=timeout_s, memory=memory)


@dataclass(frozen=True)
class CodeVerdict:
    passed: bool
    applied: bool
    returncode: int | None = None
    timed_out: bool | None = None
    duration_s: float | None = None
    stderr_tail: str | None = None  # the last line of standard error


def code_check(
    request: Request,
    answer: str,
    settings: CodeCheckSettings,
    run: Runner = sandbox_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> CodeVerdict:
    """The code check: passes a request with no `assert` line unchanged (not applied), and otherwise runs the line
    against the answer's code. A SandboxError is retried `settings.attempts` times, `wait_s` apart, then raised."""
    line = assert_line(request.prompt)
    if line is None:
        return CodeVerdict(True, False)
    code = extract_code(answer)
    if code is None or not code.strip():
        return CodeVerdict(False, True)  # as the grader: no answer, so no run
    for attempt in range(1, settings.attempts + 1):
        try:
            result = run(code_program(code, line), settings.timeout_s, settings.memory)
        except SandboxError:
            if attempt == settings.attempts:
                raise
            sleep(settings.wait_s)
            continue
        lines = result.stderr.strip().splitlines()
        return CodeVerdict(
            result.returncode == 0 and not result.timed_out, True, result.returncode, result.timed_out,
            round(result.duration_s, 2), lines[-1] if lines else None,
        )  # fmt: skip
    raise AssertionError("unreachable: the loop returns or raises")


def image_digest(image: str = IMAGE) -> str:
    """The sandbox image's ID. A locally built image has no registry digest, so this is its identifier (`sha256:...`).
    `image ls`, not `image inspect <name>`, as the sandbox does. Raises SandboxError if Docker can't say."""
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxError("docker CLI not found; install or start Docker Desktop")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}
    shown = subprocess.run(
        [docker, "image", "ls", "--no-trunc", "--format", "{{.ID}}", image], capture_output=True, text=True, env=env
    )
    ids = shown.stdout.split()
    if shown.returncode != 0 or len(ids) != 1:
        raise SandboxError(f"can't read the ID of image {image}: {shown.stderr.strip() or shown.stdout.strip()!r}")
    return ids[0]


# ---- all three ------------------------------------------------------------------------------------------------


def free_checks(
    request: Request,
    answer: str,
    stop_reason: str | None,
    settings: CodeCheckSettings,
    *,
    digest: str,
    now: Callable[[], str],
    run: Runner = sandbox_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """All three free checks on one answer, as the fields of a free-check record (`FREE_FIELDS`). Every check runs,
    so each verdict is on record; `failed_first` names the first that failed, in the order stop, schema, code. Every
    record carries the sandbox image's digest and the time, whether or not the code check applied (A17 item 4)."""
    stop = stop_check(stop_reason)
    schema_passed, schema_applied = schema_check(request, answer)
    verdict = code_check(request, answer, settings, run, sleep)
    verdicts = ((STOP, stop), (SCHEMA, schema_passed), (CODE, verdict.passed))
    failed_first = next((name for name, ok in verdicts if not ok), None)
    return dict(zip(FREE_FIELDS, (
        stop,
        schema_passed, schema_applied,
        verdict.passed, verdict.applied, verdict.returncode, verdict.timed_out, verdict.duration_s,
        verdict.stderr_tail, digest, now(),
        failed_first is None, failed_first,
    ), strict=True))  # fmt: skip
