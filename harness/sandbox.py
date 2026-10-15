"""Run untrusted Python (model-written code plus its tests) in a locked-down subprocess.

Layers, each covering a different failure:

1. macOS Seatbelt profile via `sandbox-exec`: no network, no writes outside a
   fresh temp dir, no reads of anything under your home folder except the
   Python install and venv, and no forking.
2. Clean environment: the child gets no inherited variables, so it never sees
   ANTHROPIC_API_KEY (client.py loads it into this process's environment).
3. Resource limits: a CPU-time cap, a wall-clock timeout that kills the whole
   process group, and a file-size cap that also bounds captured output.

Not covered: memory. macOS doesn't enforce address-space limits, so a runaway
allocation is only stopped by the timeout. Docker (with --memory) would cover
that if it ever matters.

This module only reports what happened. Deciding what counts as a pass is the
evaluator's job.
"""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 1_000_000  # per stream; also the file-size cap inside the sandbox

# Directories the child may read inside the home folder. sys.prefix is the
# project venv (where numpy lives); sys.base_prefix is the Python install.
_PYTHON_DIRS = sorted({os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})


@dataclass
class ExecResult:
    returncode: int | None  # None if killed on timeout; negative if killed by a signal
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float


def _symlink_hops(path: str) -> list[str]:
    """Every symlink passed through while resolving `path`.

    uv's venv python links through an intermediate folder link
    (cpython-3.13-... -> cpython-3.13.13-...), and the sandbox must be allowed
    to follow each hop.
    """
    hops = []
    pending = os.path.abspath(path)
    for _ in range(40):  # guard against link loops
        parts = Path(pending).parts
        for i in range(2, len(parts) + 1):
            prefix = str(Path(*parts[:i]))
            if os.path.islink(prefix):
                hops.append(prefix)
                target = os.path.join(os.path.dirname(prefix), os.readlink(prefix))
                pending = os.path.normpath(os.path.join(target, *parts[i:]))
                break
        else:
            return hops
    raise RuntimeError(f"Too many symlink hops resolving {path}")


_EXECUTABLE_HOPS = _symlink_hops(sys.executable)


def _ancestors(path: str) -> list[str]:
    parents = []
    p = Path(path)
    while p != p.parent:
        p = p.parent
        parents.append(str(p))
    return parents


def _profile(workdir: str) -> str:
    home = os.path.realpath(Path.home())
    allowed_reads = "".join(f'(allow file-read* (subpath "{d}"))' for d in _PYTHON_DIRS)
    # Resolving a path needs a metadata lookup on every folder above it, so the
    # folders above each allowed dir get metadata access only (no listing, no contents).
    lookups = sorted({a for d in [*_PYTHON_DIRS, *_EXECUTABLE_HOPS, workdir] for a in _ancestors(d)})
    allowed_lookups = "".join(f'(allow file-read-metadata (literal "{a}"))' for a in lookups)
    allowed_lookups += "".join(f'(allow file-read* (literal "{hop}"))' for hop in _EXECUTABLE_HOPS)
    # Later rules override earlier ones, so each deny comes before its exceptions.
    return (
        "(version 1)"
        "(allow default)"
        "(deny network*)"
        "(deny process-fork)"
        "(deny file-write*)"
        f'(allow file-write* (subpath "{workdir}"))'
        '(allow file-write* (literal "/dev/null"))'
        f'(deny file-read* (subpath "{home}"))'
        f"{allowed_reads}"
        f"{allowed_lookups}"
        f'(allow file-read* (subpath "{workdir}"))'
    )


def _limit_resources(cpu_seconds: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))

    return apply


def run_python(source: str, timeout_s: float = 10.0) -> ExecResult:
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="sandbox-"))
    try:
        program = Path(workdir) / "program.py"
        program.write_text(source)
        stdout_path = Path(workdir) / "stdout.txt"
        stderr_path = Path(workdir) / "stderr.txt"

        # -I: ignore PYTHON* variables and the user site dir. -B: no .pyc writes.
        command = ["/usr/bin/sandbox-exec", "-p", _profile(workdir), sys.executable, "-I", "-B", str(program)]
        env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "LANG": "en_US.UTF-8", "TMPDIR": workdir}

        start = time.perf_counter()
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,  # own process group, so a timeout kills everything it started
                preexec_fn=_limit_resources(int(timeout_s) + 1),
            )
            try:
                returncode = process.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = None
                timed_out = True
        duration_s = time.perf_counter() - start

        return ExecResult(
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout_path.read_bytes()[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            stderr=stderr_path.read_bytes()[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            duration_s=duration_s,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
