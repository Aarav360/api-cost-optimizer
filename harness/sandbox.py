"""Run untrusted Python (model-written code plus its tests) in a throwaway Docker container.

Each call starts a fresh container from a pinned image (sandbox/Dockerfile), so
nothing one run does can affect the next. What each container gets:

1. No network, a read-only root file system, and a small writable tmpfs at
   /work that disappears with the container.
2. None of your folders. The program arrives on standard input, so no host
   path is mounted, and the container's environment never includes
   ANTHROPIC_API_KEY (client.py loads it into this process's environment).
3. An unprivileged user with every capability dropped and no-new-privileges set.
4. Limits: 512 MB of memory by default (callers may raise it) with no swap, a cap on how many processes it can
   start, one CPU, a CPU-time limit, and a wall-clock timeout that
   force-removes the container. The host stops reading output at
   MAX_OUTPUT_BYTES per stream and removes the container if it keeps going.

This module only reports what happened. Deciding what counts as a pass is the
evaluator's job.

Build the image once with:  uv run python -m harness.sandbox --build
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "api-cost-optimizer-sandbox:py3.13.13-numpy2.5.4"
DOCKERFILE_DIR = Path(__file__).resolve().parent.parent / "sandbox"

MAX_OUTPUT_BYTES = 1_000_000  # per stream
MEMORY = "512m"
PIDS_LIMIT = 64


class SandboxError(RuntimeError):
    """Docker itself failed, so the program never ran. Never score this as a model failure."""


@dataclass
class ExecResult:
    returncode: int | None  # None if killed on timeout; negative if killed by a signal
    timed_out: bool
    stdout: str
    stderr: str
    duration_s: float


def _docker() -> str:
    path = shutil.which("docker")
    if path is None:
        raise SandboxError("docker CLI not found; install or start Docker Desktop")
    return path


def _docker_env() -> dict[str, str]:
    # The docker CLI needs HOME and PATH to find its config and context. Strip
    # Anthropic variables anyway, so a stray `-e NAME` could never forward them.
    return {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}


_image_checked = False


def _ensure_image() -> None:
    global _image_checked
    if _image_checked:
        return
    # `image ls`, not `image inspect <name>`: on Docker Desktop 29.5.3, inspect by
    # name sometimes reports "No such image" for an image that is there and runs.
    check = subprocess.run(
        [_docker(), "image", "ls", "--quiet", IMAGE], capture_output=True, text=True, env=_docker_env()
    )
    if check.returncode != 0 or not check.stdout.strip():
        raise SandboxError(
            f"Sandbox image {IMAGE} unavailable ({check.stderr.strip() or 'not built'}). "
            "Start Docker Desktop, then build it with: uv run python -m harness.sandbox --build"
        )
    _image_checked = True


def _command(name: str, timeout_s: float, memory: str) -> list[str]:
    cpu_seconds = int(timeout_s) + 1
    return [
        _docker(), "run", "--rm", "-i",
        "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/work:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--workdir", "/work",
        "--user", "10001:10001",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--memory", memory, "--memory-swap", memory,
        "--pids-limit", str(PIDS_LIMIT),
        "--cpus", "1",
        "--ulimit", f"cpu={cpu_seconds}:{cpu_seconds}",
        "--log-driver", "none",
        "-e", "HOME=/work", "-e", "TMPDIR=/work", "-e", "LANG=C.UTF-8",
        IMAGE,
        # -I: ignore PYTHON* variables and the user site dir. -B: no .pyc writes.
        # "-" reads the program from standard input.
        "python", "-I", "-B", "-",
    ]  # fmt: skip


def _force_remove(name: str) -> None:
    subprocess.run([_docker(), "rm", "-f", name], capture_output=True, env=_docker_env())


def _exit_status(code: int) -> int:
    # `docker run` passes the container's exit code through, and a process
    # killed by signal N exits as 128 + N. Report that as -N, like subprocess does.
    return -(code - 128) if 128 < code <= 128 + 64 else code


def _read_capped(stream, sink: bytearray, on_overflow) -> None:
    for chunk in iter(lambda: stream.read(65536), b""):
        room = MAX_OUTPUT_BYTES - len(sink)
        sink += chunk[:room]
        if len(chunk) > room:
            on_overflow()
            # Keep draining so the docker CLI never blocks on a full pipe.


def run_python(source: str, timeout_s: float = 10.0, memory: str = MEMORY) -> ExecResult:
    _ensure_image()
    name = f"sandbox-{uuid.uuid4().hex[:12]}"
    stdout, stderr = bytearray(), bytearray()
    killed = threading.Event()
    removers = []

    def kill_once():
        # Remove on a separate thread: Docker won't finish removing the
        # container until its pending output is read, so the reader that hit
        # the cap must keep draining instead of waiting here.
        if not killed.is_set():
            killed.set()
            remover = threading.Thread(target=_force_remove, args=(name,))
            remover.start()
            removers.append(remover)

    start = time.perf_counter()
    process = subprocess.Popen(
        _command(name, timeout_s, memory),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_docker_env(),
    )

    def feed():
        try:
            process.stdin.write(source.encode("utf-8"))
            process.stdin.close()
        except BrokenPipeError:
            pass  # container already gone; the exit status says why

    threads = [
        threading.Thread(target=feed),
        threading.Thread(target=_read_capped, args=(process.stdout, stdout, kill_once)),
        threading.Thread(target=_read_capped, args=(process.stderr, stderr, kill_once)),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_once()
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()  # the CLI should exit once its container is gone; don't hang if not
            code = process.wait()
    for thread in threads + removers:
        thread.join()
    duration_s = time.perf_counter() - start

    err_text = stderr.decode("utf-8", errors="replace")
    if code == 125 and not killed.is_set():
        # 125 means `docker run` itself failed (daemon down, bad flag, missing image).
        raise SandboxError(f"docker run failed: {err_text.strip()}")

    return ExecResult(
        returncode=None if timed_out else _exit_status(code),
        timed_out=timed_out,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=err_text,
        duration_s=duration_s,
    )


def build_image() -> None:
    subprocess.run([_docker(), "build", "-t", IMAGE, str(DOCKERFILE_DIR)], check=True, env=_docker_env())


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_image()
    else:
        print(__doc__)
