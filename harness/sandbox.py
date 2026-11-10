"""The grader's import path for the sandbox, which now lives in routers/sandbox.py (Amendment A17 item 19).

Re-exports everything the grader and its checks use, so harness/evaluators.py is untouched.

Build the image once with:  uv run python -m harness.sandbox --build
"""

import sys

from routers.sandbox import (
    DOCKERFILE_DIR,
    IMAGE,
    MAX_OUTPUT_BYTES,
    MEMORY,
    PIDS_LIMIT,
    ExecResult,
    SandboxError,
    build_image,
    run_python,
)

__all__ = [
    "DOCKERFILE_DIR",
    "IMAGE",
    "MAX_OUTPUT_BYTES",
    "MEMORY",
    "PIDS_LIMIT",
    "ExecResult",
    "SandboxError",
    "build_image",
    "run_python",
]

if __name__ == "__main__":
    if "--build" in sys.argv:
        build_image()
    else:
        print(__doc__)
