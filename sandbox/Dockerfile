# Image for harness/sandbox.py. Build with:  uv run python -m harness.sandbox --build
#
# Base pinned by digest: python:3.13.13-slim (Debian 13 trixie), pulled 2026-10-15.
# Python matches the project venv, so code behaves the same here as in analysis.
FROM python:3.13.13-slim@sha256:aa938a849bcb82dce8f49480f056ab82bf5c1c3ebc294f0430f37b6820e7f286

COPY requirements.txt /tmp/requirements.txt
RUN PIP_DISABLE_PIP_VERSION_CHECK=1 pip install --no-cache-dir --only-binary=:all: \
        --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Unprivileged user with no home folder; /work is a tmpfs mounted at run time.
RUN useradd --uid 10001 --no-create-home --home-dir /work --shell /usr/sbin/nologin sandbox
USER 10001:10001
WORKDIR /work
