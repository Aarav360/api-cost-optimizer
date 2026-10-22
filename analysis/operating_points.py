"""The operating-points file (PLAN_v4.md §8; Amendment A9, items 10 and 13).

One JSON file shared by every router. For each router, then task mix, then quality target, it holds the operating
point's threshold (null for a fallback to A), its margin, the quality target on dev, the selection method, the walk's
steps and whether it fell back. Stage 8 reads thresholds only from it.

A router's section is built from its walks (`build_section`), written with `write_section`, which leaves every other
router's section as it was, and read back with `read_section`. `section_hash` is the sha256 of a section's JSON text
with sorted keys, which is what the record hashes when Stage 4 finishes (E's section, item 18) so that any change,
a threshold moved in the fourth decimal included, is caught.

`check_sections` is the allowlist (Amendment A11, item 13): the file holds sections only for the routers an amendment
names, never F-train or a diagnostic, so Stage 8 can't pick up a threshold that was never a policy's.
"""

import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from analysis.bootstrap import RESAMPLES, SEED
from analysis.references import TARGETS
from analysis.walk import ALL_STEPS_UP_TO, QUANTILE_STEPS, Step, Walk
from harness.client import PROJECT_ROOT

# Data, not a module: routers/ holds Stage 5-7's files too, but the freeze (item 13) hashes modules, not this file.
OPERATING_POINTS_FILE = PROJECT_ROOT / "routers" / "operating_points.json"

# The routers an amendment names, by letter, and the name each has as a section of the file: E is the rules router
# (A9), F the trained router (A11). A router joins when an amendment names it (G after Stage 6). F-train and the
# diagnostics (A11, item 12) never do.
NAMED_ROUTERS = {"E": "rules", "F": "trained"}

SELECTION_METHOD = (
    f"Fixed-sequence walk from always-frontier (A) down the steps, from the highest threshold to the lowest "
    f"(Amendment A9, item 9). Steps, from the router-score table alone: every distinct breakpoint if there are "
    f"{ALL_STEPS_UP_TO} or fewer; otherwise the 0%, 5%, ..., 95% quantiles ({QUANTILE_STEPS} of them) of the pooled "
    f"breakpoints, each item's counted once per breakpoint, by the lower empirical quantile (numpy's inverted_cdf: the "
    f"smallest pooled breakpoint with at least that share of them at or below it), with duplicate steps removed. "
    f"Margin: one paired standard error of the step's quality difference from A under the mix, from {RESAMPLES} "
    f"paired bootstrap resamples with seed {SEED} (Amendment A8, items 8 and 9). A step clears if its dev quality "
    f"less the quality target is at least its margin; the walk stops at the first step that does not clear, and every "
    f"step before it is certified. Operating point: the cheapest certified step by cost per 1k under the mix, ties "
    f"to the earlier step, or A itself (threshold null, falls back to A) where no certified step costs less."
)


class OperatingPointsError(ValueError):
    """A section or file the operating-points module refuses. Nothing is written."""


def target_key(share: float) -> str:
    """A quality target's key in the file, as in the figure JSONs: "100%", "99%", "97%"."""
    return f"{share:.0%}"


def _step(step: Step) -> dict[str, Any]:
    return {"threshold": step.threshold, "quality": step.quality, "cost_per_1k": step.cost_per_1k,
            "margin": step.margin, "clears": step.clears}  # fmt: skip


def build_section(walks: Mapping[str, Mapping[float, Walk]]) -> dict[str, Any]:
    """One router's section from its walks, keyed by task mix name and then quality target share.

    Refuses (OperatingPointsError) a section with no mix, or a mix that lacks one of §11's targets: a router has
    one operating point per mix and target (Amendment A9, item 10).
    """
    if not walks:
        raise OperatingPointsError("a router's section needs at least one task mix")
    section: dict[str, Any] = {}
    for mix, by_share in walks.items():
        if set(by_share) != set(TARGETS):
            raise OperatingPointsError(f"mix {mix!r} has walks for targets {sorted(by_share)}, not {sorted(TARGETS)}")
        section[mix] = {}
        for share, result in by_share.items():
            operating = result.operating
            section[mix][target_key(share)] = {
                "threshold": operating.threshold,
                "margin": operating.margin,
                "target": result.target,
                "selection_method": SELECTION_METHOD,
                "steps": [_step(step) for step in result.steps],
                "fell_back": result.fell_back,
            }
    return section


def _text(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2)


def _read_file(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise OperatingPointsError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise OperatingPointsError(f"{path} must hold an object with one section per router")
    return loaded


def write_section(router: str, section: dict[str, Any], path: Path = OPERATING_POINTS_FILE) -> None:
    """Writes `router`'s section into the file, creating it if need be. Every other router's section is kept as
    it was, and the file is written with sorted keys, so the same sections always give the same text."""
    sections = _read_file(path) if path.exists() else {}
    sections[router] = json.loads(_text(section))  # what a reader will see: no tuple or key type JSON would change
    path.write_text(_text(sections) + "\n")


def read_section(router: str, path: Path = OPERATING_POINTS_FILE) -> dict[str, Any]:
    """`router`'s section as written. Refuses (OperatingPointsError) a router with no section in the file."""
    if router not in (sections := _read_file(path)):
        raise OperatingPointsError(f"{path} has no section for {router!r}; it holds {sorted(sections)}")
    section: dict[str, Any] = sections[router]
    return section


def section_hash(section: dict[str, Any]) -> str:
    """The sha256 of a section's JSON text with sorted keys and a two-space indent, as the file writes it, so the
    hash of a section read back equals the hash of the one written, whatever order its keys were built in."""
    return hashlib.sha256(_text(section).encode()).hexdigest()


def check_sections(path: Path = OPERATING_POINTS_FILE, required: Collection[str] = ()) -> None:
    """Refuses (OperatingPointsError) a file with a section for anything but the routers an amendment names
    (`NAMED_ROUTERS`), with F-train and every diagnostic among them, or a file that isn't there or isn't an object.

    `required` names sections that must also be present, each itself a named router: Stage 5 asks for F's once F is
    walked (`required=(NAMED_ROUTERS["E"], NAMED_ROUTERS["F"])`). By default no section is required, so a file with
    E's alone passes.
    """
    if not path.exists():
        raise OperatingPointsError(f"there is no file at {path}")
    allowed = set(NAMED_ROUTERS.values())
    if unnamed := sorted(set(required) - allowed):
        raise OperatingPointsError(
            f"required sections {unnamed} are not routers an amendment names ({sorted(allowed)})"
        )
    sections = _read_file(path)
    if extra := sorted(set(sections) - allowed):
        raise OperatingPointsError(f"{path} has sections {extra}, which are not routers an amendment names "
                                   f"({sorted(allowed)}); F-train and the diagnostics never get one")  # fmt: skip
    if missing := sorted(set(required) - set(sections)):
        raise OperatingPointsError(f"{path} has no section for {missing}; it holds {sorted(sections)}")
