"""The routers on chart and table v1 and v2: the list the chart script takes (PLAN_v4.md §12; Amendment A11, item 16).

A `RouterEntry` holds what the chart script needs of one router and nothing of the router itself: its name in the
operating-points file, its label, its short name for legends and captions, its style, how to build its dev
router-score table, how to walk it, and any extra facts for its entry in the headline JSON. `ROUTERS` is the list, in
the order the figures draw and the legend lists them; E's entry is built here, and F's joins the list when Stage 5's
router has a dev table and a section in the operating-points file. Steps, operating points, McNemar rows and the
matched-cost comparison are drawn and computed per entry, in `analysis/pareto.py`.

Each router has its own style, so that two read apart on one panel without relying on colour alone: a colour from the
data-viz reference palette's family, a marker for its steps and another for its operating points' centres, and a
line. E's is chart v1's, unchanged. F's is reserved: red, triangles and a dashed line, against E's purple, squares,
diamonds and a solid line; the figures' other colours are the configs' blue, the envelope's orange and the oracle's
green.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from analysis.bootstrap import Bootstrap
from analysis.dev_diagnostic import dev_diagnostic
from analysis.matrix import Matrix
from analysis.router_table import RouterScoreTable
from analysis.rules_table import dev_table
from analysis.rules_walk import ROUTER as RULES_NAME
from analysis.rules_walk import dev_walks
from analysis.walk import Walk

# A label's place beside an operating point's cross: points right and up from the cross, and the alignment.
Offset = tuple[int, int, str]


@dataclass(frozen=True)
class RouterStyle:
    color: str  # hex, for the line, the markers and the crosses
    marker: str  # matplotlib marker of the steps
    point_marker: str  # matplotlib marker of the operating points' centres
    linestyle: str  # matplotlib line style of the steps' line
    # Operating-point labels by (mix, label), placed by eye on the committed figure and checked against it, so a
    # different walk needs them moved; a label not listed sits just below right of its cross.
    point_offsets: dict[tuple[str, str], Offset] = field(default_factory=dict)


E_STYLE = RouterStyle(
    color="#8e5bd0", marker="s", point_marker="D", linestyle="-",
    point_offsets={
        ("realistic", "97%"): (-14, 22, "right"),
        ("realistic", "99%"): (-4, 56, "right"),
        ("realistic", "100%: falls back to A"): (24, 80, "right"),
        ("uniform", "97%"): (8, -16, "left"),
        ("uniform", "99%"): (-6, 44, "right"),
        ("uniform", "100%: falls back to A"): (24, 66, "right"),
    },
)  # fmt: skip

# F's, reserved: no entry draws it yet. Its label offsets are placed once its operating points exist.
F_STYLE = RouterStyle(color="#e34948", marker="^", point_marker="v", linestyle="--")


@dataclass(frozen=True)
class RouterEntry:
    name: str  # the router's name in the operating-points file, and its key under "routers" in the figure JSONs
    label: str  # the router's name on the chart and in its JSON entry: "E: rules router"
    short: str  # "E": the router in a legend entry or a caption
    style: RouterStyle
    # Its dev router-score table, which has passed the table's refusal, and its walks over that table on dev under
    # both task mixes for the three quality targets (`analysis/rules_walk.py`'s `dev_walks` for a table of any router).
    dev_table: Callable[[], RouterScoreTable]
    walks: Callable[[Matrix, RouterScoreTable, Bootstrap], dict[str, dict[float, Walk]]]
    # Facts about the router for its entry under "routers" in the headline JSON, between its label and its section's
    # file and hash, as strings that the script prints too.
    facts: Callable[[], dict[str, str]] = dict


RULES = RouterEntry(
    name=RULES_NAME,
    label="E: rules router",
    short="E",
    style=E_STYLE,
    dev_table=dev_table,
    walks=dev_walks,
    facts=lambda: {"task_inference_dev": dev_diagnostic()},
)

ROUTERS: tuple[RouterEntry, ...] = (RULES,)
