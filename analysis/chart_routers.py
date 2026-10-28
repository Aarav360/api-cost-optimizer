"""The routers on chart and table v1, v2 and v3: the list the chart script takes (PLAN_v4.md §12; Amendment A11, item 16;
Amendment A15, item 15).

A `RouterEntry` holds what the chart script needs of one router and nothing of the router itself: its name in the
operating-points file, its label, its short name for legends and captions, its style, how to build its dev
router-score table, how to walk it, and any extra facts for its entry in the headline JSON. `ROUTERS` is the list, in
the order the figures draw and the legend lists them; E's, F's, G's and G+'s entries are built here. Steps, operating points, McNemar rows and the
matched-cost comparison are drawn and computed per entry, in `analysis/pareto.py`.

Each router has its own style, so that two read apart on one panel without relying on colour alone: a colour, a marker
for its steps and another for its operating points' centres, and a line. E's is chart v1's, unchanged. F's is maroon,
triangles and a dashed line, against E's purple, squares, diamonds and a solid line; the figures' other colours are the
configs' blue, the envelope's orange and the oracle's green. F's maroon is at least 15 apart in OKLab (x100) from each
of those four, in normal vision and under simulated red-green colour blindness; `check_chart_routers` measures it.

G's and G+'s styles were reserved before any Jev number existed (Amendment A15, items 14 and 15) and chart v3 draws
them (Stage 6 · 12). G's is a near-black, circles, plus signs and a dash-dot line; G+'s a deep indigo, X marks,
pentagons and a dotted line. The five colours already drawn leave room for only two more that sit 15 apart from
them and each other under every simulated vision, and the darkest ones are those two: no pair of mid-tones is. A dark
line still reads against the near-white surface, and each style also has a marker and a line of its own, so no two
routers rest on colour alone. `check_jev_names` measures the distances.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from analysis.bootstrap import Bootstrap
from analysis.dev_diagnostic import dev_diagnostic
from analysis.matrix import Matrix
from analysis.router_names import G_PLUS, LABELS, NAMED_ROUTERS, E, F, G
from analysis.router_table import RouterScoreTable
from analysis.rules_table import dev_table
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

# F's: maroon, triangles and a dashed line, against E's purple, squares, diamonds and a solid line. Its label offsets are
# placed by eye on the committed figure, as E's are.
F_STYLE = RouterStyle(
    color="#9b1c31", marker="^", point_marker="v", linestyle="--",
    point_offsets={
        ("realistic", "97%"): (-6, -58, "right"),
        ("realistic", "99%"): (44, -24, "left"),
        ("realistic", "100%: falls back to A"): (74, -80, "right"),
        ("uniform", "97%"): (-84, 0, "right"),
        ("uniform", "100%, 99%: falls back to A"): (8, -92, "right"),
    },
)  # fmt: skip

# G's and G+'s: colours, markers and lines chosen before any Jev number existed. Their label offsets are placed by eye on
# the committed figure after the walks, as E's and F's were (A15, item 15: layout is presentation, not selection).
G_STYLE = RouterStyle(
    color="#1e0a14", marker="o", point_marker="P", linestyle="-.",
    point_offsets={
        ("realistic", "97%"): (-60, -40, "right"),
        ("realistic", "99%"): (30, 60, "left"),
        ("realistic", "100%"): (60, 40, "left"),
        ("uniform", "97%"): (-14, -26, "right"),
        ("uniform", "99%"): (-60, 50, "right"),
        ("uniform", "100%"): (40, 60, "left"),
    },
)  # fmt: skip
G_PLUS_STYLE = RouterStyle(
    color="#14109a", marker="X", point_marker="p", linestyle=":",
    point_offsets={
        ("realistic", "97%"): (-30, 8, "right"),
        ("realistic", "99%"): (-60, 36, "right"),
        ("realistic", "100%: falls back to A"): (66, -62, "right"),
        ("uniform", "97%"): (-50, -30, "right"),
        ("uniform", "99%"): (-60, 30, "right"),
        ("uniform", "100%"): (30, -60, "left"),
    },
)  # fmt: skip


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
    name=NAMED_ROUTERS[E],
    label=LABELS[E],
    short=E,
    style=E_STYLE,
    dev_table=dev_table,
    walks=dev_walks,
    facts=lambda: {"task_inference_dev": dev_diagnostic()},
)


def _trained_dev_table() -> RouterScoreTable:
    # Imported here: the trained router's modules pull in PyTorch and scikit-learn, which a check that only draws
    # committed numbers (CI's `check_chart_routers`) never needs.
    from analysis import trained_table

    return trained_table.dev_table()


TRAINED = RouterEntry(
    name=NAMED_ROUTERS[F],
    label=LABELS[F],
    short=F,
    style=F_STYLE,
    dev_table=_trained_dev_table,
    walks=dev_walks,
)


def _jev_dev_table(router: str) -> Callable[[], RouterScoreTable]:
    # Imported here, as F's is: the Jev tables read the stored dev records, which a check that only draws committed
    # numbers (CI's `check_chart_routers`) never needs.
    def build() -> RouterScoreTable:
        from analysis import jev_table

        return jev_table.dev_table(router)

    return build


JEV = RouterEntry(
    name=NAMED_ROUTERS[G],
    label=LABELS[G],
    short=G,
    style=G_STYLE,
    dev_table=_jev_dev_table("jev"),
    walks=dev_walks,
)

JEV_BOLSTERED = RouterEntry(
    name=NAMED_ROUTERS[G_PLUS],
    label=LABELS[G_PLUS],
    short=G_PLUS,
    style=G_PLUS_STYLE,
    dev_table=_jev_dev_table("jev_bolstered"),
    walks=dev_walks,
)

ROUTERS: tuple[RouterEntry, ...] = (RULES, TRAINED, JEV, JEV_BOLSTERED)
