"""The routers' and the cascades' names, in one place (PLAN_v4.md §8 and §12; Amendment A9, item 13; Amendment A11,
items 12, 13 and 16; Amendment A17, items 8 and 25).

A router is named here and only here: its letter, the name of its section in `routers/operating_points.json` (which is
its key under "routers" in the figure JSONs) and its label on the chart. The walks, the allowlist, the chart's router
list, the diagnostics and the checks import these, so a router renamed, or one added later, changes one place.

`E`, `F`, `G` and `G_PLUS` are also the routers' names in the readings and the comparisons ("F less E"); `F_TRAIN` and
`TASK_RATES` are diagnostics, which are no policies and so have no section and no label (A11, item 12).

The fifteen cascades (A17, item 8) are named below the routers, beside `NAMED_ROUTERS` and not inside it: several checks
hold that map to exactly the four routers. A cascade has a section key, `cascade_{start}_{gate}`, and a label such as
`haiku/Jev` or `G+/Sonnet`; `CASCADES` maps the first to the second, in item 8's order.
"""

E, F, G, G_PLUS, F_TRAIN, TASK_RATES = "E", "F", "G", "G+", "F-train", "task rates"

# The routers an amendment names, by letter, and the name each has as a section of the operating-points file: E is the
# rules router (A9), F the trained router (A11), G the Jev router and G+ the bolstered Jev router (A15, items 14 and
# 15). A router joins when an amendment names it. F-train and the diagnostics never do.
NAMED_ROUTERS = {E: "rules", F: "trained", G: "jev", G_PLUS: "jev_bolstered"}

# A section's letter, the reverse of the map above: "G" for `jev`. Code that holds a section name (the stored records'
# `router` field is one) gets its letter here instead of spelling the pair out again.
LETTER_OF = {section: letter for letter, section in NAMED_ROUTERS.items()}

# Each named router's label, on the chart and in the figure JSONs.
LABELS = {
    E: "E: rules router",
    F: "F: trained router",
    G: "G: Jev router",
    G_PLUS: "G+: bolstered Jev router",
}

# The fifteen cascades (A17, items 8 and 25). A cascade starts at `haiku` (a pure cascade) or at a router's pick (a
# router-started cascade) and ends in one gate: the free checks alone, the Jev gate or the Sonnet gate.
HAIKU = "haiku"
FREE_GATE, JEV_GATE, SONNET_GATE = "free", "jev", "sonnet"

# A key's `start` part, by the name it has on a label: `haiku` for the pure cascades, then each router's section name
# under its letter. This is item 8's order of starts.
CASCADE_STARTS = {HAIKU: HAIKU, **{section: letter for letter, section in NAMED_ROUTERS.items()}}
# A key's `gate` part, by the name it has on a label.
GATE_LABELS = {FREE_GATE: "free", JEV_GATE: "Jev", SONNET_GATE: "Sonnet"}


def cascade_section(start: str, gate: str) -> str:
    """The section key of the cascade that starts at `start` (`haiku` or a named router's section name) and ends in
    `gate`. Refuses (ValueError) a start or a gate no amendment names."""
    if start not in CASCADE_STARTS:
        raise ValueError(f"{start!r} is not a start an amendment names ({list(CASCADE_STARTS)})")
    if gate not in GATE_LABELS:
        raise ValueError(f"{gate!r} is not a gate an amendment names ({list(GATE_LABELS)})")
    return f"cascade_{start}_{gate}"


# Every cascade's section key and its label, in item 8's order: the pure cascades, then E, F, G and G+, each with free,
# Jev and Sonnet.
CASCADES = {
    cascade_section(start, gate): f"{label}/{GATE_LABELS[gate]}"
    for start, label in CASCADE_STARTS.items()
    for gate in GATE_LABELS
}
