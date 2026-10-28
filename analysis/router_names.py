"""The routers' names, in one place (PLAN_v4.md §8 and §12; Amendment A9, item 13; Amendment A11, items 12, 13 and 16).

A router is named here and only here: its letter, the name of its section in `routers/operating_points.json` (which is
its key under "routers" in the figure JSONs) and its label on the chart. The walks, the allowlist, the chart's router
list, the diagnostics and the checks import these, so a router renamed, or one added later, changes one place.

`E`, `F`, `G` and `G_PLUS` are also the routers' names in the readings and the comparisons ("F less E"); `F_TRAIN` and
`TASK_RATES` are diagnostics, which are no policies and so have no section and no label (A11, item 12).
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
