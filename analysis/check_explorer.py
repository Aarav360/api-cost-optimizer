"""Check the interactive Pareto explorer against the figure JSONs of both splits (PLAN_v4.md §12; Amendment A19, item 8;
CODING_STANDARDS.md, Numbers in documents).

Run from the project root:  uv run python -m analysis.check_explorer

Reads the committed figure JSONs (dev's chart v4 and, where committed, test's `figures/pareto_*_test.json`), the task mixes, the
template and figures/pareto_explorer.html, so CI runs it. No results matrix, never a test item.

**Before the results PR, test's JSONs don't exist in the repository**, so the committed page holds dev alone: its switch shows
test as not run, and the page opens on dev, since it can't open on numbers it doesn't hold. The check holds that rule both ways:
the committed page opens on test if and only if test's JSONs are committed, and a page built from made-up test figures (dev's JSONs
relabelled and shifted in memory, never committed, built only here) opens on test. When the results PR commits test's JSONs, the
committed page fails this check until it is rebuilt with them, which is the reminder.

- the committed page is the build from the committed JSONs, byte for byte (known positive: a JSON with one config's quality nudged by
  1e-6 builds a different page), so a rebuilt chart with a stale page fails;
- the numbers the page carries are the JSONs', read straight from the files and not through the build, for each split it holds:
  every view, every config's quality and cost as [value, low, high], every router's steps, every cascade's steps (the router-started
  ones from the router-started figure's JSON, on the two mix views), and at each mix's targets each operating point's threshold,
  cutoff, fallback, quality, cost, savings against A and matched-cost difference; the free checks' one point;
- the page opens on test when it holds test, and names the split in its title, its heading, its sub-title and its footer, and, run in
  Node where it is installed, in every readout of every view of every split: the plot's own label, the caption, the table's
  heading and each hover; a split it doesn't hold is a disabled button, and a link to it stays on the split the page is on
  (known positive: a page whose hover omits the split is found);
- each of the fifteen cascades is a series the page can switch on and off, in item 8's order, with the label and the gate's colour
  and line from `CASCADE_STYLES`; the presets name only series that exist; a task view holds the routers and the pure cascades;
- the page's own words agree with the JSONs: the split and run in its footer, the kind of interval, the targets its caption names,
  the routers' labels;
- the template: one data placeholder, Plotly pinned to an exact version on jsDelivr, and its router list is chart v3's.
"""

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from analysis.chart_routers import CASCADE_STYLES, ROUTERS
from analysis.explorer import EXPLORER, MARKERS, ORDER, PLACEHOLDER, TEMPLATE, dash_of, load_figures, page
from analysis.pareto import PANELS, TASKS
from analysis.router_names import CASCADES, GATE_LABELS
from analysis.simulator import load_mixes

problems: list[str] = []
SPREAD = ("value", "low", "high")  # the order the page reads an estimate's three numbers in


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def embedded(html: str) -> dict | None:
    """The data the page carries: the JSON after `const DATA = `, to the end of that line."""
    found = re.search(r"^const DATA = (.*);$", html, re.MULTILINE)
    return json.loads(found[1].replace("<\\/", "</")) if found else None


def spread(estimate: dict) -> list[float]:
    return [estimate[key] for key in SPREAD]


def check_cascades(name: str, view: dict, source: dict, routed: dict | None, mix: bool) -> None:
    """One view's cascades against the figure JSONs: the pure ones from the panel, the router-started ones (a mix view
    only) from the router-started panel's lines."""
    wanted = dict(source["cascades"])
    if routed is not None:
        wanted |= {key: line for router in routed["routers"].values() for key, line in router["lines"].items()
                   if key in CASCADES}  # fmt: skip
    drawn = view["cascades"]
    check(set(drawn) == set(wanted) and all(key in CASCADES for key in drawn),
          f"{name}: {len(drawn)} cascades, {'the pure three and the twelve router-started' if routed else 'the pure three'}")  # fmt: skip
    for key, part in wanted.items():
        steps = [[s["cost_per_1k"], s["quality"], s["threshold"], s["cutoff"]] for s in part["steps"]]
        check(drawn[key]["steps"] == steps, f"{name}: {CASCADES[key]}'s {len(steps)} steps, A first")
        if "point" in part:
            check(drawn[key]["point"] == {"q": spread(part["point"]["quality"]), "c": spread(part["point"]["cost_per_1k"])},
                  f"{name}: {CASCADES[key]} is one point with its intervals")  # fmt: skip
        if not mix:
            check("ops" not in drawn[key], f"{name}: {CASCADES[key]} has no operating points")
            continue
        ops = {op["target"]: op for op in drawn[key]["ops"]}
        for target, point in part["operating_points"].items():
            op, matched = ops.get(target, {}), point.get("matched_cost") or {}
            check(op.get("threshold") == point["threshold"] and op.get("cutoff") == point["cutoff"]
                  and op.get("fell_back") == point["fell_back"] and op.get("q") == spread(point["quality"])
                  and op.get("c") == spread(point["cost_per_1k"])
                  and op.get("save_share") == (point.get("savings_vs_A") or {}).get("share")
                  and op.get("matched") == ([matched[k] for k in ("difference", "low", "high")] if matched else None),
                  f"{name}: {CASCADES[key]} at {target}: threshold, cutoff, fallback, quality, cost, savings, matched cost")  # fmt: skip


def check_view(name: str, view: dict, source: dict, mix: bool, inside: bool = True) -> None:
    """One view's numbers against its panel in the figure JSON."""
    configs = {row["key"]: row for row in view["configs"]}
    check(list(configs) == list(source["configs"])
          and all(configs[c]["q"] == spread(row["quality"]) and configs[c]["c"] == spread(row["cost_per_1k"])
                  for c, row in source["configs"].items()),
          f"{name}: each config's quality and cost, with intervals")  # fmt: skip
    check((not inside) or all(low <= value <= high for row in view["configs"] for value, low, high in (row["q"], row["c"])),
          f"{name}: each config's value sits inside its interval, so the page reads [value, low, high]")  # fmt: skip
    for router in ROUTERS:
        part, drawn = source["routers"][router.name], view["routers"][router.name]
        check(drawn["steps"] == [[s["cost_per_1k"], s["quality"], s["threshold"]] for s in part["steps"]],
              f"{name}: {router.short}'s {len(part['steps'])} steps, A first")  # fmt: skip
        if not mix:
            continue
        ops = {op["target"]: op for op in drawn["ops"]}
        for target, point in part["operating_points"].items():
            op, matched = ops.get(target, {}), point.get("matched_cost") or {}
            check(op.get("threshold") == point["threshold"] and op.get("fell_back") == point["fell_back"]
                  and op.get("q") == spread(point["quality"]) and op.get("c") == spread(point["cost_per_1k"])
                  and op.get("save_share") == (point.get("savings_vs_A") or {}).get("share")
                  and op.get("matched") == ([matched[k] for k in ("difference", "low", "high")] if matched else None),
                  f"{name}: {router.short} at {target}: threshold, fallback, quality, cost, savings, matched cost")  # fmt: skip


def check_split(split: str, held: dict, figures: tuple[dict, dict, dict], label: str = "", inside: bool = True) -> None:
    """One split's views in the page against that split's three JSONs, read straight from them."""
    tag = label + split
    head, tasks, routed = figures
    views = held["views"]
    check(list(views) == [*PANELS, *TASKS], f"{tag}: the views are the mixes and the tasks, in the figures' order")
    for name in PANELS:
        check_view(f"{tag} {name}", views[name], head["panels"][name], mix=True, inside=inside)
        check_cascades(f"{tag} {name}", views[name], head["panels"][name], routed["panels"][name], mix=True)
    for name in TASKS:
        check_view(f"{tag} {name}", views[name], tasks["panels"][name], mix=False, inside=inside)
        check_cascades(f"{tag} {name}", views[name], tasks["panels"][name], None, mix=False)
        check(views[name]["items"] == tasks["panels"][name]["items"], f"{tag} {name}: its {split} item count")
    check(held["meta"] == {"split": head["split"], "n": head["n"], "prices": head["price_snapshot"],
                           "resamples": head["bootstrap"]["resamples"]} and head["split"] == split == tasks["split"] == routed["split"],
          f"{tag}'s facts: {head['n']} requests, prices of {head['price_snapshot']}, {head['bootstrap']['resamples']:,} "
          "resamples, the split named in all three JSONs")  # fmt: skip
    check(all(figure["run"] == 0 for figure in (head, tasks)) and all(
        figure["bootstrap"]["interval"] == "percentile 95%" for figure in (head, tasks)) and head["bootstrap"][
        "resampled"].startswith("source groups within each task"),
        f"{tag}: the JSONs say run 0, percentile 95% intervals, resampled by source group within each task")  # fmt: skip
    check(views["realistic"]["note"] is not None and f"{split} items" in views["realistic"]["note"]
          and ("dev items" not in views["realistic"]["note"] or split == "dev"),
          f"{tag}: the realistic mix's note counts {split} items")  # fmt: skip


def made_up_test(dev: tuple[dict, dict, dict]) -> tuple[dict, dict, dict]:
    """Made-up test figures for the check alone, never committed: dev's JSONs relabelled as test with a different item count
    and every quality and cost shifted, so no test number is dev's. The shape is the figure code's, which the rehearsal check
    (`analysis/check_final_figures.py`) holds against the real builder."""

    def shifted(value: Any) -> Any:
        if isinstance(value, dict):
            value = {
                k: v.replace(" dev items", " test items") if k == "warning" and isinstance(v, str) else v
                for k, v in value.items()
            }  # the per-task warnings count the split's items
            return {k: (v * 0.97 if k in {*SPREAD, "quality", "cost_per_1k"} and isinstance(v, float) else shifted(v))
                    for k, v in value.items()}  # fmt: skip
        if isinstance(value, list):
            return [shifted(v) for v in value]
        return value

    made = []
    for figure in dev:
        copy_ = shifted(copy.deepcopy(figure))
        copy_["split"], copy_["n"] = "test", 100
        for key in ("units",):
            copy_.pop(key, None)
        for panel in copy_["panels"].values():
            if "items" in panel:
                panel["items"] = 25
        made.append(copy_)
    return (made[0], made[1], made[2])


# The page's own script in Node with a stand-in for the page: each split's every view rendered with every series on, and what it
# showed (title, heading, caption, table heading, the plot's label and every hover) printed.
READOUTS_JS = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[1], "utf8");
const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const els = {};
const el = (id) => (els[id] ||= { id, dataset: {}, style: {}, children: [], textContent: "", innerHTML: "", disabled: false, title: "",
  appendChild(c) { this.children.push(c); }, addEventListener() {}, setAttribute() {} });
const page = {
  document: { getElementById: el, createElement: () => el("_" + Object.keys(els).length), querySelectorAll: () => [], documentElement: {}, title: "" },
  location: { hash: process.argv[2] || "" }, window: { addEventListener() {} }, localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ addEventListener() {} }), MutationObserver: class { observe() {} },
  getComputedStyle: () => ({ getPropertyValue: (name) => name }), Plotly: { react(_, traces, layout) { page.last = { traces, layout }; } },
};
vm.createContext(page);
vm.runInContext(script, page);
const out = { opened: JSON.parse(vm.runInContext("JSON.stringify({ split: state.split, view: state.view })", page)),
  title: page.document.title, h1: els.h1Split.textContent, sub: els.subSplit.textContent, foot: els.footSplit.textContent,
  buttons: Object.fromEntries(Object.values(els).filter((e) => String(e.id).startsWith("split-")).map((e) => [e.id, e.disabled])), readouts: [] };
if (process.argv[3] === "all") {
  vm.runInContext("for (const r of SERIES) state.series[r.key] = true;", page);
  for (const split of Object.keys(JSON.parse(vm.runInContext("JSON.stringify(SPLITS)", page)))) {
    const views = Object.keys(JSON.parse(vm.runInContext("JSON.stringify(SPLITS['" + split + "'].views)", page)));
    for (const view of views) {
      vm.runInContext("state.split = '" + split + "'; state.view = '" + view + "'; render();", page);
      out.readouts.push({ split, view, title: page.document.title, h1: els.h1Split.textContent, sub: els.subSplit.textContent,
        foot: els.footSplit.textContent, caption: els.caption.innerHTML, table: els.tableBox.innerHTML,
        label: page.last.layout.annotations[0].text,
        hovers: page.last.traces.flatMap((t) => t.hovertext || []) });
    }
  }
}
process.stdout.write(JSON.stringify(out));
"""


def run_page(html: str, hash_: str = "", everything: bool = False) -> dict | None:
    """What the page's script shows for a link `hash_`, or, with `everything`, for every view of every split; None without Node."""
    node = shutil.which("node")
    if node is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.html"
        path.write_text(html, encoding="utf-8")
        run = subprocess.run([node, "-e", READOUTS_JS, str(path), hash_, "all" if everything else ""], capture_output=True,
                             text=True, timeout=120)  # fmt: skip
    if run.returncode != 0:
        print(f"      node: {run.stderr.strip()[:400]}")
        return {}
    shown: dict = json.loads(run.stdout)
    return shown


def unnamed(readouts: list[dict]) -> list[str]:
    """Each readout of the page that doesn't name its split: the title, heading, sub-title, footer, the plot's own label, the caption,
    the table's heading and every hover."""
    found = []
    for r in readouts:
        split, where = r["split"], f"{r['split']} {r['view']}"
        for key in ("title", "h1", "sub", "label", "caption", "table"):
            if split not in r[key].lower():
                found.append(f"{where}: {key}")
        if r["foot"].lower() != split.capitalize().lower():
            found.append(f"{where}: footer")
        found += [f"{where}: hover {h[:40]!r}" for h in r["hovers"] if f"{split} split" not in h]
        other = next(s for s in ORDER if s != split)
        found += [f"{where}: {key} names {other}'s items" for key in ("caption", "table") if f"{other} items" in r[key]]
    return found


def main() -> int:
    figures = load_figures()
    mixes = load_mixes()
    committed = EXPLORER.read_text(encoding="utf-8") if EXPLORER.exists() else ""
    check("dev" in figures, f"chart v4's JSONs are committed: the page holds {', '.join(figures)}")
    if "dev" not in figures:
        return 1
    dev = figures["dev"]

    print("-- the committed page is the build from the committed JSONs --")
    check(page(figures, mixes) == committed,
          "figures/pareto_explorer.html is the build, byte for byte (rerun analysis.explorer after analysis.pareto, and after "
          "test's figures are committed)")  # fmt: skip
    nudged = copy.deepcopy(dev)
    nudged[0]["panels"]["realistic"]["configs"]["haiku"]["quality"]["value"] += 1e-6
    check(page({**figures, "dev": (nudged[0], nudged[1], nudged[2])}, mixes) != committed,
          "known positive: a JSON with one quality nudged by 1e-6 builds a different page")  # fmt: skip

    print("\n-- the page's numbers are the JSONs', for each split it holds --")
    data = embedded(committed)
    check(data is not None, "the page carries its data on one `const DATA = ...;` line")
    if data is None:
        print(f"\n{len(problems)} checks failed")
        return 1
    check(data["order"] == list(ORDER) and list(data["splits"]) == [s for s in ORDER if s in figures],
          f"the switch lists {', '.join(ORDER)}, and the page holds {', '.join(data['splits'])}")  # fmt: skip
    for split, held in data["splits"].items():
        check_split(split, held, figures[split])
    first = figures[next(iter(data["splits"]))]
    check(data["labels"] == {router.name: first[0]["routers"][router.name]["label"] for router in ROUTERS}
          == {router.name: router.label for router in ROUTERS}, "the routers' labels are chart v3's (analysis/chart_routers.py)")  # fmt: skip

    print("\n-- the page opens on test when it holds test, and names the split --")
    check(data["open"] == ("test" if "test" in figures else "dev"),
          f"the committed page opens on {data['open']}: test's JSONs are {'' if 'test' in figures else 'not '}committed")  # fmt: skip
    check(f"<title>Router Pareto Explorer: {data['open']} split</title>" in committed
          and f'<h1>Router Pareto Explorer: <span id="h1Split">{data["open"]}</span> split</h1>' in committed
          and f'<span id="footSplit">{data["open"].capitalize()}</span> split, run 0.' in committed
          and "{{SPLIT" not in committed,
          f"the committed page's title, heading and footer name {data['open']} before any script runs")  # fmt: skip
    made = {"dev": dev, "test": made_up_test(dev)}
    both = page(made, mixes)
    both_data = embedded(both) or {}
    check(both_data.get("open") == "test" and list(both_data.get("splits", {})) == ["test", "dev"],
          "made-up test figures beside dev's: the page opens on test and holds both splits")  # fmt: skip
    check("<title>Router Pareto Explorer: test split</title>" in both and "<h1>Router Pareto Explorer: "
          '<span id="h1Split">test</span> split</h1>' in both and '<span id="footSplit">Test</span> split, run 0.' in both,
          "and its title, heading and footer name test")  # fmt: skip
    check(page({"dev": dev}, mixes) != both and (embedded(page({"dev": dev}, mixes)) or {}).get("open") == "dev",
          "known positive: the same page without test's figures opens on dev, not test")  # fmt: skip
    for split, held in both_data.get("splits", {}).items():
        check_split(split, held, made[split], "made-up ")
    check(both_data.get("splits", {}).get("test", {}).get("views", {}).get("realistic", {}).get("note", "").count(" test items") == 1,
          "the made-up test's realistic note says test items, not dev's")  # fmt: skip
    check(refused_partial(), "a split with only some of its three figure files is refused, not drawn half")

    print("\n-- every readout names its split (the page's script, run in Node) --")
    if shutil.which("node") is None:
        print("SKIP  the page's readouts: Node is not installed")
    else:
        shown = run_page(both, everything=True) or {}
        views = len(shown.get("readouts", []))
        check(shown.get("opened", {}).get("split") == "test" and shown.get("title") == "Router Pareto Explorer: test split"
              and shown.get("h1") == "test" and shown.get("sub") == "test" and shown.get("foot") == "Test",
              "opened with no link, it is on test, and its title, heading, sub-title and footer say so")  # fmt: skip
        check(views == 12 and not unnamed(shown.get("readouts", [])),
              f"all {views} views (six on each split) name their split in the title, heading, sub-title, footer, plot label, "
              f"caption, table heading and every hover {unnamed(shown.get('readouts', []))[:3]}")  # fmt: skip
        hover_count = sum(len(r["hovers"]) for r in shown.get("readouts", []))
        check(hover_count > 500, f"{hover_count} hovers were checked, so the search sees them")
        leaky = both.replace("t.hovertext.map((h) => h + tag)", "t.hovertext.map((h) => h)")
        found = unnamed((run_page(leaky, everything=True) or {}).get("readouts", []))
        check(any("hover" in f for f in found) and leaky != both,
              "known positive: a page whose hovers leave out the split is found")  # fmt: skip
        dev_open = run_page(both, "#dev/ifeval") or {}
        check(dev_open.get("opened") == {"split": "dev", "view": "ifeval"} and dev_open.get("title") == "Router Pareto Explorer: dev split",
              "the link #dev/ifeval opens dev's IFEval view, and the title follows the split")  # fmt: skip
        check((run_page(both, "#uniform") or {}).get("opened") == {"split": "test", "view": "uniform"},
              "a link with a view alone stays on the split the page opened on")  # fmt: skip
        only = run_page(committed if "test" not in figures else page({"dev": dev}, mixes), "#test/uniform") or {}
        check(only.get("opened") == {"split": "dev", "view": "uniform"} and only.get("buttons", {}).get("split-test") is True
              and only.get("buttons", {}).get("split-dev") is False,
              "on a page that holds dev alone, test's button is disabled and a link to test stays on dev")  # fmt: skip

    print("\n-- the page's words agree with the JSONs --")
    template = TEMPLATE.read_text(encoding="utf-8")
    check(all(figure["bootstrap"]["interval"] == "percentile 95%" for figure in dev[:2])
          and "split, run 0. Intervals are percentile 95%" in template
          and "resamples of source groups within each task" in template,
          "the footer's interval kind and what is resampled are the JSONs'")  # fmt: skip
    targets = [list(dev[0]["panels"][mix]["targets"]) for mix in PANELS]
    named = f"the {', '.join(targets[0][:-1])} and {targets[0][-1]} targets"
    check(all(t == targets[0] for t in targets) and named in template, f"the caption names {named}, as both mixes "
          "hold them")  # fmt: skip
    series = data["cascades"]
    check([c["key"] for c in series] == list(CASCADES) and [c["label"] for c in series] == list(CASCADES.values()),
          "the page lists the fifteen cascades, in item 8's order, under their labels (analysis/router_names.py)")  # fmt: skip
    check(all(c["color"] == CASCADE_STYLES[next(g for g in GATE_LABELS if GATE_LABELS[g] == c["gate"])].color
              and c["dash"] == dash_of(CASCADE_STYLES[next(g for g in GATE_LABELS if GATE_LABELS[g] == c["gate"])].linestyle)
              and c["step"] in MARKERS.values() and c["point"] in MARKERS.values() for c in series),
          "each cascade has its gate's colour and line, and markers Plotly knows")  # fmt: skip
    keys = {router.name for router in ROUTERS} | set(CASCADES)
    check(all(set(p["on"]) <= keys for p in data["presets"]) and data["presets"][-1] == {"label": "All off", "on": []},
          f"the {len(data['presets'])} presets name series that exist, and the last switches everything off")  # fmt: skip

    print("\n-- the template --")
    check(template.count(PLACEHOLDER) == 1, f"the template has one data placeholder, {PLACEHOLDER}")
    scripts = re.findall(r'<script src="([^"]+)"', template)
    check(len(scripts) == 1 and re.fullmatch(r"https://cdn\.jsdelivr\.net/npm/plotly\.js-dist-min@\d+\.\d+\.\d+/"
          r"plotly\.min\.js", scripts[0]) is not None,
          f"it loads one script, Plotly at an exact version on jsDelivr: {scripts}")  # fmt: skip
    listed = re.findall(r'\{ key: "(\w+)", short: "([^"]+)"', template)
    check(listed == [(router.name, router.short) for router in ROUTERS],
          f"its router list is chart v3's, in order: {', '.join(short for _, short in listed)}")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def refused_partial() -> bool:
    """`load_figures` refuses a split whose figure files are only partly there (a made-up folder stands in for figures/)."""
    from unittest import mock

    from analysis import explorer

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "pareto_test.json").write_text("{}", encoding="utf-8")
        real = explorer.figure_paths
        with mock.patch.object(explorer, "figure_paths",
                               lambda split: real(split) if split == "dev" else tuple(folder / p.name for p in real(split))):  # fmt: skip
            try:
                explorer.load_figures()
            except FileNotFoundError:
                return True
    return False


if __name__ == "__main__":
    sys.exit(main())
