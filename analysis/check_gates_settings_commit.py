"""Check the settings commit's promises (PLAN_v4.md §9; Amendment A17, items 20.1 and 20.2; issue #68).

Run from the project root:  uv run python -m analysis.check_gates_settings_commit

Calls no Jev and no Anthropic. Stage 7's settings commit freezes every gate before any gate sees a benchmark answer. It
holds the gate settings (with the byte budget, the Dockerfile's hash and the image digest filled in), every gate's import
closure, the code that computes the cutoffs, the noise samples' drawing rule and the smoke test's record with its
script and made-up answers, so after it none of them may change:

- the settings: the byte budget, the image digest, the Dockerfile's hash and the noise section are set (known negatives
  for each), and are the smoke record's and the Dockerfile's;
- the noise samples' drawing rule, on hand-made free-check records: the first that pass in a seeded order, per task, the
  Sonnet list the first of the Jev list, deterministic;
- the smoke test's pure parts: its projection against the cap, worked by hand, and its expected free-check verdicts;
- the file list: it holds every `routers/` module a gate imports (the closure is read from the code);
- unchanged since the commit: a file of the commit's list that differs from it, in the working tree, the index or a later
  commit, is named. The guard runs first on a throwaway git repository (through `analysis.checking.git`, under git's own
  repository variables as a hook has them), then on this repository, where the commit is the one that first added the
  smoke record; before that commit exists that section says so and passes;
- the smoke answers: each request is its train item's, and (where results/ is present) no answer is a train or dev answer
  of the results matrix, since a gate may see made-up answers only;
- where Docker is running, the settings' image digest is the local image's ID.
"""

import ast
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from analysis import smoke_gates
from analysis.benchmark import read_rows
from analysis.check_jev_settings_commit import changed_since, mock_environ, settings_commit
from analysis.checking import git, refused
from analysis.collect_jev import from_row
from analysis.gate_answers import read_answers
from analysis.noise_samples import NoiseSampleError, NoiseSettings, draw, noise_lists, read_noise_settings
from harness.collect import MAIN
from routers import gates
from routers.sandbox import SandboxError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = "routers/gate_settings.json"
SMOKE_ANSWERS = "analysis/gates_smoke_answers.json"
SMOKE_RECORD = "analysis/gates_smoke_record.json"
DOCKERFILE = "sandbox/Dockerfile"
# The settings commit's files (A17 item 20.2): the settings; every gate's import closure (the free checks, the Jev gate
# and its frozen client and parser, the Sonnet gate, the cascade decision, the sandbox) and the Jev client's settings;
# the Dockerfile the image was built from; the code that computes the cutoffs and the noise lists; the smoke test's script,
# made-up answers and record.
COMMIT_FILES = (
    SETTINGS,
    "routers/__init__.py",
    "routers/cascade.py",
    "routers/gates.py",
    "routers/jev.py",
    "routers/jev_call.py",
    "routers/jev_gate.py",
    "routers/jev_settings.json",
    "routers/ladder.py",
    "routers/request.py",
    "routers/sandbox.py",
    "routers/sonnet_gate.py",
    DOCKERFILE,
    "analysis/gate_cutoff.py",
    "analysis/noise_samples.py",
    "analysis/smoke_gates.py",
    SMOKE_ANSWERS,
    SMOKE_RECORD,
)
GATE_ENTRY_POINTS = ("gates", "jev_gate", "sonnet_gate", "cascade", "sandbox")  # the modules a gate or a cascade runs
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def read_json(path: Path) -> object:
    """A JSON file's content, or None if it is missing or isn't JSON (the caller's check says which file)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def routers_closure(root: Path, entries: tuple[str, ...]) -> set[str]:
    """The `routers/` modules the entry modules import, directly or not, read from their code (`from routers import a`,
    `from routers.a import b`, `import routers.a`), with the entries and the package itself."""
    seen: set[str] = set()
    todo = list(entries)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((root / "routers" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "routers":
                todo += [a.name for a in node.names if (root / "routers" / f"{a.name}.py").exists()]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("routers."):
                todo.append(node.module.split(".", 1)[1])
            elif isinstance(node, ast.Import):
                todo += [a.name.split(".", 1)[1] for a in node.names if a.name.startswith("routers.")]
    return seen


def settings_problems(data: object, dockerfile: bytes, record: object) -> list[str]:
    """What is not yet filled in, or not what it must be, in a gate settings file: the byte budget (a whole number of at
    least 1, the smoke record's), the image digest (a `sha256:` ID, the record's), the Dockerfile's hash (the file's own,
    the record's) and the noise samples' section."""
    found: list[str] = []
    if not isinstance(data, dict):
        return ["the settings are not a JSON object"]
    measured = record.get("measured") if isinstance(record, dict) else None
    image = record.get("image") if isinstance(record, dict) else None
    jev_part = data.get("jev")
    budget = jev_part.get("budget_bytes") if isinstance(jev_part, dict) else None
    if not (type(budget) is int and budget >= 1):
        found.append(f"the Jev byte budget is not a whole number of at least 1: {budget!r}")
    elif not isinstance(measured, dict) or measured.get("jev_budget_bytes") != budget:
        found.append("the Jev byte budget is not the smoke record's")
    code = data.get("code_check")
    digest = code.get("image_digest") if isinstance(code, dict) else None
    docker = code.get("dockerfile_sha256") if isinstance(code, dict) else None
    if not (isinstance(digest, str) and DIGEST.match(digest)):
        found.append(f"the image digest is not a sha256: ID: {digest!r}")
    elif not isinstance(image, dict) or image.get("digest") != digest:
        found.append("the image digest is not the smoke record's")
    if not (isinstance(docker, str) and SHA256.match(docker)):
        found.append(f"the Dockerfile's hash is not a sha256: {docker!r}")
    elif docker != hashlib.sha256(dockerfile).hexdigest():
        found.append("the Dockerfile's hash is not the Dockerfile's sha256")
    elif not isinstance(image, dict) or image.get("dockerfile_sha256") != docker:
        found.append("the Dockerfile's hash is not the smoke record's")
    noise = data.get("noise")
    if not (isinstance(noise, dict) and noise.get("seed") == 2026 and noise.get("jev_per_task") == 25
            and noise.get("sonnet_per_task") == 20 and noise.get("config") == "haiku"):  # fmt: skip
        found.append(
            "the noise section is not seed 2026, haiku, 25 per task for the Jev gate and 20 for the Sonnet gate"
        )
    return found


GOOD_SETTINGS = {
    "jev": {"budget_bytes": 29000},
    "code_check": {"image_digest": "sha256:" + "a" * 64, "dockerfile_sha256": hashlib.sha256(b"FROM x\n").hexdigest()},
    "noise": {"seed": 2026, "config": "haiku", "jev_per_task": 25, "sonnet_per_task": 20},
}
GOOD_RECORD = {
    "measured": {"jev_budget_bytes": 29000},
    "image": {"digest": "sha256:" + "a" * 64, "dockerfile_sha256": hashlib.sha256(b"FROM x\n").hexdigest()},
}


def mutated(**changes: object) -> dict:
    data = json.loads(json.dumps(GOOD_SETTINGS))
    for path, value in changes.items():
        section, key = path.split("__")
        data.setdefault(section, {})[key] = value
    return data


def settings_checks() -> None:
    print("-- the settings are filled in --")
    check(
        settings_problems(GOOD_SETTINGS, b"FROM x\n", GOOD_RECORD) == [],
        "a filled-in file and its record raise nothing",
    )
    bad = {
        "a null byte budget": mutated(jev__budget_bytes=None),
        "a zero byte budget": mutated(jev__budget_bytes=0),
        "a float byte budget": mutated(jev__budget_bytes=29000.0),
        "a bool byte budget": mutated(jev__budget_bytes=True),
        "a budget that is not the record's": mutated(jev__budget_bytes=30000),
        "a null image digest": mutated(code_check__image_digest=None),
        "an image ID with no sha256: prefix": mutated(code_check__image_digest="a" * 64),
        "an image digest that is not the record's": mutated(code_check__image_digest="sha256:" + "b" * 64),
        "a null Dockerfile hash": mutated(code_check__dockerfile_sha256=None),
        "a Dockerfile hash that is not the file's": mutated(code_check__dockerfile_sha256="c" * 64),
        "no noise section": mutated(noise__seed=2025),
        "Jev noise of 24 per task": mutated(noise__jev_per_task=24),
        "Sonnet latency of 21 per task": mutated(noise__sonnet_per_task=21),
    }
    for what, data in bad.items():
        check(settings_problems(data, b"FROM x\n", GOOD_RECORD) != [], f"known negative: {what} is named")
    check(
        settings_problems(GOOD_SETTINGS, b"FROM y\n", GOOD_RECORD) != [],
        "known negative: a Dockerfile that changed is named",
    )
    check(settings_problems(GOOD_SETTINGS, b"FROM x\n", {}) != [] and settings_problems([], b"", {}) != [],
          "known negative: a record with no measured values, and settings that are not an object, are named")  # fmt: skip


def noise_checks() -> None:
    print("\n-- the noise samples' drawing rule --")
    failing = {"a:3", "a:5", "a:7"}
    records = []
    for task, count in (("a", 30), ("b", 12)):
        for n in range(count):
            item = f"{task}:{n}"
            records.append({"item_id": item, "config": "haiku", "passed": item not in failing})
            records.append({"item_id": item, "config": "sonnet", "passed": False})  # another config's: never read
    rule = NoiseSettings(2026, "haiku", ("a", "b"), 5, 3)
    lists = noise_lists(records, rule)
    check(len(lists["jev"]) == 10 and len(lists["sonnet"]) == 6
          and all(i.startswith("a:") for i, _ in lists["jev"][:5]) and all(i.startswith("b:") for i, _ in lists["jev"][5:]),
          "5 per task for the Jev list and 3 for the Sonnet list, tasks in the section's order")  # fmt: skip
    check(all(c == "haiku" for _, c in lists["jev"] + lists["sonnet"]) and not failing & {i for i, _ in lists["jev"]},
          "only haiku's answers, and never one that failed a free check")  # fmt: skip
    check(
        lists["sonnet"] == lists["jev"][:3] + lists["jev"][5:8], "the Sonnet list is the first of each task's Jev list"
    )
    check(noise_lists(records, rule) == lists, "and the draw is deterministic")
    pool = {f"a:{n}": f"a:{n}" not in failing for n in range(30)}
    # Known-good literals, worked once from CPython's seeded shuffle of the sorted items; a change of Python's shuffle
    # would fail here, and the rule would need re-recording.
    check(draw(pool, seed=2026, task="a", count=5) == ["a:18", "a:29", "a:17", "a:25", "a:6"],
          "a task's draw is the first that pass in random.Random('2026/a')'s shuffle of its sorted items")  # fmt: skip
    check(draw(pool, seed=2027, task="a", count=5) == ["a:10", "a:18", "a:6", "a:26", "a:13"],
          "another seed gives another sample")  # fmt: skip
    check(
        draw(pool, seed=2026, task="b", count=5) != draw(pool, seed=2026, task="a", count=5),
        "and another task another order",
    )
    check(refused(NoiseSampleError, lambda: draw({"a:1": True, "a:2": False}, seed=2026, task="a", count=2)),
          "known negative: a task with too few answers that pass is refused")  # fmt: skip
    check(refused(NoiseSampleError, lambda: noise_lists([{"item_id": "z:1", "config": "haiku", "passed": True}], rule)),
          "known negative: a task the section doesn't name is refused")  # fmt: skip
    check(refused(NoiseSampleError, lambda: noise_lists([r for r in records if not str(r["item_id"]).startswith("b:")], rule)),
          "known negative: a named task with no record is refused")  # fmt: skip


def smoke_checks() -> None:
    print("\n-- the smoke test's pure parts --")
    counts = {"train": {"a": 10, "b": 20}, "dev": {"a": 5, "b": 5}}
    # Hand-worked. Sonnet at batch prices: train 10*3*0.001 + 20*3*0.002 = 0.15, dev 5*3*0.001 + 5*3*0.002 = 0.045.
    # Jev: train 10*3*0.0001 + 20*3*0.0002 = 0.015 plus the noise pass 25*0.0001 + 25*0.0002 = 0.0075; dev 5*4*0.0001 +
    # 5*4*0.0002 = 0.006. Latency at list price 20*0.002 + 20*0.004 = 0.12. In all 0.3435.
    projection = smoke_gates.project(counts, sonnet_batch={"a": 0.001, "b": 0.002}, sonnet_list={"a": 0.002, "b": 0.004},
                                     jev={"a": 0.0001, "b": 0.0002})  # fmt: skip
    close = {"sonnet_train_usd": 0.15, "sonnet_dev_usd": 0.045, "jev_train_with_noise_usd": 0.0225, "jev_dev_usd": 0.006,
             "latency_sample_usd": 0.12, "total_usd": 0.3435}  # fmt: skip
    check(
        all(abs(projection[k] - v) < 1e-12 for k, v in close.items()),
        f"the projection is worked out: {projection['total_usd']:.4f}",
    )
    check(projection["within_cap"] and abs(projection["headroom_usd"] - (25.0 - 0.3435)) < 1e-9,
          "and it is within the $25 cap, with the headroom left")  # fmt: skip
    expensive = smoke_gates.project(
        counts, sonnet_batch={"a": 1.0, "b": 2.0}, sonnet_list={"a": 2.0, "b": 4.0}, jev={"a": 0.0, "b": 0.0}
    )
    check(not expensive["within_cap"], "known negative: a projection over $25 is not within the cap")
    check(smoke_gates.task_means([("a:1", 1.0), ("a:2", 3.0), ("b:1", 5.0)]) == {"a": 2.0, "b": 5.0},
          "a cost per check is the mean of its task's checks")  # fmt: skip
    fake = [{"item_id": i, "config": c, "passed": p, "failed_first": f} for (i, c), (p, f) in {
        ("banking77:test-0008", "haiku"): (False, "stop"), ("mbppplus:139", "opus_low"): (False, "code"),
        ("mbppplus:85", "haiku"): (False, "code"), ("mbppplus:124", "haiku"): (False, "code"),
        ("mbppplus:102", "haiku"): (True, None)}.items()]  # fmt: skip
    check(smoke_gates.free_check_mismatches(fake) == [], "the expected free-check verdicts raise nothing")
    wrong = [
        dict(fake[0], passed=True, failed_first=None),
        dict(fake[1], failed_first="stop"),
        dict(fake[4], passed=False, failed_first="code"),
    ]
    check(len(smoke_gates.free_check_mismatches(wrong)) == 3,
          "known negative: an answer that passed but should fail, one failing at the wrong check, one failing but should pass")  # fmt: skip


def copied_answers(entries: list[dict], real_by_item: dict[str, set[str]]) -> list[str]:
    """The made-up answers that are an answer of the results matrix to the same item, whichever config gave it. (A short
    label such as a Banking77 intent can equal another item's answer by chance; the same item's is the copy to catch.)"""
    return [f"{e['item_id']} {e['config']}" for e in entries if e["answer"] in real_by_item.get(e["item_id"], set())]


def copy_checks() -> None:
    print("\n-- made-up answers are not the matrix's --")
    entries = [
        {"item_id": "a:1", "config": "haiku", "answer": "x"},
        {"item_id": "a:2", "config": "sonnet", "answer": "y"},
    ]
    check(copied_answers(entries, {"a:1": {"z"}, "a:2": {"x"}}) == [], "an answer equal to another item's is no copy")
    check(copied_answers(entries, {"a:1": {"z", "x"}}) == ["a:1 haiku"],
          "known positive: an answer equal to the same item's, from any config, is named")  # fmt: skip


def closure_checks() -> None:
    print("\n-- the file list holds every gate's import closure --")
    held = {Path(f).stem for f in COMMIT_FILES if f.startswith("routers/") and f.endswith(".py")}
    closure = routers_closure(PROJECT_ROOT, GATE_ENTRY_POINTS) | {"__init__"}
    check(closure == held, f"the routers modules the gates import ({sorted(closure)}) are the ones the commit holds")
    check(routers_closure(PROJECT_ROOT, ("jev_gate",)) >= {"jev", "jev_call", "ladder", "request"},
          "known positive: the closure follows imports through the frozen client (jev_gate reaches ladder via jev_call)")  # fmt: skip
    check(not (closure - {"__init__"}) <= {"gates", "sandbox"}, "and it is not just the entry points")
    check({"rules", "trained", "decision"}.isdisjoint(closure), "the routers E and F code is no gate's closure")


def throwaway_checks() -> None:
    print("\n-- unchanged since the settings commit --")
    with (
        tempfile.TemporaryDirectory() as folder,
        mock_environ({"GIT_INDEX_FILE": str(Path(folder) / "decoy-index"), "GIT_DIR": str(Path(folder) / "decoy-git")}),
    ):
        repo = Path(folder) / "repo"
        repo.mkdir()
        broke = 0  # git commands that failed while the throwaway repository was set up and changed
        for args in (("init", "-q"), ("config", "user.email", "check@example.com"), ("config", "user.name", "check")):
            broke += git(*args, repo=repo).returncode != 0
        (repo / "other.txt").write_text("before\n")
        broke += git("add", "-A", repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "before", repo=repo).returncode != 0
        check(settings_commit(repo, SMOKE_RECORD) is None, "no commit yet adds the smoke record")
        (repo / "routers").mkdir()
        files = ("routers/gates.py", SMOKE_RECORD)
        (repo / "routers" / "gates.py").write_text("x = 1\n")
        (repo / "analysis").mkdir()
        (repo / SMOKE_RECORD).write_text("{}\n")
        broke += git("add", "-A", repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "settings", repo=repo).returncode != 0
        first = git("rev-parse", "HEAD", repo=repo).stdout.strip()
        check(
            settings_commit(repo, SMOKE_RECORD) == first,
            "the commit that added the smoke record is the settings commit",
        )
        check(changed_since(repo, first, files) == [], "nothing is named while the files are as committed")
        (repo / "other.txt").write_text("after\n")
        broke += git("commit", "-aq", "-m", "other file", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == [], "a later commit of another file names nothing")
        (repo / "routers" / "gates.py").write_text("x = 2\n")
        check(changed_since(repo, first, files) == ["routers/gates.py"], "an edit in the working tree is named")
        broke += git("add", "routers/gates.py", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["routers/gates.py"], "a staged edit is named")
        broke += git("commit", "-q", "-m", "edit", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["routers/gates.py"], "a later commit of an edit is named")
        broke += git("rm", "-q", SMOKE_RECORD, repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "remove", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["routers/gates.py", SMOKE_RECORD], "a removed file is named")
        check(changed_since(repo, first, ("never.txt",)) == ["never.txt"], "a file the commit never held is named")
        check(broke == 0, "every git command that set up and changed the throwaway repository succeeded")


def repository_checks(root: Path) -> None:
    print("\n-- this repository --")
    record = read_json(root / SMOKE_RECORD)
    if record is None:
        print("      (no smoke record yet: Stage 7 · 9 commits it, with the settings)")
        return
    data = read_json(root / SETTINGS)
    dockerfile = (root / DOCKERFILE).read_bytes()
    found = settings_problems(data, dockerfile, record)
    check(found == [], f"the settings are filled in and are the record's ({'; '.join(found)})" if found else
          "the settings are filled in: byte budget, image digest, Dockerfile hash and noise section, as the record has them")  # fmt: skip
    check(isinstance(record, dict) and record.get("free_checks", {}).get("as_expected") is True
          and record.get("fixes") is not None, "the record shows the free checks as expected, and lists its fixes")  # fmt: skip
    check(isinstance(record, dict) and 0 < record.get("spend", {}).get("total_usd", -1) <= smoke_gates.SMOKE_CAP_USD,
          f"the smoke's spend is within its ${smoke_gates.SMOKE_CAP_USD:.0f}")  # fmt: skip
    try:
        noise = read_noise_settings(root / SETTINGS)
        check(noise.seed == 2026, "the noise section is readable by the drawing code")
    except NoiseSampleError as error:
        check(False, f"the noise section is readable by the drawing code: {error}")
    answers = read_json(root / SMOKE_ANSWERS)
    rows = {r["item_id"]: from_row(r) for r in read_rows(split="train")}
    entries = answers if isinstance(answers, list) else []
    check(len(entries) >= 20 and all(e["item_id"] in rows and rows[e["item_id"]].prompt == e["prompt"]
                                     and rows[e["item_id"]].output_schema == e["output_schema"] for e in entries)
          and {e["item_id"].split(":")[0] for e in entries} == {"banking77", "mmlu_pro", "ifeval", "mbppplus"},
          f"{len(entries)} made-up answers, each to its real train item's prompt and schema, from all four tasks")  # fmt: skip
    if MAIN.matrix.exists():
        real_by_item: dict[str, set[str]] = {}
        for split in ("train", "dev"):
            for a in read_answers(split):
                real_by_item.setdefault(a.item_id, set()).add(a.answer)
        check(copied_answers(entries, real_by_item) == [],
              "no made-up answer is any answer of the results matrix to the same item")  # fmt: skip
    else:
        print("      (results/ is absent: the made-up answers are not compared with the matrix's)")
    try:
        digest = gates.image_digest(gates.read_gate_settings(root / SETTINGS).code_check.image)
        check(isinstance(data, dict) and data["code_check"]["image_digest"] == digest,
              "the settings' image digest is the local sandbox image's ID")  # fmt: skip
    except SandboxError:
        print("      (Docker is not running: the image's ID is not compared)")
    if git("rev-parse", "--git-dir", repo=root).returncode != 0:
        check(False, "git can read this tree's history")
        return
    commit = settings_commit(root, SMOKE_RECORD)
    if commit is None:
        print("      (the smoke record is not committed yet: this is the settings commit)")
    else:
        check(
            changed_since(root, commit, COMMIT_FILES) == [],
            f"the settings commit's files are unchanged since {commit[:10]}",
        )


def main() -> int:
    settings_checks()
    noise_checks()
    smoke_checks()
    copy_checks()
    closure_checks()
    throwaway_checks()
    repository_checks(PROJECT_ROOT)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
