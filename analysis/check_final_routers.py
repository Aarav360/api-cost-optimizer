"""Check E's and F's router-score tables on test, built through the handle (PLAN_v4.md Amendment A9, item 7; A11, items 9 and
17; A19, items 2, 8 and 13; issue #83).

Run from the project root:  uv run python -m analysis.check_final_routers

On the rehearsal's made-up world (`analysis/rehearsal_world.py`) with a fake embedder (`analysis/rehearsal_trained.py`), and on
a made-up world whose benchmark holds a request over F's window and one a template line empties. No results/, no model snapshot,
no Jev or Anthropic call, and nothing near the real store or a real test item.

- **The tables are built through the handle.** E's and F's router scores for the eight made-up test requests are those of the
  routers' own `score(request)` called by hand, E's by the rules router and F's by `TrainedRouter.score` on the same fake
  embedder; the tables cover the handle's IDs, each overhead is an explicit $0. A plain `"test"`, a handle built by hand and a
  step run with no unlock are refused, and so is the benchmark reader's `split="test"`;
- **each fold model's share of test** is counted from the request's own hash (this check hashes the canonical text itself, as A11
  item 9 words it) and is in the record, from the tables step on, as counts and shares; **the truncation count** is hand-worked on a
  request exactly at F's window (not truncated) and one a token over (truncated), and on a request a template line empties;
  neither the shares nor the count is in the log or the record before the tables step;
- **a resume** embeds nothing twice: the second go finds the scores stored, the embedder sees no further text and the file is
  unchanged; stored scores for other requests than the handle's are refused;
- **F-train and the diagnostics can't run on test.** Nothing on the final evaluation's path names them (a search of the modules,
  which finds them in the diagnostics' own modules as its known positive); a whole rehearsal with each of them made to fail
  never calls one; and each reader they rest on refuses test: the benchmark reader, the fold file, the embedding cache's loader,
  the results-matrix loader, the learning curve's split inputs and the operating-points file's allowlist.
"""

import ast
import dataclasses
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from analysis import fit_trained_router as fit
from analysis import learning_curve, trained_diagnostics, trained_table
from analysis import matrix as results_matrix
from analysis import rehearsal_trained as rt
from analysis import rehearsal_world as rehearsal
from analysis.backends import Backends
from analysis.benchmark import SealedItemError, read_rows
from analysis.check_final_run import rehearsed
from analysis.final_record import RECORD_FILE, Record
from analysis.final_routers import RoutersError, calls_step, read_scores, score_test, trained_facts_record
from analysis.final_run import STEPS, Outcome, Run, now_utc
from analysis.folds import FoldError, read_folds
from analysis.made_up_world import BENCHMARK_ROWS, DEFAULT, Content, made_up_world
from analysis.matrix import Matrix, MatrixError
from analysis.operating_points import OperatingPointsError, check_sections
from analysis.router_names import NAMED_ROUTERS
from analysis.unlock import Handle, HandleError, unlock
from harness.client import PROJECT_ROOT
from routers import rules, trained
from routers.ladder import SCORED
from routers.request import Request, from_row
from routers.trained import TrainedRouter

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def raises(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def independent_fold(request: Request) -> int:
    """A11 item 9, worded as the plan words it: the sha256 of the sorted-key, space-free JSON of the prompt and schema, as an
    integer, mod 5."""
    text = json.dumps({"output_schema": request.output_schema, "prompt": request.prompt}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)  # fmt: skip
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big") % 5


TEST_REQUESTS = {made["item_id"]: from_row(made) for made in rehearsal.ITEMS}
UP_TO_CALLS = tuple(s for s in STEPS if s.name in ("unlock", "calls_rules_trained"))


def tables_and_shares() -> None:
    print("-- E's and F's tables on test, through the handle --")
    with rehearsed() as r:
        outcome, _, _ = r.go()
        check(outcome == Outcome(), f"the rehearsal runs every step, the calls on test included {outcome}")
        stored = json.loads((r.world.results / "test_router_scores.json").read_text(encoding="utf-8"))
        by_hand = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=rt.SeededBackend(0.0))
        check(list(stored["rules"]) == list(TEST_REQUESTS)
              and all(stored["rules"][i] == rules.score(q) for i, q in TEST_REQUESTS.items()),
              "E's stored router scores are the rules router's own score(request), item by item, in the handle's order")  # fmt: skip
        check(all(stored["trained"][i] == by_hand.score(q) for i, q in TEST_REQUESTS.items())
              and all(set(s) == set(SCORED) for s in stored["trained"].values()),
              "F's are TrainedRouter.score(request), the hash path, on the same fake embedder, for haiku, sonnet and opus_low")  # fmt: skip
        check(all(stored["folds"][i] == independent_fold(q) for i, q in TEST_REQUESTS.items()),
              "each request's fold is the plan's hash of its canonical text, worked here independently")  # fmt: skip
        # F's scores differ by fold model: the same request scored by another fold model gives other numbers.
        item, request = next(iter(TEST_REQUESTS.items()))
        other = by_hand.score_with((stored["folds"][item] + 1) % 5, request)
        check(
            other != stored["trained"][item],
            "known positive: another fold model gives another score, so the hash matters",
        )

        record = r.record["gos"][0]["steps"]
        tables = next(s for s in record if s["name"] == "tables")["detail"]["extras"]["trained_facts"]
        counts = [independent_fold(q) for q in TEST_REQUESTS.values()]
        wanted = {str(k): counts.count(k) for k in range(5)}
        check(tables["fold_counts"] == wanted and tables["items"] == 8
              and tables["fold_shares"] == {k: n / 8 for k, n in wanted.items()} and sum(wanted.values()) == 8,
              f"the record's tables step holds each fold model's share of test, as the hash gives it {wanted}")  # fmt: skip
        check(tables["truncated"] == 0 and tables["emptied"] == 0, "and F's truncation and emptied counts (none here)")
        before = r.log.split("tables: starts")[0]
        calls = next(s for s in record if s["name"] == "calls_rules_trained")
        check("fold" not in before and "truncat" not in before and "fold" not in json.dumps(calls)
              and "truncat" not in json.dumps(calls) and "fold_shares" in r.log,
              "neither the shares nor the truncation count is in the log or the calls step's record before the tables step")  # fmt: skip
        check(calls["requests"] == {"sent": 8, "stored": 0} and calls["detail"] == {"items": 8, "stored": False},
              "the calls step records its eight requests and says nothing more")  # fmt: skip
        check(r.tables["parts"]["rules"]["mixes"]["realistic"]["99%"]["threshold"] == 0.9171597633136095
              and r.tables["parts"]["trained"]["mixes"]["uniform"]["97%"]["threshold"] == 0.9021524030177893
              and r.tables["parts"]["rules"]["mixes"]["realistic"]["100%"]["fell_back"] is True,
              "the tables hold E's and F's parts at the file's frozen thresholds, falling back where the file says")  # fmt: skip

        print("\n-- a resume embeds nothing twice --")
        before_texts = len(r.world.backends.trained.embedder.texts())
        bytes_before = (r.world.results / "test_router_scores.json").read_bytes()
        again, _, _ = r.go(UP_TO_CALLS)
        step = r.record["gos"][1]["steps"][1]
        check(again == Outcome() and step["detail"] == {"items": 8, "stored": True} and step["requests"] == {"sent": 0, "stored": 8},
              "the second go finds the eight requests' scores stored and sends none")  # fmt: skip
        check(len(r.world.backends.trained.embedder.texts()) == before_texts
              and (r.world.results / "test_router_scores.json").read_bytes() == bytes_before,
              "the embedder saw no further text, and the stored file is unchanged")  # fmt: skip
        stored["rules"].pop(next(iter(stored["rules"])))
        (r.world.results / "test_router_scores.json").write_text(json.dumps(stored), encoding="utf-8")
        refused, _, _ = r.go(UP_TO_CALLS)
        check(refused == Outcome("calls_rules_trained", "RoutersError", None),
              "stored scores for other requests than the handle's are refused, not trusted")  # fmt: skip


def refusals() -> None:
    print('\n-- a plain "test", a hand-built handle and a step with no unlock are refused --')
    with rehearsed() as r:
        world = r.world
        check(raises(HandleError, lambda: score_test("test", world)), 'the table builder refuses a plain "test"')  # type: ignore[arg-type]
        hand_built = Handle(world, ())
        check(raises(HandleError, lambda: score_test(hand_built, world)), "and a handle the unlock did not issue")
        state = Run(world, Record(world.output / RECORD_FILE), now_utc)
        check(raises(HandleError, lambda: calls_step(state)), "and the calls step run with no handle from the unlock")
        check(not (world.results / "test_router_scores.json").exists(), "nothing was stored by any of them")
    check(raises(SealedItemError, lambda: read_rows(split="test")), 'the benchmark reader still refuses split="test"')
    check(raises(RoutersError, lambda: read_scores(
        Path("/nonexistent/test_router_scores.json"), {"a": 1})), "scores that were never stored are refused")  # fmt: skip


def truncation() -> None:
    print("\n-- the truncation count, by hand --")
    window = trained.read_settings().window
    template = trained.read_settings().template_lines[0]
    # One token per character in the fake, with the two special tokens: a prompt of window - 2 characters fills the window
    # exactly, and one more character is a token over it.
    exact, over = "a" * (window - 2), "b" * (window - 1)
    rows = [dict(row) for row in BENCHMARK_ROWS]
    for row in rows:
        if row["item_id"] == "banking77:made-test-0":
            row["prompt"] = exact
        elif row["item_id"] == "mmlu_pro:made-test-1":
            row["prompt"] = over
        elif row["item_id"] == "ifeval:made-test-2":
            row["prompt"] = template
    with made_up_world(Content(rows, DEFAULT.sealed)) as made:
        handle = unlock(made.world)
        world = dataclasses.replace(made.world, backends=Backends(trained=rt.stand_in(0.0)))
        scores = score_test(handle, world)
        check(scores.truncated == 1 and scores.emptied == 1,
              f"one request at the window and one a token over it, and one a template line empties: truncated {scores.truncated}, emptied {scores.emptied}")  # fmt: skip
        embedded = world.backends.trained.embedder.embedded  # type: ignore[union-attr]
        sizes = sorted(len(ids) for ids in embedded)
        check(sizes == [2, window, window] and any(ids[0] == 1 and ids[-1] == 2 for ids in embedded),
              "three embeddings: the emptied request's two special tokens, the exact one whole and the one over cut to the window")  # fmt: skip
        record = trained_facts_record(scores)
        check(record["truncated"] == 1 and record["emptied"] == 1 and sum(record["fold_counts"].values()) == 3
              and abs(sum(record["fold_shares"].values()) - 1.0) < 1e-12,
              "the record's facts hold the counts and three fold shares that sum to one")  # fmt: skip


# ---- F-train and the diagnostics -----------------------------------------------------------------------------------

NAMES = {"f_train_table", "read_f_train_weights", "trained_diagnostics", "learning_curve", "F_TRAIN", "TASK_RATES",
         "task_rates_table"}  # fmt: skip
ON_THE_PATH = ["final_routers", "trained_drift", "offline_half", "final_run", "final_grading", "final_record",
               "operating_eval", "rehearsal_trained", "unlock", "router_panels", "final_go"]  # fmt: skip


def names_in(path: Path) -> set[str]:
    """The names a module's code uses: variables, attributes and what it imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names)
            found.update((node.module or "").split("."))
        elif isinstance(node, ast.Import):
            found.update(part for alias in node.names for part in alias.name.split("."))
    return found


class Sentinel(Exception):
    pass


def f_train_and_diagnostics() -> None:
    print("\n-- F-train and the diagnostics can't run on test --")
    folder = PROJECT_ROOT / "analysis"
    named = {name: sorted(names_in(folder / f"{name}.py") & NAMES) for name in ON_THE_PATH}
    check(not any(named.values()), f"no module on the final evaluation's path names F-train or a diagnostic {named}")
    known = names_in(folder / "trained_diagnostics.py") & NAMES
    check({"f_train_table", "F_TRAIN", "TASK_RATES"} <= known and "f_train_table" in names_in(folder / "trained_table.py"),
          "known positive: the same search finds them in the diagnostics' own modules")  # fmt: skip
    calls: list[str] = []

    def spy(name: str) -> Callable[..., None]:
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise Sentinel(name)

        return fail

    patches = [mock.patch.object(trained_table, "f_train_table", spy("f_train_table")),
               mock.patch.object(fit, "read_f_train_weights", spy("read_f_train_weights")),
               mock.patch.object(trained_diagnostics, "compute", spy("compute")),
               mock.patch.object(learning_curve, "build", spy("build"))]  # fmt: skip
    with rehearsed() as r:
        for patch in patches:
            patch.start()
        try:
            outcome, _, _ = r.go()
            check(
                outcome == Outcome() and calls == [],
                f"a whole rehearsal never calls F-train, the diagnostics or the learning curve {outcome}",
            )
            check(raises(Sentinel, trained_table.f_train_table) and raises(Sentinel, trained_diagnostics.compute)
                  and len(calls) == 2, "known positive: called by hand, each of them fires the spy")  # fmt: skip
        finally:
            for patch in patches:
                patch.stop()
    check(raises(SealedItemError, lambda: read_rows(split="test")), 'the benchmark reader refuses "test", which F-train and every diagnostic read through')  # fmt: skip
    check(
        raises(FoldError, lambda: read_folds("test")), "the fold reader has no folds for test: its fold is the hash's"
    )
    with tempfile.TemporaryDirectory() as folder_name:
        where = Path(folder_name)
        np.save(where / "test.npy", np.zeros((0, 768)))
        (where / "test_items.json").write_text(
            json.dumps({"item_ids": [], "tokens": [], "emptied": []}), encoding="utf-8"
        )
        check(raises(SealedItemError, lambda: fit.read_cache("test", where)),
              "the embedding cache's loader refuses test even where files of that name exist")  # fmt: skip
    row = {"item_id": "x", "split": "test", "source": "x", "price_snapshot": "x", "cost_usd": 0, "run": 0}
    check(raises(MatrixError, lambda: results_matrix.from_store([row])), "the results-matrix loader refuses a test row")
    check(raises(SealedItemError, lambda: learning_curve._split_inputs(trained.read_settings(), Matrix({}, {}), "test")),
          "the learning curve's split inputs refuse test")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder_name:
        path = Path(folder_name) / "operating_points.json"
        path.write_text(json.dumps({"rules": {}, "f_train": {}}), encoding="utf-8")
        check(raises(OperatingPointsError, lambda: check_sections(path)) and "f_train" not in NAMED_ROUTERS.values(),
              "the operating-points file's allowlist refuses a section for F-train, and F-train is no named router")  # fmt: skip
    check("f_train" not in (PROJECT_ROOT / "routers" / "trained.py").read_text(encoding="utf-8").lower().replace("f-train", ""),
          "the deployable router never loads F-train's weights (its code has no f_train name)")  # fmt: skip


def main() -> int:
    tables_and_shares()
    refusals()
    truncation()
    f_train_and_diagnostics()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
