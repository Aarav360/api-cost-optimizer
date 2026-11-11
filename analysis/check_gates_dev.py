"""Check Stage 7's dev run and the fifteen sections (PLAN_v4.md §9; Amendment A17, items 8 to 11, 20.5 and 20.6; issue #70).

Run from the project root:  uv run python -m analysis.check_gates_dev

Synthetic part (CI too): no results/ folder, no Docker, no Jev or Anthropic call, no benchmark answer, no test item.
- the fifteen walks from a dev gate-score table and four router-score tables, their sections in item 8's order, each the
  section the pure or router-started builder gives;
- the rules a section must keep (every certified step clears its target by its margin, the operating point is the
  cheapest certified step, the walk stops at the first miss), with known positives, and the captured fraction at most 1;
- the committed file: the fifteen cascade sections are written beside the four routers', which stay as they were,
  and the file holds those nineteen and no others.

Where results/ and the stored dev records are here (the hook only; no Jev or Anthropic call, no test item):
- the stored dev records cover every dev answer once (free checks), every passing answer once (Jev, pass 1 only) and every
  passing haiku, sonnet and opus_low answer once (Sonnet), and no other; the table rebuilds from them;
- the order: the first dev Jev call's server Date, every dev call's collection time and the dev batch's creation time come
  after the freeze commit's time (A17 item 20.6), with a known positive; every gate record names a train or dev item;
- the committed record of the run (`analysis/gates_dev_record.json`) is what the stored records give;
- the fifteen walks rebuilt from the stored records and the routers' tables are the committed sections, each keeps the
  walk's rules and every captured fraction is at most 1; E's, F's, G's and G+'s sections are byte for byte as the freeze
  commit left them, and the file holds those nineteen and no other.
"""

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

from analysis import gate_dev, gate_dev_run
from analysis.bootstrap import resample
from analysis.check_simulator import synthetic
from analysis.check_walk import table_of
from analysis.checking import refused
from analysis.gate_table import GateRow, GateScoreTable, PaidScore
from analysis.gate_walk import MIXES, pure_section, pure_walks
from analysis.operating_points import (
    OperatingPointsError,
    check_sections,
    read_section,
    section_hash,
    write_section,
)
from analysis.references import TARGETS
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, NAMED_ROUTERS, SONNET_GATE, cascade_section

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
LADDER = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
ITEMS = [f"{t}{n}" for t in TASKS for n in range(4)]
CUTOFFS = {JEV_GATE: 0.45, SONNET_GATE: 0.7}


def synthetic_dev() -> tuple:
    dev = synthetic({i: (i.rstrip("0123"), {c: (LADDER[c], int((k + j) % 3 != 0)) for j, c in enumerate(LADDER)})
                     for k, i in enumerate(ITEMS)})  # fmt: skip
    keys = [(i, c) for i in ITEMS for c in LADDER]
    rows = [GateRow(i, c, (k % 7) != 0, None if (k % 7) else "stop",
                    jev=PaidScore(0.3 + 0.01 * k, 0.0001, False) if k % 7 else None,
                    sonnet=PaidScore(0.9 - 0.01 * k, 0.0002, False) if k % 7 else None)
            for k, (i, c) in enumerate(keys)]  # fmt: skip
    table = GateScoreTable.build(rows, keys)
    routers = {
        name: table_of(
            {
                i: {c: ((k * 7 + j * 3 + seed) % 20) / 20 for j, c in enumerate(("haiku", "sonnet", "opus_low"))}
                for k, i in enumerate(ITEMS)
            },
            overhead=0.00001 * seed,
        )  # fmt: skip
        for seed, name in enumerate(NAMED_ROUTERS.values(), start=1)
    }
    return dev, table, routers


def synthetic_checks() -> None:
    dev, table, routers = synthetic_dev()
    draws = resample(dev)
    print("-- the fifteen walks --")
    walks = gate_dev.cascade_walks(dev, table, routers, CUTOFFS, draws)
    check(list(walks) == list(CASCADES) and len(walks) == 15,
          "fifteen walks, under the cascades' keys, in item 8's order: the pure three, then each router's free, Jev, Sonnet")  # fmt: skip
    check(all(list(by_mix) == list(MIXES) and all(set(by_share) == set(TARGETS) for by_share in by_mix.values())
              for by_mix in walks.values()),
          "each has a walk per mix and quality target")  # fmt: skip
    pure = pure_walks(dev, table, JEV_GATE, draws)
    check(walks[cascade_section(HAIKU, JEV_GATE)] == pure,
          "haiku/Jev's walk is the pure cascade's walk over its gate's cutoffs")  # fmt: skip
    free = walks[cascade_section(HAIKU, FREE_GATE)]
    check(all(len(w.steps) == 2 for by in free.values() for w in by.values()),
          "haiku/free's walk is A and the one step")  # fmt: skip
    sonnet = walks[cascade_section(NAMED_ROUTERS["G"], SONNET_GATE)]
    check(all(s.cutoff == 0.7 for by in sonnet.values() for w in by.values() for s in w.steps[1:]),
          "G/Sonnet's steps all sit at the fixed Sonnet cutoff")  # fmt: skip
    check(refused(ValueError, lambda: gate_dev.cascade_walks(dev, table, dict(list(routers.items())[:3]), CUTOFFS, draws)),
          "a missing router is refused")  # fmt: skip

    print("\n-- the sections --")
    sections = gate_dev.cascade_sections(walks)
    check(list(sections) == list(CASCADES), "fifteen sections under the same keys, in the same order")
    check(sections[cascade_section(HAIKU, JEV_GATE)] == pure_section(pure, JEV_GATE)
          and "threshold" not in sections[cascade_section(HAIKU, JEV_GATE)]["realistic"]["100%"],
          "a pure cascade's section is the pure builder's: a cutoff and no threshold")  # fmt: skip
    started = sections[cascade_section(NAMED_ROUTERS["F"], JEV_GATE)]["uniform"]["99%"]
    check({"threshold", "cutoff"} <= set(started),
          "a router-started cascade's section has its threshold and the fixed cutoff")  # fmt: skip
    check(all(gate_dev.section_problems(s, key) == [] for key, s in sections.items()),
          "every section keeps the walk's rules: certified steps clear their margin, the cheapest is chosen")  # fmt: skip

    print("\n-- the rules, with known positives --")
    point = sections[cascade_section(HAIKU, JEV_GATE)]["realistic"]["100%"]
    bad = copy.deepcopy(sections[cascade_section(HAIKU, JEV_GATE)])
    first = bad["realistic"]["100%"]["steps"]
    first[1]["clears"], first[1]["quality"] = True, bad["realistic"]["100%"]["target"] - 0.2
    check(gate_dev.section_problems(bad, cascade_section(HAIKU, JEV_GATE)) != [],
          "a certified step that misses its target by its margin is found")  # fmt: skip
    bad = copy.deepcopy(sections[cascade_section(HAIKU, JEV_GATE)])
    bad["realistic"]["100%"]["cutoff"] = 0.123
    check(gate_dev.section_problems(bad, cascade_section(HAIKU, JEV_GATE)) != [],
          "an operating point that is not a step is found")  # fmt: skip
    check(point["steps"][0]["cutoff"] is None, "A is the first step")
    bad = copy.deepcopy(sections[cascade_section(NAMED_ROUTERS["E"], FREE_GATE)])
    bad["uniform"]["97%"]["steps"][0]["margin"] = 0.5
    check(gate_dev.section_problems(bad, cascade_section(NAMED_ROUTERS["E"], FREE_GATE)) != [],
          "a walk that does not start at A (margin not 0) is found")  # fmt: skip
    check(gate_dev.captured_problems([0.0, 0.5, 1.0, None]) == [] and gate_dev.captured_problems([1.2]) != []
          and gate_dev.captured_problems([-0.1]) != [],
          "a captured fraction outside 0 to 1 is found, n/a is allowed")  # fmt: skip
    captured = gate_dev.captured_fractions(walks, dev, draws)
    values = [v for by_mix in captured.values() for by_share in by_mix.values() for v in by_share.values()]
    check(len(values) == 15 * 6 and gate_dev.captured_problems(values) == [],
          "each cascade has six captured fractions, all at most 1")  # fmt: skip
    check(all(captured[k]["realistic"][1.0] == 0.0
              for k, w in walks.items() if w["realistic"][1.0].fell_back),
          "a cascade that falls back to A captures nothing")  # fmt: skip

    print("\n-- the committed file --")
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "operating_points.json"
        letters = {
            name: {"realistic": {"100%": {"threshold": 0.5 + k}}} for k, name in enumerate(NAMED_ROUTERS.values())
        }
        for name, section in letters.items():
            write_section(name, section, out)
        before = {name: section_hash(read_section(name, out)) for name in letters}
        gate_dev.write_sections(sections, out)
        check(list(json.loads(out.read_text())) == sorted({*letters, *sections}),
              "the file holds the four routers' sections and the fifteen cascades', sorted")  # fmt: skip
        check({name: section_hash(read_section(name, out)) for name in letters} == before,
              "the four routers' sections are as they were, hash for hash")  # fmt: skip
        check(gate_dev.committed_problems(out) == [], "the file passes: nineteen sections and no other")
        check_sections(out, required=(*letters, *sections))
        check(True, "the allowlist passes with all nineteen required")
        check(all(read_section(k, out) == json.loads(json.dumps(v)) for k, v in sections.items()),
              "each cascade section reads back as written")  # fmt: skip
        stray = json.loads(out.read_text())
        stray["F-train"] = {}
        (Path(folder) / "stray.json").write_text(json.dumps(stray))
        check(gate_dev.committed_problems(Path(folder) / "stray.json") != [],
              "a section that is no router or cascade is found")  # fmt: skip
        short = json.loads(out.read_text())
        del short[cascade_section(HAIKU, FREE_GATE)]
        (Path(folder) / "short.json").write_text(json.dumps(short))
        check(gate_dev.committed_problems(Path(folder) / "short.json") != [],
              "a missing cascade section is found")  # fmt: skip
        check(refused(OperatingPointsError, lambda: gate_dev.write_sections({"cascade_x": {}}, out)),
              "a section no amendment names is refused before anything is written")  # fmt: skip
        check(refused(OperatingPointsError, lambda: gate_dev.write_sections({k: sections[k] for k in list(sections)[:14]}, out)),
              "fourteen sections are refused: the walks write all fifteen or none")  # fmt: skip


def free_record(item: str, config: str, passed: bool, failed: str | None = None) -> dict:
    return {"item_id": item, "config": config, "stop_passed": failed != "stop", "schema_applied": False,
            "schema_passed": True, "code_applied": failed == "code" or item.startswith("m"),
            "code_passed": failed != "code", "failed_first": failed, "passed": passed, "code_duration_s": 0.25}  # fmt: skip


def jev_record(
    item: str,
    config: str,
    seconds: float,
    *,
    tokens: int = 1000,
    new: bool = False,
    refused: bool = False,
    when: str = "2026-11-10T22:20:00-08:00",
    date: str = "Wed, 11 Nov 2026 06:20:00 GMT",
) -> dict:
    return {"item_id": item, "config": config, "pass": 1, "refused": refused, "answer_cut_bytes": 0, "prompt_cut_bytes": 0,
            "attempts": 1, "failed_attempts": [], "seconds": seconds, "new_connection": new, "usage": {"input_tokens": tokens, "output_tokens": 60},
            "model": "jev-1.14.0", "server_date": date, "collected_at": when}  # fmt: skip


def measure_checks() -> None:
    print("\n-- the dev measures, from hand-made records --")
    free = [free_record("a1", "haiku", True), free_record("a1", "opus", True), free_record("a2", "haiku", False, "stop"),
            free_record("a2", "opus", True), free_record("m1", "haiku", False, "code"), free_record("m1", "opus", True)]  # fmt: skip
    # Jev's calls, one per passing answer, in the order made: seconds .10 (the first call), .20, .30, a reconnect at .90, .40.
    jev = [jev_record("a1", "haiku", .10), jev_record("a1", "opus", .20), jev_record("a2", "opus", .30),
           jev_record("m1", "opus", .90, new=True, when="2026-11-10T22:21:00-08:00"), jev_record("a3", "opus", .40, refused=True)]  # fmt: skip
    batch = [{"item_id": "a1", "config": "haiku", "refused": False, "stop_reason": "end_turn",
              "usage": {"input_tokens": 1000, "output_tokens": 100}},
             {"item_id": "a2", "config": "haiku", "refused": True, "stop_reason": "max_tokens",
              "usage": {"input_tokens": 1000, "output_tokens": 16000}}]  # fmt: skip
    log = {"batches": [{"id": "msgbatch_x", "split": "dev", "created_at": "2026-11-10T22:30:00-08:00", "request_count": 2,
                        "custom_ids": [], "ingested_at": "2026-11-10T22:40:00-08:00", "counts": {"succeeded": 2}, "missing": 0},
                       {"id": "msgbatch_old", "split": "train", "created_at": "2026-11-10T21:31:00-08:00", "request_count": 0,
                        "custom_ids": [], "ingested_at": "2026-11-10T21:36:00-08:00", "counts": {}, "missing": 0}]}  # fmt: skip
    task_of = {"a1": "t", "a2": "t", "a3": "t", "m1": "u"}
    got = gate_dev_run.measures(free, jev, batch, log, task_of, ("haiku", "opus"),
                                jev_price=0.042, input_price=2.0, output_price=10.0, batch_discount=0.5)  # fmt: skip
    check(got["counts"]["haiku"]["answers"] == 3 and got["counts"]["haiku"]["rejected"] == 2
          and got["counts"]["haiku"]["failed_first"] == {"stop": 1, "schema": 0, "code": 1}
          and got["counts"]["opus"]["rejected"] == 0 and got["counts"]["opus"]["jev_calls"] == 4,
          "counts per config: free-check rejections by the check that failed first, Jev calls")  # fmt: skip
    check(got["counts"]["opus"]["jev_refused"] == 1 and got["counts"]["haiku"]["sonnet_refused"] == 1
          and got["counts"]["haiku"]["sonnet_cap_stops"] == 1 and got["counts"]["opus"]["sonnet_calls"] == 0,
          "refusals of both gates and the cap stop are counted, and opus has no Sonnet verdict")  # fmt: skip
    lat = got["jev_latency"]
    # The first call (.10) and the reconnect (.90) are set apart; .20 .30 .40 remain: median .30.
    check(lat["median_s"] == 0.30 and lat["count"] == 3 and lat["first_s"] == 0.10 and lat["reconnects_s"] == [0.90],
          "Jev latency leaves out the first call and the reconnect, and reports both")  # fmt: skip
    check(got["first_jev_date"] == "Wed, 11 Nov 2026 06:20:00 GMT"
          and got["batches"] == [{"id": "msgbatch_x", "created_at": "2026-11-10T22:30:00-08:00"}],
          "the first call's server Date and the dev batch's creation time, train's batch left out")  # fmt: skip
    # Jev: 5 calls x 1000 tokens at $0.042 per million = $0.000210. Batch: 2000 input at $2 + 16100 output at $10 per million
    # = $0.004 + $0.161 = $0.165 at list price, half of it as billed.
    check(math.isclose(got["spend"]["jev"], 0.000210) and math.isclose(got["spend"]["batch_list"], 0.165)
          and math.isclose(got["spend"]["batch"], 0.0825),
          "spend: Jev at its price, the batch at list price and as billed")  # fmt: skip
    check(got["code_check_median_s"] == 0.25 and got["code_checks"] == 2,
          "the code check's median is over the answers it applied to")  # fmt: skip
    check(refused(ValueError, lambda: gate_dev_run.measures(free, [], batch, log, task_of, ("haiku", "opus"),
                                                            jev_price=0.042, input_price=2.0, output_price=10.0,
                                                            batch_discount=0.5)),
          "no Jev call at all is refused")  # fmt: skip


def real_checks() -> None:
    import email.utils
    from datetime import datetime

    from analysis import collect_sonnet_gate, jev_train_report
    from analysis.benchmark import read_rows
    from analysis.check_jev_settings_commit import settings_commit
    from analysis.checking import git
    from analysis.collect_gates import read_free_records
    from analysis.collect_jev_gate import read_records as read_jev_records
    from analysis.gate_answers import read_answers
    from analysis.matrix import load as load_matrix
    from analysis.operating_points import OPERATING_POINTS_FILE
    from harness.client import PROJECT_ROOT
    from routers.ladder import SCORED

    folder = PROJECT_ROOT / "results" / "gates"
    print("\n-- the stored dev records --")
    answers = {(a.item_id, a.config) for a in read_answers("dev")}
    free = read_free_records("dev")
    passed = {(r["item_id"], r["config"]) for r in free if r["passed"]}
    check({(r["item_id"], r["config"]) for r in free} == answers and len(free) == len(answers) == 4 * 744,
          "the free checks cover each of dev's 744 x 4 answers once")  # fmt: skip
    jev_records = read_jev_records("dev")
    check({(r["item_id"], r["config"]) for r in jev_records} == passed and len(jev_records) == len(passed)
          and all(r["pass"] == 1 for r in jev_records),
          "the Jev gate has one pass-1 call on each answer that passed the free checks, and none on any other, no noise pass")  # fmt: skip
    verdicts = collect_sonnet_gate.read_sonnet_records("dev")
    check({(r["item_id"], r["config"]) for r in verdicts} == {k for k in passed if k[1] in SCORED}
          and len(verdicts) == len({(r["item_id"], r["config"]) for r in verdicts}),
          "the Sonnet gate has one verdict on each passing haiku, sonnet and opus_low answer, and none on opus")  # fmt: skip
    log = collect_sonnet_gate.load_log(collect_sonnet_gate.batch_log_path())
    dev_batches = [b for b in log["batches"] if b["split"] == "dev"]
    check(dev_batches != [] and all(b["ingested_at"] is not None and b["missing"] == 0 for b in dev_batches),
          "every dev batch is ingested with nothing missing")  # fmt: skip

    print("\n-- dev's gate-score table, rebuilt --")
    table = gate_dev_run.dev_table()
    check(len(table.rows) == 4 * 744 and {(r.item_id, r.config) for r in table.rows} == answers,
          "the table has a row for each dev answer")  # fmt: skip
    verdict = {(r["item_id"], r["config"]): (bool(r["passed"]), r["failed_first"]) for r in free}
    check(all((row.free_passed, row.failed_first) == verdict[(row.item_id, row.config)] for row in table.rows),
          "its free-check columns are the stored verdicts")  # fmt: skip
    check(all((row.jev is not None) == row.free_passed for row in table.rows)
          and all((row.sonnet is not None) == (row.free_passed and row.config in SCORED) for row in table.rows),
          "its Jev column is on every passing answer and its Sonnet column on every passing haiku, sonnet and opus_low one")  # fmt: skip
    check(sum(r.jev.refused for r in table.rows if r.jev) == sum(bool(r["refused"]) for r in jev_records)
          and sum(r.sonnet.refused for r in table.rows if r.sonnet) == sum(bool(r["refused"]) for r in verdicts),
          "its refusal counts are the stored flags'")  # fmt: skip

    print("\n-- the order (A17 item 20.6) --")
    freeze = settings_commit(PROJECT_ROOT, "routers/gate_cutoffs.json")
    check(freeze is not None, "the freeze commit is in this history")
    if freeze is None:
        return
    when = datetime.fromisoformat(git("show", "-s", "--format=%cI", freeze).stdout.strip())
    first = jev_train_report.first_date(jev_records)
    check(
        first > when,
        f"the first dev Jev call's server Date {first.isoformat()} is after the freeze commit {when.isoformat()}",
    )
    check(min(email.utils.parsedate_to_datetime(r["server_date"]) for r in jev_records) > when,
          "and so is the earliest Date among all of them")  # fmt: skip
    check(all(datetime.fromisoformat(r["collected_at"]) > when for r in jev_records + verdicts)
        and all(datetime.fromisoformat(r["code_checked_at"]) > when for r in free),
          "every dev gate record, and every free-check record's stamp, is after the freeze commit")  # fmt: skip
    check(all(datetime.fromisoformat(b["created_at"]) > when for b in dev_batches),
          "every dev batch was created after the freeze commit")  # fmt: skip
    old = datetime.fromisoformat("2026-11-10T21:51:03-08:00")
    check(not old > when, "known positive: a time one second before the freeze commit is found early")
    allowed = {r["item_id"] for split in ("train", "dev") for r in read_rows(split=split)}
    stored = sorted(folder.glob("*.jsonl"))
    check(all({json.loads(line)["item_id"] for line in path.read_text(encoding="utf-8").splitlines() if line} <= allowed
              for path in stored) and not list(folder.glob("*_test.jsonl")),
          "every gate record names a train or dev item, and no test file exists")  # fmt: skip
    check({"x:1"} - allowed == {"x:1"}, "known positive: an item that is neither is found")

    print("\n-- the record of the run --")
    record = json.loads(gate_dev_run.RECORD_FILE.read_text(encoding="utf-8"))
    check(record == json.loads(json.dumps(gate_dev_run.dev_record())),
          "analysis/gates_dev_record.json is what the stored dev records, the conditions file and the cutoffs give")  # fmt: skip
    cond = record["conditions"]
    check({"captured_at", "machine", "network"} == set(cond) and datetime.fromisoformat(cond["captured_at"]) > when
          and datetime.fromisoformat(cond["captured_at"]) <= min(datetime.fromisoformat(r["collected_at"]) for r in jev_records),
          "it records the machine and network, stamped after the freeze and before the first dev call was stored")  # fmt: skip
    lat = record["jev_latency"]
    check(lat["first_s"] == float(jev_records[0]["seconds"]) and lat["count"] > 0
          and lat["reconnects_s"] == [float(r["seconds"]) for r in jev_records[1:] if r["new_connection"]],
          "its latency sets the first call and every reconnect apart, as stored")  # fmt: skip
    counts = record["counts"]
    check(set(counts) == {"haiku", "sonnet", "opus_low", "opus"}
          and sum(c["answers"] for c in counts.values()) == len(free)
          and sum(c["jev_calls"] for c in counts.values()) == len(jev_records)
          and sum(c["sonnet_calls"] for c in counts.values()) == len(verdicts),
          "its counts cover all four configs' answers, calls and verdicts")  # fmt: skip

    print("\n-- the fifteen walks and the committed sections --")
    dev = load_matrix().split("dev")
    from analysis.bootstrap import resample

    draws = resample(dev)
    walks = gate_dev.cascade_walks(dev, table, gate_dev_run.routers_dev_tables(), gate_dev_run.read_cutoffs(), draws)
    sections = gate_dev.cascade_sections(walks)
    check(
        gate_dev.committed_problems() == [],
        "the committed file holds E's, F's, G's and G+'s sections and the fifteen cascades', no other",
    )
    committed = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    check(all(committed[k] == json.loads(json.dumps(v)) for k, v in sections.items()),
          "the walks on dev reproduce all fifteen committed sections exactly")  # fmt: skip
    check(all(section_hash(committed[k]) == section_hash(v) for k, v in sections.items()),
          "and their hashes")  # fmt: skip
    check(all(gate_dev.section_problems(committed[k], k) == [] for k in CASCADES),
          "every certified step clears its target by its margin, A first, the cheapest certified step chosen: all fifteen")  # fmt: skip
    captured = gate_dev.captured_fractions(walks, dev, draws)
    values = [v for by_mix in captured.values() for by_share in by_mix.values() for v in by_share.values()]
    check(
        len(values) == 90 and gate_dev.captured_problems(values) == [],
        "every captured fraction is at most 1 (90 of them)",
    )
    held = json.loads(git("show", f"{freeze}:routers/operating_points.json").stdout)
    check(set(held) == set(NAMED_ROUTERS.values()) and all(committed[k] == held[k] for k in held),
          "E's, F's, G's and G+'s sections are as the freeze commit held them")  # fmt: skip
    check(committed[NAMED_ROUTERS["E"]] != {} and committed[NAMED_ROUTERS["E"]] != held[NAMED_ROUTERS["F"]],
          "known positive: a router's section differs from another's")  # fmt: skip
    print(f"      (hash of the whole file: {hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest()})")


def main() -> int:
    synthetic_checks()
    measure_checks()
    from harness.client import PROJECT_ROOT

    folder = PROJECT_ROOT / "results" / "gates"
    have = (
        (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists()
        and all((folder / f"{n}_dev.jsonl").exists() for n in ("free_checks", "jev_gate", "sonnet_gate"))
        and gate_dev_run.RECORD_FILE.exists()
    )  # fmt: skip
    if have:
        real_checks()
    else:
        print(
            "\n(the dev matrix, the stored dev records or the dev record are not here: the stored-record sections are skipped)"
        )
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
