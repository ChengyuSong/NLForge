"""Apples-to-apples regression eval for `--type code-contract` against
`scripts/contract_pipeline.py`.

Same task loop, same TaskResult JSON shape so the two reports can be
diffed directly. Each function in topo order is summarized then immediately
verified by `CodeContractPass` (via `BottomUpDriver`). The safety verdict
mirrors `contract_pipeline.py:run_one_task`: `predicted_safe = (no verify
issues across functions reachable from main)`. `check_entries` results are
also collected for inspection but do not gate the verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make sibling juliet_eval importable.
sys.path.insert(0, str(Path(__file__).parent))
import juliet_eval as je  # noqa: E402

from llm_summary.code_contract.checker import check_entries  # noqa: E402
from llm_summary.code_contract.pass_ import CodeContractPass  # noqa: E402
from llm_summary.db import SummaryDB  # noqa: E402
from llm_summary.driver import BottomUpDriver  # noqa: E402
from llm_summary.llm import build_backend_kwargs, create_backend  # noqa: E402

log = logging.getLogger("code_contract_eval")


@dataclass
class TaskResult:
    yml_file: str
    i_file: str
    cwe: str
    variant: str
    expected_safe: bool
    subproperty: str
    obligations: dict[str, list[str]] = field(default_factory=dict)
    issues: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    predicted_safe: bool | None = None
    correct: bool | None = None
    classification: str = ""
    error: str | None = None
    elapsed_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _reachable_ids(db: SummaryDB, entry_name: str) -> set[int] | None:
    funcs = db.get_function_by_name(entry_name)
    if not funcs:
        return None
    entry_id = funcs[0].id
    if entry_id is None:
        return None
    driver = BottomUpDriver(db, verbose=False)
    graph, _ = driver.build_graph()
    return driver.compute_reachable({entry_id}, graph)




def _has_ir_facts(db: SummaryDB) -> bool:
    cur = db.conn.execute("SELECT 1 FROM function_ir_facts LIMIT 1")
    return cur.fetchone() is not None


def _ensure_cached_db(
    i_path: Path, cache_dir: Path, kamain_bin: str,
    *, rescan: bool, verbose: bool, emit_ir_sidecar: bool,
    data_model: str | None,
) -> Path:
    """Make sure ``cache_dir/functions.db`` exists (with sidecar facts
    when ``emit_ir_sidecar``).

    Builds the cache via ``juliet_eval.phase_scan`` + ``phase_callgraph``.
    When the DB is present but lacks ``function_ir_facts`` rows and
    sidecar emission is enabled, we re-run ``phase_callgraph`` to
    backfill — idempotent because both the call-graph importer and the
    sidecar importer upsert.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "functions.db"
    if rescan and db_path.exists():
        db_path.unlink()
    if not db_path.exists():
        db = SummaryDB(str(db_path))
        try:
            je.phase_scan(i_path, cache_dir, db)
            je.phase_callgraph(
                i_path, cache_dir, db, kamain_bin, verbose,
                emit_ir_sidecar=emit_ir_sidecar,
                data_model=data_model,
            )
        finally:
            db.close()
        return db_path
    db = SummaryDB(str(db_path))
    try:
        if emit_ir_sidecar and not _has_ir_facts(db):
            log.info("  Backfilling sidecar facts in %s", db_path)
            je.phase_callgraph(
                i_path, cache_dir, db, kamain_bin, verbose,
                emit_ir_sidecar=True,
                data_model=data_model,
            )
    finally:
        db.close()
    return db_path


def _run_one_task(
    yml_path: Path, info: dict[str, Any], llm: Any,
    *, cache_dir: Path, kamain_bin: str, svcomp: bool,
    work_dir: Path, summary_dump: dict[str, Any] | None,
    llm_log_dir: Path | None = None,
    rescan: bool = False, verbose: bool = False,
    emit_ir_sidecar: bool = True,
) -> TaskResult:
    stem = yml_path.stem
    expected_safe = bool(info["expected_verdict"])
    variant = ("bad" if "_bad" in stem else "good" if "_good" in stem else "")
    result = TaskResult(
        yml_file=yml_path.name, i_file=info["i_file"],
        cwe=stem.split("---")[0] if "---" in stem else stem.split("_")[0],
        variant=variant, expected_safe=expected_safe,
        subproperty=info["subproperty"],
    )

    # Build/backfill the cached DB (scan + callgraph + sidecar facts),
    # then copy it into the work dir so analysis tables don't leak back
    # into the cache.
    i_path = Path(info["source_path"])
    db_src = _ensure_cached_db(
        i_path, cache_dir, kamain_bin,
        rescan=rescan, verbose=verbose,
        emit_ir_sidecar=emit_ir_sidecar,
        data_model=info.get("data_model"),
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "functions.db"
    shutil.copyfile(db_src, db_path)

    db = SummaryDB(str(db_path))
    t0 = time.time()

    log_fp = None
    if llm_log_dir is not None:
        llm_log_dir.mkdir(parents=True, exist_ok=True)
        log_fp = open(llm_log_dir / f"{stem}.log", "w")

    cc_pass: CodeContractPass | None = None
    try:
        target_ids = _reachable_ids(db, "main")
        if target_ids is None:
            raise RuntimeError("no main() in DB")

        cc_pass = CodeContractPass(
            db=db, model=llm.model, llm=llm, svcomp=svcomp,
            log_fp=log_fp,
        )
        driver = BottomUpDriver(db, verbose=False)
        driver.run([cc_pass], force=True, target_ids=target_ids)

        # Verdict source-of-truth: per-function verify issues collected by
        # the pass (mirrors contract_pipeline.py:run_one_task with
        # verify=True). `check_entries` obligations are kept for inspection.
        oblig = check_entries(db, entries=["main"])
        per_prop: dict[str, list[str]] = {}
        for ob in oblig:
            per_prop.setdefault(ob.property, []).append(ob.predicate)
        result.obligations = per_prop
        result.issues = dict(cc_pass.issues)
        result.predicted_safe = len(cc_pass.issues) == 0
        result.correct = result.predicted_safe == expected_safe
        result.classification = (
            "TN" if expected_safe and result.predicted_safe else
            "FP" if expected_safe and not result.predicted_safe else
            "TP" if not expected_safe and not result.predicted_safe else
            "FN"
        )

        if summary_dump is not None:
            dump: dict[str, Any] = {}
            for fid in target_ids:
                s = db.get_code_contract_summary(fid)
                if s is None:
                    continue
                dump[s.function] = s.to_dict()
            summary_dump[stem] = dump
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
        log.exception("  -> ERROR")
    finally:
        result.elapsed_s = time.time() - t0
        if cc_pass is not None:
            result.llm_calls = cc_pass.calls
            result.input_tokens = cc_pass.input_tokens
            result.output_tokens = cc_pass.output_tokens
        db.close()
        if log_fp is not None:
            log_fp.close()

    return result


def _scoreboard(results: list[TaskResult]) -> dict[str, Any]:
    tp = sum(1 for r in results if r.classification == "TP")
    tn = sum(1 for r in results if r.classification == "TN")
    fp = sum(1 for r in results if r.classification == "FP")
    fn = sum(1 for r in results if r.classification == "FN")
    errors = sum(1 for r in results if r.error)
    total = len(results)
    scored = total - errors
    correct = tp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0)
    return {
        "total": total, "scored": scored, "correct": correct,
        "accuracy": correct / scored if scored else 0,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": errors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yml", default=None, help="Single .yml task file")
    p.add_argument("--set-file", default=None,
                   help="sv-comp .set file (one glob per line)")
    p.add_argument("--benchmarks", default=None,
                   help="Benchmark root directory")
    p.add_argument("--cwe", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--filter", default=None,
                   help="Substring filter on yml stem")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--func-scans-dir", default=je.DEFAULT_FUNC_SCANS,
                   help="Persistent scan cache (functions.db + sidecar/ "
                        "per task). Built on demand if missing.")
    p.add_argument("--kamain-bin", default=je.DEFAULT_KAMAIN,
                   help="KAMain binary used for callgraph + IR sidecar")
    p.add_argument("--rescan", action="store_true",
                   help="Re-run scan + callgraph even if a cached DB exists")
    p.add_argument("--no-ir-sidecar", action="store_true",
                   help="Disable KAMain IR fact sidecar emission. Without "
                        "sidecar facts, feature extraction degrades to the "
                        "regex fallback (baseline mode for A/B comparison).")
    p.add_argument("--rerun-from", default=None, metavar="RESULTS_JSON",
                   help="Rerun only tasks with a given classification from a "
                        "previous results JSON (use with --rerun-class)")
    p.add_argument("--rerun-class", default="FP",
                   help="Classification to rerun (default: FP). Used with "
                        "--rerun-from")
    p.add_argument("--work-root", default=None,
                   help="Where to copy DBs for the run "
                        "(defaults to a tempdir)")
    p.add_argument("--backend", default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--llm-host", default="localhost")
    p.add_argument("--llm-port", type=int, default=None)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--svcomp", action="store_true", default=True,
                   help="Seed sv-comp __VERIFIER_* helpers (default on; "
                        "this script is sv-comp-shaped)")
    p.add_argument("--no-svcomp", dest="svcomp", action="store_false")
    p.add_argument("--output", "-o", default="code_contract_eval.json")
    p.add_argument("--summary-out", default=None,
                   help="Per-function CodeContractSummary dump (JSON) "
                        "for clause-level diffing")
    p.add_argument("--llm-log-dir", default=None,
                   help="Write per-task LLM prompt/response logs here "
                        "(one .log per task, mirrors contract_pipeline.py)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.verbose:
        log.setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.yml:
        info = je.parse_yml(Path(args.yml))
        if info is None:
            log.error("No relevant property in %s", args.yml)
            sys.exit(1)
        tasks = [(Path(args.yml), info)]
    elif args.set_file:
        tasks = je.collect_tasks_from_set_file(Path(args.set_file), args.cwe)
    elif args.benchmarks:
        tasks = je.collect_tasks(Path(args.benchmarks), args.cwe, args.variant)
    else:
        log.error("One of --yml, --set-file, --benchmarks required")
        sys.exit(1)

    if args.filter:
        tasks = [(p, i) for p, i in tasks if args.filter in p.stem]
    if args.rerun_from:
        prior = json.loads(Path(args.rerun_from).read_text())
        keep = {
            r["yml_file"] for r in prior.get("results", [])
            if r.get("classification") == args.rerun_class
        }
        before = len(tasks)
        tasks = [(p, i) for p, i in tasks if p.name in keep]
        log.info("rerun-from %s [%s]: %d → %d tasks",
                 args.rerun_from, args.rerun_class, before, len(tasks))
    if args.limit > 0:
        tasks = tasks[: args.limit]

    log.info("Tasks: %d", len(tasks))

    backend_kwargs = build_backend_kwargs(
        args.backend, llm_host=args.llm_host, llm_port=args.llm_port,
        disable_thinking=args.disable_thinking,
    )
    llm = create_backend(args.backend, model=args.model, **backend_kwargs)
    log.info("Backend: %s  Model: %s", args.backend, llm.model)

    func_scans = Path(args.func_scans_dir)
    work_root = Path(args.work_root) if args.work_root else Path(
        tempfile.mkdtemp(prefix="code_contract_eval_"),
    )
    log.info("Work root: %s", work_root)

    results: list[TaskResult] = []
    summary_dump: dict[str, Any] | None = (
        {} if args.summary_out else None
    )

    for idx, (yml_path, info) in enumerate(tasks):
        stem = yml_path.stem
        cache_dir = func_scans / yml_path.parent.name / stem
        log.info("[%d/%d] %s (expected: %s, subprop: %s)",
                 idx + 1, len(tasks), stem,
                 "safe" if info["expected_verdict"] else "UNSAFE",
                 info["subproperty"])
        r = _run_one_task(
            yml_path, info, llm,
            cache_dir=cache_dir,
            kamain_bin=args.kamain_bin,
            svcomp=args.svcomp,
            work_dir=work_root / yml_path.parent.name / stem,
            summary_dump=summary_dump,
            llm_log_dir=Path(args.llm_log_dir) if args.llm_log_dir else None,
            rescan=args.rescan,
            verbose=args.verbose,
            emit_ir_sidecar=not args.no_ir_sidecar,
        )
        results.append(r)
        if r.error:
            log.error("  -> ERROR %s", r.error)
        else:
            n_issues = sum(
                len(plist)
                for props in r.issues.values()
                for plist in props.values()
            )
            log.info("  -> %s (%d issues, %d obligations, %d calls, %.1fs, "
                     "%dk in + %dk out)",
                     r.classification, n_issues,
                     sum(len(v) for v in r.obligations.values()),
                     r.llm_calls, r.elapsed_s,
                     r.input_tokens // 1000, r.output_tokens // 1000)

    scoreboard = _scoreboard(results)
    log.info("=== Results ===")
    log.info("Total: %(total)d Scored: %(scored)d Correct: %(correct)d "
             "Acc: %(accuracy).1f%%", {**scoreboard,
                                       "accuracy": scoreboard["accuracy"] * 100})
    log.info("TP: %(tp)d TN: %(tn)d FP: %(fp)d FN: %(fn)d Errors: %(errors)d",
             scoreboard)
    log.info("Precision: %.3f Recall: %.3f F1: %.3f",
             scoreboard["precision"], scoreboard["recall"], scoreboard["f1"])

    out = {
        "summary": scoreboard,
        "config": {
            "backend": args.backend, "model": llm.model,
            "approach": "code-contract-v0 (summarize + entry-check)",
            "svcomp": args.svcomp,
        },
        "results": [asdict(r) for r in results],
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.output)

    if args.summary_out and summary_dump is not None:
        Path(args.summary_out).write_text(json.dumps(summary_dump, indent=2))
        log.info("Wrote %s", args.summary_out)


if __name__ == "__main__":
    main()
