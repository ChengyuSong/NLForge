#!/usr/bin/env python3
"""Batch issue validation pipeline across projects.

Full pipeline:
1. Query link_units to find all DBs per project
2. Query each DB for functions with verification issues
3. Triage each issue (llm-summary triage) → verdict JSON
4. Validate triage verdicts (llm-summary gen-harness --validate) → harnesses
5. Run thoroupy on each built harness to find bug-proving paths

Usage:
    python scripts/batch_validate.py --backend llamacpp --llm-host 192.168.1.11 --llm-port 8001 \\
        --ko-clang-path ~/fuzzing/symsan/b3/bin/ko-clang -v
    python scripts/batch_validate.py --filter libpng --skip-run
    python scripts/batch_validate.py --tier 1 --severity high
    python scripts/batch_validate.py --filter zlib --skip-triage  # reuse existing verdicts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.db import SummaryDB
from llm_summary.link_units.pipeline import load_link_units, topo_sort_link_units
from llm_summary.llm import build_backend_kwargs, create_backend
from llm_summary.llm.base import LLMBackend
from llm_summary.models import SafetyIssue
from llm_summary.reflection import ReflectionAgent
from llm_summary.validation_consumer import classify_outcome


class PipelineError(Exception):
    """Raised when --stop-on-error is set and a stage fails."""


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
HARNESSES_DIR = REPO_ROOT / "harnesses"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"
THOROUPY_SCRIPT = SCRIPTS_DIR / "run_thoroupy.sh"


def _load_tier_map() -> dict[str, int]:
    if not GPR_PROJECTS_PATH.exists():
        return {}
    with open(GPR_PROJECTS_PATH) as f:
        projects = json.load(f)
    return {
        p["project_dir"]: p["tier"]
        for p in projects
        if "project_dir" in p
    }


def _find_dbs(project_name: str) -> list[tuple[str, Path]]:
    """Find all (target_name, db_path) for a project."""
    scan_dir = FUNC_SCANS_DIR / project_name
    results: list[tuple[str, Path]] = []

    link_units_path = scan_dir / "link_units.json"
    if link_units_path.exists():
        _, raw_units = load_link_units(link_units_path)
        for u in topo_sort_link_units(raw_units):
            name = u["name"]
            db_str = u.get("db_path")
            db_path = Path(db_str) if db_str else scan_dir / name / "functions.db"
            if db_path.exists():
                results.append((name, db_path))
    else:
        legacy = scan_dir / "functions.db"
        if legacy.exists():
            results.append((project_name, legacy))

    return results


def _find_issues(
    db_path: Path, severity: str | None, *, include_reviewed: bool = False,
) -> list[tuple[str, int]]:
    """Query DB for (function_name, issue_count) with verification issues.

    By default, skips issues that already have a non-pending review status
    (e.g. false_positive, confirmed).  Pass include_reviewed=True to include
    all issues regardless of review status.
    """
    db = SummaryDB(str(db_path))
    results: list[tuple[str, int]] = []
    try:
        for func in db.get_all_functions():
            assert func.id is not None
            vs = db.get_verification_summary_by_function_id(func.id)
            if not vs or not vs.issues:
                continue
            issues = vs.issues
            if severity:
                issues = [i for i in issues if i.severity == severity]
            if not include_reviewed and issues:
                reviews = db.get_issue_reviews(func.id)
                reviewed_fps = {
                    r["issue_fingerprint"]
                    for r in reviews
                    if r["status"] != "pending"
                }
                if reviewed_fps:
                    issues = [
                        i for i in issues
                        if i.fingerprint() not in reviewed_fps
                    ]
            if issues:
                results.append((func.name, len(issues)))
    finally:
        db.close()
    return results


def _find_compile_commands(project_name: str) -> Path | None:
    build_dir = Path("/data/csong/build-artifacts") / project_name
    cc = build_dir / "compile_commands.json"
    return cc if cc.exists() else None


def _find_project_path(project_name: str) -> Path | None:
    src = Path("/data/csong/opensource") / project_name
    return src if src.exists() else None


def _run_cmd(
    cmd: list[str], verbose: bool, timeout: int,
) -> tuple[bool, str, float]:
    """Run a command, return (success, error, duration)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, timeout=timeout,
        )
        duration = time.monotonic() - start
        if result.returncode == 0:
            return True, "", duration
        error = f"exit code {result.returncode}"
        if not verbose and result.stderr:
            error += f"\n{result.stderr[-500:]}"
        return False, error, duration
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s", time.monotonic() - start
    except FileNotFoundError:
        return False, "llm-summary not found (activate venv?)", 0.0


def run_triage(
    db_path: Path,
    func_name: str,
    output_path: Path,
    backend: str,
    model: str | None,
    llm_host: str,
    llm_port: int | None,
    severity: str | None,
    project_path: Path | None,
    verbose: bool,
    timeout: int,
) -> tuple[bool, str, float]:
    """Run llm-summary triage for a single function."""
    cmd = [
        "llm-summary", "triage",
        "--db", str(db_path),
        "--backend", backend,
        "-f", func_name,
        "-o", str(output_path),
    ]
    if model:
        cmd += ["--model", model]
    if llm_host != "localhost":
        cmd += ["--llm-host", llm_host]
    if llm_port is not None:
        cmd += ["--llm-port", str(llm_port)]
    if severity:
        cmd += ["--severity", severity]
    if project_path:
        cmd += ["--project-path", str(project_path)]
    if verbose:
        cmd.append("--verbose")
    return _run_cmd(cmd, verbose, timeout)


def run_gen_harness(
    db_path: Path,
    verdict_path: Path,
    backend: str,
    model: str | None,
    llm_host: str,
    llm_port: int | None,
    ko_clang_path: str | None,
    compile_commands: Path | None,
    project_path: Path | None,
    verbose: bool,
    timeout: int,
    issue_index: int | None = None,
) -> tuple[bool, str, float]:
    """Run llm-summary gen-harness --validate."""
    cmd = [
        "llm-summary", "gen-harness",
        "--db", str(db_path),
        "--backend", backend,
        "--validate", str(verdict_path),
    ]
    if issue_index is not None:
        cmd += ["--issue-index", str(issue_index)]
    if model:
        cmd += ["--model", model]
    if llm_host != "localhost":
        cmd += ["--llm-host", llm_host]
    if llm_port is not None:
        cmd += ["--llm-port", str(llm_port)]
    if ko_clang_path:
        cmd += ["--ko-clang-path", ko_clang_path]
    if compile_commands:
        cmd += ["--compile-commands", str(compile_commands)]
    if project_path:
        cmd += ["--project-path", str(project_path)]
    if verbose:
        cmd.append("--verbose")
    return _run_cmd(cmd, verbose, timeout)


def run_thoroupy(
    binary: Path, plan: Path, verdict: dict, timeout: int,
) -> dict:
    """Run thoroupy on a harness binary and classify the outcome.

    Args:
        binary: Path to .ucsan binary
        plan: Path to plan JSON for thoroupy
        verdict: The verdict dict this harness validates
        timeout: Max seconds for the run
    """
    result: dict = {
        "binary": str(binary),
        "plan": str(plan),
        "success": False,
        "error": None,
        "timing_seconds": 0.0,
    }

    if not THOROUPY_SCRIPT.exists():
        result["error"] = "run_thoroupy.sh not found"
        return result

    start = time.monotonic()
    try:
        subprocess.run(
            ["bash", str(THOROUPY_SCRIPT), str(binary), str(plan)],
            capture_output=True, text=True, timeout=timeout,
        )
        result["timing_seconds"] = round(time.monotonic() - start, 2)

        vr_path = binary.parent / "validation_result.json"
        if vr_path.exists():
            with open(vr_path) as f:
                validation = json.load(f)
            outcome = classify_outcome(verdict, validation)
            result["success"] = True
            result["outcome"] = outcome.to_dict()
        else:
            result["error"] = "no validation_result.json produced"
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout after {timeout}s"
        result["timing_seconds"] = round(time.monotonic() - start, 2)

    return result


def run_reflection(
    verdict: dict,
    outcome: dict,
    db_path: Path,
    harness_dir: Path,
    entry_name: str | None,
    llm: LLMBackend,
    args: argparse.Namespace,
    project_path: Path | None = None,
) -> dict[str, Any] | None:
    """Run reflection agent on a single validation outcome.

    The agent can investigate callee summaries, mark issues as
    FP/confirmed, and correct wrong summaries that cause false positives.

    Returns the reflection result dict, or None on failure.
    """
    func_name = verdict.get("function_name", "unknown")
    idx = verdict.get("issue_index", 0)
    vdir = harness_dir / func_name / f"v{idx}"

    cfg_path = None
    if entry_name:
        candidate = vdir / f"cfg_{entry_name}.txt"
        if candidate.exists():
            cfg_path = str(candidate)

    db = SummaryDB(str(db_path))
    try:
        agent = ReflectionAgent(
            db=db, llm=llm,
            verbose=args.verbose,
            project_path=project_path,
        )
        result = agent.reflect(
            verdict=verdict,
            outcome=outcome,
            cfg_dump_path=cfg_path,
            output_dir=str(vdir),
            entry_name=entry_name,
        )
        return dict(result)
    except Exception as e:
        if args.verbose:
            print(f"        reflection failed: {e}")
        return None
    finally:
        db.close()


def process_target(
    project_name: str,
    target_name: str,
    db_path: Path,
    args: argparse.Namespace,
) -> dict:
    """Process a single link-unit target: find issues → triage → validate → run."""
    result: dict = {
        "target": target_name,
        "db": str(db_path),
        "functions": [],
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    all_issues = _find_issues(
        db_path, args.severity, include_reviewed=args.force,
    )

    # Apply global skip/limit across all functions
    issues: list[tuple[str, int]] = []
    for item in all_issues:
        if args.skip > 0:
            args.skip -= 1
            continue
        if args.limit is not None and args.limit <= 0:
            break
        issues.append(item)
        if args.limit is not None:
            args.limit -= 1

    if not issues:
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    compile_commands = _find_compile_commands(project_name)
    project_path = _find_project_path(project_name)
    harness_dir = HARNESSES_DIR / target_name
    reflect_llm = None  # lazily created when first reflection is needed

    for func_name, issue_count in issues:
        func_result: dict = {
            "function": func_name,
            "issue_count": issue_count,
            "triage": {"success": False, "error": None},
            "validate": {"success": False, "error": None},
            "runs": [],
        }

        verdict_path = harness_dir / f"verdict_{func_name}.json"

        # Triage
        if not args.force and verdict_path.exists() or args.skip_triage:
            if verdict_path.exists():
                func_result["triage"]["success"] = True
                func_result["triage"]["skipped"] = True
                if args.verbose:
                    print(f"      {func_name}: reusing verdict")
            else:
                func_result["triage"]["error"] = "no existing verdict"
                if args.verbose:
                    print(f"      {func_name}: no verdict, skipped")
                result["functions"].append(func_result)
                continue
        else:
            if args.verbose:
                print(f"      {func_name}: triaging {issue_count} issues...")
            verdict_path.parent.mkdir(parents=True, exist_ok=True)
            ok, err, dur = run_triage(
                db_path=db_path,
                func_name=func_name,
                output_path=verdict_path,
                backend=args.backend,
                model=args.model,
                llm_host=args.llm_host,
                llm_port=args.llm_port,
                severity=args.severity,
                project_path=project_path,
                verbose=args.verbose,
                timeout=args.triage_timeout,
            )
            func_result["triage"]["success"] = ok
            func_result["triage"]["error"] = err if not ok else None
            func_result["triage"]["timing_seconds"] = round(dur, 2)

        if not func_result["triage"]["success"]:
            result["functions"].append(func_result)
            if args.stop_on_error:
                err = func_result["triage"].get("error", "unknown")
                raise PipelineError(f"triage failed for {func_name}: {err}")
            continue

        if args.skip_validate:
            result["functions"].append(func_result)
            continue

        # Validate + Run: per-issue loop
        if not verdict_path.exists():
            func_result["validate"]["error"] = "no verdict file"
            result["functions"].append(func_result)
            continue

        with open(verdict_path) as f:
            vdata = json.load(f)
        vlist = vdata if isinstance(vdata, list) else [vdata]

        # Build fingerprint->review lookup for this function
        reviewed_fps: dict[str, str] = {}
        if not args.force:
            db = SummaryDB(str(db_path))
            try:
                funcs = db.get_function_by_name(func_name)
                if funcs:
                    assert funcs[0].id is not None
                    for r in db.get_issue_reviews(funcs[0].id):
                        if r["status"] != "pending":
                            reviewed_fps[r["issue_fingerprint"]] = r["status"]
            finally:
                db.close()

        any_validate_ok = False
        for vi, v in enumerate(vlist):
            idx = v.get("issue_index", vi)
            vdir = harness_dir / func_name / f"v{idx}"

            # Skip already-reviewed issues
            issue_d = v.get("issue", {})
            vi_obj = SafetyIssue(
                location=issue_d.get("location", ""),
                issue_kind=issue_d.get("issue_kind", ""),
                description=issue_d.get("description", ""),
                severity=issue_d.get("severity", "medium"),
                callee=issue_d.get("callee"),
                contract_kind=issue_d.get("contract_kind"),
            )
            fp = vi_obj.fingerprint()
            if fp in reviewed_fps:
                if args.verbose:
                    print(
                        f"      {func_name}#{idx}: "
                        f"already {reviewed_fps[fp]}, skipping"
                    )
                continue

            # --- Per-issue harness generation ---
            has_binary = (
                not args.force
                and vdir.exists()
                and any(vdir.glob("*.ucsan"))
            )
            if has_binary:
                if args.verbose:
                    print(f"      {func_name}#{idx}: reusing harnesses")
            elif args.skip_validate:
                continue
            else:
                if args.verbose:
                    print(
                        f"      {func_name}#{idx}: generating harness..."
                    )
                ok, err, dur = run_gen_harness(
                    db_path=db_path,
                    verdict_path=verdict_path,
                    backend=args.backend,
                    model=args.model,
                    llm_host=args.llm_host,
                    llm_port=args.llm_port,
                    ko_clang_path=args.ko_clang_path,
                    compile_commands=compile_commands,
                    project_path=project_path,
                    verbose=args.verbose,
                    timeout=args.gen_timeout,
                    issue_index=idx,
                )
                if not ok:
                    if args.verbose:
                        print(
                            f"      {func_name}#{idx}: "
                            f"gen-harness failed: {err}"
                        )
                    if args.stop_on_error:
                        raise PipelineError(
                            f"gen-harness failed for "
                            f"{func_name}#{idx}: {err}",
                        )
                    continue
                has_binary = vdir.exists() and any(vdir.glob("*.ucsan"))

            if has_binary:
                any_validate_ok = True

            # --- Per-issue run ---
            if args.skip_run or not has_binary:
                continue

            if not vdir.exists():
                continue
            for binary in sorted(vdir.glob("*.ucsan")):
                plan = vdir / f"plan_{binary.stem}_validation.json"
                if not plan.exists():
                    continue

                # Reuse existing validation result
                vr_path = binary.parent / "validation_result.json"
                if not args.force and vr_path.exists():
                    if args.verbose:
                        print(f"        reusing {binary.name} result")
                    try:
                        with open(vr_path) as f:
                            validation = json.load(f)
                        outcome = classify_outcome(v, validation)
                        run_result: dict = {
                            "binary": str(binary),
                            "plan": str(plan),
                            "success": True,
                            "skipped": True,
                            "error": None,
                            "timing_seconds": 0.0,
                            "outcome": outcome.to_dict(),
                        }
                    except Exception as e:
                        run_result = {
                            "binary": str(binary),
                            "plan": str(plan),
                            "success": False,
                            "skipped": True,
                            "error": f"bad cached result: {e}",
                            "timing_seconds": 0.0,
                        }
                    func_result["runs"].append(run_result)
                else:
                    if args.verbose:
                        print(f"        running {binary.name}...")
                    run_result = run_thoroupy(
                        binary, plan, verdict=v, timeout=args.run_timeout,
                    )
                    func_result["runs"].append(run_result)

                oc = run_result.get("outcome", {})
                outcome_type = oc.get("outcome", "")

                # Reflect on outcomes (agent handles review + summary correction)
                if not args.skip_reflect and outcome_type:
                    if reflect_llm is None:
                        kwargs = build_backend_kwargs(
                            args.backend, args.llm_host, args.llm_port,
                        )
                        reflect_llm = create_backend(
                            args.backend, model=args.model, **kwargs,
                        )
                    entry = Path(binary).stem if binary else None
                    if args.verbose:
                        print(
                            f"        reflecting on {outcome_type}..."
                        )
                    refl = run_reflection(
                        verdict=v,
                        outcome=oc,
                        db_path=db_path,
                        harness_dir=harness_dir,
                        entry_name=entry,
                        llm=reflect_llm,
                        args=args,
                        project_path=project_path,
                    )
                    if refl:
                        run_result["reflection"] = refl
                        if args.verbose:
                            hyp = refl.get("hypothesis", "?")
                            conf = refl.get("confidence", "?")
                            act = refl.get("action", "?")
                            print(
                                f"        → {hyp} ({conf}) "
                                f"action={act}"
                            )

        func_result["validate"]["success"] = any_validate_ok
        result["functions"].append(func_result)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def process_project(
    project_name: str, args: argparse.Namespace,
) -> dict:
    """Process all targets in a project."""
    result: dict = {
        "project": project_name,
        "targets": [],
        "success": False,
        "error": None,
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    dbs = _find_dbs(project_name)
    if not dbs:
        result["error"] = "no functions.db"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    for target_name, db_path in dbs:
        if args.verbose:
            print(f"    [{target_name}]")
        target_result = process_target(
            project_name, target_name, db_path, args,
        )
        result["targets"].append(target_result)

    n_funcs = sum(len(t["functions"]) for t in result["targets"])
    result["success"] = n_funcs > 0
    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _format_result(result: dict) -> str:
    if result["error"] and not result.get("targets"):
        return f"SKIP ({result['error']})"

    n_funcs = sum(
        len(t["functions"]) for t in result.get("targets", [])
    )
    n_issues = sum(
        f["issue_count"]
        for t in result.get("targets", [])
        for f in t["functions"]
    )
    n_triage = sum(
        1 for t in result.get("targets", [])
        for f in t["functions"]
        if f["triage"]["success"]
    )
    n_validate = sum(
        1 for t in result.get("targets", [])
        for f in t["functions"]
        if f["validate"]["success"]
    )
    n_runs = sum(
        len(f.get("runs", []))
        for t in result.get("targets", [])
        for f in t["functions"]
    )

    parts = [f"{n_funcs} funcs ({n_issues} issues)"]
    parts.append(f"triage:{n_triage}")
    parts.append(f"validate:{n_validate}")
    if n_runs:
        parts.append(f"runs:{n_runs}")
    parts.append(f"{result['timing_seconds']}s")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch issue validation pipeline: "
        "find issues → triage → gen-harness → thoroupy",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        default="llamacpp",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--llm-host", type=str, default="localhost")
    parser.add_argument("--llm-port", type=int, default=None)
    parser.add_argument("--ko-clang-path", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--severity", default=None,
        choices=["high", "medium", "low"],
        help="Only process issues of this severity",
    )
    parser.add_argument(
        "--triage-timeout", type=int, default=600,
        help="Per-function triage timeout (seconds)",
    )
    parser.add_argument(
        "--gen-timeout", type=int, default=600,
        help="Per-verdict gen-harness timeout (seconds)",
    )
    parser.add_argument(
        "--run-timeout", type=int, default=60,
        help="Per-harness thoroupy timeout (seconds)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate artifacts even if they already exist",
    )
    parser.add_argument(
        "--skip-triage", action="store_true",
        help="Reuse existing verdict files, skip triage step",
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Only triage, skip harness generation and runs",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Generate harnesses but skip thoroupy runs",
    )
    parser.add_argument(
        "--skip-reflect", action="store_true",
        help="Skip reflection after thoroupy runs",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort the pipeline on the first triage or validation failure",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process projects matching this substring",
    )
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip the first N functions with issues",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N functions with issues",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    # Find projects with func-scans
    projects = sorted(
        d.name for d in FUNC_SCANS_DIR.iterdir()
        if d.is_dir() and (
            (d / "functions.db").exists()
            or (d / "link_units.json").exists()
        )
    )

    if args.filter:
        before = len(projects)
        projects = [
            p for p in projects if args.filter.lower() in p.lower()
        ]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    if args.tier is not None:
        tier_map = _load_tier_map()
        if tier_map:
            before = len(projects)
            projects = [
                p for p in projects if tier_map.get(p) == args.tier
            ]
            print(
                f"Tier {args.tier}: {len(projects)}/{before} projects",
            )

    if not projects:
        print("No projects found")
        sys.exit(0)

    stages = []
    if not args.skip_triage:
        stages.append("triage")
    if not args.skip_validate:
        stages.append("validate")
    if not args.skip_run:
        stages.append("run")
    if not args.skip_reflect:
        stages.append("reflect")

    print(f"\nProcessing {len(projects)} projects")
    print(
        f"Backend: {args.backend}"
        + (f" ({args.model})" if args.model else ""),
    )
    if args.ko_clang_path:
        print(f"ko-clang: {args.ko_clang_path}")
    print(f"Stages: {' → '.join(stages)}")
    print()

    all_results = []
    stopped_early = False
    for i, project_name in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] {project_name}...", end=" ", flush=True)
        try:
            result = process_project(project_name, args)
        except PipelineError as e:
            print(f"\nSTOPPED: {e}")
            stopped_early = True
            break
        all_results.append(result)
        print(_format_result(result))

    # Aggregate
    total_funcs = sum(
        len(t["functions"])
        for r in all_results
        for t in r.get("targets", [])
    )
    total_issues = sum(
        f["issue_count"]
        for r in all_results
        for t in r.get("targets", [])
        for f in t["functions"]
    )
    total_triage = sum(
        1 for r in all_results
        for t in r.get("targets", [])
        for f in t["functions"]
        if f["triage"]["success"]
    )
    total_validate = sum(
        1 for r in all_results
        for t in r.get("targets", [])
        for f in t["functions"]
        if f["validate"]["success"]
    )
    all_runs = [
        run
        for r in all_results
        for t in r.get("targets", [])
        for f in t["functions"]
        for run in f.get("runs", [])
    ]
    total_runs = len(all_runs)
    outcome_counts: dict[str, int] = {}
    for run in all_runs:
        oc = run.get("outcome", {}).get("outcome", "error")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects:          {len(all_results)}")
    print(f"  Functions:         {total_funcs} ({total_issues} issues)")
    print(f"  Triaged:           {total_triage}")
    print(f"  Validated:         {total_validate}")
    print(f"  Thoroupy runs:     {total_runs}")
    for oc, cnt in sorted(outcome_counts.items()):
        print(f"    {oc}: {cnt}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"validate_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "backend": args.backend,
                "model": args.model,
                "stages": stages,
                "projects": all_results,
                "totals": {
                    "projects": len(all_results),
                    "functions": total_funcs,
                    "issues": total_issues,
                    "triaged": total_triage,
                    "validated": total_validate,
                    "runs": total_runs,
                    "outcomes": outcome_counts,
                },
            },
            f,
            indent=2,
        )
    print(f"\nReport written to: {output_path}")
    if stopped_early:
        sys.exit(1)


if __name__ == "__main__":
    main()
