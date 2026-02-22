#!/usr/bin/env python3
"""Batch summarization script for all four summary passes (allocation, free, init, memsafe).

For each project in gpr_projects.json that has func-scans/<project>/functions.db,
this script runs two sequential invocations:

  1. llm-summary summarize --type allocation --type free --type init
  2. llm-summary summarize --type memsafe

memsafe is run as a separate invocation so that all post-condition summaries
(allocation, free, init) are fully committed to the DB across all functions
before memsafe begins.

Per-project allocator_candidates.json (from func-scans/<project>/) is used automatically
if present and no explicit --allocator-file is given.

Usage:
    python scripts/batch_summarize.py --backend claude --verbose
    python scripts/batch_summarize.py --tier 1 --backend ollama --model qwen3-coder:30b
    python scripts/batch_summarize.py --filter libpng --backend claude --force
    python scripts/batch_summarize.py --skip-list done.txt --success-list done.txt --backend claude
    python scripts/batch_summarize.py --types allocation free --backend claude
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.callgraph_import import CallGraphImporter
from llm_summary.db import SummaryDB
from llm_summary.link_units.pipeline import (
    build_output_index,
    load_link_units,
    resolve_dep_db_paths,
    topo_sort_link_units,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"

ALL_TYPES = ["allocation", "free", "init", "memsafe"]


def run_summarize(
    db_path: Path,
    summary_types: list[str],
    backend: str,
    model: str | None,
    force: bool,
    allocator_file: Path | None,
    deallocator_file: Path | None,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    init_stdlib: bool,
    verbose: bool,
    timeout: int,
) -> tuple[bool, str, float]:
    """Invoke llm-summary summarize for a single project. Returns (success, error, duration)."""
    cmd = ["llm-summary", "summarize", "--db", str(db_path), "--backend", backend]

    for t in summary_types:
        cmd += ["--type", t]

    if model:
        cmd += ["--model", model]
    if force:
        cmd.append("--force")
    if allocator_file and allocator_file.exists():
        cmd += ["--allocator-file", str(allocator_file)]
    if deallocator_file and deallocator_file.exists():
        cmd += ["--deallocator-file", str(deallocator_file)]
    if llm_port is not None:
        cmd += ["--llm-port", str(llm_port)]
    if llm_host != "localhost":
        cmd += ["--llm-host", llm_host]
    if log_llm:
        cmd += ["--log-llm", str(log_llm)]
    if init_stdlib:
        cmd.append("--init-stdlib")
    if verbose:
        cmd.append("--verbose")

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=timeout,
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
        return False, "llm-summary not found (activate venv?)", time.monotonic() - start
    except Exception as e:
        return False, str(e), time.monotonic() - start


PASS1_TYPES = ["allocation", "free", "init"]
MEMSAFE_TYPE = "memsafe"


def run_import_dep_summaries(
    db_path: Path,
    dep_db_paths: list[Path],
    force: bool,
    verbose: bool,
) -> tuple[bool, str]:
    """Invoke llm-summary import-dep-summaries. Returns (success, error)."""
    if not dep_db_paths:
        return True, ""
    cmd = ["llm-summary", "import-dep-summaries", "--db", str(db_path)]
    for dep in dep_db_paths:
        cmd += ["--dep-db", str(dep)]
    if force:
        cmd.append("--force")
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=300)
        if result.returncode == 0:
            return True, ""
        error = f"exit code {result.returncode}"
        if not verbose and result.stderr:
            error += f"\n{result.stderr[-300:]}"
        return False, error
    except subprocess.TimeoutExpired:
        return False, "import-dep-summaries timeout"
    except FileNotFoundError:
        return False, "llm-summary not found (activate venv?)"
    except Exception as e:
        return False, str(e)


def _summarize_target(
    target: str,
    db_path: Path,
    summary_types: list[str],
    backend: str,
    model: str | None,
    force: bool,
    allocator_file: Path | None,
    deallocator_file: Path | None,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    init_stdlib: bool,
    verbose: bool,
    timeout: int,
) -> tuple[bool, str, float]:
    """Run Pass 1 (allocation+free+init) and Pass 2 (memsafe) for one DB.

    Returns (success, error, total_duration).
    """
    total_duration = 0.0
    pass1_types = [t for t in summary_types if t in PASS1_TYPES]
    run_memsafe = MEMSAFE_TYPE in summary_types

    if pass1_types:
        if verbose:
            print(f"    [{target}] Pass 1: {pass1_types}")
        ok, err, dur = run_summarize(
            db_path=db_path,
            summary_types=pass1_types,
            backend=backend,
            model=model,
            force=force,
            allocator_file=allocator_file,
            deallocator_file=deallocator_file,
            llm_host=llm_host,
            llm_port=llm_port,
            log_llm=log_llm,
            init_stdlib=init_stdlib,
            verbose=verbose,
            timeout=timeout,
        )
        total_duration += dur
        if not ok:
            return False, f"pass1 failed: {err}", total_duration

    if run_memsafe:
        if verbose:
            print(f"    [{target}] Pass 2: memsafe")
        ok, err, dur = run_summarize(
            db_path=db_path,
            summary_types=[MEMSAFE_TYPE],
            backend=backend,
            model=model,
            force=force,
            allocator_file=None,
            deallocator_file=None,
            llm_host=llm_host,
            llm_port=llm_port,
            log_llm=log_llm,
            init_stdlib=False,
            verbose=verbose,
            timeout=timeout,
        )
        total_duration += dur
        if not ok:
            return False, f"memsafe failed: {err}", total_duration

    return True, "", total_duration


def process_project_link_units(
    project_name: str,
    link_units_path: Path,
    func_scans_dir: Path,
    summary_types: list[str],
    backend: str,
    model: str | None,
    force: bool,
    allocator_file: Path | None,
    deallocator_file: Path | None,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    init_stdlib: bool,
    verbose: bool,
    timeout: int,
) -> dict:
    """Summarize a link-unit-aware project (has link_units.json).

    Processes targets in topological order (deps before dependents).
    For each target:
      1. import-dep-summaries from intra-project dep DBs
      2. summarize (allocation+free+init, then memsafe)
    """
    result = {
        "project": project_name,
        "types": summary_types,
        "success": False,
        "error": None,
        "targets": [],
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    lu_data, raw_units = load_link_units(link_units_path)
    if not raw_units:
        result["error"] = "link_units.json has no targets"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    link_units = topo_sort_link_units(raw_units)
    by_output = build_output_index(link_units)
    project_scan_dir = func_scans_dir / project_name

    # Per-project allocator file (shared across targets)
    effective_allocator = allocator_file
    if effective_allocator is None:
        candidate = project_scan_dir / "allocator_candidates.json"
        if candidate.exists():
            effective_allocator = candidate
            if verbose:
                print(f"    Using allocator file: {candidate.name}")

    target_errors = []

    for lu in link_units:
        target = lu["name"]
        target_start = time.monotonic()

        # Resolve DB path
        db_str = lu.get("db_path")
        db_path = Path(db_str) if db_str else project_scan_dir / target / "functions.db"

        target_result: dict = {
            "target": target,
            "success": False,
            "error": None,
            "timing_seconds": 0.0,
        }

        if not db_path.exists():
            target_result["error"] = f"no_functions_db"
            target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
            result["targets"].append(target_result)
            target_errors.append(f"{target}: no_functions_db")
            continue

        # Import dep summaries from intra-project deps
        dep_db_paths = resolve_dep_db_paths(lu, by_output, project_scan_dir)
        if dep_db_paths:
            if verbose:
                names = [d.parent.name for d in dep_db_paths]
                print(f"    [{target}] Importing dep summaries from: {', '.join(names)}")
            ok, err = run_import_dep_summaries(
                db_path=db_path,
                dep_db_paths=dep_db_paths,
                force=force,
                verbose=verbose,
            )
            if not ok:
                target_result["error"] = f"import_dep_summaries failed: {err}"
                target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
                result["targets"].append(target_result)
                target_errors.append(f"{target}: {target_result['error']}")
                continue

        # Summarize
        ok, err, dur = _summarize_target(
            target=target,
            db_path=db_path,
            summary_types=summary_types,
            backend=backend,
            model=model,
            force=force,
            allocator_file=effective_allocator,
            deallocator_file=deallocator_file,
            llm_host=llm_host,
            llm_port=llm_port,
            log_llm=log_llm,
            init_stdlib=init_stdlib,
            verbose=verbose,
            timeout=timeout,
        )
        target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
        if not ok:
            target_result["error"] = err
            result["targets"].append(target_result)
            target_errors.append(f"{target}: {err}")
            continue

        target_result["success"] = True
        result["targets"].append(target_result)

    result["success"] = len(target_errors) == 0
    result["error"] = "; ".join(target_errors) if target_errors else None
    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def process_project(
    project_name: str,
    func_scans_dir: Path,
    summary_types: list[str],
    backend: str,
    model: str | None,
    force: bool,
    allocator_file: Path | None,
    deallocator_file: Path | None,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    init_stdlib: bool,
    verbose: bool,
    timeout: int,
) -> dict:
    """Process summarization for a single project. Returns a result dict.

    When link_units.json is present, delegates to process_project_link_units
    for per-target summarization in dependency order.

    Otherwise runs two sequential invocations:
      1. allocation + free + init  (post-condition passes)
      2. memsafe                   (pre-condition pass, needs full DB from pass 1)
    """
    scan_dir = func_scans_dir / project_name

    # Auto-route to link-unit-aware mode when link_units.json is present
    link_units_path = scan_dir / "link_units.json"
    if link_units_path.exists():
        if verbose:
            print(f"    link_units.json found — processing per target")
        return process_project_link_units(
            project_name=project_name,
            link_units_path=link_units_path,
            func_scans_dir=func_scans_dir,
            summary_types=summary_types,
            backend=backend,
            model=model,
            force=force,
            allocator_file=allocator_file,
            deallocator_file=deallocator_file,
            llm_host=llm_host,
            llm_port=llm_port,
            log_llm=log_llm,
            init_stdlib=init_stdlib,
            verbose=verbose,
            timeout=timeout,
        )

    # --- Legacy single-DB mode ---
    result = {
        "project": project_name,
        "types": summary_types,
        "success": False,
        "error": None,
        "timing_seconds": 0.0,
    }

    db_path = scan_dir / "functions.db"
    if not db_path.exists():
        result["error"] = "no_functions_db"
        return result

    # Ensure call graph is present; if not, attempt import from callgraph.json
    try:
        con = sqlite3.connect(db_path)
        (edge_count,) = con.execute("SELECT COUNT(*) FROM call_edges").fetchone()
        con.close()
    except Exception as e:
        result["error"] = f"db_read_failed: {e}"
        return result

    if edge_count == 0:
        callgraph_json = scan_dir / "callgraph.json"
        if not callgraph_json.exists():
            result["error"] = "no_call_edges and no callgraph.json to import"
            return result
        if verbose:
            print(f"    No call edges found, importing from {callgraph_json.name}")
        try:
            db = SummaryDB(str(db_path))
            importer = CallGraphImporter(db, verbose=verbose)
            stats = importer.import_json(callgraph_json, clear_existing=True)
            db.close()
            if verbose:
                print(f"    Imported {stats.edges_imported} edges ({stats.direct_edges} direct, {stats.indirect_edges} indirect)")
            if stats.edges_imported == 0:
                result["error"] = "callgraph.json imported but contains no edges"
                return result
        except Exception as e:
            result["error"] = f"callgraph_import_failed: {e}"
            return result

    # Auto-detect per-project allocator file when none is specified globally
    effective_allocator = allocator_file
    if effective_allocator is None:
        candidate = scan_dir / "allocator_candidates.json"
        if candidate.exists():
            effective_allocator = candidate
            if verbose:
                print(f"    Using allocator file: {candidate}")

    ok, err, total_duration = _summarize_target(
        target=project_name,
        db_path=db_path,
        summary_types=summary_types,
        backend=backend,
        model=model,
        force=force,
        allocator_file=effective_allocator,
        deallocator_file=deallocator_file,
        llm_host=llm_host,
        llm_port=llm_port,
        log_llm=log_llm,
        init_stdlib=init_stdlib,
        verbose=verbose,
        timeout=timeout,
    )
    result["success"] = ok
    result["error"] = err if not ok else None
    result["timing_seconds"] = round(total_duration, 2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch summarization (allocation, free, init, memsafe) for all projects"
    )
    parser.add_argument(
        "--projects-json", type=Path, default=GPR_PROJECTS_PATH,
        help="Path to gpr_projects.json",
    )
    parser.add_argument(
        "--func-scans-dir", type=Path, default=FUNC_SCANS_DIR,
        help="Directory containing func-scans/<project>/functions.db",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        default="claude",
        help="LLM backend to use (default: claude)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name override",
    )
    parser.add_argument(
        "--types", nargs="+",
        choices=ALL_TYPES,
        default=ALL_TYPES,
        metavar="TYPE",
        help=f"Summary types to run (default: all). Choices: {ALL_TYPES}",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-summarize even if cached",
    )
    parser.add_argument(
        "--allocator-file", type=Path, default=None,
        help="Global allocator JSON file. If not given, uses per-project allocator_candidates.json if present.",
    )
    parser.add_argument(
        "--deallocator-file", type=Path, default=None,
        help="Global deallocator JSON file (for free pass)",
    )
    parser.add_argument(
        "--llm-host", type=str, default="localhost",
        help="Hostname for local LLM backends (default: localhost)",
    )
    parser.add_argument(
        "--llm-port", type=int, default=None,
        help="Port for local LLM backends",
    )
    parser.add_argument(
        "--log-llm", type=Path, default=None,
        help="Log all LLM prompts and responses to this file",
    )
    parser.add_argument(
        "--init-stdlib", action="store_true",
        help="Auto-populate stdlib summaries before summarizing each project",
    )
    parser.add_argument(
        "--timeout", type=int, default=86400,
        help="Per-project timeout in seconds (default: 86400 = 24h)",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only process projects with this tier",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process projects matching this substring (case-insensitive)",
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip first N projects",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of projects to process",
    )
    parser.add_argument(
        "--skip-list", type=Path, default=None,
        help="File with project directory names to skip (one per line)",
    )
    parser.add_argument(
        "--success-list", type=Path, default=None,
        help="Output file for successful project names (append mode)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON report file (default: summarize_report_<timestamp>.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.projects_json.exists():
        print(f"Error: {args.projects_json} not found")
        sys.exit(1)

    if not args.func_scans_dir.exists():
        print(f"Error: {args.func_scans_dir} not found")
        sys.exit(1)

    with open(args.projects_json) as f:
        projects = json.load(f)

    print(f"Loaded {len(projects)} projects from {args.projects_json}")

    # Keep only projects with a known project_dir
    projects = [p for p in projects if p.get("project_dir")]
    print(f"Projects with project_dir: {len(projects)}")

    if args.tier is not None:
        before = len(projects)
        projects = [p for p in projects if p.get("tier") == args.tier]
        print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    if args.filter:
        filter_str = args.filter.lower()
        before = len(projects)
        projects = [p for p in projects if filter_str in p["project_dir"].lower()]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    skip_set: set[str] = set()
    if args.skip_list and args.skip_list.exists():
        with open(args.skip_list) as f:
            skip_set = {
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            }
        before = len(projects)
        projects = [p for p in projects if p["project_dir"] not in skip_set]
        print(f"Skip list: skipped {before - len(projects)}/{before} projects")

    if args.skip > 0:
        projects = projects[args.skip:]
        print(f"Skipped first {args.skip}, {len(projects)} remaining")

    if args.limit:
        projects = projects[:args.limit]
        print(f"Limited to {args.limit} projects")

    # Only keep projects that have a functions.db
    eligible = [
        p for p in projects
        if (args.func_scans_dir / p["project_dir"] / "functions.db").exists()
    ]
    skipped_no_db = len(projects) - len(eligible)
    if skipped_no_db:
        print(f"Skipped {skipped_no_db} projects with no functions.db")
    projects = eligible

    print(f"\nProcessing {len(projects)} projects")
    print(f"Backend: {args.backend}" + (f" ({args.model})" if args.model else ""))
    print(f"Types: {args.types}")
    print(f"Func-scans dir: {args.func_scans_dir}")
    print()

    all_results = []
    succeeded = 0
    failed = 0

    for i, project in enumerate(projects, 1):
        project_dir = project["project_dir"]
        print(f"[{i}/{len(projects)}] {project_dir}...", end=" ", flush=True)

        result = process_project(
            project_name=project_dir,
            func_scans_dir=args.func_scans_dir,
            summary_types=args.types,
            backend=args.backend,
            model=args.model,
            force=args.force,
            allocator_file=args.allocator_file,
            deallocator_file=args.deallocator_file,
            llm_host=args.llm_host,
            llm_port=args.llm_port,
            log_llm=args.log_llm,
            init_stdlib=args.init_stdlib,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        all_results.append(result)

        if result["success"]:
            print(f"OK ({result['timing_seconds']}s)")
            succeeded += 1
            if args.success_list:
                with open(args.success_list, "a") as f:
                    f.write(f"{project_dir}\n")
        else:
            error_preview = (result["error"] or "")[:80]
            print(f"FAIL ({error_preview})")
            failed += 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(projects)}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "model": args.model,
        "types": args.types,
        "force": args.force,
        "projects": all_results,
        "totals": {
            "succeeded": succeeded,
            "failed": failed,
            "total": len(projects),
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"summarize_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
