#!/usr/bin/env python3
"""Batch verification pass across all projects.

Runs `llm-summary summarize --type verify` on each project's DB(s).
For projects with link_units.json, runs verify on each per-target DB.
For legacy projects, runs on func-scans/<project>/functions.db.

Verify requires allocation+free+init+memsafe summaries to be present.
Run batch_summarize.py first.

Usage:
    python scripts/batch_verify.py --backend gemini --verbose
    python scripts/batch_verify.py --filter zlib --backend claude
    python scripts/batch_verify.py --tier 1 --backend gemini --force
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.link_units.pipeline import load_link_units, topo_sort_link_units

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"


def _load_tier_map() -> dict[str, int]:
    if not GPR_PROJECTS_PATH.exists():
        return {}
    with open(GPR_PROJECTS_PATH) as f:
        projects = json.load(f)
    return {p["project_dir"]: p["tier"] for p in projects if "project_dir" in p}


def run_verify(
    db_path: Path,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    verbose: bool,
    timeout: int,
) -> tuple[bool, str, float]:
    """Invoke llm-summary summarize --type verify. Returns (success, error, duration)."""
    cmd = ["llm-summary", "summarize", "--db", str(db_path), "--backend", backend, "--type", "verify"]
    if model:
        cmd += ["--model", model]
    if force:
        cmd.append("--force")
    if incremental:
        cmd.append("--incremental")
    if llm_port is not None:
        cmd += ["--llm-port", str(llm_port)]
    if llm_host != "localhost":
        cmd += ["--llm-host", llm_host]
    if log_llm:
        cmd += ["--log-llm", str(log_llm)]
    if verbose:
        cmd.append("--verbose")

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=timeout)
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


def process_project(
    project_name: str,
    func_scans_dir: Path,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    verbose: bool,
    timeout: int,
) -> dict:
    """Run verify on a single project. Returns result dict."""
    scan_dir = func_scans_dir / project_name
    result = {
        "project": project_name,
        "success": False,
        "error": None,
        "targets": [],
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    # Link-unit-aware mode
    link_units_path = scan_dir / "link_units.json"
    if link_units_path.exists():
        lu_data, raw_units = load_link_units(link_units_path)
        if not raw_units:
            result["error"] = "link_units.json has no targets"
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            return result

        link_units = topo_sort_link_units(raw_units)
        errors = []

        for lu in link_units:
            target = lu["name"]
            db_str = lu.get("db_path")
            db_path = Path(db_str) if db_str else scan_dir / target / "functions.db"

            target_result: dict = {
                "target": target,
                "success": False,
                "error": None,
                "timing_seconds": 0.0,
            }

            if not db_path.exists():
                target_result["error"] = "no_functions_db"
                result["targets"].append(target_result)
                errors.append(f"{target}: no_functions_db")
                continue

            if verbose:
                print(f"    [{target}] verify...")

            ok, err, dur = run_verify(
                db_path=db_path,
                backend=backend,
                model=model,
                force=force,
                incremental=incremental,
                llm_host=llm_host,
                llm_port=llm_port,
                log_llm=log_llm,
                verbose=verbose,
                timeout=timeout,
            )
            target_result["success"] = ok
            target_result["error"] = err if not ok else None
            target_result["timing_seconds"] = round(dur, 2)
            result["targets"].append(target_result)
            if not ok:
                errors.append(f"{target}: {err}")

        result["success"] = len(errors) == 0
        result["error"] = "; ".join(errors) if errors else None
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    # Legacy single-DB mode
    db_path = scan_dir / "functions.db"
    if not db_path.exists():
        result["error"] = "no_functions_db"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    ok, err, dur = run_verify(
        db_path=db_path,
        backend=backend,
        model=model,
        force=force,
        incremental=incremental,
        llm_host=llm_host,
        llm_port=llm_port,
        log_llm=log_llm,
        verbose=verbose,
        timeout=timeout,
    )
    result["success"] = ok
    result["error"] = err if not ok else None
    result["timing_seconds"] = round(dur, 2)
    return result


def _format_result(result: dict) -> str:
    if result["error"] and not result.get("targets"):
        return f"SKIP ({result['error']})"
    if result.get("targets"):
        n = len(result["targets"])
        ok = sum(1 for t in result["targets"] if t["success"])
        failed = [t["target"] for t in result["targets"] if not t["success"]]
        s = f"{ok}/{n} targets ({result['timing_seconds']}s)"
        if failed:
            s += f" FAILED: {', '.join(failed)}"
        return s
    if result["success"]:
        return f"ok ({result['timing_seconds']}s)"
    return f"FAILED ({result['error']})"


def main():
    parser = argparse.ArgumentParser(description="Batch verify pass for all projects")
    parser.add_argument(
        "--backend", choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        default="gemini", help="LLM backend",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("-f", "--force", action="store_true", help="Re-verify even if cached")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-verify functions with stale callee summaries")
    parser.add_argument("--llm-host", type=str, default="localhost")
    parser.add_argument("--llm-port", type=int, default=None)
    parser.add_argument("--log-llm", type=Path, default=None, help="Log LLM calls to file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--timeout", type=int, default=7200, help="Per-project timeout in seconds",
    )
    parser.add_argument(
        "--func-scans-dir", type=Path, default=FUNC_SCANS_DIR,
        help="Root of func-scans directory",
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Only process projects matching this substring")
    parser.add_argument("--tier", type=int, default=None,
                        help="Only process projects with this tier")
    parser.add_argument("--skip-list", type=str, default=None,
                        help="File with project names to skip (one per line)")
    parser.add_argument("--success-list", type=str, default=None,
                        help="Append successfully verified project names to this file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=None, help="Limit to at most N projects")
    args = parser.parse_args()

    func_scans_dir = args.func_scans_dir

    # Find projects
    projects = sorted(
        d.name for d in func_scans_dir.iterdir()
        if d.is_dir() and (
            (d / "functions.db").exists() or (d / "link_units.json").exists()
        )
    )

    if args.filter:
        before = len(projects)
        projects = [p for p in projects if args.filter.lower() in p.lower()]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    if args.tier is not None:
        tier_map = _load_tier_map()
        if not tier_map:
            print(f"Warning: {GPR_PROJECTS_PATH} not found, --tier ignored")
        else:
            before = len(projects)
            projects = [p for p in projects if tier_map.get(p) == args.tier]
            print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    if args.skip_list:
        skip_path = Path(args.skip_list)
        if not skip_path.exists():
            print(f"Error: skip list not found: {args.skip_list}")
            sys.exit(1)
        skip_names = {l.strip() for l in skip_path.read_text().splitlines() if l.strip() and not l.startswith("#")}
        before = len(projects)
        projects = [p for p in projects if p not in skip_names]
        print(f"Skip list: skipped {before - len(projects)}/{before} projects")

    if args.limit is not None:
        projects = projects[: args.limit]

    print(f"\nProcessing {len(projects)} projects")
    print(f"Backend: {args.backend}" + (f" ({args.model})" if args.model else ""))
    print()

    all_results = []
    success_list_path = Path(args.success_list) if args.success_list else None

    for i, project_name in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] {project_name}...", end=" ", flush=True)
        result = process_project(
            project_name=project_name,
            func_scans_dir=func_scans_dir,
            backend=args.backend,
            model=args.model,
            force=args.force,
            incremental=args.incremental,
            llm_host=args.llm_host,
            llm_port=args.llm_port,
            log_llm=args.log_llm,
            verbose=args.verbose,
            timeout=args.timeout,
        )
        all_results.append(result)
        print(_format_result(result))

        if result["success"] and success_list_path:
            with open(success_list_path, "a") as f:
                f.write(project_name + "\n")

    # Aggregate
    n_ok = sum(1 for r in all_results if r["success"])
    n_fail = len(all_results) - n_ok
    n_targets_ok = sum(
        sum(1 for t in r.get("targets", []) if t["success"])
        for r in all_results
    )
    n_targets_total = sum(len(r.get("targets", [])) for r in all_results)

    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects verified: {n_ok}")
    print(f"  Projects failed:   {n_fail}")
    if n_targets_total:
        print(f"  Targets verified:  {n_targets_ok}/{n_targets_total}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"verify_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "backend": args.backend,
            "model": args.model,
            "projects": all_results,
            "totals": {
                "projects_verified": n_ok,
                "projects_failed": n_fail,
                "targets_verified": n_targets_ok,
                "targets_total": n_targets_total,
            },
        }, f, indent=2)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
