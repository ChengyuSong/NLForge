#!/usr/bin/env python3
"""Batch code-contract summarization for projects in gpr_projects.json.

For each project with func-scans/<project>/functions.db (or a
link_units.json with per-target DBs), runs:

  llm-summary summarize --type code-contract --db <target_db> ...

and optionally:

  llm-summary check --db <target_db> --output <target_dir>/check_report.json

The code-contract pass subsumes allocation/free/init/memsafe/leak/
intoverflow + per-function verification (interleaved). When
`--init-stdlib` is set (default), `llm-summary init-stdlib --db <target>`
runs first to apply pre-generated libc contracts from the global
stdlib_cache.

Usage:
    python scripts/batch_code_contract.py --filter libpng \\
        --backend claude --model claude-haiku-4-5@20251001 -v
    python scripts/batch_code_contract.py --filter libpng --check -v
    python scripts/batch_code_contract.py --tier 1 --backend claude \\
        --skip-list done_cc.txt --success-list done_cc.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpr_utils import find_project_dir, get_artifact_name

from llm_summary.callgraph_import import CallGraphImporter
from llm_summary.db import SummaryDB
from llm_summary.link_units.pipeline import (
    build_output_index,
    detect_bc_alias_relations,
    load_link_units,
    resolve_dep_db_paths,
    topo_sort_link_units,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"


# ---------------------------------------------------------------------------
# llm-summary subprocess wrappers
# ---------------------------------------------------------------------------

def run_summarize_code_contract(
    db_path: Path,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    jobs: int,
    verbose: bool,
    timeout: int,
    verify_only: bool = False,
) -> tuple[bool, str, float]:
    """Invoke `llm-summary summarize --type code-contract` for one DB."""
    cmd = [
        "llm-summary", "summarize",
        "--db", str(db_path),
        "--backend", backend,
        "--type", "code-contract",
    ]
    if model:
        cmd += ["--model", model]
    if force:
        cmd.append("--force")
    if incremental:
        cmd.append("--incremental")
    if verify_only:
        cmd.append("--verify-only")
    if llm_port is not None:
        cmd += ["--llm-port", str(llm_port)]
    if llm_host != "localhost":
        cmd += ["--llm-host", llm_host]
    if log_llm:
        cmd += ["--log-llm", str(log_llm)]
    if jobs > 1:
        cmd += ["-j", str(jobs)]
    if verbose:
        cmd.append("--verbose")

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
        return False, "llm-summary not found (activate venv?)", time.monotonic() - start
    except Exception as e:
        return False, str(e), time.monotonic() - start


def run_init_stdlib(
    db_path: Path,
    verbose: bool = False,
    timeout: int = 300,
) -> tuple[bool, str]:
    """Invoke `llm-summary init-stdlib --db <db>` to apply pre-gen libc
    contracts (legacy + code-contract) from the global cache. No --backend
    is passed — cache misses surface as a non-zero exit and are reported
    upstream rather than silently triggering an LLM run."""
    cmd = ["llm-summary", "init-stdlib", "--db", str(db_path)]
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            error = f"exit code {result.returncode}"
            if not verbose and result.stderr:
                error += f"\n{result.stderr[-300:]}"
            return False, error
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"init-stdlib timeout after {timeout}s"
    except FileNotFoundError:
        return False, "llm-summary not found (activate venv?)"
    except Exception as e:
        return False, str(e)


def run_import_dep(
    db_path: Path,
    link_units_path: Path | None = None,
    target_name: str | None = None,
    scan_dir: str = "func-scans",
    verbose: bool = False,
) -> tuple[bool, str]:
    """Invoke `llm-summary import-dep` (cross-project)."""
    cmd = ["llm-summary", "import-dep", "--db", str(db_path),
           "--scan-dir", scan_dir]
    if link_units_path:
        cmd += ["--link-units", str(link_units_path)]
    if target_name:
        cmd += ["--target", target_name]
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, timeout=300,
        )
        if result.returncode == 0:
            return True, ""
        error = f"exit code {result.returncode}"
        if not verbose and result.stderr:
            error += f"\n{result.stderr[-300:]}"
        return False, error
    except subprocess.TimeoutExpired:
        return False, "import-dep timeout"
    except FileNotFoundError:
        return False, "llm-summary not found (activate venv?)"
    except Exception as e:
        return False, str(e)


def run_import_dep_summaries(
    db_path: Path,
    dep_db_paths: list[Path],
    verbose: bool = False,
) -> tuple[bool, str]:
    """Invoke `llm-summary import-dep-summaries` (intra-project)."""
    if not dep_db_paths:
        return True, ""
    cmd = ["llm-summary", "import-dep-summaries", "--db", str(db_path)]
    for dep in dep_db_paths:
        cmd += ["--dep-db", str(dep)]
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, timeout=300,
        )
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


def run_check(
    db_path: Path,
    output_path: Path,
    entries: list[str] | None = None,
    verbose: bool = False,
) -> tuple[bool, str, int]:
    """Invoke `llm-summary check`. Returns (success, error, obligation_count)."""
    cmd = ["llm-summary", "check", "--db", str(db_path),
           "--output", str(output_path)]
    for e in entries or []:
        cmd += ["--entry", e]
    try:
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, timeout=300,
        )
        if result.returncode != 0:
            error = f"exit code {result.returncode}"
            if not verbose and result.stderr:
                error += f"\n{result.stderr[-300:]}"
            return False, error, 0
        n_obl = 0
        if output_path.exists():
            try:
                report = json.loads(output_path.read_text())
                n_obl = int(report.get("obligation_count", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return True, "", n_obl
    except subprocess.TimeoutExpired:
        return False, "check timeout", 0
    except FileNotFoundError:
        return False, "llm-summary not found (activate venv?)", 0
    except Exception as e:
        return False, str(e), 0


# ---------------------------------------------------------------------------
# Per-project / per-target processing
# ---------------------------------------------------------------------------

def _process_target(
    target: str,
    target_dir: Path,
    db_path: Path,
    *,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    jobs: int,
    timeout: int,
    do_check: bool,
    do_init_stdlib: bool,
    verbose: bool,
    verify_only: bool = False,
) -> dict:
    """Run summarize (and optional check) for one target DB."""
    target_result: dict = {
        "target": target,
        "success": False,
        "error": None,
        "obligation_count": None,
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    if do_init_stdlib:
        if verbose:
            print(f"    [{target}] init-stdlib: {db_path}")
        ok_i, err_i = run_init_stdlib(db_path=db_path, verbose=verbose)
        if not ok_i:
            target_result["error"] = f"init-stdlib failed: {err_i}"
            target_result["timing_seconds"] = round(time.monotonic() - start, 2)
            return target_result

    if verbose:
        print(f"    [{target}] code-contract summarize: {db_path}")

    ok, err, _dur = run_summarize_code_contract(
        db_path=db_path,
        backend=backend,
        model=model,
        force=force,
        incremental=incremental,
        llm_host=llm_host,
        llm_port=llm_port,
        log_llm=log_llm,
        jobs=jobs,
        verbose=verbose,
        timeout=timeout,
        verify_only=verify_only,
    )
    if not ok:
        target_result["error"] = f"summarize failed: {err}"
        target_result["timing_seconds"] = round(time.monotonic() - start, 2)
        return target_result

    if do_check:
        check_out = target_dir / "check_report.json"
        if verbose:
            print(f"    [{target}] check -> {check_out}")
        ok_c, err_c, n_obl = run_check(
            db_path=db_path,
            output_path=check_out,
            entries=None,
            verbose=verbose,
        )
        if not ok_c:
            target_result["error"] = f"check failed: {err_c}"
            target_result["timing_seconds"] = round(time.monotonic() - start, 2)
            return target_result
        target_result["obligation_count"] = n_obl

    target_result["success"] = True
    target_result["timing_seconds"] = round(time.monotonic() - start, 2)
    return target_result


def process_project_link_units(
    project_name: str,
    link_units_path: Path,
    func_scans_dir: Path,
    *,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    jobs: int,
    timeout: int,
    do_check: bool,
    do_init_stdlib: bool,
    verbose: bool,
    verify_only: bool = False,
) -> dict:
    """Code-contract for each non-alias target in topological order."""
    result: dict = {
        "project": project_name,
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

    detect_bc_alias_relations(raw_units)
    link_units = topo_sort_link_units(raw_units)
    by_output = build_output_index(link_units)
    project_scan_dir = func_scans_dir / project_name

    target_errors: list[str] = []
    for lu in link_units:
        target = lu["name"]

        if lu.get("alias_of"):
            if verbose:
                print(f"    [{target}] Skipped: alias of {lu['alias_of']}")
            result["targets"].append({
                "target": target,
                "success": True,
                "alias_of": lu["alias_of"],
                "error": None,
                "timing_seconds": 0.0,
            })
            continue

        db_str = lu.get("db_path")
        target_dir = project_scan_dir / target
        db_path = Path(db_str) if db_str else target_dir / "functions.db"

        if not db_path.exists():
            project_db = project_scan_dir / "functions.db"
            if project_db.exists():
                if verbose:
                    print(f"    [{target}] Per-target DB not found, "
                          f"using project-level functions.db")
                db_path = project_db
                target_dir = project_scan_dir
            else:
                tr = {
                    "target": target,
                    "success": False,
                    "error": "no_functions_db",
                    "timing_seconds": 0.0,
                }
                result["targets"].append(tr)
                target_errors.append(f"{target}: no_functions_db")
                continue

        # Cross-project dep import (e.g. zlib contracts for libpng)
        ok, err = run_import_dep(
            db_path=db_path,
            link_units_path=link_units_path,
            target_name=target,
            scan_dir=str(func_scans_dir),
            verbose=verbose,
        )
        if not ok and verbose:
            print(f"    [{target}] import-dep warning: {err}")

        # Intra-project dep import (e.g. zlibstatic for zlib_static_example64)
        dep_db_paths = resolve_dep_db_paths(lu, by_output, project_scan_dir)
        if dep_db_paths:
            if verbose:
                names = [d.parent.name for d in dep_db_paths]
                print(f"    [{target}] Importing dep summaries from: "
                      f"{', '.join(names)}")
            ok, err = run_import_dep_summaries(
                db_path=db_path,
                dep_db_paths=dep_db_paths,
                verbose=verbose,
            )
            if not ok:
                tr = {
                    "target": target,
                    "success": False,
                    "error": f"import_dep_summaries failed: {err}",
                    "timing_seconds": 0.0,
                }
                result["targets"].append(tr)
                target_errors.append(f"{target}: {tr['error']}")
                continue

        tr = _process_target(
            target=target,
            target_dir=target_dir,
            db_path=db_path,
            backend=backend, model=model, force=force,
            incremental=incremental, llm_host=llm_host, llm_port=llm_port,
            log_llm=log_llm, jobs=jobs, timeout=timeout,
            do_check=do_check, do_init_stdlib=do_init_stdlib,
            verbose=verbose, verify_only=verify_only,
        )
        result["targets"].append(tr)
        if not tr["success"]:
            target_errors.append(f"{target}: {tr['error']}")

    result["success"] = not target_errors
    result["error"] = "; ".join(target_errors) if target_errors else None
    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def process_project(
    project_name: str,
    func_scans_dir: Path,
    *,
    backend: str,
    model: str | None,
    force: bool,
    incremental: bool,
    llm_host: str,
    llm_port: int | None,
    log_llm: Path | None,
    jobs: int,
    timeout: int,
    do_check: bool,
    do_init_stdlib: bool,
    verbose: bool,
    verify_only: bool = False,
) -> dict:
    """Code-contract for one project. Auto-routes to link-unit mode."""
    scan_dir = func_scans_dir / project_name
    link_units_path = scan_dir / "link_units.json"
    if link_units_path.exists():
        if verbose:
            print("    link_units.json found — processing per target")
        return process_project_link_units(
            project_name=project_name,
            link_units_path=link_units_path,
            func_scans_dir=func_scans_dir,
            backend=backend, model=model, force=force,
            incremental=incremental, llm_host=llm_host, llm_port=llm_port,
            log_llm=log_llm, jobs=jobs, timeout=timeout,
            do_check=do_check, do_init_stdlib=do_init_stdlib,
            verbose=verbose, verify_only=verify_only,
        )

    # --- Single-DB mode ---
    result: dict = {
        "project": project_name,
        "success": False,
        "error": None,
        "targets": [],
        "timing_seconds": 0.0,
    }
    start = time.monotonic()

    db_path = scan_dir / "functions.db"
    if not db_path.exists():
        result["error"] = "no_functions_db"
        return result

    # Sanity check: code-contract needs call edges. If empty, attempt
    # import from a sibling callgraph.json (mirrors batch_summarize.py).
    try:
        con = sqlite3.connect(db_path)
        (edge_count,) = con.execute(
            "SELECT COUNT(*) FROM call_edges",
        ).fetchone()
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
            print(f"    No call edges, importing from {callgraph_json.name}")
        try:
            db = SummaryDB(str(db_path))
            CallGraphImporter(db, verbose=verbose).import_json(
                callgraph_json, clear_existing=True,
            )
            db.close()
        except Exception as e:
            result["error"] = f"callgraph_import_failed: {e}"
            return result

    tr = _process_target(
        target=project_name,
        target_dir=scan_dir,
        db_path=db_path,
        backend=backend, model=model, force=force,
        incremental=incremental, llm_host=llm_host, llm_port=llm_port,
        log_llm=log_llm, jobs=jobs, timeout=timeout,
        do_check=do_check, do_init_stdlib=do_init_stdlib,
        verbose=verbose, verify_only=verify_only,
    )
    result["targets"].append(tr)
    result["success"] = tr["success"]
    result["error"] = tr["error"]
    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch code-contract summarization for projects",
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
        "--source-dir", type=Path,
        default=Path("/data/csong/opensource"),
        help="Root where projects are cloned (for monorepo resolution)",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        default="claude",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-summarize even if cached",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only re-summarize functions with stale callee summaries",
    )
    parser.add_argument("--llm-host", default="localhost")
    parser.add_argument("--llm-port", type=int, default=None)
    parser.add_argument(
        "--log-llm", type=Path, default=None,
        help="Log all LLM prompts/responses to this file",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Parallel LLM queries per target (default: 1)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="After summarize, run `llm-summary check` per target and "
             "write check_report.json next to functions.db",
    )
    parser.add_argument(
        "--no-init-stdlib", dest="init_stdlib", action="store_false",
        default=True,
        help="Skip the init-stdlib step before summarize (default: run it)",
    )
    parser.add_argument(
        "--timeout", type=int, default=86400,
        help="Per-target summarize timeout in seconds (default: 24h)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip contract generation; re-run verification against "
             "cached contracts and persist issues to DB.",
    )
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-list", type=Path, default=None,
        help="File with project_dir names to skip (one per line)",
    )
    parser.add_argument(
        "--success-list", type=Path, default=None,
        help="Append successful project names here",
    )
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
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

    projects = [p for p in projects if p.get("project_dir")]
    print(f"Projects with project_dir: {len(projects)}")

    if args.tier is not None:
        before = len(projects)
        projects = [p for p in projects if p.get("tier") == args.tier]
        print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    if args.filter:
        flt = args.filter.lower()
        before = len(projects)
        projects = [
            p for p in projects
            if flt in p["project_dir"].lower()
            or flt in p.get("name", "").lower()
        ]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

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

    def _has_scan_data(p: dict) -> bool:
        project_path = find_project_dir(p, args.source_dir)
        name = (
            get_artifact_name(p, project_path) if project_path
            else p["project_dir"]
        )
        scan_dir: Path = args.func_scans_dir / name
        return bool(
            (scan_dir / "functions.db").exists()
            or (scan_dir / "link_units.json").exists()
        )

    eligible = [p for p in projects if _has_scan_data(p)]
    skipped_no_db = len(projects) - len(eligible)
    if skipped_no_db:
        print(f"Skipped {skipped_no_db} projects with no functions.db")
    projects = eligible

    print(f"\nProcessing {len(projects)} projects")
    print(f"Backend: {args.backend}"
          + (f" ({args.model})" if args.model else ""))
    print("Type:    code-contract"
          + (" + check" if args.check else ""))
    print(f"Func-scans dir: {args.func_scans_dir}")
    print()

    all_results: list[dict] = []
    succeeded = 0
    failed = 0

    for i, project in enumerate(projects, 1):
        project_path = find_project_dir(project, args.source_dir)
        artifact_name = (
            get_artifact_name(project, project_path)
            if project_path
            else project["project_dir"]
        )
        # In verbose mode the inner subprocess streams to stdout, so the
        # one-line `[i/N] artifact... STATUS` format gets mangled. Emit
        # the header with a newline and the status as its own line.
        if args.verbose:
            print(f"\n[{i}/{len(projects)}] {artifact_name}", flush=True)
        else:
            print(f"[{i}/{len(projects)}] {artifact_name}...",
                  end=" ", flush=True)

        result = process_project(
            project_name=artifact_name,
            func_scans_dir=args.func_scans_dir,
            backend=args.backend,
            model=args.model,
            force=args.force,
            incremental=args.incremental,
            llm_host=args.llm_host,
            llm_port=args.llm_port,
            log_llm=args.log_llm,
            jobs=args.jobs,
            timeout=args.timeout,
            do_check=args.check,
            do_init_stdlib=args.init_stdlib,
            verbose=args.verbose,
            verify_only=args.verify_only,
        )
        all_results.append(result)

        if result["success"]:
            status = f"OK ({result['timing_seconds']}s)"
            if args.verbose:
                print(f"[{i}/{len(projects)}] {artifact_name}: {status}")
            else:
                print(status)
            succeeded += 1
            if args.success_list:
                with open(args.success_list, "a") as f:
                    f.write(f"{artifact_name}\n")
        else:
            error_preview = (result["error"] or "")[:80]
            status = f"FAIL ({error_preview})"
            if args.verbose:
                print(f"[{i}/{len(projects)}] {artifact_name}: {status}")
            else:
                print(status)
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
        "type": "code-contract",
        "check": args.check,
        "force": args.force,
        "projects": all_results,
        "totals": {
            "succeeded": succeeded,
            "failed": failed,
            "total": len(projects),
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"code_contract_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {output_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
