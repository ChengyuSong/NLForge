#!/usr/bin/env python3
"""Batch scan all built projects for indirect call targets.

Iterates over build-scripts/*/compile_commands.json and runs the Phase-1
scanner (extract functions, scan targets, find callsites) on each project.

By default, creates func-scans/<project>/functions.db for each project.
Use --dry-run to skip DB storage (in-memory only).

Produces a JSON report with per-project and aggregate statistics.

Usage:
    python scripts/batch_scan_targets.py [--verbose] [--dry-run] [-j4] [--tier 1] [--skip-list skip.txt]
"""

import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src to path so we can import llm_summary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.compile_commands import CompileCommandsDB
from llm_summary.db import SummaryDB
from llm_summary.extern_headers import extract_extern_headers
from llm_summary.extractor import FunctionExtractor
from llm_summary.indirect.callsites import IndirectCallsiteFinder
from llm_summary.indirect.scanner import AddressTakenScanner
from llm_summary.link_units.pipeline import load_link_units, topo_sort_link_units
from llm_summary.models import TargetType
from gpr_utils import resolve_compile_commands


C_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".c++"}

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BUILD_SCRIPTS_DIR = REPO_ROOT / "build-scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"


def _load_tier_map() -> dict[str, int]:
    """Load project_dir -> tier mapping from gpr_projects.json."""
    if not GPR_PROJECTS_PATH.exists():
        return {}
    with open(GPR_PROJECTS_PATH) as f:
        projects = json.load(f)
    return {
        p["project_dir"]: p["tier"]
        for p in projects
        if "project_dir" in p
    }


def _load_project_root(project_dir: Path) -> Path | None:
    """Read project_path from config.json if present."""
    config_path = project_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            p = config.get("project_path")
            if p:
                return Path(p)
        except Exception:
            pass
    return None


def _bare_stem(p: Path) -> str:
    """Strip all extensions: adler32.c.o -> adler32, adler32.bc -> adler32."""
    s = p.stem
    while "." in s:
        s = Path(s).stem
    return s



def _source_files_for_objects(
    cc_entries: list[dict],
    objects: list[str],
    build_dir: Path,
) -> list[str]:
    """Return source files whose compiled output matches an object file path.

    Used as fallback when no .bc files exist (non-LTO builds).
    Matching strategy (in order):
    1. Exact absolute output path match
    2. Resolve relative object against build_dir
    3. Stem-based fallback: strip mangled prefixes from object name
       (e.g. libtestutil-lib-opt.o -> opt) and match against source stems
    """
    # Build index: absolute output path -> source file
    idx: dict[str, str] = {}
    # Stem index for fallback: (parent_dir, bare_stem) -> source file
    # Use parent dir to disambiguate same-stem files in different directories
    stem_idx: dict[str, list[str]] = {}  # bare_stem -> [source_files]
    for entry in cc_entries:
        output = entry.get("output", "")
        src = entry.get("file", "")
        if not (src and Path(src).suffix.lower() in C_EXTENSIONS):
            continue
        if output:
            out_path = Path(output)
            if not out_path.is_absolute():
                directory = entry.get("directory", "")
                if directory:
                    out_path = Path(directory) / out_path
                else:
                    out_path = build_dir / out_path
            idx[str(out_path)] = src
        # Build stem index from source file
        src_stem = Path(src).stem
        stem_idx.setdefault(src_stem, []).append(src)

    sources: list[str] = []
    seen: set[str] = set()
    for obj in objects:
        # Try as-is first (absolute), then resolve relative against build_dir
        src = idx.get(obj)
        if not src and not Path(obj).is_absolute():
            src = idx.get(str(build_dir / obj))
        # Stem-based fallback: strip mangled prefixes
        # e.g. libtestutil-lib-opt.o -> opt, libapps-lib-app_rand.o -> app_rand
        if not src:
            obj_stem = _bare_stem(Path(obj))
            # Strip lib<name>-lib- prefix pattern (common in OpenSSL, autotools)
            if "-lib-" in obj_stem:
                obj_stem = obj_stem.split("-lib-", 1)[1]
            candidates = stem_idx.get(obj_stem, [])
            if len(candidates) == 1:
                src = candidates[0]
        if src and src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


def _scan_files(
    source_files: list[str],
    cc: CompileCommandsDB,
    db: SummaryDB,
    project_root: Path | None,
    verbose: bool,
    preprocess: bool = False,
    cc_path: str | Path | None = None,
) -> tuple[int, int, int]:
    """Extract functions, scan address-taken, find callsites. Returns (funcs, targets, callsites).

    Parses each source file once with libclang and runs all passes on the
    same translation unit.
    """
    extractor = FunctionExtractor(
        compile_commands=cc, project_root=project_root,
        enable_preprocessing=preprocess,
    )

    # Phase 1: parse each file once, extract functions + typedefs
    all_functions = []
    all_typedefs = []
    parsed_tus: list[tuple] = []  # (tu, file_path) for reuse

    for f in source_files:
        try:
            tu = extractor.parse_file(f)
            funcs = extractor.extract_from_tu(tu, f)
            all_functions.extend(funcs)
            all_typedefs.extend(extractor.extract_typedefs_from_tu(tu, f))
            parsed_tus.append((tu, f))
        except Exception:
            pass

    db.insert_functions_batch(all_functions)
    db.insert_typedefs_batch(all_typedefs)

    # Phase 2: reuse parsed TUs for address-taken scan + callsite finding
    scanner = AddressTakenScanner(db, compile_commands=cc)
    # Build function map once (normally done per scan_files call)
    for func in db.get_all_functions():
        if func.id is not None:
            scanner._function_map[func.name] = func.id

    finder = IndirectCallsiteFinder(db, compile_commands=cc)
    for func in db.get_all_functions():
        if func.id is not None:
            key = (func.file_path, func.name, func.line_start)
            finder._function_map[key] = func.id

    callsites = []
    for tu, f in parsed_tus:
        try:
            scanner.scan_tu(tu, f)
        except Exception:
            pass
        try:
            callsites.extend(finder.find_in_tu(tu, f))
        except Exception:
            pass

    atfs = db.get_address_taken_functions()

    # Extract extern declaration headers (for import-dep)
    if cc_path:
        try:
            header_map = extract_extern_headers(
                compile_commands_path=cc_path,
                project_root=project_root,
                source_files=source_files,
                verbose=verbose,
            )
            if header_map:
                updated = db.update_decl_headers(header_map)
                if verbose:
                    print(f"      Extern headers: {len(header_map)} mapped, {updated} DB rows updated")
        except Exception as e:
            if verbose:
                print(f"      Extern headers: failed ({e})")

    return len(all_functions), len(atfs), len(callsites)


def _scan_one_target(args: tuple) -> dict:
    """Scan a single link-unit target. Designed for ProcessPoolExecutor."""
    (target, source_files, match_desc, cc_entries_json_path,
     db_path_str, project_root, verbose, preprocess) = args

    target_start = time.monotonic()
    target_result: dict = {
        "target": target,
        "source_files": len(source_files),
        "functions": 0,
        "total_targets": 0,
        "callsites": 0,
        "error": None,
        "timing_seconds": 0.0,
    }

    if not source_files:
        target_result["error"] = "no_source_files_matched"
        target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
        return target_result

    cc = CompileCommandsDB(Path(cc_entries_json_path))
    db = SummaryDB(db_path_str)
    try:
        n_funcs, n_targets, n_callsites = _scan_files(
            source_files, cc, db, project_root, verbose,
            preprocess=preprocess,
            cc_path=cc_entries_json_path,
        )
    finally:
        db.close()

    target_result["functions"] = n_funcs
    target_result["total_targets"] = n_targets
    target_result["callsites"] = n_callsites
    target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
    return target_result


def scan_project_link_units(
    project_dir: Path,
    func_scans_dir: Path,
    cc_entries: list[dict],
    project_root: Path | None,
    link_units_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    preprocess: bool = False,
    jobs: int = 0,
) -> dict:
    """Scan a project with link_units.json — one DB per target.

    Args:
        jobs: Number of parallel target workers. 0 = cpu_count, 1 = sequential.
    """
    project_name = project_dir.name
    result = {
        "project": project_name,
        "link_unit_mode": True,
        "targets": [],
        "source_files": 0,
        "functions": 0,
        "total_targets": 0,
        "callsites": 0,
        "timing_seconds": 0.0,
        "error": None,
    }
    start = time.monotonic()

    lu_data, raw_units = load_link_units(link_units_path)
    if not raw_units:
        result["error"] = "link_units.json has no targets"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    link_units = topo_sort_link_units(raw_units)
    project_scan_dir = func_scans_dir / project_name
    build_dir = Path(lu_data.get("build_dir", f"/data/csong/build-artifacts/{project_name}"))

    # Write resolved compile_commands to a temp file (shared across workers)
    import tempfile as _tempfile
    tmp = _tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    import json as _json
    _json.dump(cc_entries, tmp)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        # Prepare work items for each target
        work_items = []
        for lu in link_units:
            target = lu["name"]
            objects = lu.get("objects", [])
            # Always resolve source files from objects (which come from the
            # linker command and match compile_commands output paths directly).
            # bc_files are recompiled derivatives with different stems/paths,
            # so they're unreliable for source resolution.
            source_files = _source_files_for_objects(cc_entries, objects, build_dir)
            match_desc = f"{len(objects)} objects"

            if dry_run:
                db_path_str = ":memory:"
            else:
                target_dir = project_scan_dir / target
                target_dir.mkdir(parents=True, exist_ok=True)
                db_path_str = str(target_dir / "functions.db")

            work_items.append((
                target, source_files, match_desc, str(tmp_path),
                db_path_str, project_root, verbose, preprocess,
            ))

        num_workers = jobs if jobs > 0 else (os.cpu_count() or 1)
        # Only parallelise when there are multiple targets worth scanning
        scannable = [w for w in work_items if w[1]]  # w[1] = source_files
        use_parallel = num_workers > 1 and len(scannable) > 1

        if use_parallel:
            if verbose:
                print(f"    Scanning {len(scannable)} targets with {num_workers} workers")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_scan_one_target, item): item[0]
                    for item in work_items
                }
                for future in as_completed(futures):
                    target_name = futures[future]
                    target_result = future.result()
                    result["targets"].append(target_result)
                    if verbose:
                        tr = target_result
                        if tr["error"]:
                            # Find match_desc from work_items
                            desc = next(
                                (w[2] for w in work_items if w[0] == target_name),
                                "",
                            )
                            print(f"    [{target_name}] SKIP: no source files matched {desc}")
                        else:
                            print(
                                f"    [{target_name}] {tr['functions']} funcs, "
                                f"{tr['callsites']} callsites ({tr['timing_seconds']}s)"
                            )
        else:
            for item in work_items:
                target_name = item[0]
                source_files = item[1]
                match_desc = item[2]
                if not source_files:
                    target_result = {
                        "target": target_name,
                        "source_files": 0,
                        "functions": 0,
                        "total_targets": 0,
                        "callsites": 0,
                        "error": "no_source_files_matched",
                        "timing_seconds": 0.0,
                    }
                    result["targets"].append(target_result)
                    if verbose:
                        print(f"    [{target_name}] SKIP: no source files matched {match_desc}")
                    continue

                if verbose:
                    print(f"    [{target_name}] {len(source_files)} source files from {match_desc}")
                target_result = _scan_one_target(item)
                result["targets"].append(target_result)

        for tr in result["targets"]:
            result["source_files"] += tr["source_files"]
            result["functions"] += tr["functions"]
            result["total_targets"] += tr["total_targets"]
            result["callsites"] += tr["callsites"]

    finally:
        tmp_path.unlink(missing_ok=True)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def scan_project(
    project_dir: Path,
    func_scans_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
    preprocess: bool = False,
    jobs: int = 0,
) -> dict:
    """Scan a single project and return statistics.

    Auto-routes to scan_project_link_units when link_units.json is present.
    """
    cc_path = project_dir / "compile_commands.json"
    project_name = project_dir.name

    result = {
        "project": project_name,
        "source_files": 0,
        "functions": 0,
        "targets_by_type": {},
        "total_targets": 0,
        "callsites": 0,
        "db_path": None,
        "timing_seconds": 0.0,
        "error": None,
    }

    start = time.monotonic()

    try:
        project_root = _load_project_root(project_dir)

        # Resolve Docker /workspace/ paths if needed
        DEFAULT_BUILD_ROOT = Path("/data/csong/build-artifacts")
        build_dir = DEFAULT_BUILD_ROOT / project_dir.name
        entries = resolve_compile_commands(
            cc_path,
            project_source_dir=project_root or (project_dir / "src"),
            build_dir=build_dir,
        )

        # Auto-route to link-unit mode when link_units.json is present
        link_units_path = func_scans_dir / project_name / "link_units.json"
        if link_units_path.exists():
            if verbose:
                print(f"  link_units.json found — scanning per target")
            return scan_project_link_units(
                project_dir=project_dir,
                func_scans_dir=func_scans_dir,
                cc_entries=entries,
                project_root=project_root,
                link_units_path=link_units_path,
                dry_run=dry_run,
                verbose=verbose,
                preprocess=preprocess,
                jobs=jobs,
            )

        # --- Legacy single-DB mode ---
        tmp_cc = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(entries, tmp_cc)
        tmp_cc.flush()
        tmp_cc_path = Path(tmp_cc.name)
        tmp_cc.close()

        try:
            cc = CompileCommandsDB(tmp_cc_path)
            all_files = cc.get_all_files()
            source_files = [f for f in all_files if Path(f).suffix.lower() in C_EXTENSIONS]
        finally:
            tmp_cc_path.unlink(missing_ok=True)

        if not source_files:
            result["error"] = "no_c_cpp_files"
            return result

        result["source_files"] = len(source_files)

        if verbose and project_root:
            print(f"  project_root: {project_root} (header functions will be extracted)")

        if dry_run:
            db_path_str = ":memory:"
        else:
            project_scan_dir = func_scans_dir / project_name
            project_scan_dir.mkdir(parents=True, exist_ok=True)
            db_path_str = str(project_scan_dir / "functions.db")
        result["db_path"] = None if dry_run else db_path_str

        tmp_cc2 = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(entries, tmp_cc2)
        tmp_cc2.flush()
        tmp_cc2_path = Path(tmp_cc2.name)
        tmp_cc2.close()
        try:
            cc = CompileCommandsDB(tmp_cc2_path)
            db = SummaryDB(db_path_str)
            try:
                n_funcs, n_targets, n_callsites = _scan_files(
                    source_files, cc, db, project_root, verbose,
                    preprocess=preprocess,
                    cc_path=tmp_cc2_path,
                )
                atfs = db.get_address_taken_functions()
                type_counts: Counter[str] = Counter()
                for atf in atfs:
                    type_counts[atf.target_type] += 1
                result["targets_by_type"] = dict(type_counts)
            finally:
                db.close()
        finally:
            tmp_cc2_path.unlink(missing_ok=True)

        result["functions"] = n_funcs
        result["total_targets"] = n_targets
        result["callsites"] = n_callsites

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _scan_worker(args: tuple) -> dict:
    """Worker wrapper for ProcessPoolExecutor (needs top-level picklable callable)."""
    project_dir, func_scans_dir, dry_run, verbose = args[:4]
    preprocess = args[4] if len(args) > 4 else False
    jobs = args[5] if len(args) > 5 else 0
    return scan_project(Path(project_dir), Path(func_scans_dir), dry_run=dry_run, verbose=verbose, preprocess=preprocess, jobs=jobs)


def _format_result(result: dict) -> str:
    """Format a single result for printing."""
    if result["error"]:
        return f"SKIP ({result['error']})"
    suffix = f"({result['timing_seconds']}s)"
    if result.get("link_unit_mode"):
        n_targets = len(result.get("targets", []))
        return (
            f"{n_targets} targets, "
            f"{result['functions']} funcs, "
            f"{result['callsites']} callsites {suffix}"
        )
    return (
        f"{result['functions']} funcs, "
        f"{result['total_targets']} addr-taken, "
        f"{result['callsites']} callsites {suffix}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch scan projects for indirect call targets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use in-memory DBs only (don't create func-scans/<project>/functions.db)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel workers (default: 1, 0 = cpu count)",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only scan projects with this tier (1, 2, or 3) from gpr_projects.json",
    )
    parser.add_argument(
        "--skip-list", type=str, default=None,
        help="File with project names to skip (one per line)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only scan projects whose name contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to at most N projects",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run clang -E to expand macros and store preprocessed source (pp_source)",
    )
    args = parser.parse_args()

    if not BUILD_SCRIPTS_DIR.exists():
        print(f"Error: {BUILD_SCRIPTS_DIR} not found")
        sys.exit(1)

    # Find all projects with compile_commands.json
    projects = sorted(
        d for d in BUILD_SCRIPTS_DIR.iterdir()
        if d.is_dir() and (d / "compile_commands.json").exists()
    )

    # Filter by name
    if args.filter:
        filter_str = args.filter.lower()
        before = len(projects)
        projects = [p for p in projects if filter_str in p.name.lower()]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    # Filter by tier if requested
    if args.tier is not None:
        tier_map = _load_tier_map()
        if not tier_map:
            print(f"Warning: {GPR_PROJECTS_PATH} not found, --tier ignored")
        else:
            before = len(projects)
            projects = [p for p in projects if tier_map.get(p.name) == args.tier]
            print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    # Filter by skip list if provided
    if args.skip_list is not None:
        skip_path = Path(args.skip_list)
        if not skip_path.exists():
            print(f"Error: skip list file not found: {args.skip_list}")
            sys.exit(1)
        skip_names = set()
        with open(skip_path) as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    skip_names.add(name)
        before = len(projects)
        projects = [p for p in projects if p.name not in skip_names]
        print(f"Skip list: skipped {before - len(projects)}/{before} projects")

    # Limit number of projects
    if args.limit is not None:
        projects = projects[: args.limit]

    num_workers = args.jobs if args.jobs > 0 else os.cpu_count()

    print(f"Found {len(projects)} projects with compile_commands.json")
    if args.dry_run:
        print("Dry run: using in-memory DBs (no files written)")
    else:
        print(f"DB output: {FUNC_SCANS_DIR}/<project>/[<target>/]functions.db")
    if num_workers > 1:
        print(f"Parallel workers: {num_workers}")
    print()

    # Build work items: (project_dir, func_scans_dir, dry_run, verbose, preprocess, jobs)
    work_items = [
        (str(project_dir), str(FUNC_SCANS_DIR), args.dry_run, args.verbose, args.preprocess, args.jobs)
        for project_dir in projects
    ]

    # Run scans
    results_by_name: dict[str, dict] = {}

    if num_workers <= 1:
        # Sequential
        for i, item in enumerate(work_items, 1):
            name = Path(item[0]).name
            print(f"[{i}/{len(projects)}] {name}...", end=" ", flush=True)
            result = _scan_worker(item)
            results_by_name[name] = result
            print(_format_result(result))
    else:
        # Parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_name = {}
            for item in work_items:
                name = Path(item[0]).name
                future = executor.submit(_scan_worker, item)
                future_to_name[future] = name

            for future in as_completed(future_to_name):
                completed += 1
                name = future_to_name[future]
                result = future.result()
                results_by_name[name] = result
                print(
                    f"[{completed}/{len(projects)}] {name}... "
                    f"{_format_result(result)}"
                )

    # Collect results in sorted project order
    results = []
    totals = {
        "projects_scanned": 0,
        "projects_skipped": 0,
        "total_source_files": 0,
        "total_functions": 0,
        "total_targets": 0,
        "total_callsites": 0,
        "targets_by_type": Counter(),
    }

    for project_dir in projects:
        name = project_dir.name
        result = results_by_name[name]
        results.append(result)

        if result["error"]:
            totals["projects_skipped"] += 1
        else:
            totals["projects_scanned"] += 1
            totals["total_source_files"] += result["source_files"]
            totals["total_functions"] += result["functions"]
            totals["total_targets"] += result["total_targets"]
            totals["total_callsites"] += result["callsites"]
            totals["targets_by_type"].update(result.get("targets_by_type", {}))

    # Print summary
    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects scanned: {totals['projects_scanned']}")
    print(f"  Projects skipped: {totals['projects_skipped']}")
    print(f"  Source files: {totals['total_source_files']}")
    print(f"  Functions: {totals['total_functions']}")
    print(f"  Indirect call targets: {totals['total_targets']}")
    for tt in TargetType:
        count = totals["targets_by_type"].get(tt.value, 0)
        if count > 0:
            print(f"    {tt.value}: {count}")
    print(f"  Indirect callsites: {totals['total_callsites']}")

    # Write report
    report = {
        "projects": results,
        "totals": {
            "projects_scanned": totals["projects_scanned"],
            "projects_skipped": totals["projects_skipped"],
            "total_source_files": totals["total_source_files"],
            "total_functions": totals["total_functions"],
            "total_targets": totals["total_targets"],
            "targets_by_type": dict(totals["targets_by_type"]),
            "total_callsites": totals["total_callsites"],
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"scan_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
