#!/usr/bin/env python3
"""Batch call graph generation and import for all built projects.

For each project in gpr_projects.json that has build-scripts/<project>/
with compile_commands.json, this script:

1. Resolves .bc files using compile_commands.json (3-tier strategy):
   a. Look for .bc next to .o (from -save-temps=obj)
   b. If -flto in compile flags, .o itself is LLVM bitcode — use it directly
   c. If no LTO, recompile with -emit-llvm to produce .bc
2. Extracts allocator candidates from func-scans/<project>/functions.db
3. Runs KAMain on the collected .bc files (with --allocator-file)
4. Imports the resulting JSON call graph into func-scans/<project>/functions.db

Usage:
    python scripts/batch_call_graph_gen.py --tier 1 --verbose
    python scripts/batch_call_graph_gen.py --filter libpng --verbose
    python scripts/batch_call_graph_gen.py --tier 1 --skip-list done_cg.txt --success-list done_cg.txt
    python scripts/batch_call_graph_gen.py --kamain-bin /path/to/KAMain --verbose
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.allocator import STDLIB_ALLOCATORS, AllocatorDetector
from llm_summary.callgraph_import import CallGraphImporter
from llm_summary.db import SummaryDB
from llm_summary.link_units.pipeline import (
    build_output_index,
    detect_bc_alias_relations,
    load_link_units,
    propagate_alias_db_paths,
    resolve_dep_db_paths,
    topo_sort_link_units,
    update_link_units_file,
)
from gpr_utils import find_project_dir, get_artifact_name, resolve_compile_commands

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BUILD_SCRIPTS_DIR = REPO_ROOT / "build-scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"

DEFAULT_KAMAIN_BIN = "/home/csong/project/kanalyzer/release/lib/KAMain"
DEFAULT_SOURCE_DIR = Path("/data/csong/opensource")
DEFAULT_BUILD_ROOT = Path("/data/csong/build-artifacts")

C_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".c++"}


# ---------------------------------------------------------------------------
# .bc file resolution (3-tier)
# ---------------------------------------------------------------------------

def _get_output_path(entry: dict) -> Path | None:
    """Get the resolved .o output path from a compile_commands entry."""
    output = entry.get("output")
    if not output:
        return None
    directory = entry.get("directory", "")
    o_path = Path(output)
    if not o_path.is_absolute():
        o_path = Path(directory) / o_path
    return o_path


def _derive_bc_from_output(entry: dict) -> Path | None:
    """Derive the expected .bc path from the .o output path.

    With -save-temps=obj, clang writes <stem>.bc next to <stem>.<ext>.o.
    E.g., png.c.o → png.bc in the same directory.
    """
    o_path = _get_output_path(entry)
    if o_path is None:
        return None

    source_file = entry.get("file", "")
    source_stem = Path(source_file).stem  # e.g., "png" from "png.c"
    return o_path.parent / (source_stem + ".bc")


def _has_flto(entry: dict) -> bool:
    """Check if the compile command uses -flto."""
    if "arguments" in entry:
        return any("-flto" in a for a in entry["arguments"])
    cmd = entry.get("command", "")
    return "-flto" in cmd


def _is_source_file(entry: dict) -> bool:
    """Check if the entry compiles a C/C++ source file."""
    file_path = entry.get("file", "")
    return Path(file_path).suffix.lower() in C_EXTENSIONS


def _recompile_to_bc(entry: dict, bc_output: Path, verbose: bool = False) -> bool:
    """Recompile a source file with -emit-llvm to produce .bc.

    Parses the original compile command, replaces -o target with bc_output,
    adds -emit-llvm, and runs on the host. Always substitutes the compiler
    with clang-18/clang++-18 so -emit-llvm works regardless of the original
    compiler (e.g., gcc).

    Returns True on success.
    """
    if "arguments" in entry:
        args = list(entry["arguments"])
    elif "command" in entry:
        args = shlex.split(entry["command"])
    else:
        return False

    # Determine clang replacement based on source file extension
    source_file = entry.get("file", "")
    is_cxx = Path(source_file).suffix.lower() in {".cpp", ".cxx", ".cc", ".c++"}
    clang_compiler = "clang++-18" if is_cxx else "clang-18"

    # Replace the compiler (first argument) with clang-18/clang++-18
    if args:
        args[0] = clang_compiler

    # Rebuild the command with modifications
    new_args = []
    skip_next = False
    has_emit_llvm = False
    has_c = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        # Replace -o <file> with our bc output
        if arg == "-o":
            new_args.append("-o")
            new_args.append(str(bc_output))
            skip_next = True
            continue
        if arg.startswith("-o") and len(arg) > 2:
            new_args.append(f"-o{bc_output}")
            continue

        # Strip -flto flags (incompatible with -emit-llvm)
        if arg.startswith("-flto"):
            continue

        # Strip -save-temps
        if arg.startswith("-save-temps"):
            continue

        if arg == "-emit-llvm":
            has_emit_llvm = True
        if arg == "-c":
            has_c = True

        new_args.append(arg)

    if not has_emit_llvm:
        # Insert -emit-llvm after -c (or before source file)
        if has_c:
            idx = new_args.index("-c")
            new_args.insert(idx + 1, "-emit-llvm")
        else:
            # Insert -c -emit-llvm before the source file (last arg usually)
            new_args.insert(-1, "-c")
            new_args.insert(-1, "-emit-llvm")

    directory = entry.get("directory", ".")
    bc_output.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"      Recompile: {Path(entry.get('file','')).name} -> {bc_output.name}")

    try:
        result = subprocess.run(
            new_args,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0 and bc_output.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def collect_bc_files(
    entries: list[dict],
    recompile_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[list[Path], dict]:
    """Collect .bc files for all entries using 3-tier strategy.

    Returns (list_of_bc_paths, stats_dict).
    """
    bc_files = []
    stats = {
        "total_entries": 0,
        "skipped_non_source": 0,
        "tier1_save_temps": 0,
        "tier2_lto_obj": 0,
        "tier3_recompiled": 0,
        "tier3_failed": 0,
        "not_found": 0,
    }

    seen = set()  # Deduplicate by resolved .bc path

    for entry in entries:
        if not _is_source_file(entry):
            stats["skipped_non_source"] += 1
            continue

        stats["total_entries"] += 1
        bc_path = None

        # Tier 1: Look for .bc next to .o (from -save-temps=obj)
        candidate = _derive_bc_from_output(entry)
        if candidate and candidate.exists():
            bc_path = candidate.resolve()
            stats["tier1_save_temps"] += 1

        # Tier 2: If -flto, the .o file IS bitcode
        if bc_path is None and _has_flto(entry):
            o_path = _get_output_path(entry)
            if o_path and o_path.exists():
                # Verify it's actually bitcode
                try:
                    result = subprocess.run(
                        ["file", "--brief", str(o_path)],
                        capture_output=True, text=True, timeout=5,
                    )
                    if "LLVM IR bitcode" in result.stdout:
                        bc_path = o_path.resolve()
                        stats["tier2_lto_obj"] += 1
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

        # Tier 3: Recompile with -emit-llvm
        if bc_path is None and recompile_dir is not None:
            source_file = entry.get("file", "")
            source_stem = Path(source_file).stem
            # Use a subdirectory structure to avoid collisions
            directory = entry.get("directory", "")
            # Create a relative path based on the output to avoid name collisions
            o_path = _get_output_path(entry)
            if o_path:
                try:
                    rel = o_path.relative_to(Path(directory))
                    bc_out = recompile_dir / rel.parent / (source_stem + ".bc")
                except ValueError:
                    bc_out = recompile_dir / (source_stem + ".bc")
            else:
                bc_out = recompile_dir / (source_stem + ".bc")

            if _recompile_to_bc(entry, bc_out, verbose=verbose):
                bc_path = bc_out.resolve()
                stats["tier3_recompiled"] += 1
            else:
                stats["tier3_failed"] += 1

        if bc_path is None:
            stats["not_found"] += 1
            continue

        if bc_path not in seen:
            seen.add(bc_path)
            bc_files.append(bc_path)

    return bc_files, stats


# ---------------------------------------------------------------------------
# Allocator extraction
# ---------------------------------------------------------------------------

def extract_allocator_candidates(
    db_path: str,
    project_name: str,
    output_path: Path,
    min_score: int = 5,
    include_stdlib: bool = True,
    verbose: bool = False,
) -> int:
    """Extract allocator candidates using heuristics. Returns count."""
    db = SummaryDB(db_path)
    try:
        functions = db.get_all_functions()
        if not functions:
            return 0

        detector = AllocatorDetector(
            db, llm=None, verbose=verbose, min_score=min_score,
            project_name=project_name,
        )
        scored = detector.heuristic_only()

        candidate_names = [func.name for func, _, _ in scored]
        if include_stdlib:
            for name in sorted(STDLIB_ALLOCATORS):
                if name not in candidate_names:
                    candidate_names.append(name)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"candidates": candidate_names, "confirmed": []}, f, indent=2)

        return len(candidate_names)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# KAMain execution
# ---------------------------------------------------------------------------

def run_kamain(
    bc_files: list[Path],
    output_json: Path | None,
    kamain_bin: str,
    allocator_file: Path | None = None,
    container_file: Path | None = None,
    snapshot_path: Path | None = None,
    cfl_compressed_output: Path | None = None,
    cfl_compressed_inputs: list[Path] | None = None,
    cfl_compositional: bool = False,
    verbose_level: int = 1,
    timeout: int = 3600,
    verbose: bool = False,
) -> tuple[bool, str, float]:
    """Run KAMain on .bc files. Returns (success, error_msg, duration).

    Compositional mode (two-phase):
      Phase 1 – produce compressed graph:
        run_kamain(bc_files, output_json=None,
                   cfl_compressed_output=Path("target.cflcg"))
      Phase 2 – whole-program composition:
        run_kamain(all_bc_files, output_json=Path("callgraph.json"),
                   cfl_compressed_inputs=[Path("a.cflcg"), ...],
                   cfl_compositional=True)
    """
    start = time.monotonic()

    if not bc_files:
        return False, "No .bc files to analyze", 0.0

    # Determine the working directory for bc_files.txt (use output dir or cflcg dir)
    work_dir = (output_json or cfl_compressed_output)
    if work_dir is None:
        return False, "Neither output_json nor cfl_compressed_output specified", 0.0
    work_dir = work_dir.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write .bc paths to a list file to avoid command-line length limits
    bc_list_path = work_dir / "bc_files.txt"
    bc_list_path.write_text("\n".join(str(f) for f in bc_files) + "\n")

    cmd = [kamain_bin,
        "--bc-list", str(bc_list_path),
        "--verbose", str(verbose_level),
    ]
    if output_json:
        cmd += ["--callgraph-json", str(output_json)]
    if cfl_compressed_output:
        cmd += ["--cfl-compressed-output", str(cfl_compressed_output)]
    if cfl_compositional:
        cmd.append("--cfl-compositional")
    for cflcg in (cfl_compressed_inputs or []):
        cmd += ["--cfl-compressed-input", str(cflcg)]
    if allocator_file and allocator_file.exists():
        cmd += ["--allocator-file", str(allocator_file)]
    if container_file and container_file.exists():
        cmd += ["--container-file", str(container_file)]
    if snapshot_path:
        cmd += ["--v-snapshot", str(snapshot_path)]

    if verbose:
        label = output_json.name if output_json else (cfl_compressed_output.name if cfl_compressed_output else "?")
        print(f"    KAMain: {len(bc_files)} bitcode files -> {label}")
        print(f"    cmd: {shlex.join(cmd)}")

    out_dir = output_json.parent if output_json else (cfl_compressed_output.parent if cfl_compressed_output else None)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        duration = time.monotonic() - start

        if result.returncode == 0:
            # Verify expected output was produced
            if output_json and not output_json.exists():
                return False, "KAMain exited 0 but no callgraph JSON produced", duration
            if cfl_compressed_output and not cfl_compressed_output.exists():
                return False, "KAMain exited 0 but no .cflcg file produced", duration
            return True, "", duration

        # Detect OOM kill: SIGKILL = -9 (Python) or 137 (shell convention)
        rc = result.returncode
        if rc == -9 or rc == 137:
            return False, "OOM killed (SIGKILL)", duration
        if rc < 0:
            import signal as _signal
            try:
                sig_name = _signal.Signals(-rc).name
            except (ValueError, AttributeError):
                sig_name = f"signal {-rc}"
            error = f"killed by {sig_name}"
            if result.stderr:
                error += f"\n{result.stderr[-500:]}"
            return False, error, duration

        error = f"exit code {rc}"
        if result.stderr:
            error += f"\n{result.stderr[-500:]}"
        return False, error, duration

    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s", time.monotonic() - start
    except FileNotFoundError:
        return False, f"KAMain not found: {kamain_bin}", time.monotonic() - start
    except Exception as e:
        return False, str(e), time.monotonic() - start


# ---------------------------------------------------------------------------
# Call graph import
# ---------------------------------------------------------------------------

def import_callgraph(
    json_path: Path,
    db_path: str,
    clear_edges: bool = True,
    verbose: bool = False,
) -> dict:
    """Import KAMain call graph JSON into project DB. Returns stats dict."""
    db = SummaryDB(db_path)
    try:
        importer = CallGraphImporter(db, verbose=verbose)
        stats = importer.import_json(json_path, clear_existing=clear_edges)
        return {
            "functions_in_json": stats.functions_in_json,
            "functions_matched": stats.functions_matched,
            "stubs_created": stats.stubs_created,
            "edges_imported": stats.edges_imported,
            "direct_edges": stats.direct_edges,
            "indirect_edges": stats.indirect_edges,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Per-project processing
# ---------------------------------------------------------------------------


def _process_single_link_unit(
    lu: dict,
    func_scans_dir: Path,
    project_name: str,
    kamain_bin: str,
    allocator_file: Path | None = None,
    container_file: Path | None = None,
    kamain_timeout: int = 3600,
    verbose: bool = False,
) -> list[dict]:
    """Fast path for projects with a single link unit (no compositional split).

    Runs a single KAMain invocation with --cfl-compositional but without the
    redundant compress-then-compose two-phase split.
    """
    target = lu["name"]
    bc_files = [Path(p) for p in lu.get("bc_files", []) if Path(p).exists()]
    project_scan_dir = func_scans_dir / project_name
    target_dir = project_scan_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "target": target,
        "type": lu.get("type", "unknown"),
        "bc_files": len(bc_files),
        "phase1_success": True,   # no separate phase 1
        "phase2_success": False,
        "edges_imported": 0,
        "direct_edges": 0,
        "indirect_edges": 0,
        "stubs_created": 0,
        "timing_seconds": 0.0,
        "error": None,
    }
    start = time.monotonic()

    if not bc_files:
        result["error"] = "no_bc_files"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return [result]

    callgraph_json = target_dir / "callgraph.json"
    vsnapshot_path = target_dir / f"{target}.vsnap"

    if verbose:
        print(
            f"      [{target}] Single link unit: {len(bc_files)} bc files "
            f"-> {callgraph_json.name}"
        )

    ok, err, dur = run_kamain(
        bc_files=bc_files,
        output_json=callgraph_json,
        kamain_bin=kamain_bin,
        cfl_compositional=True,
        snapshot_path=vsnapshot_path,
        allocator_file=allocator_file,
        container_file=container_file,
        timeout=kamain_timeout,
        verbose=verbose,
    )

    result["phase2_success"] = ok
    if not ok:
        result["error"] = f"kamain_failed: {err}"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return [result]

    if verbose:
        print(f"      [{target}] Done in {dur:.1f}s")

    # --- Import callgraph into per-target DB ---
    db_path = str(target_dir / "functions.db")
    if not Path(db_path).exists():
        result["error"] = "no_per_target_db (run scan --link-units first)"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return [result]

    try:
        import_stats = import_callgraph(
            json_path=callgraph_json,
            db_path=db_path,
            clear_edges=True,
            verbose=verbose,
        )
        result.update(import_stats)
        if verbose:
            print(
                f"      [{target}] Imported: {import_stats['edges_imported']} edges "
                f"({import_stats['direct_edges']}d+{import_stats['indirect_edges']}i)"
            )
    except Exception as e:
        result["error"] = f"import_failed: {e}"

    lu["db_path"] = str(target_dir / "functions.db")
    lu["callgraph_json"] = str(callgraph_json)
    lu["vsnapshot"] = str(vsnapshot_path) if vsnapshot_path.exists() else None

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return [result]


def process_project_compositional(
    project_name: str,
    link_units_path: Path,
    func_scans_dir: Path,
    kamain_bin: str,
    allocator_file: Path | None = None,
    container_file: Path | None = None,
    kamain_timeout: int = 3600,
    skip_existing: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Compositional CFL analysis for a project with link_units.json.

    For each link unit in topological order:
      Phase 1 (Level 2): kanalyzer <bc_files> --cfl-compressed-output <target>.cflcg
      Phase 2 (Level 3): kanalyzer <target+dep bc_files>
                           --cfl-compositional
                           --cfl-compressed-input <target>.cflcg
                           [--cfl-compressed-input <dep>.cflcg ...]
                           --callgraph-json <target>/callgraph.json
    Then imports the callgraph into the per-target functions.db.

    Returns a list of per-target result dicts.
    """
    lu_data, raw_units = load_link_units(link_units_path)
    if not raw_units:
        return [{"target": project_name, "error": "link_units.json has no targets"}]

    # Detect alias relations before processing (idempotent, no-op if already set)
    n_aliases = detect_bc_alias_relations(raw_units)
    if n_aliases and verbose:
        aliases = [(u["name"], u["alias_of"]) for u in raw_units if u.get("alias_of")]
        for name, target in aliases:
            print(f"    [{name}] alias of {target} (bc_files subset)")

    link_units = topo_sort_link_units(raw_units)

    # Optimisation: when there is only a single link unit, the two-phase
    # compositional split is redundant (Phase 2 would feed the exact same
    # bc_files back).  Fall back to a single direct KAMain invocation.
    if len(link_units) == 1:
        results = _process_single_link_unit(
            link_units[0],
            func_scans_dir=func_scans_dir,
            project_name=project_name,
            kamain_bin=kamain_bin,
            allocator_file=allocator_file,
            container_file=container_file,
            kamain_timeout=kamain_timeout,
            verbose=verbose,
        )
        update_link_units_file(link_units_path, lu_data)
        return results

    # Map name -> link unit (for transitive dep traversal)
    by_name = {lu["name"]: lu for lu in link_units}
    # Map output -> link unit (for dep resolution)
    by_output = build_output_index(link_units)

    # Track per-target .cflcg paths (produced in Phase 1)
    cflcg_by_name: dict[str, Path] = {}

    project_scan_dir = func_scans_dir / project_name
    results = []

    for lu in link_units:
        target = lu["name"]

        # Skip alias units — their functions and call graph are covered by the superset
        if lu.get("alias_of"):
            if verbose:
                print(f"      [{target}] Skipped: alias of {lu['alias_of']}")
            results.append({
                "target": target,
                "type": lu.get("type", "unknown"),
                "bc_files": 0,
                "phase1_success": True,
                "phase2_success": True,
                "edges_imported": 0,
                "direct_edges": 0,
                "indirect_edges": 0,
                "stubs_created": 0,
                "timing_seconds": 0.0,
                "error": None,
                "alias_of": lu["alias_of"],
            })
            continue

        bc_files = [Path(p) for p in lu.get("bc_files", []) if Path(p).exists()]
        target_dir = project_scan_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        result: dict = {
            "target": target,
            "type": lu.get("type", "unknown"),
            "bc_files": len(bc_files),
            "phase1_success": False,
            "phase2_success": False,
            "edges_imported": 0,
            "direct_edges": 0,
            "indirect_edges": 0,
            "stubs_created": 0,
            "timing_seconds": 0.0,
            "error": None,
        }
        start = time.monotonic()

        if not bc_files:
            result["error"] = "no_bc_files"
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            results.append(result)
            continue

        cflcg_path = target_dir / f"{target}.cflcg"
        callgraph_json = target_dir / "callgraph.json"

        # --- Phase 1: Produce compressed constraint graph ---
        if skip_existing and cflcg_path.exists():
            if verbose:
                print(f"      [{target}] Phase 1: skipped (existing {cflcg_path.name})")
            result["phase1_success"] = True
        else:
            if verbose:
                print(f"      [{target}] Phase 1: compress {len(bc_files)} bc files -> {cflcg_path.name}")
            ok, err, dur = run_kamain(
                bc_files=bc_files,
                output_json=None,
                kamain_bin=kamain_bin,
                cfl_compressed_output=cflcg_path,
                cfl_compositional=True,
                allocator_file=allocator_file,
                container_file=container_file,
                timeout=kamain_timeout,
                verbose=verbose,
            )
            result["phase1_success"] = ok
            if not ok:
                result["error"] = f"phase1_failed: {err}"
                result["timing_seconds"] = round(time.monotonic() - start, 2)
                results.append(result)
                continue

        cflcg_by_name[target] = cflcg_path

        # --- Phase 2: Compositional whole-program solve ---
        # Collect transitive deps: bc_files + .cflcg for this target and all deps
        seen_bc: set[Path] = set(bc_files)
        all_bc: list[Path] = list(bc_files)
        all_cflcg: list[Path] = [cflcg_path]

        def _collect_deps(lu_name: str, visited_deps: set[str]) -> None:
            lu_entry = by_name.get(lu_name)
            if not lu_entry:
                return
            for dep_output in lu_entry.get("link_deps", []):
                dep_lu = by_output.get(dep_output)
                if dep_lu and dep_lu["name"] not in visited_deps:
                    visited_deps.add(dep_lu["name"])
                    for p in dep_lu.get("bc_files", []):
                        bp = Path(p)
                        if bp.exists() and bp not in seen_bc:
                            all_bc.append(bp)
                            seen_bc.add(bp)
                    dep_cflcg = cflcg_by_name.get(dep_lu["name"])
                    if dep_cflcg and dep_cflcg.exists():
                        all_cflcg.append(dep_cflcg)
                    _collect_deps(dep_lu["name"], visited_deps)

        _collect_deps(target, {target})

        vsnapshot_path = target_dir / f"{target}.vsnap"

        # --- Phase 2: Compositional whole-program solve ---
        if skip_existing and callgraph_json.exists():
            if verbose:
                print(f"      [{target}] Phase 2: skipped (existing {callgraph_json.name})")
            result["phase2_success"] = True
        else:
            if verbose:
                print(
                    f"      [{target}] Phase 2: compose {len(all_bc)} bc, "
                    f"{len(all_cflcg)} cflcg -> {callgraph_json.name}"
                )
            ok, err, dur = run_kamain(
                bc_files=all_bc,
                output_json=callgraph_json,
                kamain_bin=kamain_bin,
                cfl_compressed_inputs=all_cflcg,
                cfl_compositional=True,
                snapshot_path=vsnapshot_path,
                allocator_file=allocator_file,
                container_file=container_file,
                timeout=kamain_timeout,
                verbose=verbose,
            )
            result["phase2_success"] = ok
            if not ok:
                result["error"] = f"phase2_failed: {err}"
                result["timing_seconds"] = round(time.monotonic() - start, 2)
                results.append(result)
                continue
            if verbose:
                print(f"      [{target}] Phase 2 done in {dur:.1f}s")

        # --- Import callgraph into per-target DB ---
        db_path = str(target_dir / "functions.db")
        if not Path(db_path).exists():
            result["error"] = "no_per_target_db (run scan --link-units first)"
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            results.append(result)
            continue

        try:
            import_stats = import_callgraph(
                json_path=callgraph_json,
                db_path=db_path,
                clear_edges=True,
                verbose=verbose,
            )
            result.update(import_stats)
            if verbose:
                print(
                    f"      [{target}] Imported: {import_stats['edges_imported']} edges "
                    f"({import_stats['direct_edges']}d+{import_stats['indirect_edges']}i)"
                )
        except Exception as e:
            result["error"] = f"import_failed: {e}"

        # Record output paths on the link unit entry (written back to link_units.json below)
        lu["db_path"] = str(target_dir / "functions.db")
        lu["callgraph_json"] = str(callgraph_json)
        lu["cflcg"] = str(cflcg_path)
        lu["vsnapshot"] = str(vsnapshot_path) if vsnapshot_path.exists() else None

        result["timing_seconds"] = round(time.monotonic() - start, 2)
        results.append(result)

    # Propagate db_path from superset units to their alias units, then persist
    propagate_alias_db_paths(link_units)
    update_link_units_file(link_units_path, lu_data)

    return results


def _populate_link_unit_bc_files(
    link_units_path: Path,
    cc_path: Path,
    scan_dir: Path,
    source_dir: Path,
    build_root: Path,
    project_scripts_dir: Path,
    project_name: str,
    recompile: bool = True,
    verbose: bool = False,
) -> None:
    """Populate bc_files in link_units.json for units that have objects but no bc_files.

    Loads compile_commands.json, filters entries to those whose output matches
    a link unit's objects, and runs the 3-tier bc collection (including tier-3
    recompilation if enabled).  Updates link_units.json in place.
    """
    lu_data, raw_units = load_link_units(link_units_path)
    needs_bc = [lu for lu in raw_units if lu.get("objects") and not lu.get("bc_files")]
    if not needs_bc:
        return

    # Resolve compile_commands paths
    config_path = project_scripts_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        project_source_dir = Path(config.get("project_path", str(source_dir / project_name)))
    else:
        project_source_dir = source_dir / project_name
    build_dir = build_root / project_name

    try:
        entries = resolve_compile_commands(cc_path, project_source_dir, build_dir)
    except Exception:
        return

    if not entries:
        return

    # Build index: output path -> compile_commands entry
    output_to_entry: dict[str, dict] = {}
    for entry in entries:
        output = entry.get("output")
        if output:
            output_to_entry[str(Path(output).resolve())] = entry
            # Also index by basename for flat ar-t member matching
            output_to_entry[Path(output).name] = entry

    recompile_dir = (scan_dir / "recompiled_bc") if recompile else None
    updated = False

    for lu in needs_bc:
        # Find compile_commands entries matching this unit's objects
        matching_entries = []
        for obj in lu["objects"]:
            resolved = str(Path(obj).resolve())
            entry = output_to_entry.get(resolved) or output_to_entry.get(Path(obj).name)
            if entry and entry not in matching_entries:
                matching_entries.append(entry)

        if not matching_entries:
            continue

        bc_files, stats = collect_bc_files(matching_entries, recompile_dir=recompile_dir, verbose=False)
        if bc_files:
            lu["bc_files"] = [str(p) for p in bc_files]
            updated = True
            if verbose:
                print(
                    f"    [{lu['name']}] Populated {len(bc_files)} bc files "
                    f"(T1:{stats['tier1_save_temps']} T2:{stats['tier2_lto_obj']} "
                    f"T3:{stats['tier3_recompiled']})"
                )

    if updated:
        update_link_units_file(link_units_path, lu_data)


def process_project(
    project_name: str,
    build_scripts_dir: Path,
    func_scans_dir: Path,
    source_dir: Path,
    build_root: Path,
    kamain_bin: str,
    min_score: int = 5,
    include_stdlib: bool = True,
    kamain_timeout: int = 3600,
    recompile: bool = True,
    compositional: bool | None = None,  # None = auto-detect from link_units.json
    skip_existing: bool = False,
    verbose: bool = False,
) -> dict:
    """Process a single project end-to-end.

    When link_units.json is present (and compositional is not False), delegates
    to process_project_compositional for per-link-unit analysis.  Returns a
    wrapper dict with a 'targets' key containing per-target results.
    """
    result = {
        "project": project_name,
        "bc_files": 0,
        "bc_tier1": 0,
        "bc_tier2": 0,
        "bc_tier3": 0,
        "bc_tier3_failed": 0,
        "allocator_candidates": 0,
        "kamain_success": False,
        "edges_imported": 0,
        "direct_edges": 0,
        "indirect_edges": 0,
        "functions_in_json": 0,
        "stubs_created": 0,
        "timing_seconds": 0.0,
        "error": None,
    }

    start = time.monotonic()

    project_scripts_dir = build_scripts_dir / project_name
    scan_dir = func_scans_dir / project_name

    # --- Auto-route to compositional mode when link_units.json is present ---
    link_units_path = scan_dir / "link_units.json"
    if compositional is not False and link_units_path.exists():
        if verbose:
            print(f"    link_units.json found — using compositional CFL analysis")

        # Populate bc_files for link units that have objects but no bc_files
        cc_path = project_scripts_dir / "compile_commands.json"
        if cc_path.exists():
            _populate_link_unit_bc_files(
                link_units_path, cc_path, scan_dir,
                source_dir, build_root, project_scripts_dir, project_name,
                recompile=recompile, verbose=verbose,
            )

        # Extract project-level allocator candidates if a project-level DB exists
        allocator_json = scan_dir / "allocator_candidates.json"
        project_db = scan_dir / "functions.db"
        if project_db.exists() and not allocator_json.exists():
            try:
                extract_allocator_candidates(
                    db_path=str(project_db),
                    project_name=project_name,
                    output_path=allocator_json,
                    min_score=min_score,
                    include_stdlib=include_stdlib,
                    verbose=verbose,
                )
            except Exception:
                allocator_json = None
        container_file = None
        for candidate in [
            project_scripts_dir / "containers.json",
            scan_dir / "containers.json",
        ]:
            if candidate.exists():
                container_file = candidate
                break
        target_results = process_project_compositional(
            project_name=project_name,
            link_units_path=link_units_path,
            func_scans_dir=func_scans_dir,
            kamain_bin=kamain_bin,
            allocator_file=allocator_json if allocator_json and allocator_json.exists() else None,
            container_file=container_file,
            kamain_timeout=kamain_timeout,
            skip_existing=skip_existing,
            verbose=verbose,
        )
        total_edges = sum(r.get("edges_imported", 0) for r in target_results)
        total_direct = sum(r.get("direct_edges", 0) for r in target_results)
        total_indirect = sum(r.get("indirect_edges", 0) for r in target_results)
        total_stubs = sum(r.get("stubs_created", 0) for r in target_results)
        errors = [r["error"] for r in target_results if r.get("error")]
        result["targets"] = target_results
        result["edges_imported"] = total_edges
        result["direct_edges"] = total_direct
        result["indirect_edges"] = total_indirect
        result["stubs_created"] = total_stubs
        result["kamain_success"] = not errors
        result["error"] = "; ".join(errors) if errors else None
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    cc_path = project_scripts_dir / "compile_commands.json"
    db_path = str(scan_dir / "functions.db")

    # --- Prerequisites ---
    if not cc_path.exists():
        result["error"] = "no_compile_commands"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    if not Path(db_path).exists():
        result["error"] = "no_functions_db"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    # --- Resolve paths ---
    # Determine project source dir and build dir for Docker path translation
    config_path = project_scripts_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        project_source_dir = Path(config.get("project_path", str(source_dir / project_name)))
    else:
        project_source_dir = source_dir / project_name

    build_dir = build_root / project_name

    try:
        entries = resolve_compile_commands(cc_path, project_source_dir, build_dir)
    except Exception as e:
        result["error"] = f"cc_parse_failed: {e}"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    if not entries:
        result["error"] = "empty_compile_commands"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    # --- Step 1: Collect .bc files ---
    recompile_dir = (scan_dir / "recompiled_bc") if recompile else None
    bc_files, bc_stats = collect_bc_files(entries, recompile_dir=recompile_dir, verbose=verbose)

    result["bc_files"] = len(bc_files)
    result["bc_tier1"] = bc_stats["tier1_save_temps"]
    result["bc_tier2"] = bc_stats["tier2_lto_obj"]
    result["bc_tier3"] = bc_stats["tier3_recompiled"]
    result["bc_tier3_failed"] = bc_stats["tier3_failed"]

    if verbose:
        print(
            f"    Bitcode: {len(bc_files)} files "
            f"(T1:{bc_stats['tier1_save_temps']} "
            f"T2:{bc_stats['tier2_lto_obj']} "
            f"T3:{bc_stats['tier3_recompiled']}/"
            f"{bc_stats['tier3_recompiled'] + bc_stats['tier3_failed']})"
        )

    if not bc_files:
        result["error"] = "no_bc_files"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    # --- Step 2: Extract allocator candidates ---
    allocator_json = scan_dir / "allocator_candidates.json"
    try:
        n_candidates = extract_allocator_candidates(
            db_path=db_path,
            project_name=project_name,
            output_path=allocator_json,
            min_score=min_score,
            include_stdlib=include_stdlib,
            verbose=verbose,
        )
        result["allocator_candidates"] = n_candidates
        if verbose:
            print(f"    Allocator candidates: {n_candidates}")
    except Exception as e:
        if verbose:
            print(f"    Warning: allocator extraction failed: {e}")
        allocator_json = None

    # --- Step 3: Run KAMain ---
    callgraph_json = scan_dir / "callgraph.json"

    # Look for container file
    container_file = None
    for candidate in [
        project_scripts_dir / "containers.json",
        scan_dir / "containers.json",
    ]:
        if candidate.exists():
            container_file = candidate
            break

    snapshot_path = build_root / f"{project_name}.vsnapt"

    success, error_msg, duration = run_kamain(
        bc_files=bc_files,
        output_json=callgraph_json,
        kamain_bin=kamain_bin,
        allocator_file=allocator_json if allocator_json and allocator_json.exists() else None,
        container_file=container_file,
        snapshot_path=snapshot_path,
        timeout=kamain_timeout,
        verbose=verbose,
    )

    result["kamain_success"] = success

    if not success:
        result["error"] = f"kamain_failed: {error_msg}"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    if verbose:
        print(f"    KAMain completed in {duration:.1f}s")

    # --- Step 4: Import call graph ---
    try:
        import_stats = import_callgraph(
            json_path=callgraph_json,
            db_path=db_path,
            clear_edges=True,
            verbose=verbose,
        )
        result.update(import_stats)
        if verbose:
            print(
                f"    Imported: {import_stats['edges_imported']} edges "
                f"({import_stats['direct_edges']} direct, "
                f"{import_stats['indirect_edges']} indirect)"
            )
    except Exception as e:
        result["error"] = f"import_failed: {e}"

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _format_result(r: dict) -> str:
    """Format a single result for console output."""
    if r["error"]:
        return f"SKIP ({r['error'][:200]})"
    parts = [
        f"{r['bc_files']} bc",
        f"T1:{r['bc_tier1']}/T2:{r['bc_tier2']}/T3:{r['bc_tier3']}",
        f"{r['allocator_candidates']} alloc",
        f"{r['edges_imported']} edges ({r['direct_edges']}d+{r['indirect_edges']}i)",
        f"{r['stubs_created']} stubs",
        f"({r['timing_seconds']}s)",
    ]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch call graph generation and import using KAMain"
    )
    parser.add_argument(
        "--projects-json", type=Path, default=GPR_PROJECTS_PATH,
        help="Path to gpr_projects.json",
    )
    parser.add_argument(
        "--build-scripts-dir", type=Path, default=BUILD_SCRIPTS_DIR,
        help="Directory containing build-scripts/<project>/",
    )
    parser.add_argument(
        "--func-scans-dir", type=Path, default=FUNC_SCANS_DIR,
        help="Directory containing func-scans/<project>/functions.db",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help=f"Root directory where projects are cloned (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--build-root", type=Path, default=DEFAULT_BUILD_ROOT,
        help=f"Root directory for build outputs (default: {DEFAULT_BUILD_ROOT})",
    )
    parser.add_argument(
        "--kamain-bin", type=str, default=DEFAULT_KAMAIN_BIN,
        help=f"Path to KAMain binary (default: {DEFAULT_KAMAIN_BIN})",
    )
    parser.add_argument(
        "--kamain-timeout", type=int, default=3600,
        help="KAMain timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--min-score", type=int, default=5,
        help="Minimum heuristic score for allocator candidates (default: 5)",
    )
    parser.add_argument(
        "--no-stdlib", action="store_true",
        help="Do not include well-known stdlib allocators in allocator file",
    )
    parser.add_argument(
        "--no-recompile", action="store_true",
        help="Skip tier-3 recompilation (only use existing .bc/.o files)",
    )
    comp_group = parser.add_mutually_exclusive_group()
    comp_group.add_argument(
        "--compositional", action="store_true", default=None,
        help="Force compositional CFL analysis (requires link_units.json)",
    )
    comp_group.add_argument(
        "--no-compositional", action="store_true",
        help="Force monolithic analysis even when link_units.json is present",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only process projects with this tier (1, 2, or 3)",
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
        help="File with project names to skip (one per line)",
    )
    parser.add_argument(
        "--success-list", type=Path, default=None,
        help="Output file for successful project names (append mode)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip KAMain phases for link units that already have artifacts: "
            "phase 1 if .cflcg exists, phase 2 if callgraph.json exists. "
            "Import is still attempted when callgraph.json is present."
        ),
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON report file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate
    if not args.projects_json.exists():
        print(f"Error: {args.projects_json} not found")
        sys.exit(1)

    if not args.build_scripts_dir.exists():
        print(f"Error: {args.build_scripts_dir} not found")
        sys.exit(1)

    # Load projects
    with open(args.projects_json) as f:
        projects = json.load(f)

    print(f"Loaded {len(projects)} projects from {args.projects_json}")

    # Filter to projects with project_dir
    projects = [p for p in projects if p.get("project_dir")]
    print(f"Projects with project_dir: {len(projects)}")

    # Filter by tier
    if args.tier is not None:
        before = len(projects)
        projects = [p for p in projects if p.get("tier") == args.tier]
        print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    # Filter by name
    if args.filter:
        filter_str = args.filter.lower()
        before = len(projects)
        projects = [p for p in projects if filter_str in p["name"].lower()]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    # Load skip list
    skip_set = set()
    if args.skip_list and args.skip_list.exists():
        with open(args.skip_list) as f:
            skip_set = {
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            }
        before = len(projects)
        projects = [p for p in projects if p["project_dir"] not in skip_set]
        print(f"Skip list: skipped {before - len(projects)}/{before} projects")

    # Apply skip and limit
    if args.skip > 0:
        projects = projects[args.skip:]
        print(f"Skipped first {args.skip}, {len(projects)} remaining")

    if args.limit:
        projects = projects[:args.limit]
        print(f"Limited to {args.limit} projects")

    # Validate KAMain
    if not Path(args.kamain_bin).exists():
        print(f"Warning: KAMain binary not found at {args.kamain_bin}")

    print(f"\nProcessing {len(projects)} projects")
    print(f"KAMain: {args.kamain_bin}")
    print(f"Source dir: {args.source_dir}")
    print(f"Build root: {args.build_root}")
    print(f"Allocator min_score: {args.min_score}")
    print(f"Recompile (tier 3): {'no' if args.no_recompile else 'yes'}")
    print()

    # Process each project
    all_results = []
    totals = {
        "projects_processed": 0,
        "projects_skipped": 0,
        "total_bc_files": 0,
        "total_tier1": 0,
        "total_tier2": 0,
        "total_tier3": 0,
        "total_tier3_failed": 0,
        "total_allocator_candidates": 0,
        "total_edges": 0,
        "total_direct_edges": 0,
        "total_indirect_edges": 0,
        "total_stubs": 0,
    }

    compositional: bool | None = (
        False if args.no_compositional else (True if args.compositional else None)
    )

    for i, project in enumerate(projects, 1):
        project_dir = project["project_dir"]
        project_path = find_project_dir(project, args.source_dir)
        artifact_name = (
            get_artifact_name(project, project_path)
            if project_path
            else project_dir
        )
        print(f"[{i}/{len(projects)}] {artifact_name}...", end=" ", flush=True)

        result = process_project(
            project_name=artifact_name,
            build_scripts_dir=args.build_scripts_dir,
            func_scans_dir=args.func_scans_dir,
            source_dir=args.source_dir,
            build_root=args.build_root,
            kamain_bin=args.kamain_bin,
            min_score=args.min_score,
            include_stdlib=not args.no_stdlib,
            kamain_timeout=args.kamain_timeout,
            recompile=not args.no_recompile,
            compositional=compositional,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )

        all_results.append(result)
        print(_format_result(result))

        if result["error"]:
            totals["projects_skipped"] += 1
        else:
            totals["projects_processed"] += 1
            totals["total_bc_files"] += result["bc_files"]
            totals["total_tier1"] += result["bc_tier1"]
            totals["total_tier2"] += result["bc_tier2"]
            totals["total_tier3"] += result["bc_tier3"]
            totals["total_tier3_failed"] += result["bc_tier3_failed"]
            totals["total_allocator_candidates"] += result["allocator_candidates"]
            totals["total_edges"] += result["edges_imported"]
            totals["total_direct_edges"] += result["direct_edges"]
            totals["total_indirect_edges"] += result["indirect_edges"]
            totals["total_stubs"] += result["stubs_created"]

            if args.success_list:
                with open(args.success_list, "a") as f:
                    f.write(f"{artifact_name}\n")

    # Print summary
    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects processed: {totals['projects_processed']}")
    print(f"  Projects skipped:   {totals['projects_skipped']}")
    print(f"  Bitcode files:      {totals['total_bc_files']}")
    print(f"    Tier 1 (save-temps .bc): {totals['total_tier1']}")
    print(f"    Tier 2 (LTO .o as bc):   {totals['total_tier2']}")
    print(f"    Tier 3 (recompiled):     {totals['total_tier3']}")
    if totals["total_tier3_failed"]:
        print(f"    Tier 3 failed:           {totals['total_tier3_failed']}")
    print(f"  Allocator candidates: {totals['total_allocator_candidates']}")
    print(f"  Call edges:           {totals['total_edges']}")
    print(f"    Direct:  {totals['total_direct_edges']}")
    print(f"    Indirect: {totals['total_indirect_edges']}")
    print(f"  Stubs created:        {totals['total_stubs']}")

    # Write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "kamain_bin": args.kamain_bin,
        "min_score": args.min_score,
        "recompile": not args.no_recompile,
        "projects": all_results,
        "totals": totals,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"callgraph_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
