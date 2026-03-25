#!/usr/bin/env python3
"""Prepare CGC challenges for the llm-summary pipeline.

For each challenge:
1. Filters compile_commands.json (exclude patched/pov entries, remap paths)
2. Writes func-scans/cgc/<name>/compile_commands.json
3. Runs the function scan (FunctionExtractor)
4. Generates cgc_projects.json for batch_summarize.py

Usage:
    python scripts/cgc_prepare.py [--filter Palindrome] [--limit 10] [--verbose]
    python scripts/cgc_prepare.py --scan-only  # skip compile_commands filtering
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.compile_commands import CompileCommandsDB
from llm_summary.db import SummaryDB
from llm_summary.extractor import FunctionExtractor
from llm_summary.indirect.callsites import IndirectCallsiteFinder
from llm_summary.indirect.scanner import AddressTakenScanner
from llm_summary.models import CallEdge

from batch_call_graph_gen import collect_bc_files, run_kamain, import_callgraph

REPO_ROOT = Path(__file__).resolve().parent.parent


FUNC_SCANS_DIR = REPO_ROOT / "func-scans" / "cgc"
C_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".c++"}

_PATCHED_START_RE = re.compile(r"#\s*(ifdef|ifndef)\s+PATCHED(?:_\d+)?\s*$")
_PATCHED_END_RE = re.compile(r"#\s*endif\b")
_PATCHED_NESTED_RE = re.compile(r"#\s*(ifdef|ifndef|if)\b")


def strip_patched_ifdefs(source: str) -> str:
    """Remove entire #ifdef/#ifndef PATCHED blocks (both branches) from source.

    The preprocessed pp_source already has the correct resolved code path.
    Stripping the raw #ifdef blocks prevents the macro-diff annotation from
    leaking the other branch as ``// (macro)`` comments to the LLM.
    """
    lines = source.splitlines()
    result: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = _PATCHED_START_RE.match(s)
        if not m:
            result.append(lines[i])
            i += 1
            continue

        # Skip the entire #ifdef/#ifndef PATCHED ... #endif block
        depth = 1
        end_idx = None
        j = i + 1
        while j < len(lines) and depth > 0:
            sj = lines[j].strip()
            if _PATCHED_NESTED_RE.match(sj):
                depth += 1
            elif _PATCHED_END_RE.match(sj):
                depth -= 1
                if depth == 0:
                    end_idx = j
            j += 1

        if end_idx is None:
            # Malformed — keep as-is
            result.append(lines[i])
            i += 1
            continue

        i = end_idx + 1

    return "\n".join(result)


def strip_patched_ifdefs_in_db(db_path: Path, verbose: bool = False) -> int:
    """Strip #ifdef PATCHED blocks from all function sources in a DB.

    Returns the number of functions updated.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    updated = 0
    rows = conn.execute("SELECT id, name, source FROM functions").fetchall()
    for row in rows:
        old = row["source"]
        if not old or "PATCHED" not in old:
            continue
        new = strip_patched_ifdefs(old)
        if new != old:
            conn.execute(
                "UPDATE functions SET source = ? WHERE id = ?",
                (new, row["id"]),
            )
            updated += 1
            if verbose:
                print(f"    {row['name']}: stripped #ifdef PATCHED from source")
    conn.commit()
    conn.close()
    return updated


def _infer_project_root(source_files: list[str]) -> Path | None:
    """Infer project root as common parent of all source files."""
    if not source_files:
        return None
    parents = [Path(f).resolve().parent for f in source_files]
    root = parents[0]
    for p in parents[1:]:
        # Walk up until we find a common ancestor
        while root != p and root not in p.parents:
            root = root.parent
    return root


def find_challenges(cgc_dir: Path) -> list[str]:
    """Find single-binary challenge names (skip multi-binary)."""
    challenges_dir = cgc_dir / "challenges"
    results = []
    for d in sorted(challenges_dir.iterdir()):
        if d.is_dir() and not (d / "cb_1").exists():
            results.append(d.name)
    return results


def remap_path(path: str, cgc_dir: str) -> str:
    """Remap /workspace/ Docker path to host path."""
    return path.replace("/workspace/", cgc_dir.rstrip("/") + "/")


def filter_compile_commands(
    all_entries: list[dict],
    challenge_name: str,
    cgc_dir: str,
    patched: bool = False,
) -> list[dict]:
    """Filter and remap compile_commands entries for a single challenge.

    Args:
        patched: If False (default), keep unpatched entries (no -DPATCHED).
                 If True, keep patched entries (with -DPATCHED).
    """
    result = []
    challenge_pattern = f"/challenges/{challenge_name}/"

    for entry in all_entries:
        file_path = entry.get("file", "")

        # Only keep entries for this challenge
        if challenge_pattern not in file_path:
            continue

        command = entry.get("command", "")
        has_patched = "-DPATCHED" in command

        # Filter based on patched flag
        if patched and not has_patched:
            continue
        if not patched and has_patched:
            continue

        # Exclude pov files
        if "/pov" in file_path.lower():
            continue

        # Remap /workspace/ paths
        remapped = {}
        for key in ("directory", "file", "output"):
            if key in entry:
                remapped[key] = remap_path(entry[key], cgc_dir)
        if "command" in entry:
            remapped["command"] = remap_path(entry["command"], cgc_dir)
        if "arguments" in entry:
            remapped["arguments"] = [
                remap_path(a, cgc_dir) for a in entry["arguments"]
            ]

        result.append(remapped)

    return result


def scan_challenge(
    name: str,
    func_scans_dir: Path,
    patched: bool = False,
    verbose: bool = False,
) -> dict:
    """Run function extraction scan for a single challenge.

    Args:
        patched: If True, use patched compile_commands and keep patched code.
    """
    challenge_dir = func_scans_dir / name
    cc_name = "compile_commands_patched.json" if patched else "compile_commands.json"
    db_name = "functions_patched.db" if patched else "functions.db"
    cc_path = challenge_dir / cc_name
    db_path = challenge_dir / db_name

    result = {
        "challenge": name,
        "source_files": 0,
        "functions": 0,
        "targets": 0,
        "callsites": 0,
        "error": None,
        "timing_seconds": 0.0,
    }

    if not cc_path.exists():
        result["error"] = f"no {cc_name}"
        return result

    start = time.monotonic()

    try:
        cc = CompileCommandsDB(cc_path)
        all_files = cc.get_all_files()
        source_files = [
            f for f in all_files if Path(f).suffix.lower() in C_EXTENSIONS
        ]
        if not source_files:
            result["error"] = "no source files"
            return result

        result["source_files"] = len(source_files)

        db = SummaryDB(str(db_path))
        try:
            project_root = _infer_project_root(source_files)
            extractor = FunctionExtractor(
                compile_commands=cc, enable_preprocessing=True,
                project_root=project_root,
            )

            all_functions = []
            all_typedefs = []
            parsed_tus = []

            for f in source_files:
                try:
                    tu = extractor.parse_file(f)
                    funcs = extractor.extract_from_tu(tu, f)
                    all_functions.extend(funcs)
                    all_typedefs.extend(
                        extractor.extract_typedefs_from_tu(tu, f)
                    )
                    parsed_tus.append((tu, f))
                except Exception as e:
                    if verbose:
                        print(f"      parse error {f}: {e}")

            db.insert_functions_batch(all_functions)
            db.insert_typedefs_batch(all_typedefs)
            db.close()

            # Strip #ifdef PATCHED blocks from raw source so the LLM
            # only sees the code path resolved by the preprocessor.
            n_stripped = strip_patched_ifdefs_in_db(db_path, verbose=verbose)
            if verbose and n_stripped:
                print(f"      stripped PATCHED ifdefs from {n_stripped} functions")

            db = SummaryDB(str(db_path))

            # Address-taken scan + callsite finding
            scanner = AddressTakenScanner(db, compile_commands=cc)
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

            # Populate call_edges from callsites (direct calls).
            # batch_summarize.py requires call_edges > 0.
            func_name_to_id = {
                func.name: func.id
                for func in db.get_all_functions()
                if func.id is not None
            }
            edges = []
            for func in db.get_all_functions():
                if func.id is None or not func.callsites:
                    continue
                for cs in func.callsites:
                    callee_name = cs.get("callee")
                    callee_id = func_name_to_id.get(callee_name)
                    if callee_id is not None:
                        edges.append(CallEdge(
                            caller_id=func.id,
                            callee_id=callee_id,
                            is_indirect=False,
                            file_path=func.file_path,
                            line=cs.get("line"),
                        ))
            if edges:
                db.add_call_edges_batch(edges)

            result["functions"] = len(all_functions)
            result["targets"] = len(atfs)
            result["callsites"] = len(callsites)
            result["call_edges"] = len(edges)

        finally:
            db.close()

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def patch_rescan(
    name: str,
    func_scans_dir: Path,
    verbose: bool = False,
) -> dict:
    """Copy unpatched DB, re-scan with patched code, delete summaries for changed functions.

    1. Copy functions.db → functions_patched.db
    2. Re-extract functions from patched compile_commands
    3. For each function: compare source_hash, if changed → delete all summaries
    """
    import shutil

    challenge_dir = func_scans_dir / name
    unpatched_db = challenge_dir / "functions.db"
    patched_db = challenge_dir / "functions_patched.db"
    patched_cc = challenge_dir / "compile_commands_patched.json"

    result = {
        "challenge": name,
        "functions_changed": 0,
        "functions_unchanged": 0,
        "error": None,
    }

    if not unpatched_db.exists():
        result["error"] = "no functions.db"
        return result
    if not patched_cc.exists():
        result["error"] = "no compile_commands_patched.json"
        return result

    # Step 1: Copy DB (only if patched DB doesn't exist yet)
    if not patched_db.exists():
        shutil.copy2(str(unpatched_db), str(patched_db))

    # Step 2: Load existing source hashes
    db = SummaryDB(str(patched_db))
    try:
        old_hashes: dict[int, str | None] = {}
        for func in db.get_all_functions():
            if func.id is not None:
                old_hashes[func.id] = db.get_function_source_hash(func.id)

        # Step 3: Re-extract from patched compile_commands
        cc = CompileCommandsDB(patched_cc)
        all_files = cc.get_all_files()
        source_files = [
            f for f in all_files if Path(f).suffix.lower() in C_EXTENSIONS
        ]
        project_root = _infer_project_root(source_files)
        extractor = FunctionExtractor(
            compile_commands=cc, enable_preprocessing=True,
            project_root=project_root,
        )

        all_functions = []
        for f in source_files:
            try:
                tu = extractor.parse_file(f)
                funcs = extractor.extract_from_tu(tu, f)
                all_functions.extend(funcs)
            except Exception:
                pass

        # Re-insert (ON CONFLICT updates source, pp_source, source_hash)
        db.insert_functions_batch(all_functions)
        db.close()

        # Strip #ifdef PATCHED blocks from raw source
        n_stripped = strip_patched_ifdefs_in_db(patched_db, verbose=verbose)
        if verbose and n_stripped:
            print(f"    stripped PATCHED ifdefs from {n_stripped} functions")

        db = SummaryDB(str(patched_db))

        # Step 4: Compare hashes, delete summaries for changed functions
        summary_tables = [
            "allocation_summaries", "free_summaries", "init_summaries",
            "memsafe_summaries", "verification_summaries",
        ]
        for func in db.get_all_functions():
            if func.id is None:
                continue
            new_hash = db.get_function_source_hash(func.id)
            old_hash = old_hashes.get(func.id)
            if old_hash is not None and new_hash != old_hash:
                result["functions_changed"] += 1
                # Delete all summaries for this function
                for table in summary_tables:
                    db.conn.execute(
                        f"DELETE FROM {table} WHERE function_id = ?",
                        (func.id,),
                    )
                if verbose:
                    print(f"    {func.name}: source changed, summaries cleared")
            else:
                result["functions_unchanged"] += 1

        db.conn.commit()

    finally:
        db.close()

    return result


def callgraph_challenge(
    name: str,
    func_scans_dir: Path,
    kamain_bin: str = "kanalyzer",
    verbose: bool = False,
) -> dict:
    """Run call graph generation for a single challenge."""
    challenge_dir = func_scans_dir / name
    cc_path = challenge_dir / "compile_commands.json"
    db_path = challenge_dir / "functions.db"

    result = {
        "challenge": name,
        "bc_files": 0,
        "edges_imported": 0,
        "error": None,
        "timing_seconds": 0.0,
    }

    if not cc_path.exists() or not db_path.exists():
        result["error"] = "missing compile_commands.json or functions.db"
        return result

    start = time.monotonic()

    try:
        with open(cc_path) as f:
            entries = json.load(f)

        recompile_dir = challenge_dir / "recompiled_bc"
        bc_files, bc_stats = collect_bc_files(
            entries, recompile_dir=recompile_dir, verbose=verbose,
        )
        result["bc_files"] = len(bc_files)

        if not bc_files:
            result["error"] = "no_bc_files"
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            return result

        if verbose:
            print(
                f"    bc: {len(bc_files)} "
                f"(T1:{bc_stats['tier1_save_temps']} "
                f"T2:{bc_stats['tier2_lto_obj']} "
                f"T3:{bc_stats['tier3_recompiled']})"
            )

        callgraph_json = challenge_dir / "callgraph.json"
        ok, err, dur = run_kamain(
            bc_files=bc_files,
            output_json=callgraph_json,
            kamain_bin=kamain_bin,
            cfl_compositional=True,
            verbose=verbose,
        )

        if not ok:
            result["error"] = f"kamain_failed: {err}"
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            return result

        import_stats = import_callgraph(
            json_path=callgraph_json,
            db_path=str(db_path),
            clear_edges=True,
            verbose=verbose,
        )
        result["edges_imported"] = import_stats["edges_imported"]

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CGC challenges for llm-summary pipeline"
    )
    parser.add_argument(
        "--cgc-dir", type=Path,
        default=Path("/data/csong/cgc/cb-multios"),
        help="Path to cb-multios directory",
    )
    parser.add_argument(
        "--func-scans-dir", type=Path, default=FUNC_SCANS_DIR,
        help="Output directory for per-challenge scan data",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process challenges matching this substring",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Skip compile_commands filtering, only run scan",
    )
    parser.add_argument(
        "--no-scan", action="store_true",
        help="Only filter compile_commands, skip scan",
    )
    parser.add_argument(
        "--callgraph", action="store_true", default=True,
        help="Run call graph generation after scan (default)",
    )
    parser.add_argument(
        "--no-callgraph", action="store_true",
        help="Skip call graph generation",
    )
    parser.add_argument(
        "--kamain-bin", type=str, default="kanalyzer",
        help="Path to KAMain/kanalyzer binary",
    )
    parser.add_argument(
        "--patch", action="store_true",
        help="Create patched variant: copy DB, re-scan with -DPATCHED, "
             "clear summaries for changed functions",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    challenges = find_challenges(args.cgc_dir)
    print(f"Found {len(challenges)} single-binary challenges")

    if args.filter:
        filt = args.filter.lower()
        challenges = [c for c in challenges if filt in c.lower()]
        print(f"Filter '{args.filter}': {len(challenges)} challenges")

    if args.limit:
        challenges = challenges[: args.limit]

    # Load compile_commands.json once
    all_entries = None
    if not args.scan_only:
        cc_path = args.cgc_dir / "build" / "compile_commands.json"
        if not cc_path.exists():
            print(f"Error: {cc_path} not found")
            sys.exit(1)
        with open(cc_path) as f:
            all_entries = json.load(f)
        print(f"Loaded {len(all_entries)} compile_commands entries")

    cgc_dir_str = str(args.cgc_dir)
    args.func_scans_dir.mkdir(parents=True, exist_ok=True)

    prepared = 0
    scanned = 0
    errors = 0

    for i, name in enumerate(challenges, 1):
        print(f"[{i}/{len(challenges)}] {name}...", end=" ", flush=True)

        challenge_dir = args.func_scans_dir / name
        challenge_dir.mkdir(parents=True, exist_ok=True)

        if args.patch:
            # Patch mode: filter patched compile_commands + copy DB + rescan
            if not args.scan_only and all_entries is not None:
                entries = filter_compile_commands(
                    all_entries, name, cgc_dir_str, patched=True,
                )
                if not entries:
                    print("SKIP (no patched compile_commands entries)")
                    continue
                cc_out = challenge_dir / "compile_commands_patched.json"
                with open(cc_out, "w") as f:
                    json.dump(entries, f, indent=2)
                prepared += 1

            pr = patch_rescan(name, args.func_scans_dir, verbose=args.verbose)
            if pr["error"]:
                print(f"patch ERROR: {pr['error']}")
                errors += 1
            else:
                print(
                    f"{pr['functions_changed']} changed, "
                    f"{pr['functions_unchanged']} unchanged"
                )
                scanned += 1
            continue

        # Step 1: Filter compile_commands (unpatched)
        if not args.scan_only and all_entries is not None:
            entries = filter_compile_commands(all_entries, name, cgc_dir_str)
            if not entries:
                print("SKIP (no compile_commands entries)")
                continue
            cc_out = challenge_dir / "compile_commands.json"
            with open(cc_out, "w") as f:
                json.dump(entries, f, indent=2)
            prepared += 1

        # Step 2: Run scan
        if not args.no_scan:
            result = scan_challenge(name, args.func_scans_dir, verbose=args.verbose)
            if result["error"]:
                print(f"scan ERROR: {result['error']}")
                errors += 1
                continue
            else:
                msg = (
                    f"{result['functions']} funcs, "
                    f"{result['callsites']} callsites"
                )
                scanned += 1

                # Step 3: Run call graph
                if not args.no_callgraph:
                    cg_result = callgraph_challenge(
                        name, args.func_scans_dir,
                        kamain_bin=args.kamain_bin,
                        verbose=args.verbose,
                    )
                    if cg_result["error"]:
                        msg += f", cg ERROR: {cg_result['error']}"
                    else:
                        msg += f", {cg_result['edges_imported']} edges"

                msg += f" ({result['timing_seconds']}s)"
                print(msg)
        else:
            entries_count = len(entries) if not args.scan_only else "?"
            print(f"{entries_count} entries")

    # Generate cgc_projects.json
    projects = []
    for name in sorted(challenges):
        db_path = args.func_scans_dir / name / "functions.db"
        if db_path.exists():
            projects.append({"project_dir": name})

    projects_json_path = REPO_ROOT / "scripts" / "cgc_projects.json"
    with open(projects_json_path, "w") as f:
        json.dump(projects, f, indent=2)

    print(f"\nSummary:")
    print(f"  Prepared: {prepared}")
    print(f"  Scanned: {scanned}")
    print(f"  Errors: {errors}")
    print(f"  Projects JSON: {projects_json_path} ({len(projects)} entries)")


if __name__ == "__main__":
    main()
