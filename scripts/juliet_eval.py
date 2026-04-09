#!/usr/bin/env python3
"""Evaluate llm-summary verify pass on SV-COMP Juliet benchmarks.

Phases:
  0  scan          Extract functions from .i files
  1  callgraph     Compile to .bc, run KAMain, import call graph
  2  summarize     Bottom-up passes: alloc, free, init, memsafe
  3  verify        Verification pass
  4  evaluate      Collect issues, score against ground truth

Usage:
    source ~/project/llm-summary/venv/bin/activate
    python scripts/juliet_eval.py \
        --benchmarks /data/csong/opensource/sv-benchmarks/c/Juliet_Test \
        --cwe CWE415 \
        --variant 01 \
        --backend claude \
        --output juliet_eval_results.json \
        -v

    # Re-run only verify + evaluate (skip scan/callgraph/summarize):
    python scripts/juliet_eval.py \
        --benchmarks ... --cwe CWE415 --variant 01 \
        --backend claude --from-phase 3 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_summary.callgraph_import import CallGraphImporter
from llm_summary.cli import _load_compile_commands
from llm_summary.db import SummaryDB
from llm_summary.driver import (
    AllocationPass,
    BottomUpDriver,
    FreePass,
    InitPass,
    IntegerOverflowPass,
    LeakPass,
    MemsafePass,
    SummaryPass,
    VerificationPass,
)
from llm_summary.extractor import FunctionExtractor
from llm_summary.free_summarizer import FreeSummarizer
from llm_summary.init_summarizer import InitSummarizer
from llm_summary.integer_overflow_summarizer import IntegerOverflowSummarizer
from llm_summary.leak_summarizer import LeakSummarizer
from llm_summary.llm import LLMBackend, build_backend_kwargs, create_backend
from llm_summary.memsafe_summarizer import MemsafeSummarizer
from llm_summary.summarizer import AllocationSummarizer
from llm_summary.verification_summarizer import VerificationSummarizer

log = logging.getLogger("juliet_eval")

# Token-tracking keys shared across all summarizers
_TOKEN_KEYS = ("llm_calls", "input_tokens", "output_tokens")


def _merge_stats(*summarizers: Any) -> dict[str, int]:
    """Sum token-tracking keys across multiple summarizers."""
    merged: dict[str, int] = dict.fromkeys(_TOKEN_KEYS, 0)
    for s in summarizers:
        st = s.stats
        for k in _TOKEN_KEYS:
            merged[k] += int(st.get(k, 0))
    return merged

DEFAULT_KAMAIN = "/home/csong/project/kanalyzer/release/lib/KAMain"
DEFAULT_FUNC_SCANS = "func-scans/sv-benchmarks"

# ── SV-COMP property -> our issue_kind mapping ─────────────────────────────
SUBPROP_TO_KINDS: dict[str, set[str]] = {
    "valid-free": {"double_free", "use_after_free", "invalid_free"},
    "valid-deref": {"null_deref", "buffer_overflow", "use_after_free", "uninitialized_use"},
    "valid-memtrack": {"memory_leak"},
    "valid-memsafety": {
        "double_free", "use_after_free", "invalid_free",
        "null_deref", "buffer_overflow", "memory_leak", "uninitialized_use",
    },
    "no-overflow": {"integer_overflow", "division_by_zero", "shift_ub"},
}

# Boilerplate functions shared across all Juliet .i files (identical source)
BOILERPLATE_FUNCTIONS = {
    "assume_abort_if_not", "reach_error", "internal_start",
    # ldv_ wrappers
    "ldv_asprintf", "ldv_calloc", "ldv_error", "ldv_exit", "ldv_exit_1",
    "ldv_exit_2", "ldv_free", "ldv_malloc", "ldv_realloc",
    "ldv_reference_calloc", "ldv_reference_free", "ldv_reference_malloc",
    "ldv_reference_realloc", "ldv_reference_xmalloc",
    "ldv_reference_xzalloc", "ldv_reference_zalloc",
    "ldv_strcpy", "ldv_strdup", "ldv_strcpy_1", "ldv_strlen", "ldv_strncpy",
    "ldv_undef_int", "ldv_undef_int_negative", "ldv_undef_int_nonpositive",
    "ldv_undef_int_positive", "ldv_undef_long", "ldv_undef_uint",
    "ldv_undef_ulong", "ldv_undef_ulonglong",
    "ldv_xmalloc", "ldv_xzalloc", "ldv_zalloc",
    # print helpers
    "printBytesLine", "printDoubleLine", "printFloatLine",
    "printHexCharLine", "printHexUnsignedCharLine", "printIntLine",
    "printLine", "printLongLine", "printLongLongLine", "printShortLine",
    "printSizeTLine", "printStructLine", "printUnsignedLine",
    "printWLine", "printWcharLine",
    # decode helpers
    "decodeHexChars", "decodeHexWChars",
    # global predicates
    "globalReturnsFalse", "globalReturnsTrue", "globalReturnsTrueOrFalse",
    "staticReturnsFalse",
    # threading stubs
    "stdThreadCreate", "stdThreadDestroy", "stdThreadJoin",
    "stdThreadLockAcquire", "stdThreadLockCreate",
    "stdThreadLockDestroy", "stdThreadLockRelease",
    # bswap intrinsics
    "__bswap_16", "__bswap_32", "__bswap_64",
    "__uint16_identity", "__uint32_identity", "__uint64_identity",
}


@dataclass
class TaskResult:
    """Result for one Juliet task (.yml file)."""

    yml_file: str
    i_file: str
    cwe: str
    variant: str  # "bad" or "good"
    expected_safe: bool  # True = no bug, False = has bug
    subproperty: str
    issues_found: list[dict[str, Any]] = field(default_factory=list)
    predicted_safe: bool | None = None
    correct: bool | None = None
    classification: str = ""  # TP, TN, FP, FN
    error: str | None = None
    elapsed_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def parse_yml(yml_path: Path) -> dict[str, Any] | None:
    """Parse an SV-COMP .yml task file. Returns None if no relevant property."""
    with open(yml_path) as f:
        data = yaml.safe_load(f)

    # input_files is relative to the yml's directory
    i_file = data["input_files"]
    source_path = yml_path.parent / i_file
    props = data.get("properties", [])

    # Properties we can evaluate (valid-memsafety has sub-properties)
    relevant_props = {"valid-memsafety", "no-overflow"}

    expected_verdict = True
    subproperty = ""
    found = False
    for prop in props:
        if "expected_verdict" not in prop:
            continue
        pf = prop.get("property_file", "")
        base = Path(pf).stem  # e.g., "valid-memsafety", "no-overflow"
        sub = prop.get("subproperty", "")
        if base in relevant_props or sub in SUBPROP_TO_KINDS:
            expected_verdict = prop["expected_verdict"]
            subproperty = sub if sub else base
            found = True
            break

    if not found:
        return None

    return {
        "i_file": i_file,
        "source_path": str(source_path),
        "expected_verdict": expected_verdict,
        "subproperty": subproperty,
    }


def find_juliet_target_functions(db: SummaryDB, variant: str) -> list[str]:
    """Find Juliet CWE-specific functions to check.

    For _bad files: the CWE*_bad named function.
    For _good files: static helpers like goodG2B, goodB2G.
    """
    funcs = db.get_all_functions()
    targets = []
    for f in funcs:
        name = f.name
        if name == "main":
            continue
        if name in BOILERPLATE_FUNCTIONS:
            continue
        # Skip empty stubs (bad1-bad9, good1-good9)
        if re.match(r"^(bad|good)\d+$", name):
            continue
        if variant == "bad":
            if name.endswith("_bad"):
                targets.append(name)
        else:
            # Static helpers (goodG2B, goodB2G, etc.) and wrapper
            if name.startswith("good") or name.endswith("_good"):
                targets.append(name)

    return targets


def task_dir_name(yml_stem: str) -> str:
    """Derive a directory name from the .yml stem.

    E.g. 'CWE415_..._char_44_bad' -> same string, used as subdir name.
    The full stem is unique per task so we use it directly.
    """
    return yml_stem


# ── Phase 0: scan ───────────────────────────────────────────────────────────

def phase_scan(
    i_path: Path,
    work_dir: Path,
    db: SummaryDB,
) -> int:
    """Extract functions from source file into DB. Returns function count."""
    cc_json = [{
        "directory": str(i_path.parent),
        "file": str(i_path),
        "command": f"clang -c {i_path.name}",
    }]
    cc_path = work_dir / "compile_commands.json"
    cc_path.write_text(json.dumps(cc_json))

    compile_commands, _ = _load_compile_commands(str(cc_path), None)

    extractor = FunctionExtractor(
        compile_commands=compile_commands,
        enable_preprocessing=(i_path.suffix != ".i"),
    )
    functions = extractor.extract_from_file(i_path)
    db.insert_functions_batch(functions)
    typedefs = extractor.extract_typedefs_from_file(i_path)
    if typedefs:
        db.insert_typedefs_batch(typedefs)
    return len(functions)


# ── Phase 1: callgraph ─────────────────────────────────────────────────────

def phase_callgraph(
    i_path: Path,
    work_dir: Path,
    db: SummaryDB,
    kamain_bin: str,
    verbose: bool,
) -> int:
    """Compile .i to .bc, run KAMain, import call graph. Returns edge count."""
    bc_path = work_dir / "input.bc"
    cg_json = work_dir / "callgraph.json"
    vsnap_path = work_dir / "v-snapshot.vsnap"

    compile_cmd = [
        "clang-18", "-emit-llvm", "-c", "-Wno-everything",
        str(i_path), "-o", str(bc_path),
    ]
    result = subprocess.run(
        compile_cmd, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"clang -emit-llvm failed: {result.stderr}")

    kamain_cmd = [
        kamain_bin, str(bc_path),
        "--callgraph-json", str(cg_json),
        "--v-snapshot", str(vsnap_path),
        "--verbose", "0",
    ]
    result = subprocess.run(
        kamain_cmd, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"KAMain failed: {result.stderr}")

    importer = CallGraphImporter(db, verbose=verbose)
    stats = importer.import_json(cg_json, clear_existing=True)
    log.debug(
        "  KAMain: %d edges (%d direct, %d indirect)",
        stats.edges_imported, stats.direct_edges, stats.indirect_edges,
    )
    return int(stats.edges_imported)


# ── Phase 2: summarize ──────────────────────────────────────────────────────

def phase_summarize(
    db: SummaryDB,
    llm: LLMBackend,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
    alias_builder: Any | None = None,
    cache_mode: str = "none",
) -> dict[str, int]:
    """Run alloc/free/init/memsafe passes. Returns token stats."""

    alloc_s = AllocationSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    free_s = FreeSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    init_s = InitSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    memsafe_s = MemsafeSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )

    passes: list[SummaryPass] = [
        AllocationPass(alloc_s, db, llm.model),
        FreePass(free_s, db, llm.model),
        InitPass(init_s, db, llm.model),
        MemsafePass(memsafe_s, db, llm.model, alias_builder=alias_builder),
    ]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(passes, force=force, target_ids=reachable_ids)

    return _merge_stats(alloc_s, free_s, init_s, memsafe_s)


# ── Phase 2.5: leak detection ────────────────────────────────────────────────

def phase_leak(
    db: SummaryDB,
    llm: LLMBackend,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
    entry_functions: set[str] | None = None,
    cache_mode: str = "none",
) -> dict[str, int]:
    """Run leak detection pass. Returns token stats."""

    leak_s = LeakSummarizer(
        db, llm, verbose=verbose, log_file=log_llm,
        entry_functions=entry_functions,
    )

    l_passes: list[SummaryPass] = [
        LeakPass(leak_s, db, llm.model),
    ]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(l_passes, force=force, target_ids=reachable_ids)

    return _merge_stats(leak_s)


# ── Phase 2.6: integer overflow detection ───────────────────────────────────

def phase_overflow(
    db: SummaryDB,
    llm: LLMBackend,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
    cache_mode: str = "none",
) -> dict[str, int]:
    """Run integer overflow detection pass. Returns token stats."""

    ovf_s = IntegerOverflowSummarizer(
        db, llm, verbose=verbose, log_file=log_llm,
    )

    o_passes: list[SummaryPass] = [
        IntegerOverflowPass(ovf_s, db, llm.model),
    ]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(o_passes, force=force, target_ids=reachable_ids)

    return _merge_stats(ovf_s)


# ── Phase 3: verify ─────────────────────────────────────────────────────────

def phase_verify(
    db: SummaryDB,
    llm: LLMBackend,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
    entry_functions: set[str] | None = None,
    alias_builder: Any | None = None,
    cache_mode: str = "none",
) -> dict[str, int]:
    """Run verification pass. Returns token stats."""

    verify_s = VerificationSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
        entry_functions=entry_functions,
    )

    v_passes: list[SummaryPass] = [
        VerificationPass(verify_s, db, llm.model, alias_builder=alias_builder),
    ]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(v_passes, force=force, target_ids=reachable_ids)

    return _merge_stats(verify_s)


# ── Phase 4: evaluate ───────────────────────────────────────────────────────

def _collect_issues(
    db: SummaryDB,
    target_names: list[str],
) -> list[dict[str, Any]]:
    """Collect verification issues from named functions."""
    issues: list[dict[str, Any]] = []
    for tname in target_names:
        for f in db.get_function_by_name(tname):
            assert f.id is not None
            vsm = db.get_verification_summary_by_function_id(f.id)
            if vsm and vsm.issues:
                for issue in vsm.issues:
                    issues.append(issue.to_dict())
            lsm = db.get_leak_summary_by_function_id(f.id)
            if lsm and lsm.issues:
                for issue in lsm.issues:
                    issues.append(issue.to_dict())
            osm = db.get_integer_overflow_summary_by_function_id(f.id)
            if osm and osm.issues:
                for issue in osm.issues:
                    issues.append(issue.to_dict())
    return issues


def phase_evaluate(
    db: SummaryDB,
    variant: str | None,
    reachable_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Collect issues from target functions. Returns issues list."""
    if variant:
        # Juliet: use variant-specific target selection
        targets = find_juliet_target_functions(db, variant)
    else:
        # General: collect issues from all reachable functions with source
        targets = []
        for f in db.get_all_functions():
            if not f.source:
                continue
            if reachable_ids and f.id not in reachable_ids:
                continue
            targets.append(f.name)
    return _collect_issues(db, targets)


# ── Summary cache for boilerplate functions ─────────────────────────────────

SUMMARY_TABLES = [
    "allocation_summaries",
    "free_summaries",
    "init_summaries",
    "memsafe_summaries",
    "verification_summaries",
    "leak_summaries",
    "integer_overflow_summaries",
]

# Key: (func_name, source_hash) → {table_name: summary_json}
SummaryCache = dict[tuple[str, str], dict[str, str]]


def build_summary_cache(db: SummaryDB) -> SummaryCache:
    """Extract boilerplate function summaries, keyed by (name, source_hash)."""
    cache: SummaryCache = {}
    for func in db.get_all_functions():
        if func.id is None or not func.source_hash:
            continue
        if func.name not in BOILERPLATE_FUNCTIONS:
            continue
        key = (func.name, func.source_hash)
        blobs: dict[str, str] = {}
        for table in SUMMARY_TABLES:
            row = db.conn.execute(
                f"SELECT summary_json FROM {table} WHERE function_id = ?",  # noqa: S608
                (func.id,),
            ).fetchone()
            if row:
                blobs[table] = row[0]
        if blobs:
            cache[key] = blobs
    return cache


def apply_summary_cache(db: SummaryDB, cache: SummaryCache) -> int:
    """Inject cached summaries into a DB for matching functions. Returns count."""
    injected = 0
    for func in db.get_all_functions():
        if func.id is None or not func.source_hash:
            continue
        key = (func.name, func.source_hash)
        if key not in cache:
            continue
        blobs = cache[key]
        for table, summary_json in blobs.items():
            existing = db.conn.execute(
                f"SELECT id FROM {table} WHERE function_id = ?",  # noqa: S608
                (func.id,),
            ).fetchone()
            if existing:
                continue
            db.conn.execute(
                f"INSERT INTO {table} (function_id, summary_json, model_used) "  # noqa: S608
                "VALUES (?, ?, ?)",
                (func.id, summary_json, "cached"),
            )
        injected += 1
    db.conn.commit()
    return injected


# ── Task runner ──────────────────────────────────────────────────────────────

def run_one_task(
    i_path: Path,
    variant: str | None,
    work_dir: Path,
    backend: str,
    model: str | None,
    verbose: bool,
    kamain_bin: str = DEFAULT_KAMAIN,
    log_llm: str | None = None,
    from_phase: int = 0,
    to_phase: int = 4,
    force: bool = False,
    summary_cache: SummaryCache | None = None,
    disable_thinking: bool = False,
    llm_host: str = "localhost",
    llm_port: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run pipeline phases on one .i file using persistent work_dir.

    Returns (issues_list, stats_dict).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "functions.db"
    db = SummaryDB(str(db_path))
    total_stats: dict[str, int] = dict.fromkeys(_TOKEN_KEYS, 0)

    # Create one backend instance for all phases
    backend_kwargs = build_backend_kwargs(
        backend, llm_host=llm_host, llm_port=llm_port,
        disable_thinking=disable_thinking,
    )
    llm = create_backend(backend, model=model, **backend_kwargs)
    cache_mode = "source" if backend == "claude" else "none"

    def _add_stats(phase_stats: dict[str, int]) -> None:
        for k in _TOKEN_KEYS:
            total_stats[k] += phase_stats.get(k, 0)

    try:
        # Phase 0: scan
        if from_phase <= 0:
            n_funcs = phase_scan(i_path, work_dir, db)
            log.debug("  Extracted %d functions", n_funcs)

        # Phase 1: callgraph
        if from_phase <= 1:
            n_edges = phase_callgraph(
                i_path, work_dir, db, kamain_bin, verbose,
            )
            log.debug("  Call edges: %d", n_edges)

        # Compute reachable functions from main
        reachable_ids: set[int] | None = None
        main_funcs = db.get_function_by_name("main")
        if main_funcs:
            driver = BottomUpDriver(db, verbose=False)
            graph, _ = driver.build_graph()
            main_id = main_funcs[0].id
            assert main_id is not None
            reachable_ids = driver.compute_reachable({main_id}, graph)
            log.debug("  Reachable from main: %d functions", len(reachable_ids))

        # Init stdlib summaries for stubs (malloc, free, realloc, etc.)
        if from_phase <= 1:
            from llm_summary.stdlib import (
                STDLIB_ATTRIBUTES,
                get_all_stdlib_free_summaries,
                get_all_stdlib_init_summaries,
                get_all_stdlib_memsafe_summaries,
                get_all_stdlib_summaries,
            )

            for name, summary in get_all_stdlib_summaries().items():
                # SV-COMP assumes allocators never fail
                for alloc in summary.allocations:
                    if alloc.may_be_null:
                        alloc.may_be_null = False
                funcs = db.get_function_by_name(name)
                if funcs:
                    f = funcs[0]
                    attrs = STDLIB_ATTRIBUTES.get(name, "")
                    if attrs and not f.attributes:
                        f.attributes = attrs
                        db.conn.execute(
                            "UPDATE functions SET attributes = ? "
                            "WHERE id = ?", (attrs, f.id),
                        )
                    db.upsert_summary(f, summary, model_used="builtin")
            for name, fs in get_all_stdlib_free_summaries().items():
                funcs = db.get_function_by_name(name)
                if funcs:
                    db.upsert_free_summary(funcs[0], fs, model_used="builtin")
            for name, isum in get_all_stdlib_init_summaries().items():
                funcs = db.get_function_by_name(name)
                if funcs:
                    db.upsert_init_summary(funcs[0], isum, model_used="builtin")
            for name, msum in get_all_stdlib_memsafe_summaries().items():
                funcs = db.get_function_by_name(name)
                if funcs:
                    db.upsert_memsafe_summary(
                        funcs[0], msum, model_used="builtin",
                    )
            # Update attributes for attribute-only functions
            for name in STDLIB_ATTRIBUTES:
                funcs = db.get_function_by_name(name)
                if funcs and not funcs[0].attributes:
                    attrs = STDLIB_ATTRIBUTES[name]
                    db.conn.execute(
                        "UPDATE functions SET attributes = ? "
                        "WHERE id = ?", (attrs, funcs[0].id),
                    )
            # SV-COMP __VERIFIER_nondet_* builtins: full-range nondeterministic
            _verifier_nondet = {
                "__VERIFIER_nondet_int": ("int", "sizeof(int)"),
                "__VERIFIER_nondet_uint": ("unsigned int", "sizeof(unsigned int)"),
                "__VERIFIER_nondet_long": ("long", "sizeof(long)"),
                "__VERIFIER_nondet_ulong": ("unsigned long", "sizeof(unsigned long)"),
                "__VERIFIER_nondet_short": ("short", "sizeof(short)"),
                "__VERIFIER_nondet_ushort": (
                    "unsigned short", "sizeof(unsigned short)",
                ),
                "__VERIFIER_nondet_char": ("char", "sizeof(char)"),
                "__VERIFIER_nondet_uchar": (
                    "unsigned char", "sizeof(unsigned char)",
                ),
                "__VERIFIER_nondet_bool": ("_Bool", "sizeof(_Bool)"),
                "__VERIFIER_nondet_float": ("float", "sizeof(float)"),
                "__VERIFIER_nondet_double": ("double", "sizeof(double)"),
            }
            from llm_summary.models import (
                InitOp,
                InitSummary,
                MemsafeSummary,
                OutputRange,
            )
            for vname, (vtype, vsize) in _verifier_nondet.items():
                funcs = db.get_function_by_name(vname)
                if funcs:
                    isum = InitSummary(
                        function_name=vname,
                        inits=[InitOp(
                            target="return value",
                            target_kind="return_value",
                            initializer="nondeterministic",
                            byte_count=vsize,
                        )],
                        output_ranges=[OutputRange(
                            target="return value",
                            range=f"full {vtype} range",
                            description=(
                                f"Any valid {vtype} value; "
                                "arithmetic on this value may overflow"
                            ),
                        )],
                        description=f"Returns a nondeterministic {vtype} value "
                        f"(any value in the full {vtype} range).",
                    )
                    db.upsert_init_summary(funcs[0], isum, model_used="builtin")
                    msum = MemsafeSummary(
                        function_name=vname,
                        contracts=[],
                        description="No safety pre-conditions required.",
                    )
                    db.upsert_memsafe_summary(
                        funcs[0], msum, model_used="builtin",
                    )
            db.conn.commit()

        # Load alias context from vsnapshot if available
        alias_builder = None
        vsnap_path = work_dir / "v-snapshot.vsnap"
        if vsnap_path.exists():
            from llm_summary.alias_context import AliasContextBuilder
            alias_builder = AliasContextBuilder(str(vsnap_path), db)

        # Inject cached summaries before summarization
        if summary_cache and not force:
            n_cached = apply_summary_cache(db, summary_cache)
            if n_cached:
                log.debug("  Injected %d cached summaries", n_cached)

        # Phase 2: summarize
        if from_phase <= 2 and to_phase >= 2:
            _add_stats(phase_summarize(
                db, llm, verbose, log_llm,
                force=force,
                reachable_ids=reachable_ids,
                alias_builder=alias_builder,
                cache_mode=cache_mode,
            ))

        # Phase 3: leak + overflow + verify
        if from_phase <= 3 and to_phase >= 3:
            _add_stats(phase_leak(
                db, llm, verbose, log_llm,
                force=force,
                reachable_ids=reachable_ids,
                entry_functions={"main"} if main_funcs else None,
                cache_mode=cache_mode,
            ))
            _add_stats(phase_overflow(
                db, llm, verbose, log_llm,
                force=force,
                reachable_ids=reachable_ids,
                cache_mode=cache_mode,
            ))
            _add_stats(phase_verify(
                db, llm, verbose, log_llm,
                force=force,
                reachable_ids=reachable_ids,
                entry_functions={"main"} if main_funcs else None,
                alias_builder=alias_builder,
                cache_mode=cache_mode,
            ))

        if to_phase < 4:
            db.close()
            return [], total_stats

        # Phase 4: evaluate
        issues = phase_evaluate(db, variant, reachable_ids=reachable_ids)
        log.debug("  Issues: %d", len(issues))
        db.close()
        return issues, total_stats

    except Exception:
        db.close()
        raise


def collect_tasks_from_set_file(
    set_file: Path,
    cwe_filter: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Collect tasks from a .set file (glob patterns relative to set file dir)."""
    base_dir = set_file.parent
    seen: set[Path] = set()
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for line in set_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for yml_path in sorted(base_dir.glob(line)):
            if yml_path in seen:
                continue
            seen.add(yml_path)
            name = yml_path.stem
            if cwe_filter:
                if not any(name.startswith(c) for c in cwe_filter.split(",")):
                    continue
            try:
                info = parse_yml(yml_path)
            except Exception as e:
                log.warning("Failed to parse %s: %s", yml_path, e)
                continue
            if info is None:
                continue
            tasks.append((yml_path, info))
    return tasks


def collect_tasks(
    benchmarks_dir: Path,
    cwe_filter: str | None,
    variant_filter: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Collect all matching .yml tasks from the Juliet_Test directory."""
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for yml_path in sorted(benchmarks_dir.glob("*.yml")):
        name = yml_path.stem

        if cwe_filter:
            if not any(name.startswith(c) for c in cwe_filter.split(",")):
                continue

        if variant_filter:
            if (f"_{variant_filter}_bad" not in name
                    and f"_{variant_filter}_good" not in name):
                continue

        try:
            info = parse_yml(yml_path)
        except Exception as e:
            log.warning("Failed to parse %s: %s", yml_path, e)
            continue
        if info is None:
            continue

        tasks.append((yml_path, info))

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate llm-summary on Juliet benchmarks",
    )
    parser.add_argument(
        "--benchmarks", default=None,
        help="Path to benchmark directory (required without --set-file)",
    )
    parser.add_argument(
        "--cwe", default=None,
        help="CWE filter (e.g., CWE415 or CWE415,CWE476)",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Variant number filter (e.g., 01)",
    )
    parser.add_argument("--backend", default="claude", help="LLM backend")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--llm-host", default="localhost", help="LLM server host")
    parser.add_argument("--llm-port", type=int, default=None, help="LLM server port")
    parser.add_argument(
        "--output", "-o", default="juliet_eval_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tasks to run (0=all)",
    )
    parser.add_argument(
        "--kamain-bin", default=DEFAULT_KAMAIN,
        help="Path to KAMain binary",
    )
    parser.add_argument(
        "--func-scans-dir", default=DEFAULT_FUNC_SCANS,
        help="Persistent work directory (default: %(default)s)",
    )
    parser.add_argument(
        "--set-file", default=None,
        help="SV-COMP .set file with glob patterns (e.g., Juliet.set)",
    )
    parser.add_argument(
        "--from-phase", type=int, default=0,
        help="Start from phase N (0=scan, 1=callgraph, 2=summarize, "
             "3=verify, 4=evaluate-only)",
    )
    parser.add_argument(
        "--to-phase", type=int, default=4,
        help="Stop after phase N (0=scan, 1=callgraph, 2=summarize, "
             "3=verify, 4=evaluate)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-run even if cached",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Only run tasks whose stem contains this substring",
    )
    parser.add_argument(
        "--rerun-from", default=None, metavar="RESULTS_JSON",
        help="Rerun only tasks with a given classification from a previous "
             "results JSON (use with --rerun-class)",
    )
    parser.add_argument(
        "--rerun-class", default="FP",
        help="Classification to rerun (default: FP). Used with --rerun-from",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="Disable thinking/reasoning for thinking models (Gemini, etc.)",
    )
    parser.add_argument(
        "--llm-log-dir", default=None,
        help="Directory for per-target LLM interaction logs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.verbose:
        log.setLevel(logging.DEBUG)

    func_scans = Path(args.func_scans_dir)

    if args.set_file:
        set_path = Path(args.set_file)
        if not set_path.exists():
            log.error("Set file not found: %s", set_path)
            sys.exit(1)
        tasks = collect_tasks_from_set_file(set_path, args.cwe)
    elif args.benchmarks:
        benchmarks_dir = Path(args.benchmarks)
        if not benchmarks_dir.exists():
            log.error("Benchmarks directory not found: %s", benchmarks_dir)
            sys.exit(1)
        tasks = collect_tasks(benchmarks_dir, args.cwe, args.variant)
    else:
        log.error("Either --benchmarks or --set-file is required")
        sys.exit(1)
    log.info("Collected %d tasks", len(tasks))

    if args.filter:
        tasks = [
            (p, info) for p, info in tasks if args.filter in p.stem
        ]
        log.info("Filtered to %d tasks matching '%s'", len(tasks), args.filter)

    if args.rerun_from:
        with open(args.rerun_from) as _rf:
            prev = json.load(_rf)
        rerun_ymls = {
            r["yml_file"]
            for r in prev["results"]
            if r["classification"] == args.rerun_class
        }
        tasks = [
            (p, info) for p, info in tasks if p.name in rerun_ymls
        ]
        log.info(
            "Rerunning %d %s tasks from %s",
            len(tasks), args.rerun_class, args.rerun_from,
        )

    if args.limit > 0:
        tasks = tasks[: args.limit]
        log.info("Limited to %d tasks", len(tasks))

    if args.from_phase > 0:
        log.info("Starting from phase %d", args.from_phase)

    results: list[TaskResult] = []
    tp = tn = fp = fn = errors = 0
    total_funcs = 0
    total_reachable = 0
    summary_cache: SummaryCache = {}

    for idx, (yml_path, info) in enumerate(tasks):
        i_path = Path(info["source_path"])
        expected_safe: bool = info["expected_verdict"]
        subprop: str = info["subproperty"]

        # Juliet variant detection (optional)
        variant: str | None = None
        if "_bad" in yml_path.stem:
            variant = "bad"
        elif "_good" in yml_path.stem:
            variant = "good"

        stem = yml_path.stem
        work_dir = func_scans / yml_path.parent.name / stem

        result = TaskResult(
            yml_file=yml_path.name,
            i_file=info["i_file"],
            cwe=stem.split("---")[0] if "---" in stem else stem.split("_")[0],
            variant=variant or "",
            expected_safe=expected_safe,
            subproperty=subprop,
        )

        log.info(
            "[%d/%d] %s (expected: %s)", idx + 1, len(tasks),
            stem, "safe" if expected_safe else "UNSAFE",
        )

        t0 = time.time()
        log_llm: str | None = None
        if args.llm_log_dir:
            log_dir = Path(args.llm_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_llm = str(log_dir / f"{stem}.log")

        try:
            issues, task_stats = run_one_task(
                i_path, variant, work_dir,
                args.backend, args.model,
                args.verbose, args.kamain_bin, log_llm,
                from_phase=args.from_phase, to_phase=args.to_phase,
                force=args.force,
                summary_cache=summary_cache,
                disable_thinking=args.disable_thinking,
                llm_host=args.llm_host,
                llm_port=args.llm_port,
            )
            result.elapsed_s = time.time() - t0
            result.llm_calls = task_stats["llm_calls"]
            result.input_tokens = task_stats["input_tokens"]
            result.output_tokens = task_stats["output_tokens"]
            result.issues_found = issues

            # Grow summary cache from completed task
            db_path = work_dir / "functions.db"
            if db_path.exists():
                sdb = SummaryDB(str(db_path))
                new_entries = build_summary_cache(sdb)
                sdb.close()
                added = 0
                for k, v in new_entries.items():
                    if k not in summary_cache:
                        summary_cache[k] = v
                        added += 1
                if added:
                    log.debug("  Cache: +%d entries (total %d)",
                              added, len(summary_cache))

            # Collect scan stats from DB
            if db_path.exists():
                sdb = SummaryDB(str(db_path))
                all_funcs = sdb.get_all_functions()
                total_funcs += len(all_funcs)
                main_funcs = sdb.get_function_by_name("main")
                if main_funcs:
                    drv = BottomUpDriver(sdb, verbose=False)
                    graph, _ = drv.build_graph()
                    mid = main_funcs[0].id
                    assert mid is not None
                    reachable = drv.compute_reachable({mid}, graph)
                    total_reachable += len(reachable)
                sdb.close()

            if args.to_phase >= 4:
                # Filter issues by subproperty if applicable
                if subprop and subprop in SUBPROP_TO_KINDS:
                    relevant_kinds = SUBPROP_TO_KINDS[subprop]
                    relevant_issues = [
                        i for i in issues
                        if i.get("issue_kind", "") in relevant_kinds
                    ]
                else:
                    relevant_issues = issues

                result.predicted_safe = len(relevant_issues) == 0
                result.correct = result.predicted_safe == expected_safe

                if expected_safe and result.predicted_safe:
                    result.classification = "TN"
                    tn += 1
                elif expected_safe and not result.predicted_safe:
                    result.classification = "FP"
                    fp += 1
                elif not expected_safe and not result.predicted_safe:
                    result.classification = "TP"
                    tp += 1
                else:
                    result.classification = "FN"
                    fn += 1

                log.info(
                    "  -> %s (%d issues, %.1fs, %d LLM calls, %dk in + %dk out)",
                    result.classification, len(relevant_issues),
                    result.elapsed_s, result.llm_calls,
                    result.input_tokens // 1000, result.output_tokens // 1000,
                )

        except Exception as e:
            result.elapsed_s = time.time() - t0
            result.error = str(e)
            errors += 1
            log.error("  -> ERROR: %s", e)

        results.append(result)

    # -- Summary --
    total = len(results)
    log.info("")

    if args.to_phase < 4:
        log.info("=== Scan Stats (phase 0-%d) ===", args.to_phase)
        log.info("Tasks: %d  Total functions: %d  Reachable: %d",
                 total, total_funcs, total_reachable)
        log.info("Errors: %d", errors)
    else:
        correct = tp + tn
        accuracy = correct / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        summary: dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "errors": errors,
        }

        log.info("=== Results ===")
        log.info(
            "Total: %d  Correct: %d  Accuracy: %.1f%%",
            total, correct, accuracy * 100,
        )
        log.info(
            "TP: %d  TN: %d  FP: %d  FN: %d  Errors: %d",
            tp, tn, fp, fn, errors,
        )
        log.info(
            "Precision: %.3f  Recall: %.3f  F1: %.3f",
            precision, recall, f1,
        )

    # Token totals
    total_llm_calls = sum(r.llm_calls for r in results)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    log.info(
        "Tokens: %d LLM calls, %dk input, %dk output",
        total_llm_calls, total_input // 1000, total_output // 1000,
    )

    output: dict[str, Any] = {
        "summary": {
            "total_tasks": total,
            "total_functions": total_funcs,
            "total_reachable": total_reachable,
            "errors": errors,
        } if args.to_phase < 4 else summary,
        "config": {
            "backend": args.backend,
            "model": args.model,
            "cwe": args.cwe,
            "variant": getattr(args, "variant", None),
            "to_phase": args.to_phase,
        },
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
