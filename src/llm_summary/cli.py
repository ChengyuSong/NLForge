"""Command-line interface for LLM-based allocation summary analysis."""

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .asm_extractor import ASM_EXTENSIONS, extract_asm_functions
from .compile_commands import CompileCommandsDB
from .db import SummaryDB
from .driver import BottomUpDriver, SummaryPass
from .extractor import C_EXTENSIONS, FunctionExtractor
from .indirect import AddressTakenScanner, IndirectCallsiteFinder
from .llm import create_backend
from .ordering import ProcessingOrderer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .llm.base import LLMBackend
    from .stdlib_cache import StdlibCache

console = Console()


def _build_backend_kwargs(
    backend: str,
    llm_host: str,
    llm_port: int | None,
    disable_thinking: bool,
) -> dict:
    """Build kwargs dict for create_backend() from CLI options."""
    from .llm import build_backend_kwargs
    return build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)


@click.group()
@click.version_option()
def main():
    """LLM-based memory allocation summary analysis for C/C++ code."""
    pass


@main.command()
@click.option(
    "--db", "db_path", required=True,
    help="Database file path (must have functions + call_edges)",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["claude", "openai", "ollama", "llamacpp", "gemini"]
    ),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option(
    "--llm-host", default="localhost",
    help="Hostname for local LLM backends (llamacpp, ollama)",
)
@click.option(
    "--llm-port", default=None, type=int,
    help="Port for local LLM backends "
         "(llamacpp: 8080, ollama: 11434)",
)
@click.option(
    "--disable-thinking", is_flag=True,
    help="Disable thinking/reasoning mode for llamacpp "
         "(useful for structured output)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--force", "-f", is_flag=True,
    help="Force re-summarize even if summary exists",
)
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log all LLM prompts and responses to file",
)
@click.option(
    "--svcomp", is_flag=True,
    help="Seed sv-comp __VERIFIER_* helpers and never-NULL malloc model.",
)
@click.option("-j", "jobs", default=1, type=int, help="Parallel LLM queries (default: 1)")
@click.option(
    "--cache-mode",
    type=click.Choice(["none", "instructions", "source"]),
    default=None,
    help="Prompt caching mode: none, instructions, source. "
         "Auto-selects 'source' for claude backend.",
)
@click.option(
    "--function", "function_names", multiple=True,
    help="Only summarize these function(s). Can be specified multiple times.",
)
@click.option(
    "--incremental", is_flag=True,
    help="Only re-summarize dirty functions (missing, source-changed, or callee-updated) "
         "and their transitive callers.",
)
@click.option(
    "--exclude-unreachable-from", "entry_functions", multiple=True,
    help="Only process functions reachable from the given function(s). "
         "Excludes dead code not in the call graph of these roots.",
)
@click.option(
    "--verify-only", is_flag=True,
    help="(code-contract only) Skip contract generation; re-run verification "
         "against cached contracts and persist issues to DB.",
)
@click.option(
    "--vsnap", "vsnap_path", type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="V-snapshot file from KAMain. Enables whole-program alias context "
         "in code-contract prompts.",
)
def summarize(
    db_path, backend, model, llm_host, llm_port,
    disable_thinking, verbose, force, log_llm, svcomp,
    jobs, cache_mode, function_names, incremental, entry_functions,
    verify_only, vsnap_path,
):
    """Generate per-function code contracts on a pre-populated database.

    Requires a database that already has functions and call_edges
    (populated via 'scan' and/or 'import-callgraph').

    Runs the code-contract pass, producing Hoare-style requires/ensures
    contracts per function.

    Example:
        llm-summary summarize --db functions.db --backend claude -v
    """
    db = SummaryDB(db_path)

    try:
        # Validate DB has required data
        stats = db.get_stats()
        func_count = stats["functions"]
        edge_count = stats["call_edges"]

        if func_count == 0:
            console.print("[red]No functions in database. Run 'scan' first.[/red]")
            return

        if edge_count == 0:
            console.print(
                "[red]Error: No call edges in database."
                " Run call graph import first.[/red]"
            )
            sys.exit(1)

        console.print(f"Database: {db_path}")
        console.print(f"  Functions: {func_count}")
        console.print(f"  Call edges: {edge_count}")

        # Show call graph stats
        edges = db.get_all_call_edges()
        if edges:
            graph: dict[int, list[int]] = {}
            for edge in edges:
                if edge.caller_id not in graph:
                    graph[edge.caller_id] = []
                graph[edge.caller_id].append(edge.callee_id)

            orderer = ProcessingOrderer(graph)
            cg_stats = orderer.get_stats()
            console.print(
                f"  SCCs: {cg_stats['sccs']} "
                f"({cg_stats['recursive_sccs']} recursive, "
                f"largest: {cg_stats['largest_scc']})"
            )

        # Create LLM backend
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        # Auto-select cache mode for backends that support prompt caching
        if cache_mode is None:
            cache_mode = "source" if backend == "claude" else "none"
        if cache_mode != "none":
            console.print(f"  Prompt cache mode: {cache_mode}")

        from .code_contract.pass_ import CodeContractPass

        alias_builder = None
        if vsnap_path:
            from .alias_context import AliasContextBuilder
            alias_builder = AliasContextBuilder(vsnap_path, db)
            console.print(f"  Alias context: {vsnap_path}")

        code_contract_pass = CodeContractPass(
            db=db, model=llm.model, llm=llm,
            svcomp=svcomp,
            cache_system=(cache_mode != "none"),
            verbose=verbose,
            verify_only=verify_only,
            log_file=log_llm,
            alias_builder=alias_builder,
        )
        passes: list[SummaryPass] = [code_contract_pass]

        console.print(f"\n[bold]Running pass: {code_contract_pass.name}[/bold]")

        # Resolve --function names to IDs
        target_ids = None
        if function_names:
            target_ids = set()
            for fname in function_names:
                found = db.get_function_by_name(fname)
                if found:
                    for fn in found:
                        assert fn.id is not None
                        target_ids.add(fn.id)
                else:
                    console.print(f"[yellow]Warning: function '{fname}' not found in DB[/yellow]")
            if not target_ids:
                console.print("[red]No matching functions found.[/red]")
                return
            console.print(f"  Targeting {len(target_ids)} function(s): {', '.join(function_names)}")
            force = True  # always re-summarize targeted functions

        # Restrict to functions reachable from entry points
        if entry_functions and not function_names:
            entry_ids: set[int] = set()
            for ename in entry_functions:
                found = db.get_function_by_name(ename)
                if found:
                    for fn in found:
                        assert fn.id is not None
                        entry_ids.add(fn.id)
                else:
                    console.print(f"[yellow]Warning: entry '{ename}' not found in DB[/yellow]")
            if entry_ids:
                tmp_driver = BottomUpDriver(db, verbose=False)
                graph, _ = tmp_driver.build_graph()
                reachable = tmp_driver.compute_reachable(entry_ids, graph)
                target_ids = reachable
                console.print(
                    f"  Entry points: {', '.join(entry_functions)}"
                    f" → {len(reachable)} reachable functions"
                )

        # Compute dirty_ids for incremental mode (per-pass)
        dirty_ids = None
        per_pass_dirty: dict[str, set[int]] | None = None
        if incremental and not function_names:
            from .driver import PASS_TABLE_MAP
            per_pass_dirty = {}
            dirty_ids = set()
            for p in passes:
                table = PASS_TABLE_MAP.get(p.name)
                if table:
                    pd = db.find_dirty_function_ids(table)
                    per_pass_dirty[p.name] = pd
                    dirty_ids |= pd
            console.print(f"  Incremental: {len(dirty_ids)} dirty function(s) detected")
            if not dirty_ids:
                console.print("  Nothing to do — all summaries up to date.")
                return

        pool = None
        if jobs > 1:
            from .llm.pool import LLMPool
            pool = LLMPool(max_workers=jobs)
            if verbose:
                console.print(f"  Thread pool: {jobs} workers")

        driver = BottomUpDriver(db, verbose=verbose, pool=pool)
        try:
            results = driver.run(passes, force=force, dirty_ids=dirty_ids,
                                target_ids=target_ids,
                                per_pass_dirty=per_pass_dirty)
        finally:
            if pool is not None:
                pool.shutdown()

        cc_results = results["code_contract"]
        console.print("\nCode-contract pass complete:")
        console.print(f"  Functions processed: {len(cc_results)}")
        with_reqs = sum(
            1 for sm in cc_results.values()
            if any(sm.has_requires(p) for p in sm.properties)
        )
        with_ens = sum(
            1 for sm in cc_results.values()
            if any(sm.has_ensures(p) for p in sm.properties)
        )
        console.print(f"  Functions with non-trivial requires: {with_reqs}")
        console.print(f"  Functions with non-trivial ensures:  {with_ens}")
        noret = sum(1 for sm in cc_results.values() if sm.noreturn)
        if noret:
            console.print(f"  Functions marked noreturn: {noret}")
        if code_contract_pass.struggle_retries:
            console.print(
                f"  Struggle retries accepted: "
                f"{code_contract_pass.struggle_retries}"
            )

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--name", help="Filter by function name")
@click.option("--file", "file_path", help="Filter by file path")
@click.option("--allocating-only", is_flag=True, help="Only show allocating functions")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--limit", type=int, default=0, help="Limit number of results (0 = unlimited)")
@click.option("--offset", type=int, default=0, help="Skip first N results")
def show(db_path, name, file_path, allocating_only, fmt, limit, offset):
    """Show stored summaries."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()

        if name:
            functions = [f for f in functions if name in f.name]

        if file_path:
            functions = [f for f in functions if file_path in f.file_path]

        if offset:
            functions = functions[offset:]
        if limit:
            functions = functions[:limit]

        if fmt == "json":
            output: list[dict] = []
            for func in functions:
                assert func.id is not None
                summary = db.get_summary_by_function_id(func.id)
                if summary:
                    if allocating_only and not summary.allocations:
                        continue
                entry: dict[str, Any] = {
                    "function": func.name,
                    "file": func.file_path,
                    "line": func.line_start,
                }
                if summary:
                    entry["summary"] = summary.to_dict()
                cc = db.get_code_contract_summary(func.id)
                if cc:
                    entry["code_contract"] = cc.to_dict()
                if summary or cc:
                    output.append(entry)
            console.print(json.dumps(output, indent=2))

        else:
            has_cc = any(
                db.get_code_contract_summary(f.id) is not None
                for f in functions if f.id is not None
            )
            if has_cc:
                table = Table(title="Code Contract Summaries")
                table.add_column("Function", style="cyan")
                table.add_column("File", style="dim")
                table.add_column("Properties", style="green")
                table.add_column("Requires")
                table.add_column("Ensures")

                for func in functions:
                    assert func.id is not None
                    cc = db.get_code_contract_summary(func.id)
                    if not cc or (not cc.properties and not cc.inline_body):
                        continue
                    if cc.inline_body:
                        table.add_row(
                            func.name,
                            Path(func.file_path).name,
                            "(inlined)", "", "",
                        )
                        continue
                    props = ", ".join(cc.properties)
                    req_parts = []
                    for p in cc.properties:
                        clauses = [c for c in cc.requires.get(p, [])
                                   if c.strip().lower() not in ("true", "")]
                        if clauses:
                            req_parts.append(
                                f"[{p}] " + "; ".join(clauses))
                    ens_parts = []
                    for p in cc.properties:
                        clauses = [c for c in cc.ensures.get(p, [])
                                   if c.strip().lower() not in
                                   ("true", "", "(no observable effect)")]
                        if clauses:
                            ens_parts.append(
                                f"[{p}] " + "; ".join(clauses))
                    req_str = "\n".join(req_parts) if req_parts else "true"
                    ens_str = "\n".join(ens_parts) if ens_parts else "-"
                    table.add_row(
                        func.name,
                        Path(func.file_path).name,
                        props,
                        req_str[:120] + "..." if len(req_str) > 120 else req_str,
                        ens_str[:120] + "..." if len(ens_str) > 120 else ens_str,
                    )
            else:
                table = Table(title="Allocation Summaries")
                table.add_column("Function", style="cyan")
                table.add_column("File", style="dim")
                table.add_column("Allocations", style="green")
                table.add_column("Description")

                for func in functions:
                    assert func.id is not None
                    summary = db.get_summary_by_function_id(func.id)
                    if summary:
                        if allocating_only and not summary.allocations:
                            continue

                        alloc_str = ", ".join(
                            a.source for a in summary.allocations) or "-"
                        table.add_row(
                            func.name,
                            Path(func.file_path).name,
                            alloc_str,
                            (summary.description[:50] + "..."
                             if len(summary.description) > 50
                             else summary.description),
                        )

            console.print(table)

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
def stats(db_path):
    """Show database statistics."""
    db = SummaryDB(db_path)

    try:
        stats = db.get_stats()

        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for key, value in stats.items():
            table.add_row(key, str(value))

        console.print(table)

        # Call graph stats
        edges = db.get_all_call_edges()
        if edges:
            graph: dict[int, list[int]] = {}
            for edge in edges:
                if edge.caller_id not in graph:
                    graph[edge.caller_id] = []
                graph[edge.caller_id].append(edge.callee_id)

            orderer = ProcessingOrderer(graph)
            cg_stats = orderer.get_stats()

            console.print("\nCall Graph Statistics:")
            console.print(f"  Nodes: {cg_stats['nodes']}")
            console.print(f"  Edges: {cg_stats['edges']}")
            console.print(f"  SCCs: {cg_stats['sccs']}")
            console.print(f"  Recursive SCCs: {cg_stats['recursive_sccs']}")
            console.print(f"  Largest SCC: {cg_stats['largest_scc']}")

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--entry", "entries", multiple=True,
    help="Restrict the check to these entry function(s). "
         "If omitted, every function with no callers is treated as an entry.",
)
@click.option(
    "--output", "output_path", type=click.Path(), default=None,
    help="Write JSON report to this path instead of stdout.",
)
def check(db_path, entries, output_path):
    """Phase 4 entry-point check (no LLM).

    Reads `code_contract_summaries` for each entry function and surfaces
    every non-trivial `requires` clause as an obligation, with a witness
    chain back to the leaf operation that produced it.
    """
    from .code_contract.checker import check_entries

    db = SummaryDB(db_path)
    try:
        obligations = check_entries(
            db, entries=list(entries) if entries else None,
        )

        report: dict[str, Any] = {
            "db": db_path,
            "entries": list(entries) if entries else None,
            "obligation_count": len(obligations),
            "obligations": [o.to_dict() for o in obligations],
        }
        text = json.dumps(report, indent=2)

        if output_path:
            Path(output_path).write_text(text)
            console.print(
                f"Wrote {len(obligations)} obligation(s) to {output_path}"
            )
        else:
            click.echo(text)
    finally:
        db.close()


@main.command()
@click.argument("name")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--signature", help="Function signature for disambiguation")
def lookup(name, db_path, signature):
    """Look up the code-contract summary for a specific function."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_function_by_name(name)
        if signature:
            functions = [f for f in functions if f.signature == signature]

        if not functions:
            console.print(f"[yellow]Function '{name}' not found in database.[/yellow]")
            return

        results: list[dict[str, Any]] = []
        for func in functions:
            assert func.id is not None
            cc = db.get_code_contract_summary(func.id)
            if cc is None:
                continue
            results.append({
                "name": func.name,
                "signature": func.signature,
                "file": func.file_path,
                "line_start": func.line_start,
                "line_end": func.line_end,
                "code_contract": cc.to_dict(),
            })

        if results:
            console.print(json.dumps(results, indent=2))
        else:
            console.print(
                f"[yellow]No code-contract summary found for {name}. "
                f"Run 'summarize' to generate.[/yellow]"
            )

    finally:
        db.close()


@main.command("init-stdlib")
@click.option("--db", "db_path", default=None, help="Target project database file path")
@click.option(
    "--abilist",
    "extra_abilists",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional abilist file(s) to merge with the bundled list (repeatable)",
)
@click.option(
    "--stdlib-db",
    "stdlib_db",
    default=None,
    help="Stdlib cache path (default: ~/.llm-summary/stdlib_cache.db)",
)
@click.option(
    "--seed-from",
    "seed_from_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Seed stdlib cache from this project DB before applying (repeatable)",
)
@click.option(
    "--force", "-f", is_flag=True,
    help="Overwrite existing stdlib cache entries when seeding with --seed-from",
)
@click.option(
    "--extra-aliases",
    "extra_alias_files",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional libc-alias JSON file(s) to merge with the bundled map "
         "(repeatable). User entries override built-ins.",
)
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default=None,
    help="LLM backend to use for generating summaries of uncached externals",
)
@click.option("--model", default=None, help="Model name override for --backend")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log LLM prompts/responses to file",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def init_stdlib(
    db_path, extra_abilists, stdlib_db, seed_from_paths, force, extra_alias_files,
    backend, model, llm_host, llm_port, log_llm, verbose,
):
    """Populate external-function summaries using a persistent global cache.

    \b
    Seeding the cache (--seed-from):
      Import summaries from a reviewed project DB (e.g. musl) into the global
      stdlib cache so they are reused for all future projects.

        llm-summary init-stdlib --seed-from func-scans/musl/c/functions.db

    Applying to a project (--db):
      For each function in the project DB that has no source body:

        1. If it is in the known-externals list (bundled abilist + --abilist):
             a. Cache hit  → copy summaries from the stdlib cache.
             b. Cache miss → generate via LLM, save to cache, then copy.
        2. Not in known-externals list → skip (main summarizer handles it).

    Both steps can be combined in one invocation.  --stdlib-db overrides the
    default cache location (~/.llm-summary/stdlib_cache.db).
    """
    from .external_summarizer import ExternalFunctionSummarizer
    from .models import Function, VerificationSummary
    from .stdlib_cache import StdlibCache, load_known_externals, load_libc_aliases

    if not db_path and not seed_from_paths:
        console.print("[red]Error: provide --db and/or --seed-from.[/red]")
        sys.exit(1)

    # 1. Load known-externals — needed both for filtering seed-from and for apply
    known_externals = load_known_externals(list(extra_abilists) if extra_abilists else None)
    console.print(f"Known-externals registry: {len(known_externals):,} function names")

    # 1b. Load libc alias map (glibc abilist names -> musl source names).
    #     Validate orphans (musl=null) up front: every such entry must have a
    #     hand-crafted code-contract in code_contract/stdlib.py.
    libc_aliases = load_libc_aliases(
        list(extra_alias_files) if extra_alias_files else None
    )
    if libc_aliases:
        from .code_contract.stdlib import STDLIB_CONTRACTS as _CC_BUILTINS
        orphans = sorted(
            n for n, musl in libc_aliases.items()
            if musl is None and n not in _CC_BUILTINS
        )
        if orphans:
            console.print(
                f"\n[red]Error: {len(orphans)} libc-alias orphan(s) "
                f"(musl=null) have no builtin contract in "
                f"code_contract/stdlib.py:[/red]"
            )
            for n in orphans:
                console.print(f"  {n}")
            sys.exit(1)
        n_resolvable = sum(1 for v in libc_aliases.values() if v is not None)
        n_orphan = len(libc_aliases) - n_resolvable
        console.print(
            f"Libc-alias map: {len(libc_aliases):,} entries "
            f"({n_resolvable} mapped to musl, {n_orphan} orphan-with-builtin)"
        )

    # 2. Open the stdlib cache
    cache = StdlibCache(stdlib_db)

    # 3. Seed from project DBs (--seed-from) — runs before builtins so reviewed
    #    contracts from a real libc DB take priority over hand-crafted entries.
    #    Only exported (known-external) names are seeded; internal statics are skipped.
    summary_tables = [
        ("allocation_summaries", "allocation_json"),
        ("free_summaries",       "free_json"),
        ("init_summaries",       "init_json"),
        ("memsafe_summaries",    "memsafe_json"),
    ]
    for src_path in seed_from_paths:
        src_db = SummaryDB(src_path)
        tag = f"libc:{Path(src_path).parent.name}"
        added = skipped_cached = skipped_internal = alias_fallbacks = 0

        # Build name->id map for fast weak_alias fallback lookup (__name -> name)
        all_src_funcs = src_db.get_all_functions()
        name_to_ids: dict[str, list[int]] = {}
        for sf in all_src_funcs:
            assert sf.id is not None
            name_to_ids.setdefault(sf.name, []).append(sf.id)

        def _fetch_blobs(func_id: int, _db: SummaryDB = src_db) -> dict[str, str | None]:
            blobs: dict[str, str | None] = {col: None for _, col in summary_tables}
            for table, col in summary_tables:
                row = _db.conn.execute(
                    f"SELECT summary_json FROM {table} WHERE function_id = ?",
                    (func_id,),
                ).fetchone()
                if row:
                    blobs[col] = row[0]
            # Code-contract lives in its own table with its own row shape.
            row = _db.conn.execute(
                "SELECT summary_json FROM code_contract_summaries "
                "WHERE function_id = ?",
                (func_id,),
            ).fetchone()
            blobs["code_contract_json"] = row[0] if row else None
            return blobs

        cc_added = 0
        try:
            for src_func in all_src_funcs:
                if src_func.name not in known_externals:
                    skipped_internal += 1
                    continue
                if not force and cache.has(src_func.name):
                    skipped_cached += 1
                    continue
                assert src_func.id is not None
                blobs = _fetch_blobs(src_func.id)
                if not any(blobs.values()):
                    # weak_alias fallback: try __<name> (e.g. pthread_exit -> __pthread_exit)
                    private_name = f"__{src_func.name}"
                    for fid in name_to_ids.get(private_name, []):
                        candidate = _fetch_blobs(fid)
                        if any(candidate.values()):
                            blobs = candidate
                            alias_fallbacks += 1
                            if verbose:
                                console.print(
                                    f"  [alias] {src_func.name} -> {private_name}"
                                )
                            break
                if not any(blobs.values()):
                    continue
                cache.put(
                    name=src_func.name,
                    allocation_json=blobs["allocation_json"],
                    free_json=blobs["free_json"],
                    init_json=blobs["init_json"],
                    memsafe_json=blobs["memsafe_json"],
                    model_used=tag,
                    code_contract_json=blobs["code_contract_json"],
                    code_contract_model=tag if blobs["code_contract_json"] else None,
                )
                added += 1
                if blobs["code_contract_json"]:
                    cc_added += 1
                if verbose:
                    console.print(f"  [seed] {src_func.name} ({tag})")

            # Second pass: libc alias map. For each (glibc_name, musl_name)
            # entry, copy the musl source's contracts under the glibc name
            # so abilist-driven lookups hit the cache.
            via_alias_map = 0
            via_alias_cc = 0
            for glibc_name, musl_name in libc_aliases.items():
                if musl_name is None:
                    continue  # orphan; handled by seed_builtins
                if glibc_name not in known_externals:
                    continue
                if not force and cache.has(glibc_name):
                    continue
                fids = name_to_ids.get(musl_name)
                if not fids:
                    continue
                blobs = _fetch_blobs(fids[0])
                if not any(blobs.values()):
                    continue
                cache.put(
                    name=glibc_name,
                    allocation_json=blobs["allocation_json"],
                    free_json=blobs["free_json"],
                    init_json=blobs["init_json"],
                    memsafe_json=blobs["memsafe_json"],
                    model_used=tag,
                    code_contract_json=blobs["code_contract_json"],
                    code_contract_model=tag if blobs["code_contract_json"] else None,
                )
                via_alias_map += 1
                if blobs["code_contract_json"]:
                    via_alias_cc += 1
                if verbose:
                    console.print(
                        f"  [libc-alias] {glibc_name} -> {musl_name} ({tag})"
                    )
        finally:
            src_db.close()
        console.print(
            f"Seeded from {src_path}: {added} exported symbols added"
            + (f" ({cc_added} with code-contract)" if cc_added else "")
            + (f" ({alias_fallbacks} via weak_alias)" if alias_fallbacks else "")
            + (
                f", {via_alias_map} via libc-alias"
                + (f" ({via_alias_cc} with code-contract)" if via_alias_cc else "")
                if via_alias_map else ""
            )
            + (f", {skipped_cached} already cached" if skipped_cached else "")
            + (f", {skipped_internal} internal symbols skipped" if skipped_internal else "")
        )

    # 4. Seed hand-crafted builtins — always wins over DB-seeded entries so
    #    carefully reviewed contracts take priority over LLM-generated ones.
    seeded = cache.seed_builtins(force=True)
    if seeded:
        console.print(f"  Seeded {seeded} hand-crafted builtin entries into cache (priority)")

    # 4b. Resolve aliases against builtins already in cache (e.g. fopen64→fopen).
    #     This covers the case where no --seed-from musl DB is given.
    alias_seeded = 0
    for glibc_name, musl_name in libc_aliases.items():
        if musl_name is None:
            continue
        if cache.has(glibc_name, "code_contract"):
            continue
        entry = cache.get(musl_name)
        if entry is None:
            continue
        cache.put(
            name=glibc_name,
            allocation_json=entry.allocation_json,
            free_json=entry.free_json,
            init_json=entry.init_json,
            memsafe_json=entry.memsafe_json,
            model_used=entry.model_used,
            code_contract_json=entry.code_contract_json,
            code_contract_model=entry.code_contract_model,
        )
        alias_seeded += 1
    if alias_seeded:
        console.print(f"  Resolved {alias_seeded} alias(es) from cached builtins")

    if not db_path:
        cache.close()
        return

    db = SummaryDB(db_path)
    try:
        # 4. Identify functions that have no source and need summaries
        all_funcs = db.get_all_functions()
        sourceless = [f for f in all_funcs if not f.source]

        def _needs_summary(f: Function) -> bool:
            if force:
                return True  # re-apply all when --force is given
            assert f.id is not None
            return db.get_code_contract_summary(f.id) is None

        needs = [f for f in sourceless if _needs_summary(f)]

        if not needs:
            console.print("All external functions already have summaries.")
            return

        in_known = [f for f in needs if f.name in known_externals]
        not_known = [f for f in needs if f.name not in known_externals]

        cache_hits = [f for f in in_known if cache.has(f.name, "code_contract")]
        cache_misses = [f for f in in_known if not cache.has(f.name, "code_contract")]

        console.print(
            f"  Functions needing summaries: {len(needs)} total  "
            f"({len(in_known)} known-external, {len(not_known)} unknown)"
        )
        console.print(
            f"  Known-externals breakdown: {len(cache_hits)} cache hits, "
            f"{len(cache_misses)} cache misses"
        )

        # 5. Require --backend if there are cache misses
        if cache_misses and not backend:
            names = sorted(f.name for f in cache_misses)
            console.print(
                f"\n[red]Error: {len(cache_misses)} known-external function(s) have no cached "
                f"code-contract summary and no --backend was given.[/red]"
            )
            console.print("  Re-run with --backend <claude|gemini|…> to generate them.")
            console.print(f"  Missing: {', '.join(names)}")
            sys.exit(1)

        # 6. Helper: get-or-create a function stub in the project DB
        def _get_or_create(name: str) -> Function:
            existing = db.get_function_by_name(name)
            if existing:
                result: Function = existing[0]
                return result
            stub = Function(
                name=name,
                file_path="<stdlib>",
                line_start=0,
                line_end=0,
                source="",
                signature=f"{name}(...)",
            )
            stub.id = db.insert_function(stub)
            return stub

        # 7. Helper: apply a cache entry to the project DB
        def _apply_entry(func: Function, entry) -> None:
            model_used = entry.model_used

            if entry.allocation_json:
                alloc_obj = db._json_to_summary(entry.allocation_json)
                db.upsert_summary(func, alloc_obj, model_used=model_used)

            if entry.free_json:
                free_obj = db._json_to_free_summary(entry.free_json)
                db.upsert_free_summary(func, free_obj, model_used=model_used)

            if entry.init_json:
                init_obj = db._json_to_init_summary(entry.init_json)
                db.upsert_init_summary(func, init_obj, model_used=model_used)

            if entry.memsafe_json:
                memsafe_obj = db._json_to_memsafe_summary(entry.memsafe_json)
                db.upsert_memsafe_summary(func, memsafe_obj, model_used=model_used)
                # Mirror contracts into verification_summary for bottom-up callers
                ver = VerificationSummary(
                    function_name=func.name,
                    simplified_contracts=memsafe_obj.contracts,
                    issues=[],
                    description=memsafe_obj.description,
                )
                db.upsert_verification_summary(func, ver, model_used=model_used)

            if entry.code_contract_json:
                from .code_contract.models import CodeContractSummary
                cc = CodeContractSummary.from_dict(json.loads(entry.code_contract_json))
                cc.function = func.name  # match the project DB stub's name
                db.store_code_contract_summary(
                    func, cc,
                    model_used=entry.code_contract_model or model_used,
                )

        # 8. Apply cache hits
        if cache_hits:
            console.print(f"\nApplying {len(cache_hits)} cached summaries...")
            for f in cache_hits:
                entry = cache.get(f.name)
                func = _get_or_create(f.name)
                _apply_entry(func, entry)
                if verbose:
                    console.print(f"  [cache] {f.name}")

        # 9. LLM-generate cache misses
        if cache_misses and backend:
            backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, False)
            llm = create_backend(backend, model=model, **backend_kwargs)
            gen = ExternalFunctionSummarizer(llm, verbose=verbose, log_file=log_llm)
            console.print(
                f"\nGenerating summaries for {len(cache_misses)} uncached functions "
                f"via {backend} ({llm.model})..."
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                console=console,
            ) as prog:
                task = prog.add_task("", total=len(cache_misses))
                for f in cache_misses:
                    prog.update(task, description=f.name)
                    result = gen.generate(f.name)
                    cache.put(
                        name=f.name,
                        allocation_json=result.allocation_json,
                        free_json=result.free_json,
                        init_json=result.init_json,
                        memsafe_json=result.memsafe_json,
                        model_used=llm.model,
                    )
                    func = _get_or_create(f.name)
                    entry = cache.get(f.name)
                    _apply_entry(func, entry)
                    prog.advance(task)

            s = gen.stats
            console.print(
                f"  LLM calls: {s['llm_calls']}, processed: {s['functions_processed']}, "
                f"errors: {s['errors']}"
            )

        # 10. Report skipped unknowns
        if not_known:
            names = sorted(f.name for f in not_known)
            console.print(
                f"\n[yellow]{len(not_known)} sourceless function(s) not in known-externals list "
                f"(skipped — main summarizer will handle them):[/yellow]"
            )
            if verbose:
                for n in names:
                    console.print(f"  {n}")
            else:
                console.print(f"  {', '.join(names)}")

        console.print("\n[green]Done.[/green]")

    finally:
        db.close()
        cache.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
def export(db_path, output):
    """Export all code-contract summaries to JSON."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()
        output_data = []

        for func in functions:
            assert func.id is not None
            cc = db.get_code_contract_summary(func.id)
            if cc:
                output_data.append({
                    "name": func.name,
                    "signature": func.signature,
                    "file": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "code_contract": cc.to_dict(),
                })

        json_str = json.dumps(output_data, indent=2)

        if output:
            with open(output, "w") as f:
                f.write(json_str)
            console.print(f"Exported {len(output_data)} summaries to {output}")
        else:
            console.print(json_str)

    finally:
        db.close()


@main.command("callgraph")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--format", "fmt",
    type=click.Choice(["tuples", "csv", "json"]),
    default="tuples",
    help="Output format"
)
@click.option("--no-header", is_flag=True, help="Omit header row (for csv/tuples)")
def callgraph(db_path, output, fmt, no_header):
    """Export call graph as (caller, callsite, callee) tuples."""
    db = SummaryDB(db_path)

    try:
        edges = db.get_all_call_edges()
        functions = {f.id: f for f in db.get_all_functions()}

        lines = []

        if fmt == "json":
            output_data = []
            for edge in edges:
                caller = functions.get(edge.caller_id)
                callee = functions.get(edge.callee_id)
                if caller and callee:
                    output_data.append({
                        "caller": caller.name,
                        "callsite": {
                            "file": edge.file_path,
                            "line": edge.line,
                            "column": edge.column,
                        },
                        "callee": callee.name,
                        "is_indirect": edge.is_indirect,
                    })
            lines.append(json.dumps(output_data, indent=2))

        else:
            # tuples or csv format
            separator = "," if fmt == "csv" else "\t"

            if not no_header:
                lines.append(separator.join(["caller", "callsite", "callee"]))

            for edge in edges:
                caller = functions.get(edge.caller_id)
                callee = functions.get(edge.callee_id)
                if caller and callee:
                    callsite = edge.callsite_str() or f"{edge.file_path}:{edge.line}"
                    lines.append(separator.join([caller.name, callsite, callee.name]))

        result = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(result)
                f.write("\n")
            console.print(f"Exported {len(edges)} call edges to {output}")
        else:
            console.print(result)

    finally:
        db.close()


def _issue_fingerprint(issue_dict: dict) -> str:
    """Compute issue fingerprint from a verdict's issue dict."""
    from .models import SafetyIssue
    si = SafetyIssue(
        location=issue_dict.get("location", ""),
        issue_kind=issue_dict.get("issue_kind", ""),
        description=issue_dict.get("description", ""),
        severity=issue_dict.get("severity", "medium"),
        callee=issue_dict.get("callee"),
        contract_kind=issue_dict.get("contract_kind"),
    )
    return si.fingerprint()


def _verdict_dir(base: Path, func_name: str, idx: int, issue_dict: dict) -> Path:
    """Build per-verdict output dir with fingerprint for staleness detection."""
    fp = _issue_fingerprint(issue_dict)
    return base / func_name / f"v{idx}_{fp}"


def _find_entry_functions(db: SummaryDB, relevant: list[str]) -> list[str]:
    """Find entry functions: those with no callers within the relevant set.

    Multiple entries mean multiple call chains to validate separately.
    """
    name_to_id: dict[str, int] = {}
    for name in relevant:
        funcs = db.get_function_by_name(name)
        if funcs and funcs[0].id is not None:
            name_to_id[name] = funcs[0].id

    id_to_name = {v: k for k, v in name_to_id.items()}

    has_relevant_caller: set[str] = set()
    for name, fid in name_to_id.items():
        callee_ids = db.get_callees(fid)
        for cid in callee_ids:
            callee_name = id_to_name.get(cid)
            if callee_name and callee_name != name:
                has_relevant_caller.add(callee_name)

    candidates = [n for n in relevant if n not in has_relevant_caller and n in name_to_id]
    return candidates if candidates else [relevant[0]]



def _load_compile_commands(
    compile_commands_path: str,
    project_path: str | None = None,
    build_dir: str | None = None,
) -> tuple["CompileCommandsDB", str | None]:
    """Load compile_commands.json, remapping Docker paths when project_path is given.

    Returns (db, tmp_path).  tmp_path is a temp file that the caller must
    unlink when done; it is None when no remapping was needed.

    Raises click.UsageError when source file paths do not exist on disk and
    no project_path was provided to perform remapping.
    """
    import json as _json
    import tempfile

    from .docker_paths import is_docker_path as _is_docker_path
    from .docker_paths import remap_path as _remap_path_str
    from .docker_paths import translate_compiler_arg as _translate_arg

    with open(compile_commands_path) as f:
        entries = _json.load(f)

    # Sample the first few entries to detect bad/Docker paths.
    sample = [e for e in entries[:5] if e.get("file")]
    has_docker = any(
        _is_docker_path(e.get("file", "")) or _is_docker_path(e.get("directory", ""))
        for e in sample
    )
    has_missing = (
        not has_docker
        and bool(sample)
        and not Path(sample[0]["file"]).exists()
    )

    if has_docker or has_missing:
        if not project_path:
            example = sample[0]["file"] if sample else "?"
            if has_docker:
                hint = (
                    f"compile_commands.json uses Docker container paths "
                    f"(e.g. {example!r}). "
                    f"Pass --project-path <host-source-dir> to remap "
                    f"/workspace/src to the host source directory."
                )
            else:
                hint = (
                    f"Source file not found on host: {example!r}. "
                    f"Pass --project-path <host-source-dir> to remap paths."
                )
            raise click.UsageError(hint)

        proj_dir = Path(project_path)
        if build_dir:
            build_dir_path = Path(build_dir)
        else:
            raise click.UsageError(
                "compile_commands.json requires path remapping but "
                "--build-dir was not provided.  Pass "
                "--build-dir <host-build-dir> explicitly."
            )

        resolved = []
        for e in entries:
            e = dict(e)
            if _is_docker_path(e.get("directory", "")):
                e["directory"] = _remap_path_str(e["directory"], proj_dir, build_dir_path)
            if _is_docker_path(e.get("file", "")):
                e["file"] = _remap_path_str(e["file"], proj_dir, build_dir_path)
            if "output" in e and _is_docker_path(e["output"]):
                e["output"] = _remap_path_str(e["output"], proj_dir, build_dir_path)
            if "arguments" in e:
                e["arguments"] = [
                    _translate_arg(a, proj_dir, build_dir_path)
                    for a in e["arguments"]
                ]
            if "command" in e:
                import shlex
                parts = shlex.split(e["command"])
                parts = [_translate_arg(a, proj_dir, build_dir_path) for a in parts]
                e["command"] = " ".join(shlex.quote(p) for p in parts)
            resolved.append(e)

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        _json.dump(resolved, tmp)
        tmp.flush()
        tmp.close()
        console.print(f"Remapped compile_commands.json: /workspace/src -> {proj_dir}")
        return CompileCommandsDB(tmp.name), tmp.name

    return CompileCommandsDB(compile_commands_path), None


def _asm_sources_from_objects(
    objects: list[str],
    compile_commands_path: str,
    build_dir: str | None = None,
) -> set[str]:
    """Derive assembly source paths from a link unit's object list.

    For each object in the list, looks up the compile_commands entry and
    returns source files with assembly extensions (.s/.S/.asm).  Used when
    link_units.json has no explicit 'asm_sources' field (e.g. older files
    populated by batch_call_graph_gen before this field existed).
    """
    import json as _json

    if not objects or not compile_commands_path:
        return set()

    asm_extensions = {".s", ".S", ".asm"}
    try:
        with open(compile_commands_path) as f:
            entries = _json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()

    # Build output -> source map
    out_to_src: dict[str, str] = {}
    for entry in entries:
        output = entry.get("output", "")
        src = entry.get("file", "")
        directory = entry.get("directory", "")
        if not output or not src:
            continue
        out_path = Path(output) if Path(output).is_absolute() else Path(directory) / output
        src_path = Path(src) if Path(src).is_absolute() else Path(directory) / src
        out_to_src[str(out_path.resolve())] = str(src_path.resolve())
        # Also index by basename for flat ar-t member names like "longjmp.lo"
        out_to_src[out_path.name] = str(src_path.resolve())

    build_dir_path = Path(build_dir) if build_dir else None
    result: set[str] = set()
    for obj in objects:
        obj_path = Path(obj)
        if not obj_path.is_absolute() and build_dir_path:
            obj_path = build_dir_path / obj_path
        src = (
            out_to_src.get(str(obj_path.resolve()))
            or out_to_src.get(obj_path.name)
        )
        if src and Path(src).suffix in asm_extensions:
            result.add(src)
    return result


def _source_files_for_target(
    compile_commands_path: str,
    bc_files: set[str],
    build_dir: str | None = None,
) -> set[str]:
    """Return resolved source file paths whose .bc output is in bc_files.

    For each compile_commands.json entry, derives the expected .bc path from
    the output .o path using the -save-temps=obj naming convention
    (foo.c.o -> foo.bc in the same directory), then checks membership in
    bc_files.  Also handles the -flto case where the .o file itself is bitcode.

    When build_dir is provided (from link_units.json), matching is done on
    relative paths so that container build paths (e.g. /workspace/build) are
    correctly mapped to host bc_file paths (e.g. /data/.../build-artifacts).
    """
    import json as _json

    with open(compile_commands_path) as f:
        entries = _json.load(f)

    # Build a set of relative bc paths by stripping build_dir prefix.
    # This handles container/host path mismatches: compile_commands.json may
    # use /workspace/build while bc_files use the host build-artifacts path.
    rel_bc_files: set[str] = set()
    build_dir_path = Path(build_dir) if build_dir else None
    for bc in bc_files:
        if build_dir_path:
            try:
                rel_bc_files.add(str(Path(bc).relative_to(build_dir_path)))
                continue
            except ValueError:
                pass
        rel_bc_files.add(bc)  # fallback: use as absolute

    result: set[str] = set()
    for entry in entries:
        output = entry.get("output")
        if not output:
            continue
        directory = entry.get("directory", "")
        output_path = Path(output)

        # Derive relative bc path from the output .o path.
        # The output field is typically relative to directory; use it as-is
        # to stay in build-dir-relative space for container-safe matching.
        if output_path.is_absolute():
            # If absolute, try to make it relative to its directory so the
            # relative suffix can be compared against rel_bc_files.
            try:
                output_path = output_path.relative_to(directory)
            except ValueError:
                pass  # keep absolute

        name = output_path.name
        if name.endswith(".o"):
            # Tier 1: -save-temps=obj  foo.c.o -> foo.bc
            source_stem = Path(name[:-2]).stem  # strip .o then .c/.cpp etc.
            rel_bc = str(output_path.parent / (source_stem + ".bc"))
            match = rel_bc in rel_bc_files
        else:
            # Tier 2: -flto, the .o itself is bitcode
            rel_bc = str(output_path)
            match = rel_bc in rel_bc_files

        if match:
            file_path = entry.get("file", "")
            if file_path:
                if not Path(file_path).is_absolute():
                    file_path = str(Path(directory) / file_path)
                result.add(str(Path(file_path).resolve()))

    return result


@main.command()
@click.option(
    "--compile-commands", "compile_commands_path",
    type=click.Path(exists=True), required=True,
    help="Path to compile_commands.json",
)
@click.option(
    "--link-units", "link_units_path",
    type=click.Path(exists=True), default=None,
    help="Path to link_units.json "
         "(from discover-link-units). "
         "Restricts scan to the named target.",
)
@click.option(
    "--target", "target_name", default=None,
    help="Link-unit target name to scan "
         "(required when --link-units is given).",
)
@click.option(
    "--project-path", "project_path",
    type=click.Path(), default=None,
    help="Host path to project source root. Required "
         "when compile_commands.json uses Docker "
         "container paths (/workspace/src/...). "
         "Maps /workspace/src -> project_path and "
         "/workspace/build -> build_dir.",
)
@click.option(
    "--db", "db_path", default="summaries.db",
    help="Database file path",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--preprocess", is_flag=True,
    help="Run clang -E to expand macros and "
         "store preprocessed source",
)
def scan(
    compile_commands_path: str, link_units_path: str | None, target_name: str | None,
    project_path: str | None, db_path: str, verbose: bool, preprocess: bool,
) -> None:
    """Extract functions, scan indirect call targets, and find callsites (no LLM).

    This command runs the pre-LLM scanner phases:
    1. Extract all functions from source files
    2. Scan for indirect call targets (address-taken, virtual, attributes)
    3. Find indirect callsites

    To scope the scan to a single link unit, pass --link-units and --target:

        llm-summary scan \\
          --compile-commands build/compile_commands.json \\
          --link-units func-scans/zlib/link_units.json \\
          --target zlibstatic \\
          --project-path /data/csong/opensource/zlib \\
          --db func-scans/zlib/zlibstatic/functions.db

    When compile_commands.json was generated in a Docker container, pass
    --project-path to remap /workspace/src/ paths to host paths.

    Without --link-units, all C/C++ source files in compile_commands.json are scanned.
    """
    import json as _json
    from collections import Counter

    from rich.progress import BarColumn, MofNCompleteColumn, TimeElapsedColumn

    from .models import TargetType

    # Validate --link-units / --target pairing
    if link_units_path and not target_name:
        console.print("[red]--target is required when --link-units is given[/red]")
        return
    if target_name and not link_units_path:
        console.print("[red]--link-units is required when --target is given[/red]")
        return

    # Resolve bc_files filter from link_units.json
    bc_files_filter: set[str] | None = None
    asm_sources_filter: set[str] | None = None
    lu_build_dir: str | None = None
    if link_units_path:
        with open(link_units_path) as f:
            lu_data = _json.load(f)
        targets_map = {t["name"]: t for t in lu_data.get("link_units", lu_data.get("targets", []))}
        if target_name not in targets_map:
            available = ", ".join(targets_map.keys()) or "(none)"
            console.print(f"[red]Target '{target_name}' not found in {link_units_path}[/red]")
            console.print(f"  Available targets: {available}")
            return
        target_entry = targets_map[target_name]
        bc_files_filter = set(target_entry.get("bc_files", []))
        lu_build_dir = lu_data.get("build_dir")

        # asm_sources: prefer explicit field, fall back to deriving from objects
        if "asm_sources" in target_entry:
            asm_sources_filter = set(target_entry["asm_sources"])
        else:
            asm_sources_filter = _asm_sources_from_objects(
                target_entry.get("objects", []),
                compile_commands_path,
                lu_build_dir,
            )
            if asm_sources_filter and verbose:
                console.print(
                    f"  Derived {len(asm_sources_filter)} asm sources from objects list"
                )

        console.print(
            f"Link-unit filter: target '{target_name}', "
            f"{len(bc_files_filter)} bc files"
            + (f", {len(asm_sources_filter)} asm sources" if asm_sources_filter else "")
        )

    # Load compile_commands.json, remapping Docker paths if needed.
    try:
        compile_commands, _tmp_cc_file = _load_compile_commands(
            compile_commands_path,
            project_path=project_path,
            build_dir=lu_build_dir,
        )
        resolved_cc_path = _tmp_cc_file if _tmp_cc_file else compile_commands_path
    except click.UsageError:
        raise
    except Exception as e:
        console.print(f"[red]Failed to load compile_commands.json: {e}[/red]")
        return

    # Filter to C/C++ source files and assembly files
    all_files = compile_commands.get_all_files()

    if bc_files_filter is not None:
        # Use the resolved compile_commands path so source file paths match
        scoped = _source_files_for_target(resolved_cc_path, bc_files_filter, build_dir=lu_build_dir)
        source_files = [
            f for f in all_files
            if f in scoped
            and Path(f).suffix.lower() in C_EXTENSIONS
        ]
        # Assembly files don't produce .bc — use asm_sources_filter from link_units.json
        if asm_sources_filter:
            asm_files = [
                f for f in all_files
                if Path(f).suffix in ASM_EXTENSIONS and f in asm_sources_filter
            ]
        else:
            asm_files = []
    else:
        source_files = [f for f in all_files if Path(f).suffix.lower() in C_EXTENSIONS]
        asm_files = [f for f in all_files if Path(f).suffix in ASM_EXTENSIONS]

    file_counts = f"{len(source_files)} C/C++ source files"
    if asm_files:
        file_counts += f", {len(asm_files)} assembly files"
    if bc_files_filter is not None:
        file_counts += f" (scoped to target '{target_name}')"
    console.print(f"Loaded compile_commands.json: {len(all_files)} entries, {file_counts}")

    if not source_files and not asm_files:
        console.print("[red]No source files found in compile_commands.json[/red]")
        if _tmp_cc_file:
            Path(_tmp_cc_file).unlink(missing_ok=True)
        return

    db = SummaryDB(db_path)

    try:
        # Phase 1: Extract functions
        console.print("\n[bold]Phase 1: Extracting functions[/bold]")

        project_root = Path(project_path).resolve() if project_path else None
        extractor = FunctionExtractor(
            compile_commands=compile_commands,
            project_root=project_root,
            enable_preprocessing=preprocess,
        )
        all_functions = []
        all_typedefs = []
        extract_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(source_files))

            for src_file in source_files:
                try:
                    functions = extractor.extract_from_file(src_file)
                    all_functions.extend(functions)
                    typedefs = extractor.extract_typedefs_from_file(src_file)
                    all_typedefs.extend(typedefs)
                    if verbose:
                        progress.console.print(
                            f"  {Path(src_file).name}: "
                            f"{len(functions)} functions, "
                            f"{len(typedefs)} typedefs"
                        )
                except Exception as e:
                    extract_errors += 1
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(src_file).name}: {e}[/yellow]")
                progress.advance(task)

        db.insert_functions_batch(all_functions)
        db.insert_typedefs_batch(all_typedefs)
        console.print(f"  Functions: {len(all_functions)}")
        console.print(f"  Typedefs: {len(all_typedefs)}")
        if extract_errors:
            console.print(f"  [yellow]Errors: {extract_errors}[/yellow]")

        # Phase 1b: Extract assembly functions
        if asm_files:
            console.print("\n[bold]Phase 1b: Extracting assembly functions[/bold]")

            # Build source -> output mapping from raw compile_commands JSON
            asm_output_map: dict[str, str] = {}
            cc_json_path = Path(resolved_cc_path or compile_commands_path)
            if cc_json_path.exists():
                try:
                    with open(cc_json_path) as f:
                        raw_cc = json.load(f)
                    for entry in raw_cc:
                        src = entry.get("file", "")
                        out = entry.get("output", "")
                        if src and out:
                            src_resolved = (
                                str(Path(src).resolve())
                                if Path(src).is_absolute()
                                else src
                            )
                            asm_output_map[src_resolved] = out
                except (json.JSONDecodeError, OSError):
                    pass

            asm_functions = []
            asm_errors = 0
            for asm_file in asm_files:
                try:
                    obj_path = asm_output_map.get(asm_file)
                    funcs = extract_asm_functions(
                        Path(asm_file),
                        Path(obj_path) if obj_path else None,
                    )
                    asm_functions.extend(funcs)
                    if verbose:
                        console.print(f"  {Path(asm_file).name}: {len(funcs)} functions")
                except Exception as e:
                    asm_errors += 1
                    if verbose:
                        console.print(f"  [yellow]{Path(asm_file).name}: {e}[/yellow]")

            if asm_functions:
                db.insert_functions_batch(asm_functions)
                all_functions.extend(asm_functions)
            console.print(f"  Assembly functions: {len(asm_functions)}")
            if asm_errors:
                console.print(f"  [yellow]Errors: {asm_errors}[/yellow]")

        # Phase 2: Scan for indirect call targets
        console.print("\n[bold]Phase 2: Scanning indirect call targets[/bold]")

        scanner = AddressTakenScanner(db, compile_commands=compile_commands)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning...", total=len(source_files))

            for src_file in source_files:
                try:
                    scanner.scan_files([src_file])
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(src_file).name}: {e}[/yellow]")
                progress.advance(task)

        # Count targets by type
        atfs = db.get_address_taken_functions()
        type_counts: Counter[str] = Counter()
        for atf in atfs:
            type_counts[atf.target_type] += 1

        console.print(f"  Total targets: {len(atfs)}")
        for tt in TargetType:
            count = type_counts.get(tt.value, 0)
            if count > 0:
                console.print(f"    {tt.value}: {count}")

        # Phase 3: Find indirect callsites
        console.print("\n[bold]Phase 3: Finding indirect callsites[/bold]")

        finder = IndirectCallsiteFinder(db, compile_commands=compile_commands)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Finding callsites...", total=len(source_files))

            all_callsites = []
            for src_file in source_files:
                try:
                    callsites = finder.find_in_files([src_file])
                    all_callsites.extend(callsites)
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(src_file).name}: {e}[/yellow]")
                progress.advance(task)

        console.print(f"  Indirect callsites: {len(all_callsites)}")

        # Phase 4: Extract extern declaration headers
        console.print("\n[bold]Phase 4: Extracting extern declaration headers[/bold]")

        from .extern_headers import extract_extern_headers

        extern_header_map, _preprocess_failed = extract_extern_headers(
            compile_commands_path=resolved_cc_path or compile_commands_path,
            project_root=project_root,
            source_files=source_files,
            verbose=verbose,
        )
        if extern_header_map:
            updated = db.update_decl_headers(extern_header_map)
            # Group by header for display
            from collections import Counter as _Counter
            header_counts = _Counter(extern_header_map.values())
            n_funcs = len(extern_header_map)
            n_hdrs = len(header_counts)
            console.print(f"  Mapped {n_funcs} extern functions to {n_hdrs} headers")
            if verbose:
                for hdr, cnt in header_counts.most_common(10):
                    console.print(f"    {hdr}: {cnt} functions")
            console.print(f"  Updated {updated} function rows in DB")
        else:
            console.print("  No extern declarations found")

        # Summary
        console.print("\n[bold]Summary[/bold]")
        console.print(f"  Source files: {len(source_files)}")
        console.print(f"  Functions: {len(all_functions)}")
        console.print(f"  Indirect call targets: {len(atfs)}")
        console.print(f"  Indirect callsites: {len(all_callsites)}")
        console.print(f"  Extern headers mapped: {len(extern_header_map)}")
        console.print(f"  Database: {db_path}")

    finally:
        db.close()
        if _tmp_cc_file:
            Path(_tmp_cc_file).unlink(missing_ok=True)


@main.command("build-learn")
@click.option(
    "--project-path", type=click.Path(exists=True),
    required=True, help="Path to the project to build",
)
@click.option(
    "--build-dir", type=click.Path(), default=None,
    help="Custom build directory "
         "(default: <project-path>/build)",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["claude", "openai", "ollama", "llamacpp", "gemini"]
    ),
    default="claude",
    help="LLM backend for incremental learning",
)
@click.option("--model", default=None, help="Model name (default depends on backend)")
@click.option(
    "--llm-host", default="localhost",
    help="Hostname for local LLM backends "
         "(llamacpp, ollama)",
)
@click.option(
    "--llm-port", default=None, type=int,
    help="Port for local LLM backends "
         "(llamacpp: 8080, ollama: 11434)",
)
@click.option(
    "--disable-thinking", is_flag=True,
    help="Disable thinking/reasoning mode for llamacpp "
         "(useful for structured output)",
)
@click.option("--max-retries", default=3, help="Maximum build attempts")
@click.option("--container-image", default="llm-summary-builder:latest", help="Docker image to use")
@click.option("--enable-lto/--no-lto", default=True, help="Enable LLVM LTO")
@click.option("--prefer-static/--no-static", default=True, help="Prefer static linking")
@click.option("--generate-ir/--no-ir", default=True, help="Generate and save LLVM IR artifacts")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log all LLM prompts and responses to file",
)
@click.option("--ccache-dir", type=click.Path(), default="~/.cache/llm-summary-ccache",
              help="Host ccache directory (default: ~/.cache/llm-summary-ccache)")
@click.option("--no-ccache", is_flag=True, help="Disable ccache")
@click.option("--source-subdir", default=None,
              help="Subdirectory containing CMakeLists.txt (for monorepos like llvm-project)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def build_learn(
    project_path,
    build_dir,
    backend,
    model,
    llm_host,
    llm_port,
    disable_thinking,
    max_retries,
    container_image,
    enable_lto,
    prefer_static,
    generate_ir,
    db_path,
    log_llm,
    ccache_dir,
    no_ccache,
    source_subdir,
    verbose,
):
    """Learn how to build a project and generate reusable build script."""
    from .builder import Builder
    from .builder.script_generator import ScriptGenerator

    project_path = Path(project_path).resolve()

    console.print("\n[bold]Build Agent System[/bold]")
    console.print(f"Project: {project_path}")
    if build_dir:
        console.print(f"Build directory: {build_dir}")
    else:
        console.print(f"Build directory: {project_path}/build")
    console.print(f"Backend: {backend}")
    if model:
        console.print(f"Model: {model}")
    console.print()

    # Initialize LLM backend
    console.print("\n[bold]Initializing LLM backend...[/bold]")
    try:
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using model: [bold]{llm.model}[/bold]")
    except Exception as e:
        console.print(f"[red]Error initializing LLM backend: {e}[/red]")
        sys.exit(1)

    # Resolve ccache directory
    ccache_path = None if no_ccache else Path(ccache_dir).expanduser()

    # Initialize unified builder
    builder = Builder(
        llm=llm,
        container_image=container_image,
        build_dir=Path(build_dir) if build_dir else None,
        build_scripts_dir=Path("build-scripts"),
        max_retries=max_retries,
        enable_lto=enable_lto,
        prefer_static=prefer_static,
        generate_ir=generate_ir,
        verbose=verbose,
        log_file=log_llm,
        ccache_dir=ccache_path,
        source_subdir=source_subdir,
    )

    # Set artifact name for monorepo sub-projects
    if source_subdir:
        # For monorepo sub-projects, use the directory leaf of project_path
        # combined with source_subdir isn't useful; instead use build_dir name
        # or let user set it explicitly. Default: build_dir basename if set.
        if build_dir:
            builder.artifact_name = Path(build_dir).name
        # Otherwise falls back to project_path.name in Builder

    # Learn and build
    console.print("\n[bold]Learning build configuration...[/bold]")
    try:
        result = builder.learn_and_build(project_path)
    except Exception as e:
        console.print(f"[red]Error during build learning: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    if not result["success"]:
        console.print(f"\n[red]Build failed after {result['attempts']} attempts[/red]")
        console.print("\n[bold]Error messages:[/bold]")
        for i, error in enumerate(result["error_messages"], 1):
            console.print(f"\n[yellow]Attempt {i}:[/yellow]")
            console.print(error[:500] + "..." if len(error) > 500 else error)
        sys.exit(1)

    console.print(f"\n[green]Build successful after {result['attempts']} attempts![/green]")

    # Generate build script
    console.print("\n[bold]Generating reusable build script...[/bold]")
    project_name = builder.artifact_name or project_path.name
    generator = ScriptGenerator()

    flags = result.get("flags", [])
    build_system_used = result.get("build_system_used", "unknown")
    use_build_dir = result.get("use_build_dir", True)
    dependencies = result.get("dependencies", []) or []
    build_script = result.get("build_script")

    try:
        paths = generator.generate(
            project_name=project_name,
            project_path=project_path,
            flags=flags,
            container_image=container_image,
            build_system=build_system_used,
            enable_ir=generate_ir,
            use_build_dir=use_build_dir,
            dependencies=dependencies,
            build_script=build_script,
        )

        console.print(f"Script: [bold]{paths['script']}[/bold]")
        console.print(f"Config: {paths['config']}")
        console.print(f"Artifacts: {paths['artifacts_dir']}")

        # Extract compile_commands.json to build-scripts/<project>/
        console.print("\n[bold]Extracting compile_commands.json...[/bold]")
        try:
            project_dir = paths['script'].parent
            compile_commands_path = builder.extract_compile_commands(
                project_path, output_dir=project_dir, use_build_dir=use_build_dir
            )
            console.print(f"Extracted to: {compile_commands_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract compile_commands.json: {e}[/yellow]")
            compile_commands_path = None

        # Generate README if it doesn't exist
        readme_path = generator.scripts_base_dir / "README.md"
        if not readme_path.exists():
            generator.generate_readme()
            console.print(f"README: {readme_path}")

    except Exception as e:
        console.print(f"[red]Error generating build script: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    # Store in database
    console.print("\n[bold]Storing build configuration in database...[/bold]")
    db = SummaryDB(db_path)
    try:
        config_dict = {
            "flags": flags,
            "use_build_dir": use_build_dir,
        }
        if dependencies:
            config_dict["dependencies"] = dependencies

        db.add_build_config(
            project_path=str(project_path),
            project_name=project_name,
            build_system=build_system_used,
            configuration=config_dict,
            script_path=str(paths["script"]),
            artifacts_dir=str(paths["artifacts_dir"]),
            compile_commands_path=str(compile_commands_path) if compile_commands_path else None,
            llm_backend=backend,
            llm_model=llm.model,
            build_attempts=result["attempts"],
        )
        console.print(f"Database: {db_path}")
    finally:
        db.close()

    # Summary
    console.print("\n[bold green]✓ Build learning complete![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Run the build script:")
    console.print(f"   [cyan]{paths['script']}[/cyan]")
    console.print("\n2. Scan functions with llm-summary:")
    if compile_commands_path:
        console.print(
            f"   [cyan]llm-summary scan --compile-commands "
            f"{compile_commands_path} --db {db_path}[/cyan]"
        )
    else:
        console.print(
            f"   [cyan]llm-summary scan --compile-commands "
            f"<path-to-compile_commands.json> --db {db_path}[/cyan]"
        )

    if generate_ir:
        console.print("\n3. LLVM IR artifacts will be in:")
        console.print(f"   [cyan]{paths['artifacts_dir']}[/cyan]")


@main.command("import-callgraph")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True),
              help="Path to KAMain callgraph JSON")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--clear-edges", is_flag=True, help="Clear existing call_edges before import")
@click.option(
    "--link-units", "link_units_path",
    type=click.Path(exists=True), default=None,
    help="Path to link_units.json. With --target, copies shared "
         "functions/edges from each unit named in the target's "
         "imported_from and from each .a entry in its link_deps "
         "before importing the callgraph (avoids creating stubs for "
         "code that lives in linked-in static archives or subset units).",
)
@click.option(
    "--target", "target_name", default=None,
    help="Link-unit target name (required when --link-units is given).",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def import_callgraph(json_path, db_path, clear_edges, link_units_path,
                     target_name, verbose):
    """Import a KAMain call graph JSON into the database.

    Parses the call graph, matches functions to existing DB entries,
    creates stubs for unmatched functions, and populates call_edges.

    With --link-units / --target, dependency function rows are imported
    from each dep DB first so that calls into linked-in static archives
    or smaller subset units don't materialize as stubs. Both
    ``imported_from`` (subset deps; uses the unit's ``imported_files``
    file scope) and ``link_deps`` ending in ``.a`` (statically linked
    archives; uses every file_path present in the dep DB) are handled.
    """
    from .callgraph_import import CallGraphImporter

    if link_units_path and not target_name:
        console.print("[red]--target is required when --link-units is given[/red]")
        return
    if target_name and not link_units_path:
        console.print("[red]--link-units is required when --target is given[/red]")
        return

    json_path = Path(json_path)
    db = SummaryDB(db_path)

    try:
        # Show before stats
        before_stats = db.get_stats()
        console.print(f"Database: {db_path}")
        console.print(f"  Functions before: {before_stats['functions']}")
        console.print(f"  Call edges before: {before_stats['call_edges']}")

        if link_units_path:
            _import_link_unit_deps(
                db,
                Path(link_units_path),
                target_name,
                verbose=verbose,
            )

        importer = CallGraphImporter(db, verbose=verbose)
        stats = importer.import_json(json_path, clear_existing=clear_edges)

        console.print("\n[bold]Import complete[/bold]")
        console.print(stats.summary())

        # When edges were cleared and reimported, bump updated_at on all callee
        # stub summaries so that --incremental detects their callers as stale.
        if clear_edges:
            touched = db.touch_stub_summaries()
            if touched and verbose:
                console.print(
                    f"  Touched {touched} stub summary timestamps"
                    " (incremental staleness)"
                )

        # Show after stats
        after_stats = db.get_stats()
        console.print("\nDatabase after:")
        console.print(f"  Functions: {after_stats['functions']}")
        console.print(f"  Call edges: {after_stats['call_edges']}")

    finally:
        db.close()


def _import_link_unit_deps(
    db: "SummaryDB",
    link_units_path: Path,
    target_name: str,
    *,
    verbose: bool,
) -> None:
    """Copy dep functions/edges into ``db`` for the named target.

    Reads ``link_units_path``, finds ``target_name``, and runs
    ``db.import_unit_data`` once per dep:
      - ``imported_from``: file scope = the target's ``imported_files``.
      - ``link_deps`` ending in ``.a``: file scope = every distinct
        ``file_path`` present in the dep DB (entire archive embedded by
        the static link).
    Deps already covered by ``imported_from`` are not re-imported.
    """
    import json as _json
    import sqlite3

    lu_data = _json.loads(link_units_path.read_text())
    link_units_list = lu_data.get("link_units", [])
    by_name = {u.get("name"): u for u in link_units_list}
    by_output: dict[str, dict] = {}
    for u in link_units_list:
        # Aliases redirect to canonical so file resolution lines up.
        canonical = by_name.get(u.get("alias_of") or u.get("name"))
        if canonical is None:
            continue
        out = u.get("output")
        if out:
            by_output[out] = canonical
            by_output[Path(out).name] = canonical

    target_lu = by_name.get(target_name)
    if target_lu is None:
        console.print(
            f"[red]Target '{target_name}' not found in {link_units_path}[/red]"
        )
        return

    project_scan_dir = link_units_path.parent
    seen: set[str] = set()

    def _dep_db(dep: dict) -> str:
        return dep.get("db_path") or str(
            project_scan_dir / dep["name"] / "functions.db"
        )

    # 1) imported_from: use the unit's pre-computed imported_files.
    imported_files = target_lu.get("imported_files") or []
    for dep_name in target_lu.get("imported_from") or []:
        dep_lu = by_name.get(dep_name)
        if dep_lu is None:
            console.print(
                f"[yellow]  imported_from '{dep_name}' not in "
                f"link_units.json — skipping[/yellow]"
            )
            continue
        dep_db_str = _dep_db(dep_lu)
        if not Path(dep_db_str).exists():
            console.print(
                f"[yellow]  dep DB {dep_db_str} missing — skipping[/yellow]"
            )
            continue
        if not imported_files:
            if verbose:
                console.print(
                    f"  imported_from '{dep_name}' has empty imported_files"
                    " — skipping"
                )
            continue
        stats_imp = db.import_unit_data(dep_db_str, imported_files)
        seen.add(dep_name)
        console.print(
            f"  imported {stats_imp.functions} functions, "
            f"{stats_imp.call_edges} call edges from {dep_name} "
            f"(imported_from)"
        )

    # 2) .a link_deps: pull every file_path from the dep DB.
    for dep_output in target_lu.get("link_deps") or []:
        if not dep_output.endswith(".a"):
            continue
        dep_lu = by_output.get(dep_output) or by_output.get(
            Path(dep_output).name
        )
        if dep_lu is None:
            if verbose:
                console.print(
                    f"  .a link_dep '{dep_output}' has no matching unit"
                    " — skipping"
                )
            continue
        dep_name = dep_lu["name"]
        if dep_name in seen or dep_name == target_name:
            continue
        dep_db_str = _dep_db(dep_lu)
        if not Path(dep_db_str).exists():
            console.print(
                f"[yellow]  dep DB {dep_db_str} missing — skipping[/yellow]"
            )
            continue
        try:
            with sqlite3.connect(f"file:{dep_db_str}?mode=ro", uri=True) as c:
                files = [
                    r[0] for r in c.execute(
                        "SELECT DISTINCT file_path FROM functions"
                    )
                ]
        except sqlite3.Error as e:
            console.print(
                f"[yellow]  could not read file_paths from {dep_db_str}: "
                f"{e} — skipping[/yellow]"
            )
            continue
        if not files:
            if verbose:
                console.print(
                    f"  .a link_dep '{dep_name}' DB has no functions"
                    " — skipping"
                )
            continue
        stats_imp = db.import_unit_data(dep_db_str, files)
        seen.add(dep_name)
        console.print(
            f"  imported {stats_imp.functions} functions, "
            f"{stats_imp.call_edges} call edges from {dep_name} "
            f"(.a link_dep)"
        )


@main.command("discover-link-units")
@click.option(
    "--project-name", default=None,
    help="Project name "
         "(default: inferred from project path)",
)
@click.option(
    "--project-path", type=click.Path(exists=True),
    required=True, help="Path to the project source",
)
@click.option(
    "--build-dir", type=click.Path(exists=True),
    required=True, help="Path to the build directory",
)
@click.option(
    "--compile-commands", "compile_commands_path",
    type=click.Path(exists=True), default=None,
    help="Path to compile_commands.json "
         "(auto-detected if not given)",
)
@click.option(
    "--build-system", default=None,
    help="Build system hint (cmake, autotools, make)",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output path "
         "(default: func-scans/<project>/link_units.json)",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["claude", "openai", "ollama", "llamacpp", "gemini"]
    ),
    default="claude",
    help="LLM backend (only needed for non-Ninja builds)",
)
@click.option("--model", default=None, help="Model name")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking mode")
@click.option("--container-image", default="llm-summary-builder:latest",
              help="Docker image for sandboxed commands (agent mode only)")
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log LLM prompts/responses to file",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def discover_link_units(
    project_name, project_path, build_dir, compile_commands_path,
    build_system, output, backend, model, llm_host, llm_port,
    disable_thinking, container_image, log_llm, verbose,
):
    """Discover link units (libraries and executables) from build artifacts.

    Parses build artifacts to identify which object files and .bc files
    belong to each library and executable. For CMake+Ninja builds, this
    runs deterministically with no LLM calls.

    \b
    Examples:
        # Deterministic fast path (CMake+Ninja, no LLM needed)
        llm-summary discover-link-units \\
          --project-path /data/csong/opensource/zlib \\
          --build-dir /data/csong/build-artifacts/zlib -v

        # Agent path (autotools, needs LLM)
        llm-summary discover-link-units \\
          --project-path /data/csong/opensource/binutils-gdb \\
          --build-dir /data/csong/build-artifacts/binutils \\
          --backend gemini -v
    """
    from .link_units.skills import discover_deterministic

    project_path = Path(project_path).resolve()
    build_dir = Path(build_dir).resolve()

    # Infer project name
    if not project_name:
        project_name = project_path.name

    # Auto-detect compile_commands.json
    if compile_commands_path is None:
        # Check build-scripts/<project>/compile_commands.json
        candidate = Path("build-scripts") / project_name / "compile_commands.json"
        if candidate.exists():
            compile_commands_path = str(candidate)
        else:
            # Check build_dir/compile_commands.json
            candidate = build_dir / "compile_commands.json"
            if candidate.exists():
                compile_commands_path = str(candidate)

    if compile_commands_path:
        compile_commands_path = Path(compile_commands_path).resolve()

    # Auto-detect build system and in-tree build flag from config.json
    use_build_dir = True
    config_path = Path("build-scripts") / project_name / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if build_system is None:
                build_system = config.get("build_system")
            use_build_dir = config.get("use_build_dir", True)
        except (json.JSONDecodeError, OSError):
            pass

    # For in-tree builds (use_build_dir=false), scan the source tree for artifacts
    # Also auto-detect: if build_dir has no compile_commands.json but project_path does
    if not use_build_dir or (
        not (build_dir / "compile_commands.json").exists()
        and (project_path / "compile_commands.json").exists()
    ):
        use_build_dir = False
        build_dir = project_path
        if verbose:
            console.print("[yellow]In-tree build detected — using source dir as build dir[/yellow]")

    # Determine output path
    if output is None:
        output_dir = Path("func-scans") / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / "link_units.json")

    console.print("[bold]Link Unit Discovery[/bold]")
    console.print(f"  Project: {project_name}")
    console.print(f"  Source:  {project_path}")
    console.print(f"  Build:   {build_dir}")
    if compile_commands_path:
        console.print(f"  CC:      {compile_commands_path}")
    if build_system:
        console.print(f"  System:  {build_system}")
    console.print(f"  Output:  {output}")

    # Try deterministic fast path first (build.ninja)
    console.print("\n[bold]Attempting deterministic discovery...[/bold]")
    result = discover_deterministic(
        build_dir=build_dir,
        compile_commands_path=compile_commands_path,
        project_name=project_name,
        project_path=project_path,
        verbose=verbose,
    )

    if result is not None:
        console.print("[green]Deterministic discovery succeeded (no LLM needed)[/green]")
    else:
        # Try heuristic path (prescan + Makefile parsing)
        console.print("[yellow]No build.ninja — trying heuristic discovery...[/yellow]")

        from .link_units.skills import discover_heuristic

        heuristic_result, unresolved = discover_heuristic(
            build_dir, verbose=verbose, compile_commands_path=compile_commands_path,
        )
        heuristic_result["project"] = project_name

        if not unresolved:
            console.print("[green]Heuristic discovery fully resolved (no LLM needed)[/green]")
            result = heuristic_result
        else:
            console.print(
                f"[yellow]Heuristic resolved {len(heuristic_result['link_units'])} units, "
                f"{len(unresolved)} unresolved — falling back to LLM agent[/yellow]"
            )

            # Initialize LLM backend
            backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
            try:
                llm = create_backend(backend, model=model, **backend_kwargs)
            except Exception as e:
                console.print(f"[red]Failed to initialize LLM backend: {e}[/red]")
                console.print("[yellow]Using heuristic results only[/yellow]")
                result = heuristic_result
                llm = None

            if llm is not None:
                console.print(f"  Using {backend} backend ({llm.model})")

                from .link_units.discoverer import LinkUnitDiscoverer

                discoverer = LinkUnitDiscoverer(
                    llm=llm,
                    container_image=container_image,
                    verbose=verbose,
                    log_file=log_llm,
                )

                try:
                    agent_result = discoverer.discover(
                        project_name=project_name,
                        project_path=project_path,
                        build_dir=build_dir,
                        build_system=build_system,
                        heuristic_result=heuristic_result,
                        unresolved_objects=unresolved,
                    )
                    result = agent_result
                except Exception as e:
                    console.print(f"[red]Agent failed: {e}[/red]")
                    console.print("[yellow]Using heuristic results only[/yellow]")
                    result = heuristic_result

    # Populate bc_files for any link unit that only has objects (heuristic/agent path)
    if result is None:
        console.print("[red]No result from any discovery path[/red]")
        return
    from .link_units.skills import map_objects_to_bc
    result_build_dir = Path(result.get("build_dir", str(build_dir)))
    for lu in result.get("link_units", []):
        if not lu.get("bc_files") and lu.get("objects"):
            bc_result = map_objects_to_bc(lu["objects"], result_build_dir)
            lu["bc_files"] = [bc for bc in bc_result["mappings"].values() if bc is not None]
            if verbose:
                s = bc_result["stats"]
                console.print(
                    f"  [bc] {lu['name']}: {len(lu['bc_files'])} bc files "
                    f"(tier1={s['tier1']}, tier2={s['tier2']}, missing={s['not_found']})"
                )

    # Record compile_commands.json mtime for staleness detection
    cc_path = compile_commands_path or (build_dir / "compile_commands.json")
    if isinstance(cc_path, str):
        cc_path = Path(cc_path)
    if cc_path.exists():
        result["compile_commands_mtime"] = cc_path.stat().st_mtime

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    link_units = result.get("link_units", [])
    libs = [u for u in link_units if u.get("type") in ("static_library", "shared_library")]
    exes = [u for u in link_units if u.get("type") == "executable"]
    total_obj = sum(len(u.get("objects", [])) for u in link_units)
    total_bc = sum(len(u.get("bc_files", [])) for u in link_units)

    console.print("\n[bold]Results[/bold]")
    console.print(
        f"  Link units: {len(link_units)} "
        f"({len(libs)} libraries, "
        f"{len(exes)} executables)"
    )
    console.print(f"  Total objects: {total_obj}")
    if total_bc:
        console.print(f"  Total .bc files: {total_bc}")
    console.print(f"  Written to: {output_path}")

    if verbose and link_units:
        console.print("\n[bold]Link Units:[/bold]")
        for u in link_units:
            obj_count = len(u.get("objects", []))
            bc_count = len(u.get("bc_files", []))
            deps = u.get("link_deps", [])
            parts = [f"{obj_count} objects"]
            if bc_count:
                parts.append(f"{bc_count} bc")
            if deps:
                parts.append(f"deps=\\[{', '.join(deps)}]")
            utype = u.get("type", "unknown")
            console.print(f"  {utype:16s} {u['name']}: {', '.join(parts)}")


@main.command("import-dep-summaries")
@click.option("--db", "db_path", required=True, help="Target database path")
@click.option("--dep-db", "dep_db_paths", multiple=True, required=True,
              help="Source (dependency) database path(s). Repeat for multiple deps.")
@click.option("--force", "-f", is_flag=True,
              help="Overwrite existing summaries in target DB")
@click.option("--verbose", "-v", is_flag=True)
def import_dep_summaries(db_path, dep_db_paths, force, verbose):
    """Copy function summaries from dependency DBs into the target DB.

    For each function in a dep DB that has at least one summary, finds or
    creates a matching stub in the target DB and copies all available
    summaries (allocation, free, init, memsafe, verification).

    Existing summaries in the target are skipped unless --force is given.
    Imported summaries are tagged with model_used='dep:<dep_db_stem>'.

    Example:
        llm-summary import-dep-summaries \\
            --db func-scans/libpng/libpng16/functions.db \\
            --dep-db func-scans/zlib/zlibstatic/functions.db
    """
    # Per-function summary tables that share the (function_id, summary_json,
    # model_used) row shape. code_contract_summaries is handled separately
    # below because it has additional columns.
    summary_tables = [
        "allocation_summaries",
        "free_summaries",
        "init_summaries",
        "memsafe_summaries",
        "verification_summaries",
        "leak_summaries",
        "integer_overflow_summaries",
    ]

    target_db = SummaryDB(db_path)
    total_funcs = 0
    total_summaries = 0
    total_edges = 0

    try:
        for dep_path in dep_db_paths:
            dep_db = SummaryDB(dep_path)
            dep_tag = f"dep:{Path(dep_path).parent.name}"
            copied_funcs = 0
            copied_summaries = 0

            try:
                dep_funcs = dep_db.get_all_functions()

                for src_func in dep_funcs:
                    # Collect summaries available for this function in the dep DB
                    src_summaries: dict[str, tuple[str, str]] = {}
                    for table in summary_tables:
                        row = dep_db.conn.execute(
                            f"SELECT summary_json, model_used FROM {table}"
                            " WHERE function_id = ?",
                            (src_func.id,),
                        ).fetchone()
                        if row:
                            src_summaries[table] = (row[0], dep_tag)

                    # Code-contract has a richer row shape; fetch separately.
                    cc_row = dep_db.conn.execute(
                        "SELECT summary_json, model FROM code_contract_summaries"
                        " WHERE function_id = ?",
                        (src_func.id,),
                    ).fetchone()
                    cc_payload: tuple[str, str] | None = (
                        (cc_row[0], dep_tag) if cc_row else None
                    )

                    if not src_summaries and cc_payload is None:
                        continue

                    # Find or create matching functions in target DB.
                    # Multiple candidates can share a name (C++ overloads /
                    # template instantiations) — import to ALL of them.
                    candidates = target_db.get_function_by_name(src_func.name)
                    if not candidates:
                        stub_id = target_db.insert_function_stub(
                            name=src_func.name,
                            file_path=src_func.file_path,
                            line_start=src_func.line_start,
                            line_end=src_func.line_end,
                        )
                        stub = target_db.get_function(stub_id)
                        candidates = [stub] if stub else []

                    func_imported = False
                    for tgt_func in candidates:
                        if tgt_func is None:
                            continue
                        for table, (summary_json, model_used) in src_summaries.items():
                            existing = target_db.conn.execute(
                                f"SELECT id FROM {table} WHERE function_id = ?",
                                (tgt_func.id,),
                            ).fetchone()
                            if existing and not force:
                                continue

                            target_db.conn.execute(
                                f"INSERT INTO {table} (function_id, summary_json, model_used)"
                                " VALUES (?, ?, ?)"
                                " ON CONFLICT(function_id) DO UPDATE SET"
                                "   summary_json = excluded.summary_json,"
                                "   updated_at   = CURRENT_TIMESTAMP,"
                                "   model_used   = excluded.model_used",
                                (tgt_func.id, summary_json, model_used),
                            )
                            copied_summaries += 1
                            func_imported = True

                        if cc_payload is not None:
                            assert tgt_func.id is not None
                            existing_cc = target_db.get_code_contract_summary(tgt_func.id)
                            if existing_cc is None or force:
                                from .code_contract.models import CodeContractSummary
                                cc = CodeContractSummary.from_dict(
                                    json.loads(cc_payload[0])
                                )
                                cc.function = tgt_func.name
                                target_db.store_code_contract_summary(
                                    tgt_func, cc, model_used=cc_payload[1],
                                )
                                copied_summaries += 1
                                func_imported = True

                    if func_imported:
                        copied_funcs += 1
                        if verbose:
                            kinds = [t.replace("_summaries", "") for t in src_summaries]
                            if cc_payload is not None:
                                kinds.append("code_contract")
                            console.print(f"  {src_func.name}: {', '.join(kinds)}")

                target_db.conn.commit()

                # --- Copy call edges ---
                # Build src→tgt ID mapping for all functions present in both DBs
                id_map: dict[int, int] = {}
                for src_func in dep_funcs:
                    if src_func.id is None:
                        continue
                    candidates = target_db.get_function_by_name(src_func.name)
                    if candidates and candidates[0].id is not None:
                        id_map[src_func.id] = candidates[0].id

                # Fetch dep call edges and remap
                copied_edges = 0
                if id_map:
                    dep_edges = dep_db.conn.execute(
                        "SELECT caller_id, callee_id, is_indirect, file_path, line, \"column\""
                        " FROM call_edges"
                    ).fetchall()
                    for edge in dep_edges:
                        src_caller = edge[0]
                        src_callee = edge[1]
                        tgt_caller = id_map.get(src_caller)
                        tgt_callee = id_map.get(src_callee)
                        if tgt_caller is None or tgt_callee is None:
                            continue
                        # Skip if edge already exists
                        existing = target_db.conn.execute(
                            "SELECT id FROM call_edges"
                            " WHERE caller_id = ? AND callee_id = ?",
                            (tgt_caller, tgt_callee),
                        ).fetchone()
                        if existing:
                            continue
                        target_db.conn.execute(
                            "INSERT INTO call_edges"
                            " (caller_id, callee_id, is_indirect, file_path, line, \"column\")"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (tgt_caller, tgt_callee, edge[2], edge[3], edge[4], edge[5]),
                        )
                        copied_edges += 1
                    target_db.conn.commit()
            finally:
                dep_db.close()

            total_funcs += copied_funcs
            total_summaries += copied_summaries
            total_edges += copied_edges
            console.print(
                f"  {Path(dep_path).name}: "
                f"{copied_funcs} functions, {copied_summaries} summaries,"
                f" {copied_edges} call edges imported"
            )

        console.print(
            f"\n[bold]Total:[/bold] {total_funcs} functions, "
            f"{total_summaries} summaries, {total_edges} call edges"
            f" imported into {db_path}"
        )
    finally:
        target_db.close()


@main.command("import-dep")
@click.option("--db", "db_path", required=True, help="Target project database path")
@click.option(
    "--from", "from_db_path", default=None,
    help="Source dependency database path "
         "(e.g. func-scans/zlib/zlibstatic/functions.db)",
)
@click.option(
    "--scan-dir", "scan_dir", default="func-scans",
    help="Directory to auto-discover dependency DBs "
         "(default: func-scans)",
)
@click.option(
    "--link-units", "link_units_path", default=None,
    type=click.Path(),
    help="link_units.json to record resolved dep_dbs into",
)
@click.option(
    "--target", "target_name", default=None,
    help="Link-unit target name (for recording dep_dbs in link_units.json)",
)
@click.option("--force", "-f", is_flag=True,
              help="Overwrite existing summaries in target DB")
@click.option("--dry-run", is_flag=True,
              help="Show what would be imported without writing")
@click.option("--verbose", "-v", is_flag=True)
def import_dep(db_path, from_db_path, scan_dir, link_units_path, target_name,
               force, dry_run, verbose):
    """Import summaries from dependency project DBs using decl_header matching.

    Uses the decl_header column (populated by 'scan' Phase 4) to determine
    which external functions belong to which dependency, then copies summaries
    from the dependency's DB.  Standard library functions (libc/libm/POSIX)
    are always skipped — use init-stdlib for those.

    Resolved header -> dep DB mappings are cached globally in
    ~/.llm-summary/stdlib_cache.db so they don't need re-resolving.

    \b
    Explicit mode (--from):
        llm-summary import-dep \\
            --db func-scans/libpng/png_static/functions.db \\
            --from func-scans/zlib/zlibstatic/functions.db

    Auto-discover mode (no --from):
        llm-summary import-dep \\
            --db func-scans/libpng/png_static/functions.db

    Record deps in link_units.json:
        llm-summary import-dep \\
            --db func-scans/libpng/png_static/functions.db \\
            --from func-scans/zlib/zlibstatic/functions.db \\
            --link-units func-scans/libpng/link_units.json \\
            --target png_static
    """
    from .extern_headers import classify_header
    from .stdlib_cache import StdlibCache

    summary_tables = [
        "allocation_summaries",
        "free_summaries",
        "init_summaries",
        "memsafe_summaries",
        "verification_summaries",
        "leak_summaries",
        "integer_overflow_summaries",
    ]

    target_db = SummaryDB(db_path)
    cache = StdlibCache()

    try:
        # 1. Find externals with decl_header set (populated by scan Phase 4)
        all_funcs = target_db.get_all_functions()
        externals = [
            f for f in all_funcs
            if (not f.source) and f.decl_header
        ]

        if not externals:
            rows = target_db.conn.execute(
                "SELECT id, name, decl_header FROM functions "
                "WHERE (source IS NULL OR source = '') AND decl_header IS NOT NULL"
            ).fetchall()
            ext_info = [(r[0], r[1], r[2]) for r in rows]
        else:
            ext_info = [(f.id, f.name, f.decl_header) for f in externals]

        if not ext_info:
            console.print(
                "[yellow]No external functions with decl_header found.[/yellow]\n"
                "Run 'llm-summary scan' first to populate declaration headers."
            )
            return

        # 2. Separate stdlib vs third-party
        thirdparty: list[tuple[int, str, str]] = []  # (func_id, name, header)
        stdlib_count = 0
        for func_id, name, header in ext_info:
            if classify_header(header):
                stdlib_count += 1
            else:
                thirdparty.append((func_id, name, header))

        if not thirdparty:
            console.print(
                f"All {stdlib_count} externals with headers are stdlib — "
                "nothing to import (use init-stdlib for libc)."
            )
            return

        # Group by header
        from collections import defaultdict
        by_header: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for func_id, name, header in thirdparty:
            by_header[header].append((func_id, name))

        console.print(
            f"External functions: {len(ext_info)} total, "
            f"{stdlib_count} stdlib (skipped), "
            f"{len(thirdparty)} third-party across {len(by_header)} headers"
        )
        for hdr, funcs in sorted(by_header.items()):
            console.print(f"  {hdr}: {len(funcs)} functions")

        # 3. Resolve headers to dependency DBs
        if from_db_path:
            # Explicit: all third-party functions come from this DB
            dep_mapping: dict[str, str] = dict.fromkeys(by_header, from_db_path)
            # Cache the explicit mapping
            for hdr in by_header:
                basename = Path(hdr).stem
                cache.put_dep_header(hdr, basename, from_db_path, resolved_by="explicit")
        else:
            # Auto-discover: check global cache first, then heuristic
            dep_mapping = _resolve_dep_dbs(by_header.keys(), scan_dir, db_path,
                                           cache, verbose)

        if not dep_mapping:
            console.print("[yellow]No dependency DBs found for the headers.[/yellow]")
            return

        # 4. Import summaries
        total_funcs = 0
        total_summaries = 0

        for header, dep_path in sorted(dep_mapping.items()):
            funcs_for_header = by_header.get(header, [])
            if not funcs_for_header:
                continue

            dep_db = SummaryDB(dep_path)
            dep_tag = f"dep:{Path(dep_path).parent.name}/{Path(dep_path).stem}"
            copied_funcs = 0
            copied_summaries = 0

            try:
                for func_id, func_name in funcs_for_header:
                    # Find the function in the dep DB (by name, with source)
                    dep_candidates = dep_db.get_function_by_name(func_name)
                    # Prefer the one with source (real implementation)
                    dep_func = None
                    for c in dep_candidates:
                        if c.source:
                            dep_func = c
                            break
                    if dep_func is None and dep_candidates:
                        dep_func = dep_candidates[0]
                    if dep_func is None:
                        if verbose:
                            console.print(f"  [dim]{func_name}: not found in {dep_path}[/dim]")
                        continue

                    # Collect legacy summaries from dep DB
                    src_summaries: dict[str, str] = {}
                    for table in summary_tables:
                        row = dep_db.conn.execute(
                            f"SELECT summary_json FROM {table} WHERE function_id = ?",
                            (dep_func.id,),
                        ).fetchone()
                        if row:
                            src_summaries[table] = row[0]

                    # Code-contract summary (different schema)
                    cc_row = dep_db.conn.execute(
                        "SELECT summary_json FROM code_contract_summaries"
                        " WHERE function_id = ?",
                        (dep_func.id,),
                    ).fetchone()

                    if not src_summaries and not cc_row:
                        if verbose:
                            console.print(f"  [dim]{func_name}: no summaries in {dep_path}[/dim]")
                        continue

                    if dry_run:
                        kinds = [t.replace("_summaries", "") for t in src_summaries]
                        if cc_row:
                            kinds.append("code_contract")
                        console.print(f"  [dry-run] {func_name}: would import {', '.join(kinds)}")
                        copied_funcs += 1
                        copied_summaries += len(src_summaries) + (1 if cc_row else 0)
                        continue

                    func_imported = False
                    for table, summary_json in src_summaries.items():
                        existing = target_db.conn.execute(
                            f"SELECT id FROM {table} WHERE function_id = ?",
                            (func_id,),
                        ).fetchone()
                        if existing and not force:
                            continue
                        target_db.conn.execute(
                            f"INSERT INTO {table} (function_id, summary_json, model_used)"
                            " VALUES (?, ?, ?)"
                            " ON CONFLICT(function_id) DO UPDATE SET"
                            "   summary_json = excluded.summary_json,"
                            "   updated_at   = CURRENT_TIMESTAMP,"
                            "   model_used   = excluded.model_used",
                            (func_id, summary_json, dep_tag),
                        )
                        copied_summaries += 1
                        func_imported = True

                    if cc_row:
                        existing_cc = target_db.conn.execute(
                            "SELECT function_id FROM code_contract_summaries"
                            " WHERE function_id = ?",
                            (func_id,),
                        ).fetchone()
                        if not existing_cc or force:
                            target_db.conn.execute(
                                "INSERT INTO code_contract_summaries"
                                " (function_id, summary_json, model)"
                                " VALUES (?, ?, ?)"
                                " ON CONFLICT(function_id) DO UPDATE SET"
                                "   summary_json = excluded.summary_json,"
                                "   model = excluded.model,"
                                "   updated_at = CURRENT_TIMESTAMP",
                                (func_id, cc_row[0], dep_tag),
                            )
                            copied_summaries += 1
                            func_imported = True

                    if func_imported:
                        copied_funcs += 1
                        if verbose:
                            kinds = [t.replace("_summaries", "") for t in src_summaries]
                            if cc_row:
                                kinds.append("code_contract")
                            console.print(f"  {func_name}: {', '.join(kinds)}")

                if not dry_run:
                    target_db.conn.commit()
            finally:
                dep_db.close()

            total_funcs += copied_funcs
            total_summaries += copied_summaries
            console.print(
                f"  {Path(dep_path).name} ({header}): "
                f"{copied_funcs} functions, {copied_summaries} summaries"
                + (" (dry-run)" if dry_run else "")
            )

        # 5. Record dep_dbs in link_units.json if requested
        if link_units_path and not dry_run:
            _record_dep_dbs_in_link_units(
                link_units_path, target_name, dep_mapping, verbose,
            )

        action = "would import" if dry_run else "imported"
        console.print(
            f"\n[bold]Total:[/bold] {total_funcs} functions, "
            f"{total_summaries} summaries {action}"
        )
    finally:
        target_db.close()
        cache.close()


def _resolve_dep_dbs(
    headers: "Iterable[str]",
    scan_dir: str,
    target_db_path: str,
    cache: "StdlibCache",
    verbose: bool,
) -> dict[str, str]:
    """Map header paths to dependency DB paths.

    Checks the global cache first, then falls back to basename heuristic.
    Resolved mappings are persisted in the global cache for future use.
    """
    import glob as _glob

    target_db_resolved = str(Path(target_db_path).resolve())
    scan_root = Path(scan_dir)
    mapping: dict[str, str] = {}

    # Check global cache first
    headers_to_resolve: list[str] = []
    for header in headers:
        cached = cache.get_dep_header(header)
        if cached:
            lib_name, dep_db_path, resolved_by = cached
            if dep_db_path and Path(dep_db_path).exists():
                mapping[header] = dep_db_path
                if verbose:
                    console.print(f"  [cache] {header} -> {dep_db_path} ({resolved_by})")
            else:
                # Cached path no longer exists, re-resolve
                headers_to_resolve.append(header)
        else:
            headers_to_resolve.append(header)

    if not headers_to_resolve:
        return mapping

    # Build index of available projects from func-scans/
    available: dict[str, list[str]] = {}
    for db_file in _glob.glob(str(scan_root / "*/*/functions.db")):
        if str(Path(db_file).resolve()) == target_db_resolved:
            continue
        parts = Path(db_file).relative_to(scan_root).parts
        project_name = parts[0]
        available.setdefault(project_name, []).append(db_file)

    for header in headers_to_resolve:
        basename = Path(header).stem
        parent = Path(header).parent.name

        candidates = []
        if basename in available:
            candidates.append(basename)
        if parent not in ("include", "local") and parent in available:
            candidates.append(parent)
        if f"lib{basename}" in available:
            candidates.append(f"lib{basename}")

        if len(candidates) == 1:
            project = candidates[0]
            dbs = available[project]
            best = max(dbs, key=lambda p: Path(p).stat().st_size)
            mapping[header] = best
            cache.put_dep_header(header, project, best, resolved_by="heuristic")
            if verbose:
                console.print(f"  [resolve] {header} -> {best}")
        elif len(candidates) > 1:
            console.print(
                f"  [yellow]Ambiguous: {header} matches projects: "
                f"{', '.join(candidates)}[/yellow]"
            )
        else:
            console.print(
                f"  [yellow]No match: {header} "
                f"(tried: {basename}, {parent}, lib{basename})[/yellow]"
            )

    return mapping


def _record_dep_dbs_in_link_units(
    link_units_path: str,
    target_name: str | None,
    dep_mapping: dict[str, str],
    verbose: bool,
) -> None:
    """Record resolved dep_dbs in link_units.json for the target link unit."""
    lu_path = Path(link_units_path)
    if not lu_path.exists():
        console.print(f"  [yellow]link_units.json not found: {lu_path}[/yellow]")
        return

    with open(lu_path) as f:
        lu_data = json.load(f)

    link_units = lu_data.get("link_units", lu_data.get("targets", []))

    # Collect unique dep DB paths
    dep_db_paths = sorted(set(dep_mapping.values()))

    if target_name:
        # Update specific target
        for lu in link_units:
            if lu.get("name") == target_name:
                lu["dep_dbs"] = dep_db_paths
                break
        else:
            console.print(f"  [yellow]Target '{target_name}' not found in {lu_path}[/yellow]")
            return
    else:
        # Update all link units
        for lu in link_units:
            lu.setdefault("dep_dbs", [])
            for p in dep_db_paths:
                if p not in lu["dep_dbs"]:
                    lu["dep_dbs"].append(p)

    with open(lu_path, "w") as f:
        json.dump(lu_data, f, indent=2)
        f.write("\n")

    if verbose:
        console.print(f"  [link-units] Recorded {len(dep_db_paths)} dep DBs in {lu_path}")
        for p in dep_db_paths:
            console.print(f"    {p}")


@main.command("review-issue")
@click.argument("function_name")
@click.argument("issue_index", type=int)
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--status",
    "review_status",
    required=True,
    type=click.Choice(["pending", "confirmed", "false_positive", "wontfix"]),
    help="Review status to set",
)
@click.option("--reason", default=None, help="Reviewer explanation")
@click.option("--signature", default=None, help="Function signature for disambiguation")
def review_issue(
    function_name: str, issue_index: int, db_path: str,
    review_status: str, reason: str | None, signature: str | None,
) -> None:
    """Mark a verification issue as confirmed, false_positive, or wontfix.

    Example:
        llm-summary review-issue sftp_path_append 0 \\
          --status false_positive \\
          --reason "short-circuit eval guards this" \\
          --db func-scans/openssh/scp/functions.db
    """
    db = SummaryDB(db_path)

    try:
        # Resolve function
        functions = db.get_function_by_name(function_name)
        if not functions:
            console.print(f"[red]Function '{function_name}' not found in database.[/red]")
            sys.exit(1)

        if signature:
            functions = [f for f in functions if f.signature == signature]
            if not functions:
                console.print(
                    f"[red]No function '{function_name}' "
                    f"with signature "
                    f"'{signature}'.[/red]"
                )
                sys.exit(1)

        if len(functions) > 1:
            console.print(f"[yellow]Multiple functions named '{function_name}':[/yellow]")
            for f in functions:
                console.print(f"  {f.signature}  ({f.file_path}:{f.line_start})")
            console.print("Use --signature to disambiguate.")
            sys.exit(1)

        func = functions[0]
        assert func.id is not None

        # Load verification summary
        vsummary = db.get_verification_summary_by_function_id(func.id)
        if not vsummary:
            console.print(
                f"[red]No verification summary for "
                f"'{function_name}'. "
                f"Run verify pass first.[/red]"
            )
            sys.exit(1)

        if not vsummary.issues:
            console.print(f"[yellow]'{function_name}' has no issues.[/yellow]")
            return

        if issue_index < 0 or issue_index >= len(vsummary.issues):
            console.print(
                f"[red]Invalid issue_index {issue_index}. "
                f"'{function_name}' has "
                f"{len(vsummary.issues)} issue(s) "
                f"(0-{len(vsummary.issues) - 1}).[/red]"
            )
            sys.exit(1)

        issue = vsummary.issues[issue_index]
        fp = issue.fingerprint()

        db.upsert_issue_review(
            function_id=func.id,
            issue_index=issue_index,
            fingerprint=fp,
            status=review_status,
            reason=reason,
        )

        console.print(
            f"[green]Marked issue #{issue_index} of '{function_name}' as {review_status}[/green]"
        )
        console.print(f"  kind: {issue.issue_kind}")
        console.print(f"  location: {issue.location}")
        console.print(f"  fingerprint: {fp}")
        if reason:
            console.print(f"  reason: {reason}")

    finally:
        db.close()


@main.command("show-issues")
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option("--name", default=None, help="Filter by function name")
@click.option(
    "--status",
    "filter_status",
    default=None,
    type=click.Choice(["pending", "confirmed", "false_positive", "wontfix"]),
    help="Filter by review status",
)
@click.option(
    "--severity", default=None,
    type=click.Choice(["high", "medium", "low"]),
    help="Filter by severity",
)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_issues(
    db_path: str, name: str | None, filter_status: str | None,
    severity: str | None, fmt: str,
) -> None:
    """List all verification issues with their review status.

    Example:
        llm-summary show-issues --db func-scans/openssh/scp/functions.db
        llm-summary show-issues --db ... --status false_positive
        llm-summary show-issues --db ... --severity high --status pending
        llm-summary show-issues --db ... --name sftp_path_append
    """
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()

        if name:
            functions = [f for f in functions if name in f.name]

        rows = []
        for func in functions:
            assert func.id is not None
            vsummary = db.get_verification_summary_by_function_id(func.id)
            leak_summary = db.get_leak_summary_by_function_id(func.id)
            # Merge leak issues into verification issues
            all_issues: list = []
            if vsummary and vsummary.issues:
                all_issues.extend(vsummary.issues)
            if leak_summary and leak_summary.issues:
                all_issues.extend(leak_summary.issues)
            if not all_issues:
                continue

            # Build fingerprint->review lookup
            fingerprints = [iss.fingerprint() for iss in all_issues]
            reviews = db.get_issue_reviews_by_fingerprints(func.id, fingerprints)

            for idx, issue in enumerate(all_issues):
                fp = issue.fingerprint()
                review = reviews.get(fp)
                issue_status = review["status"] if review else "pending"
                issue_reason = review["reason"] if review else None

                if severity and issue.severity != severity:
                    continue
                if filter_status and issue_status != filter_status:
                    continue

                rows.append({
                    "function": func.name,
                    "file": Path(func.file_path).name,
                    "index": idx,
                    "severity": issue.severity,
                    "kind": issue.issue_kind,
                    "status": issue_status,
                    "description": issue.description,
                    "reason": issue_reason,
                    "fingerprint": fp,
                    "location": issue.location,
                })

        if fmt == "json":
            console.print(json.dumps(rows, indent=2))
        else:
            table = Table(title=f"Verification Issues ({len(rows)})")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("#", justify="right")
            table.add_column("Sev", style="bold")
            table.add_column("Kind")
            table.add_column("Status")
            table.add_column("Description")

            status_styles = {
                "pending": "",
                "confirmed": "red",
                "false_positive": "green",
                "wontfix": "yellow",
            }

            for r in rows:
                sev = str(r["severity"])
                sev_style = (
                    "red" if sev == "high"
                    else ("yellow" if sev == "medium"
                          else "dim")
                )
                status = str(r["status"])
                status_style = status_styles.get(status, "")
                desc = str(r["description"])
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                table.add_row(
                    str(r["function"]),
                    str(r["file"]),
                    str(r["index"]),
                    f"[{sev_style}]{sev}[/{sev_style}]",
                    str(r["kind"]),
                    (f"[{status_style}]{status}"
                     f"[/{status_style}]"
                     if status_style
                     else status),
                    desc,
                )

            console.print(table)

    finally:
        db.close()


@main.command("triage")
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option("--disable-thinking", is_flag=True, help="Disable extended thinking")
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "-f", "--function", "function_names", multiple=True,
    help="Function name(s) to triage. If omitted, triages all functions with issues.",
)
@click.option(
    "--severity", default=None,
    type=click.Choice(["high", "medium", "low"]),
    help="Only triage issues of this severity.",
)
@click.option(
    "--issue-index", default=None, type=int,
    help="Triage a specific issue index (requires -f with a single function).",
)
@click.option(
    "-o", "--output", default=None,
    help="Write JSON results to file (default: print to stdout).",
)
@click.option(
    "--project-path", default=None,
    help="Project source root (enables git_show/git_grep/git_ls_tree tools).",
)
@click.option(
    "--limit", default=None, type=int,
    help="Max number of issues to triage (useful for testing).",
)
@click.option(
    "--build-script-dir", default=None,
    help="Path to build-scripts/<project>/ directory. Enables Stage 4: "
         "witness generation for feasible verdicts. "
         "compile_commands.json is auto-loaded from this dir.",
)
@click.option(
    "-d", "--harness-dir", default=None,
    help="Output directory for witness harnesses "
         "(default: harnesses/<project>/).",
)
@click.option(
    "--build-dir", default=None,
    help="Host directory for ASan rebuild artifacts. Mounted into Docker "
         "for build and test execution. Required with --build-script-dir.",
)
@click.option(
    "--force", is_flag=True,
    help="Re-triage issues regardless of existing review status.",
)
def triage(
    db_path: str, backend: str, model: str | None,
    llm_host: str, llm_port: int | None, disable_thinking: bool,
    verbose: bool, function_names: tuple[str, ...],
    severity: str | None, issue_index: int | None,
    output: str | None, project_path: str | None,
    limit: int | None,
    build_script_dir: str | None,
    harness_dir: str | None,
    build_dir: str | None,
    force: bool,
) -> None:
    """Triage verification issues: prove safety or feasibility.

    The triage agent analyzes each issue by reading caller/callee context
    from the DB and produces a proof:

    \b
    - safe: updated contracts showing the issue cannot manifest
    - feasible: execution path showing the violation is reachable

    Examples:
        llm-summary triage --db func-scans/zlib/functions.db -f deflate -v
        llm-summary triage --db ... --severity high --backend gemini
        llm-summary triage --db ... -f gz_write --issue-index 0 -v
    """
    from .triage import TriageAgent

    if issue_index is not None and len(function_names) != 1:
        console.print("[red]--issue-index requires exactly one -f function[/red]")
        sys.exit(1)

    kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
    db = SummaryDB(db_path)

    def llm_factory() -> "LLMBackend":
        return create_backend(backend, model=model, **kwargs)

    try:
        from pathlib import Path as _Path
        out_path = _Path(output) if output else None

        # Stage 4 (witness) setup
        compile_commands = None
        bsd_path = _Path(build_script_dir) if build_script_dir else None
        if bsd_path is not None:
            cc_path = bsd_path / "compile_commands.json"
            if cc_path.exists():
                with open(cc_path) as f:
                    compile_commands = json.load(f)
                if verbose:
                    console.print(
                        f"Loaded compile_commands: "
                        f"{len(compile_commands)} entries",
                    )

        harness_out: _Path | None = None
        if harness_dir:
            harness_out = _Path(harness_dir)
        elif bsd_path is not None:
            db_p = _Path(db_path)
            proj_name = db_p.parent.name
            harness_out = _Path("harnesses") / proj_name

        agent = TriageAgent(
            db, llm_factory, verbose=verbose,
            project_path=_Path(project_path) if project_path else None,
            output_path=out_path,
            compile_commands=compile_commands,
            build_script_dir=bsd_path,
            harness_output_dir=harness_out,
            build_dir=_Path(build_dir) if build_dir else None,
        )

        # Resolve functions to triage
        if function_names:
            functions = []
            for fn in function_names:
                found = db.get_function_by_name(fn)
                if not found:
                    console.print(f"[red]Function '{fn}' not found[/red]")
                    sys.exit(1)
                functions.append(found[0])
        else:
            # All functions with verification issues
            functions = []
            for func in db.get_all_functions():
                assert func.id is not None
                vs = db.get_verification_summary_by_function_id(func.id)
                if vs and vs.issues:
                    functions.append(func)

        severity_filter = {severity} if severity else None
        all_results: list[Any] = []

        for func in functions:
            if limit is not None and len(all_results) >= limit:
                break
            assert func.id is not None

            if issue_index is not None:
                # Triage a specific issue
                vs = db.get_verification_summary_by_function_id(func.id)
                if not vs or not vs.issues:
                    console.print(f"[yellow]No issues for {func.name}[/yellow]")
                    continue
                if issue_index >= len(vs.issues):
                    console.print(
                        f"[red]Issue index {issue_index} out of range "
                        f"(0..{len(vs.issues) - 1})[/red]"
                    )
                    sys.exit(1)
                issue = vs.issues[issue_index]
                result = agent.triage_issue(func, issue, issue_index)
                all_results.append(result)
            else:
                results = agent.triage_function(
                    func, severity_filter=severity_filter,
                    force=force,
                )
                remaining = (
                    limit - len(all_results) if limit is not None
                    else len(results)
                )
                all_results.extend(results[:remaining])

        # Output (agent saves incrementally via output_path)
        if output:
            console.print(f"[green]Wrote {len(all_results)} results to {output}[/green]")
        else:
            # Print summary table to stdout
            safe_count = sum(1 for r in all_results if r.hypothesis == "safe")
            gap_count = sum(1 for r in all_results if r.hypothesis == "contract_gap")
            feasible_count = sum(1 for r in all_results if r.hypothesis == "feasible")
            console.print(
                f"\n[bold]Triage Results[/bold]: {len(all_results)} issues — "
                f"[green]{safe_count} safe[/green], "
                f"[yellow]{gap_count} contract_gap[/yellow], "
                f"[red]{feasible_count} feasible[/red]"
            )
            for r in all_results:
                style = (
                    "green" if r.hypothesis == "safe"
                    else "yellow" if r.hypothesis == "contract_gap"
                    else "red"
                )
                console.print(
                    f"  [{style}]{r.hypothesis}[/{style}] "
                    f"{r.function_name}#{r.issue_index}: "
                    f"{r.issue.issue_kind} — {r.reasoning[:120]}"
                )

            if verbose:
                console.print("\n[dim]Full JSON:[/dim]")
                console.print(json.dumps(
                    [r.to_dict() for r in all_results], indent=2,
                ))

    finally:
        db.close()


@main.command("contract-check")
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--project-path", required=True,
    help="Project source root (the library's git repo).",
)
@click.option(
    "--library", default=None,
    help="Library name (defaults to derived from --db parent directory).",
)
@click.option(
    "--target", default=None,
    help="Build target / link unit (defaults to derived from --db).",
)
@click.option(
    "--max-hypothesis-turns", default=80, type=int,
    help="Max ReAct turns for the hypothesis stage (default 80).",
)
@click.option(
    "--max-audit-turns", default=60, type=int,
    help="Max ReAct turns per audit (default 60). Each candidate gets "
         "its own fresh-context audit with this budget.",
)
@click.option(
    "--limit", "audit_limit", default=None, type=int,
    help="Audit only the first N candidates from the hypothesis stage "
         "(testing/cost control). Default: audit all.",
)
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option("--disable-thinking", is_flag=True, help="Disable extended thinking")
@click.option(
    "-o", "--output", default=None,
    help="Write JSON gap catalog to file (default: print to stdout).",
)
@click.option("-v", "--verbose", is_flag=True)
def contract_check(
    db_path: str, project_path: str,
    library: str | None, target: str | None,
    max_hypothesis_turns: int, max_audit_turns: int,
    audit_limit: int | None,
    backend: str, model: str | None,
    llm_host: str, llm_port: int | None, disable_thinking: bool,
    output: str | None, verbose: bool,
) -> None:
    """Hunt for hazardous behaviors not warned about in the library's docs.

    Two-stage pipeline: a HYPOTHESIS agent walks the public API and emits
    a list of candidate hazards (null inputs, ordering, lifecycle,
    noreturn, error returns, ...). Then a fresh AUDIT agent runs PER
    candidate to check whether the manual / public header / example
    program warns about it (lean strict — vague mentions count as
    undocumented). Documented hazards are dropped; the rest become the
    final catalog. Hypothesis flags incomplete contract-DB rows too,
    driving both upstream doc patches and DB seeding.

    \b
    Example:
        llm-summary contract-check \\
            --db func-scans/libpng/png_static/functions.db \\
            --project-path /data/csong/opensource/libpng \\
            --backend claude -o gaps.json -v
    """
    from .contract_check import ContractCheckAgent

    db_p = Path(db_path)
    if library is None or target is None:
        # Heuristic: .../<library>/<target>/functions.db
        parts = db_p.resolve().parts
        if len(parts) >= 3:
            target = target or parts[-2]
            library = library or parts[-3]
        else:
            library = library or db_p.stem
            target = target or "default"

    kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)

    def _llm_factory() -> Any:
        return create_backend(backend, model=model, **kwargs)

    db = SummaryDB(db_path)

    try:
        agent = ContractCheckAgent(
            db, _llm_factory, verbose=verbose,
            project_path=Path(project_path),
        )
        result = agent.check_library(
            library=library, target=target,
            max_hypothesis_turns=max_hypothesis_turns,
            max_audit_turns=max_audit_turns,
            audit_limit=audit_limit,
            output_path=Path(output) if output else None,
        )

        result_json = result.to_dict()

        if output:
            # check_library already wrote the final JSON to `output`
            # incrementally; just confirm to the user.
            console.print(
                f"[green]Wrote {len(result.gaps)} gaps to {output}[/green]",
            )
        else:
            status = (
                "[green]completed[/green]" if result.completed
                else "[red]incomplete[/red]"
            )
            console.print(
                f"\n[bold]Contract Check[/bold] "
                f"{result.library}/{result.target}: {status} — "
                f"{len(result.gaps)} gap(s)"
            )
            by_cat: dict[str, int] = {}
            by_kind: dict[str, int] = {}
            for g in result.gaps:
                for cat in g.categories:
                    by_cat[cat] = by_cat.get(cat, 0) + 1
                if g.hazard_kind:
                    by_kind[g.hazard_kind] = by_kind.get(g.hazard_kind, 0) + 1
            for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                console.print(f"  {cat}: {n}")
            if by_kind:
                console.print("[dim]by hazard:[/dim]")
                for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
                    console.print(f"  {kind}: {n}")
            if verbose:
                console.print("\n[dim]Full JSON:[/dim]")
                console.print(json.dumps(result_json, indent=2))
    finally:
        db.close()


@main.command("gen-harness")
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option("--disable-thinking", is_flag=True, help="Disable extended thinking")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--log-llm", default=None, help="Log LLM interactions to file")
@click.option(
    "-o", "--output-dir", default=None,
    help="Output directory for harness files (default: harnesses/<project>/)",
)
@click.option(
    "-f", "--function", "function_names", multiple=True,
    help="Function name(s) to generate harnesses for. "
         "If omitted, generates for all functions with memsafe contracts.",
)
@click.option(
    "--ko-clang-path", default=None,
    help="Path to ko-clang binary. When set, compiles the shim and "
         "uses LLM to fix compilation errors (up to 3 attempts).",
)
@click.option(
    "--symsan-dir", default=None,
    help="Path to SymSan install directory (auto-detected from ko-clang-path if not set).",
)
@click.option(
    "--compile-commands", "compile_commands_path",
    type=click.Path(exists=True), default=None,
    help="Path to compile_commands.json for re-compiling source to bitcode.",
)
@click.option(
    "--project-path", default=None,
    help="Host path to project source root. Required when "
         "compile_commands.json uses Docker container paths.",
)
@click.option(
    "--build-dir", default=None,
    help="Host path to build directory (for remapping container build paths).",
)
@click.option(
    "--bc-file", default=None,
    help="Path to pre-compiled project bitcode (.bc). "
         "If not set, re-compiles from compile_commands.json.",
)
@click.option(
    "--plan", is_flag=True, default=False,
    help="After generating harnesses, also generate LLM trace plans "
         "(annotate source with BB IDs and query LLM for exploration strategy).",
)
@click.option(
    "--plan-only", is_flag=True, default=False,
    help="Only generate trace plans (skip harness generation). "
         "Requires existing harness files in the output directory.",
)
@click.option(
    "--validate", default=None, type=click.Path(exists=True),
    help="Path to triage verdict JSON. Generates a harness to validate "
         "the triage conclusion using ucsan. Entry function and scope are "
         "derived from the verdict's relevant_functions list.",
)
@click.option(
    "--issue-index", "issue_index", default=None, type=int,
    help="When used with --validate, only process the verdict with this "
         "issue_index. Skips all other verdicts in the file.",
)
@click.option(
    "--seeds-only", is_flag=True, default=False,
    help="Only generate seeds for an existing harness (skip shim generation). "
         "Requires --validate and an existing shim + config in the output dir.",
)
def gen_harness(
    db_path, backend, model, llm_host, llm_port,
    disable_thinking, verbose, log_llm, output_dir, function_names,
    ko_clang_path, symsan_dir, compile_commands_path, project_path,
    build_dir, bc_file, plan, plan_only, validate, issue_index, seeds_only,
):
    """Generate test harnesses for contract-guided symbolic execution.

    Generates thin C shims with __dfsw_ callee stubs and test() entry points
    that link against instrumented project bitcode via SymSan/ucsan.

    Example:
        llm-summary gen-harness --db func-scans/zlib/zlibstatic/functions.db \\
            -f gzputc --ko-clang-path ~/fuzzing/symsan/b3/bin/ko-clang \\
            --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \\
            --project-path /data/csong/opensource/zlib -v

        # Plan only (skip shim regeneration):
        llm-summary gen-harness --db func-scans/zlib/zlibstatic/functions.db \\
            -f gzputc --plan-only --symsan-dir ~/fuzzing/symsan/b3 \\
            --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \\
            --project-path /data/csong/opensource/zlib -v
    """
    from pathlib import Path

    from .harness_generator import HarnessGenerator

    db = SummaryDB(db_path)
    tmp_cc_path = None
    try:
        # Determine output directory
        if output_dir is None:
            db_p = Path(db_path)
            project_name = db_p.parent.name
            output_dir = str(Path("harnesses") / project_name)

        # Load compile_commands.json with path remapping
        compile_commands = None
        if not compile_commands_path:
            # Auto-detect: build-scripts/<project>/ then build-dir/
            db_p2 = Path(db_path)
            proj_name = db_p2.parent.parent.name  # func-scans/<proj>/<unit>/
            candidates = [
                Path("build-scripts") / proj_name / "compile_commands.json",
            ]
            if build_dir:
                candidates.append(Path(build_dir) / "compile_commands.json")
            for cc_candidate in candidates:
                if cc_candidate.exists():
                    compile_commands_path = str(cc_candidate)
                    if verbose:
                        console.print(
                            f"Auto-detected compile_commands: {cc_candidate}")
                    break
        if compile_commands_path:
            compile_commands, tmp_cc_path = _load_compile_commands(
                compile_commands_path, project_path, build_dir,
            )
            if verbose:
                console.print(f"Loaded compile_commands: {len(compile_commands)} entries")

        # Build LLM backend
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        generator = HarnessGenerator(
            db, llm, verbose=verbose, log_file=log_llm,
            ko_clang_path=ko_clang_path,
            symsan_dir=symsan_dir,
            compile_commands=compile_commands,
            project_path=project_path,
        )

        if seeds_only and not validate:
            console.print("[red]--seeds-only requires --validate[/red]")
            return

        # Validate a triage verdict
        if validate:
            with open(validate) as vf:
                verdict_data = json.load(vf)

            # Support both single verdict and list of verdicts
            verdicts = verdict_data if isinstance(verdict_data, list) else [verdict_data]

            for vi, v in enumerate(verdicts):
                relevant = v.get("relevant_functions", [])
                func_name = v.get("function_name", "")
                issue_idx = v.get("issue_index", vi)

                # Filter by --issue-index if provided
                if issue_index is not None and issue_idx != issue_index:
                    continue

                if not relevant:
                    relevant = [func_name] if func_name else []
                if not relevant:
                    console.print("[red]No relevant_functions in verdict[/red]")
                    continue

                # Build test cases from validation_plan or fallback
                vplan = v.get("validation_plan", [])
                if vplan:
                    test_cases: list[list[str]] = []
                    for tc in vplan:
                        tc_entries = tc.get("entries", [])
                        # Remove functions called by other functions
                        # in THIS entry list, preserving order
                        tc_non_entries = (
                            set(tc_entries)
                            - set(_find_entry_functions(db, tc_entries))
                        )
                        filtered = [
                            e for e in tc_entries
                            if e not in tc_non_entries
                        ]
                        if filtered:
                            test_cases.append(filtered)
                else:
                    # Legacy: one test case per entry function
                    entries = _find_entry_functions(db, relevant)
                    test_cases = [[e] for e in entries]

                # Per-verdict output dir — fingerprint prevents stale reuse
                verdict_dir = str(_verdict_dir(
                    Path(output_dir), func_name, issue_idx, v.get("issue", {}),
                ))
                Path(verdict_dir).mkdir(parents=True, exist_ok=True)
                console.print(
                    f"Validating {func_name}#{issue_idx} "
                    f"({v.get('hypothesis', '?')}): "
                    f"test_cases={test_cases} -> {verdict_dir}"
                )

                for tc_entries in test_cases:
                    # Harness named after the last entry (function under test)
                    harness_name = tc_entries[-1]
                    # Scope: all entries + non-entry relevant functions
                    all_entry_fns = {
                        e for tc in test_cases for e in tc
                    }
                    entry_scope = [
                        f for f in relevant
                        if f in tc_entries or f not in all_entry_fns
                    ]

                    out = Path(verdict_dir)

                    if plan_only:
                        # Skip harness generation, just regenerate plan
                        cfg_path = out / f"cfg_{harness_name}.txt"
                        if cfg_path.exists():
                            plan_result = generator.generate_validation_plan(
                                v, str(out),
                                cfg_dump=str(cfg_path),
                                entry_name=harness_name,
                                scope_functions=entry_scope,
                            )
                            if plan_result:
                                console.print(
                                    f"  [green]Validation plan: "
                                    f"{len(plan_result.get('traces', []))} "
                                    f"traces[/green]"
                                )
                            else:
                                console.print(
                                    "[yellow]  Failed to generate "
                                    "validation plan[/yellow]"
                                )
                        else:
                            console.print(
                                f"[yellow]  No CFG dump at {cfg_path} — "
                                f"run without --plan-only first[/yellow]"
                            )
                        continue

                    if seeds_only:
                        shim_path = out / f"shim_{harness_name}.c"
                        if not shim_path.exists():
                            console.print(
                                f"[red]No shim at {shim_path} — "
                                f"run without --seeds-only first[/red]"
                            )
                            continue
                        triage_ctx = {
                            "hypothesis": v.get("hypothesis", ""),
                            "reasoning": v.get("reasoning", ""),
                            "severity": v.get("issue", {}).get("severity", ""),
                            "issue_kind": v.get("issue", {}).get("issue_kind", ""),
                            "issue_description": v.get("issue", {}).get("description", ""),
                            "assumptions": v.get("assumptions", []),
                            "assertions": v.get("assertions", []),
                        }
                        seed_results = generator.generate_seeds(
                            harness_name, triage_ctx, str(out),
                        )
                        if seed_results:
                            console.print(
                                f"  [green]Generated "
                                f"{len(seed_results)} seed(s)[/green]"
                            )
                            for seed_path, script_path in seed_results:
                                seed_build = subprocess.run(
                                    ["bash", str(script_path)],
                                    capture_output=True, text=True,
                                )
                                if seed_build.returncode == 0:
                                    console.print(
                                        f"  [green]Built: "
                                        f"{seed_path.stem}.ucsan[/green]"
                                    )
                                else:
                                    console.print(
                                        f"  [yellow]Seed build failed: "
                                        f"{seed_path.name}[/yellow]"
                                    )
                                    if verbose:
                                        err = (
                                            seed_build.stderr
                                            + seed_build.stdout
                                        ).strip()
                                        console.print(
                                            f"  [dim]{err[:300]}[/dim]"
                                        )
                        else:
                            console.print("[yellow]No seeds generated[/yellow]")
                        continue

                    triage_ctx = {
                        "hypothesis": v.get("hypothesis", ""),
                        "reasoning": v.get("reasoning", ""),
                        "severity": v.get("issue", {}).get("severity", ""),
                        "issue_kind": v.get("issue", {}).get("issue_kind", ""),
                        "issue_description": v.get("issue", {}).get("description", ""),
                        "assumptions": v.get("assumptions", []),
                        "assertions": v.get("assertions", []),
                        "real_functions": entry_scope,
                        "test_sequence": tc_entries,
                    }

                    result = generator.validate_triage(
                        harness_name,
                        triage_context=triage_ctx,
                        output_dir=verdict_dir,
                        bc_file=bc_file,
                    )
                    if result:
                        console.print(
                            f"[green]Harness generated: {harness_name}[/green]"
                        )

                        # Auto-build to get CFG dump, then generate
                        # validation plan with counter-example traces
                        script = out / f"build_{harness_name}.sh"
                        if script.exists():
                            console.print(f"  Building {harness_name}...")
                            build_ok = False
                            for build_attempt in range(3):
                                build_result = subprocess.run(
                                    ["bash", str(script)],
                                    capture_output=True, text=True,
                                )
                                if build_result.returncode == 0:
                                    build_ok = True
                                    break
                                # Link/build error — feed back to fix agent
                                link_err = (
                                    build_result.stderr + build_result.stdout
                                ).strip()
                                shim_name = f"shim_{harness_name}.c"
                                shim_mentioned = shim_name in link_err
                                if (
                                    shim_mentioned
                                    and generator._can_use_fix_agent()
                                    and build_attempt < 2
                                ):
                                    console.print(
                                        f"  [yellow]Build failed "
                                        f"(attempt {build_attempt + 1}), "
                                        f"retrying with fix agent...[/yellow]"
                                    )
                                    shim_path = out / shim_name
                                    cfg_yaml = out / f"config_{harness_name}.yaml"
                                    if shim_path.exists() and cfg_yaml.exists():
                                        fixed = generator._fix_with_agent(
                                            shim_path.read_text(),
                                            cfg_yaml.read_text(),
                                            None,
                                            link_err,
                                        )
                                        if fixed:
                                            shim_path.write_text(fixed)
                                            console.print(
                                                "  Fix agent updated shim"
                                            )
                                            continue
                                console.print(
                                    f"[red]  Build failed: "
                                    f"{link_err[:200]}[/red]"
                                )
                                break

                            if build_ok:
                                console.print(
                                    f"  [green]Built: "
                                    f"{harness_name}.ucsan[/green]"
                                )
                                cfg_path = out / f"cfg_{harness_name}.txt"
                                if cfg_path.exists():
                                    plan_result = generator.generate_validation_plan(
                                        v, str(out),
                                        cfg_dump=str(cfg_path),
                                        entry_name=harness_name,
                                        scope_functions=entry_scope,
                                    )
                                    if plan_result:
                                        console.print(
                                            f"  [green]Validation plan: "
                                            f"{len(plan_result.get('traces', []))} "
                                            f"traces[/green]"
                                        )
                                    else:
                                        console.print(
                                            "[yellow]  Failed to generate "
                                            "validation plan[/yellow]"
                                        )
                                else:
                                    console.print(
                                        "[yellow]  No CFG dump — "
                                        "skipping plan[/yellow]"
                                    )

                                # Auto-generate seeds
                                seed_results = generator.generate_seeds(
                                    harness_name, triage_ctx, str(out),
                                )
                                if seed_results:
                                    console.print(
                                        f"  [green]Generated "
                                        f"{len(seed_results)} seed(s)[/green]"
                                    )
                                    for seed_path, script_path in seed_results:
                                        seed_build = subprocess.run(
                                            ["bash", str(script_path)],
                                            capture_output=True, text=True,
                                        )
                                        if seed_build.returncode == 0:
                                            console.print(
                                                f"  [green]Built: "
                                                f"{seed_path.stem}.ucsan"
                                                f"[/green]"
                                            )
                                        else:
                                            console.print(
                                                f"  [yellow]Seed build "
                                                f"failed: {seed_path.name}"
                                                f"[/yellow]"
                                            )
                    else:
                        console.print(f"[red]Failed: {harness_name}[/red]")
            return

        # Determine which functions to process
        if function_names:
            targets = list(function_names)
        else:
            rows = db.conn.execute(
                "SELECT f.name FROM functions f "
                "JOIN code_contract_summaries c ON c.function_id = f.id"
            ).fetchall()
            targets = [r[0] for r in rows]

        # Generate harnesses (unless --plan-only)
        if not plan_only:
            console.print(f"Generating harnesses for {len(targets)} function(s)")
            console.print(f"Output: {output_dir}/")

            successes = 0
            for func_name in targets:
                gen_result = generator.generate(func_name, output_dir=output_dir,
                                                bc_file=bc_file)
                if gen_result:
                    successes += 1

            stats = generator.stats
            fix_msg = f", {stats['fix_attempts']} fix attempts" if stats.get('fix_attempts') else ""
            console.print(
                f"\nDone: {successes}/{len(targets)} harnesses generated, "
                f"{stats['llm_calls']} LLM calls, {stats['errors']} errors{fix_msg}"
            )
        else:
            successes = len(targets)  # assume harnesses exist

        # Generate trace plans
        if (plan or plan_only) and successes > 0:
            console.print("\nGenerating trace plans...")
            plan_ok = 0
            for func_name in targets:
                plan_result = generator.generate_plan(
                    func_name, output_dir=output_dir, bc_file=bc_file,
                )
                if plan_result:
                    plan_ok += 1
                    plan_file = Path(output_dir) / f"plan_{func_name}.json"
                    console.print(f"  Plan: {plan_file}")
            console.print(f"Plans generated: {plan_ok}/{len(targets)}")

    finally:
        db.close()
        if tmp_cc_path:
            Path(tmp_cc_path).unlink(missing_ok=True)


@main.command("reflect")
@click.option("--verdict", "-v", required=True, help="Path to verdict JSON file")
@click.option(
    "--harness-dir", "-d", required=True,
    help="Base harness directory containing per-verdict subdirs",
)
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost")
@click.option("--llm-port", default=None, type=int)
@click.option("--disable-thinking", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--output", "-o", default=None, help="Output JSON path")
@click.option("--project-path", "-p", default=None, help="Project source path for call site search")
def reflect_cmd(
    verdict: str, harness_dir: str, db_path: str,
    backend: str, model: str | None, llm_host: str,
    llm_port: int | None, disable_thinking: bool,
    verbose: bool, output: str | None, project_path: str | None,
) -> None:
    """Reflect on validation outcomes that need investigation.

    Analyzes mismatches between triage verdicts and thoroupy validation
    evidence (e.g., different crash type than predicted, infeasible traces
    with unrelated crashes).

    Example:
        llm-summary reflect \\
            -v harnesses/png_static/verdict_png_build_gamma_table.json \\
            -d harnesses/png_static \\
            --db func-scans/libpng/png_static/functions.db \\
            --backend claude -v
    """
    from pathlib import Path

    from .reflection import reflect
    from .validation_consumer import consume_validation_dir

    verdict_path = Path(verdict)
    harness_path = Path(harness_dir)

    # Load verdicts
    with open(verdict_path) as f:
        verdicts_data = json.load(f)
    if not isinstance(verdicts_data, list):
        verdicts_data = [verdicts_data]

    # Classify outcomes
    outcomes = consume_validation_dir(verdict_path, harness_path)
    if not outcomes:
        console.print("[yellow]No validation results found[/yellow]")
        return

    # Build LLM
    kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
    llm = create_backend(backend, model=model, **kwargs)
    console.print(f"Using {backend} backend ({llm.model})")

    db = SummaryDB(db_path)
    try:
        verdict_by_idx = {
            v.get("issue_index", i): v
            for i, v in enumerate(verdicts_data)
        }

        results = []
        for oc in outcomes:
            v = verdict_by_idx.get(oc["issue_index"])
            if not v:
                continue

            # Find CFG dump and output dir for this verdict
            idx = oc["issue_index"]
            func_name = oc["function"]
            vdir = _verdict_dir(harness_path, func_name, idx, v.get("issue", {}))

            # Determine entry name from binary path
            binary = Path(oc.get("binary", ""))
            entry_name = binary.stem if binary.stem else None

            cfg_path = None
            if entry_name:
                candidate = vdir / f"cfg_{entry_name}.txt"
                if candidate.exists():
                    cfg_path = str(candidate)

            status = (
                "[green]CONFIRMED[/green]" if oc["confirmed"]
                else "[red]REJECTED[/red]"
            )
            console.print(
                f"\n  {func_name}[{idx}] {oc['hypothesis']} → {status}: "
                f"{oc['summary']}"
            )

            proj_path = Path(project_path) if project_path else None
            assessment = reflect(
                verdict=v,
                outcome=oc,
                db=db,
                llm=llm,
                cfg_dump_path=cfg_path,
                output_dir=str(vdir),
                entry_name=entry_name,
                project_path=proj_path,
                verbose=verbose,
            )
            results.append({
                "function": func_name,
                "issue_index": idx,
                "outcome": oc["outcome"],
                "reflection": assessment,
            })

            hyp = assessment.get("hypothesis", "?")
            conf = assessment.get("confidence", "?")
            action = assessment.get("action", "?")
            console.print(
                f"  → {hyp} ({conf}) action={action}: "
                f"{assessment.get('reasoning', '')[:200]}"
            )

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\nResults written to: {output}")
        else:
            console.print(json.dumps(results, indent=2))
    finally:
        db.close()


@main.command("bug-report")
@click.option("--verdict", "-v", required=True, help="Path to verdict JSON file")
@click.option(
    "--project-path", "-p", required=True,
    help="Path to the project source directory",
)
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--compile-commands", default=None,
    help="Path to compile_commands.json for include dirs",
)
@click.option("--output-dir", "-o", default=None, help="Output directory for report")
@click.option("--target-name", "-t", default=None, help="Link unit target name (e.g. png_static)")
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]),
    default="claude",
)
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost")
@click.option("--llm-port", default=None, type=int)
@click.option("--disable-thinking", is_flag=True)
@click.option("--verbose", is_flag=True)
def bug_report_cmd(
    verdict: str, project_path: str, db_path: str,
    compile_commands: str | None, output_dir: str | None,
    target_name: str | None,
    backend: str, model: str | None, llm_host: str,
    llm_port: int | None, disable_thinking: bool,
    verbose: bool,
) -> None:
    """Generate a bug report for a feasible-confirmed verdict.

    Searches the project for existing harnesses (OSS-Fuzz, tests),
    generates a standalone PoC harness, compiles with ASan, runs it,
    and writes a markdown bug report.

    Example:
        llm-summary bug-report \\
            -v harnesses/png_static/verdict_png_chunk_warning.json \\
            -p /data/csong/opensource/libpng \\
            --db func-scans/libpng/png_static/functions.db \\
            --backend llamacpp
    """
    from pathlib import Path

    from .bug_report import run_bug_report

    verdict_path = Path(verdict)
    proj_path = Path(project_path)

    with open(verdict_path) as f:
        verdicts_data = json.load(f)
    if not isinstance(verdicts_data, list):
        verdicts_data = [verdicts_data]

    kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
    llm = create_backend(backend, model=model, **kwargs)
    console.print(f"Using {backend} backend ({llm.model})")

    db = SummaryDB(db_path)
    cc_path = Path(compile_commands) if compile_commands else None

    try:
        for v in verdicts_data:
            func_name = v["function_name"]
            hypothesis = v.get("hypothesis", "unknown")

            if hypothesis != "feasible":
                if verbose:
                    console.print(
                        f"[dim]Skipping {func_name}: "
                        f"hypothesis={hypothesis} (not feasible)[/dim]",
                    )
                continue

            idx = v.get("issue_index", 0)
            out = Path(output_dir) if output_dir else _verdict_dir(
                Path("harnesses") / proj_path.name, func_name, idx,
                v.get("issue", {}),
            )

            console.print(f"\n[bold]{func_name}[/bold] [{idx}]")

            result = run_bug_report(
                verdict=v,
                db=db,
                llm=llm,
                project_path=proj_path,
                output_dir=out,
                compile_commands_path=cc_path,
                target_name=target_name,
                verbose=verbose,
            )

            if result["success"]:
                console.print(
                    f"  [green]PoC confirmed crash![/green] "
                    f"Report: {result['report_path']}",
                )
            elif result.get("compile_error"):
                console.print(
                    f"  [red]Compile failed[/red]: "
                    f"{result['compile_error'][:200]}",
                )
            else:
                console.print(
                    f"  [yellow]No crash from PoC[/yellow] "
                    f"Report: {result['report_path']}",
                )
    finally:
        db.close()


@main.command("consume-validation")
@click.option("--verdict", "-v", required=True, help="Path to verdict JSON file")
@click.option(
    "--harness-dir", "-d", required=True,
    help="Base harness directory containing per-verdict subdirs",
)
@click.option("--db", "db_path", default=None, help="DB path to auto-review validated outcomes")
@click.option("--output", "-o", default=None, help="Output JSON path (default: stdout)")
def consume_validation(
    verdict: str, harness_dir: str, db_path: str | None, output: str | None,
) -> None:
    """Classify validation outcomes for triage verdicts.

    Reads validation_result.json files produced by thoroupy and classifies
    each verdict as confirmed or rejected.  When --db is provided,
    safe_confirmed outcomes are auto-reviewed as false_positive;
    feasible_confirmed outcomes are auto-reviewed as confirmed.

    Example:
        llm-summary consume-validation \\
            -v harnesses/png_static/verdict_png_build_16to8_table.json \\
            -d harnesses/png_static \\
            --db func-scans/png_static/functions.db
    """
    from pathlib import Path

    from .models import SafetyIssue
    from .validation_consumer import consume_validation_dir

    verdict_path = Path(verdict)
    results = consume_validation_dir(verdict_path, Path(harness_dir))

    if not results:
        console.print("[yellow]No validation results found[/yellow]")
        return

    # Load verdicts for issue fingerprint computation
    with open(verdict_path) as f:
        verdicts_data = json.load(f)
    if not isinstance(verdicts_data, list):
        verdicts_data = [verdicts_data]
    verdict_by_idx = {v.get("issue_index", i): v for i, v in enumerate(verdicts_data)}

    for r in results:
        status = "[green]CONFIRMED[/green]" if r["confirmed"] else "[red]REJECTED[/red]"
        console.print(
            f"  {r['function']}[{r['issue_index']}] "
            f"{r['hypothesis']} → {status}: {r['summary']}"
        )

    # Auto-review safe_confirmed / feasible_confirmed
    outcome_to_status = {
        "safe_confirmed": "false_positive",
        "feasible_confirmed": "confirmed",
    }
    if db_path:
        db = SummaryDB(db_path)
        try:
            reviewed = 0
            for r in results:
                review_status = outcome_to_status.get(r["outcome"])
                if not review_status:
                    continue
                v = verdict_by_idx.get(r["issue_index"])
                if not v:
                    continue
                issue_d = v.get("issue", {})
                vi_obj = SafetyIssue(
                    location=issue_d.get("location", ""),
                    issue_kind=issue_d.get("issue_kind", ""),
                    description=issue_d.get("description", ""),
                    severity=issue_d.get("severity", "medium"),
                    callee=issue_d.get("callee"),
                    contract_kind=issue_d.get("contract_kind"),
                )
                funcs = db.get_function_by_name(r["function"])
                if funcs:
                    assert funcs[0].id is not None
                    db.upsert_issue_review(
                        function_id=funcs[0].id,
                        issue_index=r["issue_index"],
                        fingerprint=vi_obj.fingerprint(),
                        status=review_status,
                        reason=r.get("summary", ""),
                    )
                    reviewed += 1
            if reviewed:
                console.print(
                    f"\n[green]Auto-reviewed {reviewed} issue(s)[/green]"
                )
        finally:
            db.close()

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults written to: {output}")
    else:
        console.print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
