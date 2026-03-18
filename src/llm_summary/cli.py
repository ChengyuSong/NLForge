"""Command-line interface for LLM-based allocation summary analysis."""

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .callgraph import CallGraphBuilder
from .compile_commands import CompileCommandsDB
from .db import SummaryDB
from .driver import (
    AllocationPass,
    BottomUpDriver,
    FreePass,
    InitPass,
    MemsafePass,
    SummaryPass,
    VerificationPass,
)
from .extractor import FunctionExtractor
from .indirect import (
    AddressTakenScanner,
    FlowSummarizer,
    IndirectCallResolver,
    IndirectCallsiteFinder,
)
from .llm import create_backend
from .ordering import ProcessingOrderer
from .stdlib import (
    STDLIB_ATTRIBUTES,
    get_all_stdlib_free_summaries,
    get_all_stdlib_init_summaries,
    get_all_stdlib_memsafe_summaries,
    get_all_stdlib_summaries,
)
from .summarizer import AllocationSummarizer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .stdlib_cache import StdlibCache

console = Console()


def _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking):
    """Build kwargs dict for create_backend() from CLI options."""
    kwargs = {}
    if backend == "llamacpp":
        kwargs["host"] = llm_host
        kwargs["port"] = llm_port if llm_port is not None else 8080
    elif backend == "ollama":
        if llm_port is None:
            llm_port = 11434
        kwargs["base_url"] = f"http://{llm_host}:{llm_port}"
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return kwargs


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
    "--init-stdlib", is_flag=True,
    help="Auto-populate stdlib summaries before starting",
)
@click.option(
    "--allocator-file", type=click.Path(exists=True), default=None,
    help="JSON file with custom allocator names "
         "(e.g. from find-allocator-candidates)",
)
@click.option(
    "--type", "summary_types", multiple=True,
    type=click.Choice(
        ["allocation", "free", "init", "memsafe", "verify"]
    ),
    help="Summary pass(es) to run (default: allocation). "
         "Can be specified multiple times.",
)
@click.option("--deallocator-file", type=click.Path(exists=True), default=None,
              help="JSON file with custom deallocator names (for free pass)")
@click.option("--vsnap", type=click.Path(exists=True), default=None,
              help="V-snapshot (.vsnap) file for alias context in memsafe/verify passes")
@click.option("-j", "jobs", default=1, type=int, help="Parallel LLM queries (default: 1)")
@click.option(
    "--cache-mode",
    type=click.Choice(["none", "instructions", "source"]),
    default="none",
    help="Prompt caching mode: none (default), "
         "instructions (cache task instructions), "
         "source (cache function source)",
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
def summarize(
    db_path, backend, model, llm_host, llm_port,
    disable_thinking, verbose, force, log_llm, init_stdlib,
    allocator_file, summary_types, deallocator_file, vsnap,
    jobs, cache_mode, function_names, incremental,
):
    """Generate allocation, free, init, memsafe, and/or verify
    summaries on a pre-populated database.

    Requires a database that already has functions and call_edges
    (populated via 'extract', 'scan', and/or 'import-callgraph').

    Use --type to select which summary passes to run:
        --type allocation  (default)
        --type free
        --type init
        --type memsafe
        --type verify      (requires all four prior passes)
        --type allocation --type free --type init --type memsafe  (run all)

    Example:
        llm-summary summarize --db func-scans/libpng/functions.db --backend ollama --model qwen3 -v
        llm-summary summarize --db func-scans/libpng/functions.db --type free --backend llamacpp -v
        llm-summary summarize --db func-scans/libpng/functions.db --type verify --backend ollama -v
    """
    # Default to allocation if no --type given
    if not summary_types:
        summary_types = ("allocation",)
    db = SummaryDB(db_path)

    try:
        # Validate DB has required data
        stats = db.get_stats()
        func_count = stats["functions"]
        edge_count = stats["call_edges"]

        if func_count == 0:
            console.print("[red]No functions in database. Run 'extract' or 'scan' first.[/red]")
            return

        if edge_count == 0:
            console.print(
                "[red]Error: No call edges in database."
                " Run call graph import first.[/red]"
            )
            sys.exit(1)

        # Prerequisite check for verify pass
        if "verify" in summary_types:
            missing = []
            req_tables = [
                "allocation_summaries", "free_summaries",
                "init_summaries", "memsafe_summaries",
            ]
            for req_table in req_tables:
                if stats.get(req_table, 0) == 0:
                    missing.append(req_table.replace("_summaries", "").replace("_", " "))
            if missing:
                console.print(
                    f"[red]Error: --type verify requires "
                    f"all four prior passes. "
                    f"Missing: {', '.join(missing)}. "
                    f"Run --type allocation --type free "
                    f"--type init --type memsafe "
                    f"first.[/red]"
                )
                return

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

        # Init stdlib if requested
        if init_stdlib:
            from .models import Function

            def _find_stdlib_func(name: str) -> Function | None:
                """Find an existing stdlib stub in the DB (created by callgraph import).

                Never creates new stubs — only functions already referenced in the
                call graph (and thus already present as stubs) should receive summaries.
                """
                existing = db.get_function_by_name(name)
                if not existing:
                    return None
                func: Function = existing[0]
                # Update attributes if not already set (callgraph import may have missed them)
                attrs = STDLIB_ATTRIBUTES.get(name, "")
                if attrs and not func.attributes:
                    func.attributes = attrs
                    db.conn.execute(
                        "UPDATE functions SET attributes = ? WHERE id = ?",
                        (attrs, func.id),
                    )
                    db.conn.commit()
                return func

            # Allocation summaries
            alloc_summaries = get_all_stdlib_summaries()
            for name, summary in alloc_summaries.items():
                func = _find_stdlib_func(name)
                if func is not None:
                    db.upsert_summary(func, summary, model_used="builtin")

            # Free summaries
            free_summaries = get_all_stdlib_free_summaries()
            for name, fsummary in free_summaries.items():
                func = _find_stdlib_func(name)
                if func is not None:
                    db.upsert_free_summary(func, fsummary, model_used="builtin")

            # Init summaries
            init_summaries = get_all_stdlib_init_summaries()
            for name, isummary in init_summaries.items():
                func = _find_stdlib_func(name)
                if func is not None:
                    db.upsert_init_summary(func, isummary, model_used="builtin")

            # Memsafe summaries
            memsafe_summaries = get_all_stdlib_memsafe_summaries()
            for name, msummary in memsafe_summaries.items():
                func = _find_stdlib_func(name)
                if func is not None:
                    db.upsert_memsafe_summary(func, msummary, model_used="builtin")

            # Update attributes for attribute-only functions (e.g., exit, abort) if they exist
            for name in STDLIB_ATTRIBUTES:
                _find_stdlib_func(name)

            total = len(set(
                list(alloc_summaries) + list(free_summaries) +
                list(init_summaries) + list(memsafe_summaries) +
                list(STDLIB_ATTRIBUTES)
            ))
            console.print(f"  Stdlib functions initialized: {total}")

        # Create LLM backend
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")
        if cache_mode != "none":
            console.print(f"  Prompt cache mode: {cache_mode}")

        # Load alias context builder early (needed for candidate confirmation + memsafe/verify)
        alias_builder = None
        if vsnap:
            from .alias_context import AliasContextBuilder
            alias_builder = AliasContextBuilder(vsnap, db)
            if verbose:
                console.print(f"  V-snapshot loaded: {vsnap} "
                              f"(nodes={alias_builder.snap.node_count}, "
                              f"reps={alias_builder.snap.rep_count}, "
                              f"named={len(alias_builder.snap.named_entries)})")

        # Build passes list
        passes: list[SummaryPass] = []
        alloc_summarizer = None
        free_summarizer = None

        if "allocation" in summary_types:
            # Load custom allocators if provided
            allocators = []
            if allocator_file:
                with open(allocator_file) as f:
                    alloc_data = json.load(f)
                allocators = alloc_data.get("confirmed", [])
                if not allocators:
                    allocators = alloc_data.get("candidates", [])
                if allocators:
                    console.print(
                        f"Custom allocators: "
                        f"{len(allocators)} loaded "
                        f"from {allocator_file}"
                    )

                # Confirm candidates via vsnapshot alias analysis
                if allocators and alias_builder:
                    from .allocator import vsnapshot_confirm_allocators
                    confirmed, remaining = (
                        vsnapshot_confirm_allocators(
                            alias_builder.snap, allocators
                        )
                    )
                    console.print(f"  Vsnapshot alloc confirmation: {len(confirmed)} confirmed, "
                                  f"{len(remaining)} unconfirmed (dropped)")
                    allocators = confirmed

            alloc_summarizer = AllocationSummarizer(
                db, llm, verbose=verbose, log_file=log_llm,
                allocators=allocators, cache_mode=cache_mode,
            )
            passes.append(AllocationPass(alloc_summarizer, db, llm.model))

        if "free" in summary_types:
            from .free_summarizer import FreeSummarizer

            # Load custom deallocators if provided
            deallocators = []
            if deallocator_file:
                with open(deallocator_file) as f:
                    dealloc_data = json.load(f)
                deallocators = dealloc_data.get("confirmed", [])
                if not deallocators:
                    deallocators = dealloc_data.get("candidates", [])
                if deallocators:
                    console.print(
                        f"Custom deallocators: "
                        f"{len(deallocators)} loaded "
                        f"from {deallocator_file}"
                    )

                # Confirm candidates via vsnapshot alias analysis
                if deallocators and alias_builder:
                    from .allocator import vsnapshot_confirm_deallocators
                    dconfirmed, dremaining = (
                        vsnapshot_confirm_deallocators(
                            alias_builder.snap, deallocators
                        )
                    )
                    console.print(f"  Vsnapshot dealloc confirmation: {len(dconfirmed)} confirmed, "
                                  f"{len(dremaining)} unconfirmed (dropped)")
                    deallocators = dconfirmed

            free_summarizer = FreeSummarizer(
                db, llm, verbose=verbose, log_file=log_llm,
                deallocators=deallocators,
                cache_mode=cache_mode,
            )
            passes.append(FreePass(free_summarizer, db, llm.model))

        init_summarizer = None
        if "init" in summary_types:
            from .init_summarizer import InitSummarizer

            init_summarizer = InitSummarizer(
                db, llm, verbose=verbose, log_file=log_llm,
                cache_mode=cache_mode,
            )
            passes.append(InitPass(init_summarizer, db, llm.model))

        memsafe_summarizer = None
        if "memsafe" in summary_types:
            from .memsafe_summarizer import MemsafeSummarizer

            memsafe_summarizer = MemsafeSummarizer(
                db, llm, verbose=verbose, log_file=log_llm,
                cache_mode=cache_mode,
            )
            passes.append(MemsafePass(
                memsafe_summarizer, db, llm.model,
                alias_builder=alias_builder,
            ))

        verification_summarizer = None
        if "verify" in summary_types:
            from .verification_summarizer import VerificationSummarizer

            verification_summarizer = VerificationSummarizer(
                db, llm, verbose=verbose, log_file=log_llm,
                cache_mode=cache_mode,
            )
            passes.append(VerificationPass(
                verification_summarizer, db, llm.model,
                alias_builder=alias_builder,
            ))

        pass_names = " + ".join(p.name for p in passes)
        console.print(f"\n[bold]Running passes: {pass_names}[/bold]")

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

        # Compute dirty_ids for incremental mode
        dirty_ids = None
        if incremental and not function_names:
            from .driver import PASS_TABLE_MAP
            dirty_ids = set()
            for p in passes:
                table = PASS_TABLE_MAP.get(p.name)
                if table:
                    dirty_ids |= db.find_dirty_function_ids(table)
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
            results = driver.run(passes, force=force, dirty_ids=dirty_ids, target_ids=target_ids)
        finally:
            if pool is not None:
                pool.shutdown()

        # Print stats per pass
        if alloc_summarizer is not None:
            summaries = results["allocation"]
            s = alloc_summarizer.stats
            console.print("\nAllocation summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            has_cache_tok = (
                s.get("cache_read_tokens")
                or s.get("cache_creation_tokens")
            )
            if cache_mode != "none" and has_cache_tok:
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            allocating = sum(1 for sm in summaries.values() if sm.allocations)
            console.print(f"\nFunctions with allocations: {allocating}")

        if free_summarizer is not None:
            free_results = results["free"]
            s = free_summarizer.stats
            console.print("\nFree summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            has_cache_tok = (
                s.get("cache_read_tokens")
                or s.get("cache_creation_tokens")
            )
            if cache_mode != "none" and has_cache_tok:
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            freeing = sum(1 for sm in free_results.values() if sm.frees)
            releasing = sum(1 for sm in free_results.values() if sm.resource_releases)
            console.print(f"\nFunctions with frees: {freeing}")
            if releasing:
                console.print(f"Functions with resource releases: {releasing}")

        if init_summarizer is not None:
            init_results = results["init"]
            s = init_summarizer.stats
            console.print("\nInit summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            has_cache_tok = (
                s.get("cache_read_tokens")
                or s.get("cache_creation_tokens")
            )
            if cache_mode != "none" and has_cache_tok:
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            initializing = sum(1 for sm in init_results.values() if sm.inits)
            console.print(f"\nFunctions with inits: {initializing}")

        if memsafe_summarizer is not None:
            memsafe_results = results["memsafe"]
            s = memsafe_summarizer.stats
            console.print("\nMemsafe summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            has_cache_tok = (
                s.get("cache_read_tokens")
                or s.get("cache_creation_tokens")
            )
            if cache_mode != "none" and has_cache_tok:
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            with_contracts = sum(1 for sm in memsafe_results.values() if sm.contracts)
            console.print(f"\nFunctions with safety contracts: {with_contracts}")

        if verification_summarizer is not None:
            verify_summaries = results["verify"]
            s = verification_summarizer.stats
            console.print("\nVerification complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            has_cache_tok = (
                s.get("cache_read_tokens")
                or s.get("cache_creation_tokens")
            )
            if cache_mode != "none" and has_cache_tok:
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            console.print(f"  Contracts simplified: {s['contracts_simplified']}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            with_issues = sum(1 for sm in verify_summaries.values() if sm.issues)
            total_issues = sum(len(sm.issues) for sm in verify_summaries.values())
            high_issues = sum(
                1 for sm in verify_summaries.values()
                for issue in sm.issues if issue.severity == "high"
            )
            console.print(f"\nFunctions with issues: {with_issues}")
            issue_msg = f"Total issues: {total_issues}"
            if high_issues > 0:
                issue_msg += f" ({high_issues} high severity)"
            console.print(issue_msg)

    finally:
        db.close()


@main.command()
@click.argument("path_arg", type=click.Path(exists=True), required=False, default=None)
@click.option(
    "--path", "path_opt", type=click.Path(exists=True),
    default=None, help="Path to extract from",
)
@click.option(
    "--db", "db_path", default="summaries.db",
    help="Database file path",
)
@click.option(
    "--compile-commands", "compile_commands_path",
    type=click.Path(exists=True), default=None,
    help="Path to compile_commands.json for proper "
         "macro/include handling",
)
@click.option(
    "--project-path", "project_path",
    type=click.Path(), default=None,
    help="Host path to project source root. Required when "
         "compile_commands.json uses Docker "
         "container paths (/workspace/src/...).",
)
@click.option("--recursive/--no-recursive", default=True)
@click.option(
    "--preprocess", is_flag=True,
    help="Run clang -E to expand macros and store "
         "preprocessed source",
)
def extract(
    path_arg, path_opt, db_path,
    compile_commands_path, project_path,
    recursive, preprocess,
):
    """Extract functions and build call graph (no LLM)."""
    # Accept path as either positional argument or --path option
    path = path_opt or path_arg
    if not path:
        raise click.UsageError("PATH is required (provide as argument or use --path)")

    path = Path(path).resolve()

    # Load compile_commands.json if provided
    compile_commands = None
    _tmp_cc = None
    if compile_commands_path:
        try:
            compile_commands, _tmp_cc = _load_compile_commands(compile_commands_path, project_path)
            console.print(f"Loaded compile_commands.json ({len(compile_commands)} entries)")
        except click.UsageError:
            raise
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load compile_commands.json: {e}[/yellow]")

    # Determine file extensions based on whether compile_commands is available
    # Without compile_commands, include headers to get more complete coverage
    if compile_commands:
        extensions = [".c", ".cpp", ".cc", ".cxx"]
    else:
        extensions = [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"]
        console.print("[dim]No compile_commands.json - including header files[/dim]")

    if path.is_file():
        files = [path]
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in extensions]

    if not files:
        console.print("[red]No source files found[/red]")
        return

    console.print(f"Found {len(files)} source file(s)")

    db = SummaryDB(db_path)

    try:
        extractor = FunctionExtractor(
            compile_commands=compile_commands,
            enable_preprocessing=preprocess,
        )
        all_functions = []

        for f in files:
            try:
                functions = extractor.extract_from_file(f)
                all_functions.extend(functions)
                console.print(f"  {f.name}: {len(functions)} functions")
            except Exception as e:
                console.print(f"  [yellow]Warning: {f.name}: {e}[/yellow]")

        db.insert_functions_batch(all_functions)
        console.print(f"\nExtracted {len(all_functions)} functions")

        # Build call graph
        cg_builder = CallGraphBuilder(db, compile_commands=compile_commands)
        edges = cg_builder.build_from_files(files)
        console.print(f"Found {len(edges)} call edges")

        # Show stats
        stats = db.get_stats()
        console.print("\nDatabase statistics:")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")

    finally:
        db.close()
        if _tmp_cc:
            Path(_tmp_cc).unlink(missing_ok=True)


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
                    output.append({
                        "function": func.name,
                        "file": func.file_path,
                        "line": func.line_start,
                        "summary": summary.to_dict(),
                    })
            console.print(json.dumps(output, indent=2))

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

                    alloc_str = ", ".join(a.source for a in summary.allocations) or "-"
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
@click.argument("name")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--signature", help="Function signature for disambiguation")
def lookup(name, db_path, signature):
    """Look up summary for a specific function."""
    db = SummaryDB(db_path)

    try:
        summary = db.get_summary(name, signature)

        if summary:
            console.print(json.dumps(summary.to_dict(), indent=2))
        else:
            console.print(f"[yellow]No summary found for {name}[/yellow]")

            # Check if function exists
            functions = db.get_function_by_name(name)
            if functions:
                console.print("Function exists but has no summary. Run 'analyze' to generate.")
            else:
                console.print("Function not found in database.")

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
def clear(db_path):
    """Clear all data from the database."""
    if not click.confirm("This will delete all data. Continue?"):
        return

    db = SummaryDB(db_path)
    try:
        db.clear_all()
        console.print("Database cleared.")
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
    db_path, extra_abilists, stdlib_db, seed_from_paths, force, backend, model,
    llm_host, llm_port, log_llm, verbose,
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
    from .stdlib_cache import StdlibCache, load_known_externals

    if not db_path and not seed_from_paths:
        console.print("[red]Error: provide --db and/or --seed-from.[/red]")
        sys.exit(1)

    # 1. Load known-externals — needed both for filtering seed-from and for apply
    known_externals = load_known_externals(list(extra_abilists) if extra_abilists else None)
    console.print(f"Known-externals registry: {len(known_externals):,} function names")

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
            return blobs

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
                )
                added += 1
                if verbose:
                    console.print(f"  [seed] {src_func.name} ({tag})")
        finally:
            src_db.close()
        console.print(
            f"Seeded from {src_path}: {added} exported symbols added"
            + (f" ({alias_fallbacks} via weak_alias)" if alias_fallbacks else "")
            + (f", {skipped_cached} already cached" if skipped_cached else "")
            + (f", {skipped_internal} internal symbols skipped" if skipped_internal else "")
        )

    # 4. Seed hand-crafted builtins — always wins over DB-seeded entries so
    #    carefully reviewed contracts take priority over LLM-generated ones.
    seeded = cache.seed_builtins(force=True)
    if seeded:
        console.print(f"  Seeded {seeded} hand-crafted builtin entries into cache (priority)")

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
            return (
                db.get_summary_by_function_id(f.id) is None
                or db.get_free_summary_by_function_id(f.id) is None
                or db.get_init_summary_by_function_id(f.id) is None
            )

        needs = [f for f in sourceless if _needs_summary(f)]

        if not needs:
            console.print("All external functions already have summaries.")
            return

        in_known = [f for f in needs if f.name in known_externals]
        not_known = [f for f in needs if f.name not in known_externals]

        cache_hits = [f for f in in_known if cache.has(f.name)]
        cache_misses = [f for f in in_known if not cache.has(f.name)]

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
                f"summary and no --backend was given.[/red]"
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
    """Export all summaries to JSON."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()
        output_data = []

        for func in functions:
            assert func.id is not None
            summary = db.get_summary_by_function_id(func.id)
            if summary:
                output_data.append({
                    "name": func.name,
                    "signature": func.signature,
                    "file": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "summary": summary.to_dict(),
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


@main.command("indirect-analyze")
@click.argument("path_arg", type=click.Path(exists=True), required=False, default=None)
@click.option(
    "--path", "path_opt", type=click.Path(exists=True),
    default=None, help="Path to analyze",
)
@click.option(
    "--db", "db_path", default="summaries.db",
    help="Database file path",
)
@click.option(
    "--compile-commands", "compile_commands_path",
    type=click.Path(exists=True), default=None,
    help="Path to compile_commands.json for proper "
         "macro/include handling",
)
@click.option(
    "--project-path", "project_path",
    type=click.Path(), default=None,
    help="Host path to project source root. Required "
         "when compile_commands.json uses Docker "
         "container paths (/workspace/src/...).",
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
@click.option(
    "--recursive/--no-recursive", default=True,
    help="Scan directories recursively",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--force", "-f", is_flag=True,
    help="Force re-analysis (ignore cache)",
)
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log all LLM prompts and responses to file",
)
@click.option(
    "--pass1-only", is_flag=True,
    help="Only run Pass 1 (flow summarization)",
)
@click.option(
    "--pass2-only", is_flag=True,
    help="Only run Pass 2 (resolution), "
         "requires Pass 1 already done",
)
def indirect_analyze(
    path_arg, path_opt, db_path,
    compile_commands_path, project_path, backend,
    model, llm_host, llm_port, disable_thinking,
    recursive, verbose, force, log_llm,
    pass1_only, pass2_only,
):
    """
    Analyze indirect calls using LLM-based two-pass approach.

    Pass 1: Summarize where address-taken function pointers flow.
    Pass 2: Resolve indirect callsites using flow summaries.

    Example:
        llm-summary indirect-analyze --path src/ --db out.db \\
            --compile-commands compile_commands.json
    """
    # Accept path as either positional argument or --path option
    path = path_opt or path_arg
    if not path:
        raise click.UsageError("PATH is required (provide as argument or use --path)")

    path = Path(path).resolve()

    # Load compile_commands.json if provided
    compile_commands = None
    _tmp_cc = None
    if compile_commands_path:
        try:
            compile_commands, _tmp_cc = _load_compile_commands(compile_commands_path, project_path)
            console.print(f"Loaded compile_commands.json ({len(compile_commands)} entries)")
        except click.UsageError:
            raise
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load compile_commands.json: {e}[/yellow]")

    # Determine file extensions based on whether compile_commands is available
    # Without compile_commands, include headers to get more complete coverage
    if compile_commands:
        extensions = [".c", ".cpp", ".cc", ".cxx"]
    else:
        extensions = [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"]
        console.print("[dim]No compile_commands.json - including header files[/dim]")

    # Collect files
    if path.is_file():
        files = [path]
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in extensions]

    if not files:
        console.print("[red]No source files found[/red]")
        return

    console.print(f"Found {len(files)} source file(s)")

    # Initialize database
    db = SummaryDB(db_path)

    try:
        # Check if we need to extract functions first
        existing_funcs = db.get_all_functions()
        if not existing_funcs:
            console.print("No functions in database. Running extraction first...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Extracting functions...", total=None)

                extractor = FunctionExtractor(compile_commands=compile_commands)
                all_functions = []

                for f in files:
                    try:
                        functions = extractor.extract_from_file(f)
                        all_functions.extend(functions)
                        if verbose:
                            progress.console.print(f"  {f.name}: {len(functions)} functions")
                    except Exception as e:
                        progress.console.print(f"  [yellow]Warning: {f.name}: {e}[/yellow]")

                progress.update(task, completed=True)

            db.insert_functions_batch(all_functions)
            console.print(f"Extracted {len(all_functions)} functions")

        # Scan for address-taken functions and indirect callsites
        # Skip scanning if --pass2-only (use existing data from DB)
        if pass2_only:
            atf_count = len(db.get_address_taken_functions())
            callsite_count = len(db.get_indirect_callsites())
            console.print(
                f"Using existing data: "
                f"{atf_count} address-taken functions, "
                f"{callsite_count} indirect callsites"
            )

            if atf_count == 0:
                console.print(
                    "[red]No address-taken functions in "
                    "database. Run without --pass2-only "
                    "first.[/red]"
                )
                return

            if callsite_count == 0:
                console.print(
                    "[red]No indirect callsites in database."
                    " Run without --pass2-only first.[/red]"
                )
                return
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Scanning for address-taken functions...", total=None)

                scanner = AddressTakenScanner(db, compile_commands=compile_commands)
                scanner.scan_files(files)

                progress.update(task, completed=True)

            atf_count = len(db.get_address_taken_functions())
            console.print(f"Found {atf_count} address-taken functions")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Finding indirect call sites...", total=None)

                finder = IndirectCallsiteFinder(db, compile_commands=compile_commands)
                callsites = finder.find_in_files(files)

                progress.update(task, completed=True)

            console.print(f"Found {len(callsites)} indirect call sites")

            if atf_count == 0:
                console.print(
                    "[yellow]No address-taken functions "
                    "found. Nothing to analyze.[/yellow]"
                )
                return

            if len(callsites) == 0:
                console.print("[yellow]No indirect call sites found. Nothing to resolve.[/yellow]")
                return

        # Initialize LLM backend
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        # Pass 1: Flow summarization
        if not pass2_only:
            console.print("\n[bold]Pass 1: Flow Summarization[/bold]")

            flow_summarizer = FlowSummarizer(db, llm, verbose=verbose, log_file=log_llm)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Summarizing function pointer flows...", total=None)

                flow_summarizer.summarize_all(force=force)

                progress.update(task, completed=True)

            stats1 = flow_summarizer.stats
            console.print("Pass 1 complete:")
            console.print(f"  Functions processed: {stats1['functions_processed']}")
            console.print(f"  LLM calls: {stats1['llm_calls']}")
            console.print(f"  Cache hits: {stats1['cache_hits']}")
            if stats1["errors"] > 0:
                console.print(f"  [yellow]Errors: {stats1['errors']}[/yellow]")

        if pass1_only:
            console.print("\n[green]Pass 1 complete. Use --pass2-only to run resolution.[/green]")
            return

        # Pass 2: Indirect call resolution
        console.print("\n[bold]Pass 2: Indirect Call Resolution[/bold]")

        resolver = IndirectCallResolver(db, llm, verbose=verbose, log_file=log_llm)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Resolving indirect calls...", total=None)

            resolutions = resolver.resolve_all_callsites(force=force)

            progress.update(task, completed=True)

        stats2 = resolver.stats
        console.print("Pass 2 complete:")
        console.print(f"  Callsites processed: {stats2['callsites_processed']}")
        console.print(f"  LLM calls: {stats2['llm_calls']}")
        console.print(f"  Cache hits: {stats2['cache_hits']}")
        if stats2["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats2['errors']}[/yellow]")

        # Summary
        total_targets = sum(len(targets) for targets in resolutions.values())
        high_conf = sum(
            1 for targets in resolutions.values()
            for t in targets if t.confidence == "high"
        )
        console.print("\n[green]Analysis complete![/green]")
        console.print(f"  Total resolved targets: {total_targets}")
        console.print(f"  High confidence matches: {high_conf}")

        # Show database stats
        stats = db.get_stats()
        console.print("\nDatabase statistics:")
        console.print(f"  Address-taken functions: {stats['address_taken_functions']}")
        console.print(f"  Address flow summaries: {stats['address_flow_summaries']}")
        console.print(f"  Indirect callsites: {stats['indirect_callsites']}")
        console.print(f"  Resolved targets: {stats['indirect_call_targets']}")

    finally:
        db.close()
        if _tmp_cc:
            Path(_tmp_cc).unlink(missing_ok=True)


@main.command("show-indirect")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_indirect(db_path, fmt):
    """Show indirect call resolution results."""
    db = SummaryDB(db_path)

    try:
        callsites = db.get_indirect_callsites()
        functions = {f.id: f for f in db.get_all_functions()}

        if fmt == "json":
            output = []
            for cs in callsites:
                assert cs.id is not None
                targets = db.get_indirect_call_targets(cs.id)
                caller = functions.get(cs.caller_function_id)

                target_list = []
                for t in targets:
                    target_func = functions.get(t.target_function_id)
                    target_list.append({
                        "function": (
                            target_func.name
                            if target_func
                            else f"ID:{t.target_function_id}"
                        ),
                        "confidence": t.confidence,
                        "reasoning": t.llm_reasoning,
                    })

                output.append({
                    "callsite": {
                        "caller": caller.name if caller else "unknown",
                        "expression": cs.callee_expr,
                        "file": cs.file_path,
                        "line": cs.line_number,
                        "signature": cs.signature,
                    },
                    "targets": target_list,
                })

            console.print(json.dumps(output, indent=2))

        else:
            table = Table(title="Indirect Call Resolutions")
            table.add_column("Caller", style="cyan")
            table.add_column("Expression", style="yellow")
            table.add_column("Location", style="dim")
            table.add_column("Targets", style="green")
            table.add_column("Confidence", style="magenta")

            for cs in callsites:
                assert cs.id is not None
                targets = db.get_indirect_call_targets(cs.id)
                caller = functions.get(cs.caller_function_id)

                if not targets:
                    table.add_row(
                        caller.name if caller else "?",
                        cs.callee_expr,
                        f"{Path(cs.file_path).name}:{cs.line_number}",
                        "[dim]unresolved[/dim]",
                        "-",
                    )
                else:
                    for i, t in enumerate(targets):
                        target_func = functions.get(t.target_function_id)
                        table.add_row(
                            caller.name if caller and i == 0 else "",
                            cs.callee_expr if i == 0 else "",
                            f"{Path(cs.file_path).name}:{cs.line_number}" if i == 0 else "",
                            target_func.name if target_func else f"ID:{t.target_function_id}",
                            t.confidence,
                        )

            console.print(table)

    finally:
        db.close()


@main.command("container-analyze")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option(
    "--backend",
    type=click.Choice(
        ["claude", "openai", "ollama", "llamacpp", "gemini"]
    ),
    default="ollama",
)
@click.option("--model", default=None, help="Model name to use")
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
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis (ignore cache)")
@click.option(
    "--min-score", default=5, type=int,
    help="Minimum heuristic score for LLM confirmation "
         "(default: 5)",
)
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log all LLM prompts and responses to file",
)
@click.option("--heuristic-only", is_flag=True, help="Only run heuristic scoring, skip LLM")
@click.option("--project-name", default=None, help="Project name (default: inferred from DB path)")
def container_analyze(
    db_path, backend, model, llm_host, llm_port, disable_thinking,
    verbose, force, min_score, log_llm, heuristic_only, project_name
):
    """Detect container/collection functions (hash tables, lists, trees, etc.).

    Uses a heuristic pre-filter to find candidates, then confirms with LLM.
    Results are stored in the database for downstream analysis.

    Example:
        llm-summary container-analyze --db summaries.db --backend ollama --model qwen3 -v
        llm-summary container-analyze --db summaries.db --heuristic-only -v
    """
    from .container import ContainerDetector

    # Infer project name from DB path: func-scans/<project>/functions.db -> <project>
    if not project_name:
        db_dir = Path(db_path).resolve().parent
        project_name = db_dir.name if db_dir.name != "." else Path(db_path).stem

    db = SummaryDB(db_path)

    try:
        # Check for functions in DB
        functions = db.get_all_functions()
        if not functions:
            console.print("[red]No functions in database. Run 'extract' or 'scan' first.[/red]")
            return

        console.print(f"Database: {db_path} ({len(functions)} functions, project: {project_name})")

        if heuristic_only:
            # Heuristic-only mode: no LLM
            detector = ContainerDetector(
                db, llm=None, verbose=verbose, min_score=min_score
            )
            candidates = detector.heuristic_only()

            if not candidates:
                console.print("[yellow]No candidates found above threshold.[/yellow]")
                return

            # Display results as a table
            table = Table(title=f"Container Candidates (score >= {min_score})")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Signals")

            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)

            for func, score, signals in candidates:
                signal_strs = []
                for s in signals:
                    # Shorten signal display
                    tag = s.split(": ", 1)[0] if ": " in s else s
                    signal_strs.append(tag)
                table.add_row(
                    str(score),
                    func.name,
                    Path(func.file_path).name,
                    ", ".join(signal_strs),
                )

            console.print(table)
            console.print(f"\nTotal candidates: {len(candidates)}")

            stats = detector.stats
            console.print(f"Functions scanned: {stats['functions_scanned']}")
            return

        # Full mode: heuristic + LLM
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        detector = ContainerDetector(
            db, llm=llm, verbose=verbose, log_file=log_llm,
            min_score=min_score, project_name=project_name,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting container functions...", total=None)
            detector.detect_all(force=force)
            progress.update(task, completed=True)

        stats = detector.stats
        console.print("\nContainer detection complete:")
        console.print(f"  Functions scanned: {stats['functions_scanned']}")
        console.print(f"  Candidates (score >= {min_score}): {stats['candidates']}")
        console.print(f"  LLM calls: {stats['llm_calls']}")
        console.print(f"  Cache hits: {stats['cache_hits']}")
        console.print(f"  Containers found: {stats['containers_found']}")
        if stats["input_tokens"] > 0 or stats["output_tokens"] > 0:
            total_tok = stats['input_tokens'] + stats['output_tokens']
            console.print(
                f"  Tokens: {total_tok:,} "
                f"({stats['input_tokens']:,} in + "
                f"{stats['output_tokens']:,} out)"
            )
        if stats["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats['errors']}[/yellow]")

    finally:
        db.close()


@main.command("show-containers")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_containers(db_path, fmt):
    """Show detected container/collection functions."""
    db = SummaryDB(db_path)

    try:
        summaries = db.get_all_container_summaries()
        functions = {f.id: f for f in db.get_all_functions()}

        if not summaries:
            console.print(
                "[yellow]No container summaries found. "
                "Run 'container-analyze' first.[/yellow]"
            )
            return

        if fmt == "json":
            output = []
            for cs in summaries:
                func = functions.get(cs.function_id)
                entry = cs.to_dict()
                if func:
                    entry["function_name"] = func.name
                    entry["file_path"] = func.file_path
                    entry["signature"] = func.signature
                output.append(entry)
            console.print(json.dumps(output, indent=2))

        else:
            table = Table(title="Container Functions")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Type", style="green")
            table.add_column("Container", justify="center")
            table.add_column("Store", justify="center")
            table.add_column("Load", justify="center")
            table.add_column("Conf.", style="magenta")
            table.add_column("Score", justify="right")

            for cs in summaries:
                func = functions.get(cs.function_id)
                table.add_row(
                    func.name if func else f"ID:{cs.function_id}",
                    Path(func.file_path).name if func else "?",
                    cs.container_type,
                    f"arg[{cs.container_arg}]",
                    ",".join(str(a) for a in cs.store_args) if cs.store_args else "-",
                    "ret" if cs.load_return else "-",
                    cs.confidence,
                    str(cs.heuristic_score),
                )

            console.print(table)
            console.print(f"\nTotal: {len(summaries)} container functions")

    finally:
        db.close()


@main.command("find-allocator-candidates")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option(
    "--output", "-o", "output_path", required=True,
    type=click.Path(), help="Output JSON path",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["claude", "openai", "ollama", "llamacpp", "gemini"]
    ),
    default="ollama",
)
@click.option("--model", default=None, help="Model name to use")
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
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode")
@click.option("--min-score", default=5, type=int, help="Minimum heuristic score (default: 5)")
@click.option(
    "--heuristic-only", is_flag=True,
    help="Skip LLM, output all candidates above threshold",
)
@click.option(
    "--include-stdlib", is_flag=True,
    help="Include well-known stdlib allocators in "
         "confirmed list",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--log-llm", type=click.Path(), default=None,
    help="Log all LLM prompts and responses to file",
)
def find_allocator_candidates(
    db_path, output_path, backend, model, llm_host, llm_port, disable_thinking,
    min_score, heuristic_only, include_stdlib, verbose, log_llm
):
    """Find allocator function candidates for KAMain's --allocator-file.

    Uses heuristic scoring to find candidates, optionally confirms with LLM.
    Outputs JSON with {candidates: [...], confirmed: [...]}.

    Example:
        llm-summary find-allocator-candidates \\
            --db functions.db -o alloc.json \\
            --heuristic-only -v
        llm-summary find-allocator-candidates \\
            --db functions.db -o alloc.json \\
            --backend ollama --model qwen3
    """
    from .allocator import STDLIB_ALLOCATORS, STDLIB_DEALLOCATORS, AllocatorDetector

    # Infer project name from DB path
    db_dir = Path(db_path).resolve().parent
    project_name = db_dir.name if db_dir.name != "." else Path(db_path).stem

    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()
        if not functions:
            console.print("[red]No functions in database. Run 'extract' or 'scan' first.[/red]")
            return

        console.print(f"Database: {db_path} ({len(functions)} functions, project: {project_name})")

        if heuristic_only:
            detector = AllocatorDetector(
                db, llm=None, verbose=verbose, min_score=min_score,
                project_name=project_name,
            )
            alloc_scored, dealloc_scored = detector.heuristic_only()

            if not alloc_scored and not dealloc_scored:
                console.print("[yellow]No candidates found above threshold.[/yellow]")
                output: dict[str, list] = {
                    "candidates": [], "confirmed": [],
                    "dealloc_candidates": [], "dealloc_confirmed": [],
                }
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
                console.print(f"Wrote empty result to {output_path}")
                return

            # Display allocator results
            if alloc_scored:
                table = Table(title=f"Allocator Candidates (score >= {min_score})")
                table.add_column("Score", justify="right", style="green")
                table.add_column("Function", style="cyan")
                table.add_column("File", style="dim")
                table.add_column("Signals")

                alloc_scored.sort(key=lambda x: x[1], reverse=True)

                for func, score, signals in alloc_scored:
                    signal_strs = [s.split(": ", 1)[0] if ": " in s else s for s in signals]
                    table.add_row(
                        str(score),
                        func.name,
                        Path(func.file_path).name if func.file_path else "?",
                        ", ".join(signal_strs),
                    )

                console.print(table)

            # Display deallocator results
            if dealloc_scored:
                dtable = Table(title=f"Deallocator Candidates (score >= {min_score})")
                dtable.add_column("Score", justify="right", style="green")
                dtable.add_column("Function", style="cyan")
                dtable.add_column("File", style="dim")
                dtable.add_column("Signals")

                dealloc_scored.sort(key=lambda x: x[1], reverse=True)

                for func, score, signals in dealloc_scored:
                    signal_strs = [s.split(": ", 1)[0] if ": " in s else s for s in signals]
                    dtable.add_row(
                        str(score),
                        func.name,
                        Path(func.file_path).name if func.file_path else "?",
                        ", ".join(signal_strs),
                    )

                console.print(dtable)

            candidate_names = [func.name for func, _, _ in alloc_scored]
            dealloc_candidate_names = [func.name for func, _, _ in dealloc_scored]
            if include_stdlib:
                for name in sorted(STDLIB_ALLOCATORS):
                    if name not in candidate_names:
                        candidate_names.append(name)
                for name in sorted(STDLIB_DEALLOCATORS):
                    if name not in dealloc_candidate_names:
                        dealloc_candidate_names.append(name)

            output = {
                "candidates": candidate_names,
                "confirmed": [],
                "dealloc_candidates": dealloc_candidate_names,
                "dealloc_confirmed": [],
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            stats = detector.stats
            console.print(f"\nAlloc candidates: {len(candidate_names)}")
            console.print(f"Dealloc candidates: {len(dealloc_candidate_names)}")
            console.print(f"Functions scanned: {stats['functions_scanned']}")
            console.print(f"Wrote {output_path}")
            return

        # Full mode: heuristic + LLM
        backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, disable_thinking)
        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        detector = AllocatorDetector(
            db, llm=llm, verbose=verbose, log_file=log_llm,
            min_score=min_score, project_name=project_name,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting allocator/deallocator functions...", total=None)
            candidates, confirmed, dealloc_candidates, dealloc_confirmed = detector.detect_all(
                include_stdlib=include_stdlib
            )
            progress.update(task, completed=True)

        # All go into candidates for KAMain to verify;
        # confirmed is left empty for KAMain to populate
        all_candidates = confirmed + candidates
        all_dealloc = dealloc_confirmed + dealloc_candidates
        output = {
            "candidates": all_candidates,
            "confirmed": [],
            "dealloc_candidates": all_dealloc,
            "dealloc_confirmed": [],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        stats = detector.stats
        console.print("\nAllocator/deallocator detection complete:")
        console.print(f"  Functions scanned: {stats['functions_scanned']}")
        console.print(f"  Alloc candidates (score >= {min_score}): {stats['candidates']}")
        console.print(f"  Confirmed allocators: {stats['confirmed']}")
        console.print(f"  Dealloc candidates (score >= {min_score}): {stats['dealloc_candidates']}")
        console.print(f"  Confirmed deallocators: {stats['dealloc_confirmed']}")
        console.print(f"  LLM calls: {stats['llm_calls']}")
        if stats["input_tokens"] > 0 or stats["output_tokens"] > 0:
            total_tok = stats['input_tokens'] + stats['output_tokens']
            console.print(
                f"  Tokens: {total_tok:,} "
                f"({stats['input_tokens']:,} in + "
                f"{stats['output_tokens']:,} out)"
            )
        if stats["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats['errors']}[/yellow]")
        console.print(f"  Wrote {output_path}")

    finally:
        db.close()


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

    from .link_units.skills import _is_docker_path, _resolve_host_path, _translate_arg

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
        build_dir_path = Path(build_dir) if build_dir else proj_dir

        resolved = []
        for e in entries:
            e = dict(e)
            if _is_docker_path(e.get("directory", "")):
                e["directory"] = str(_resolve_host_path(e["directory"], proj_dir, build_dir_path))
            if _is_docker_path(e.get("file", "")):
                e["file"] = str(_resolve_host_path(e["file"], proj_dir, build_dir_path))
            if "output" in e and _is_docker_path(e["output"]):
                e["output"] = str(_resolve_host_path(e["output"], proj_dir, build_dir_path))
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
    c_extensions = {".c", ".cpp", ".cc", ".cxx", ".c++"}
    asm_extensions = {".s", ".S", ".asm"}
    all_files = compile_commands.get_all_files()

    if bc_files_filter is not None:
        # Use the resolved compile_commands path so source file paths match
        scoped = _source_files_for_target(resolved_cc_path, bc_files_filter, build_dir=lu_build_dir)
        source_files = [
            f for f in all_files
            if f in scoped
            and Path(f).suffix.lower() in c_extensions
        ]
        # Assembly files don't produce .bc — use asm_sources_filter from link_units.json
        if asm_sources_filter:
            asm_files = [
                f for f in all_files
                if Path(f).suffix in asm_extensions and f in asm_sources_filter
            ]
        else:
            asm_files = []
    else:
        source_files = [f for f in all_files if Path(f).suffix.lower() in c_extensions]
        asm_files = [f for f in all_files if Path(f).suffix in asm_extensions]

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
            from .asm_extractor import extract_asm_functions

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
    console.print("\n2. Analyze with llm-summary:")
    console.print(f"   [cyan]llm-summary extract --path {project_path} --db {db_path}[/cyan]")

    if generate_ir:
        console.print("\n3. LLVM IR artifacts will be in:")
        console.print(f"   [cyan]{paths['artifacts_dir']}[/cyan]")


@main.command("generate-kanalyzer-script")
@click.option("--project", required=True, help="Project name (looks up build-scripts/<project>/)")
@click.option(
    "--artifacts-dir", default=None,
    help="Artifacts directory "
         "(default: build-scripts/<project>/artifacts)",
)
@click.option(
    "--output-json", required=True,
    help="Output path for KAMain JSON call graph",
)
@click.option(
    "--kamain-bin",
    default="/home/csong/project/kanalyzer/release"
            "/lib/KAMain",
    help="Path to KAMain binary",
)
@click.option(
    "--allocator-file", default=None,
    help="Path to allocator candidates JSON",
)
@click.option(
    "--container-file", default=None,
    help="Path to container functions JSON",
)
@click.option(
    "--verbose-level", default=1, type=int,
    help="KAMain verbose level (default: 1)",
)
@click.option(
    "--output", "-o", default=None,
    help="Output script path (default: stdout)",
)
def generate_kanalyzer_script(
    project, artifacts_dir, output_json, kamain_bin,
    allocator_file, container_file, verbose_level,
    output,
):
    """Generate a shell script to run KAMain on a project's .bc files."""
    scripts_dir = Path("build-scripts") / project

    if not scripts_dir.exists():
        console.print(f"[red]Project directory not found: {scripts_dir}[/red]")
        return

    if artifacts_dir is None:
        artifacts_dir = str(scripts_dir / "artifacts")

    lines = [
        "#!/bin/bash",
        f"# KAMain call graph analysis for {project}",
        "# Generated by llm-summary generate-kanalyzer-script",
        "",
        "set -e",
        "",
        f'ARTIFACTS_DIR="{artifacts_dir}"',
        f'OUTPUT_JSON="{output_json}"',
        f'KAMAIN="{kamain_bin}"',
        "",
        'if [ ! -x "$KAMAIN" ]; then',
        '    echo "Error: KAMain not found at $KAMAIN"',
        '    exit 1',
        'fi',
        "",
        '# Find all .bc files',
        'BC_FILES=$(find "$ARTIFACTS_DIR" -name "*.bc" -type f)',
        "",
        'if [ -z "$BC_FILES" ]; then',
        '    echo "Error: No .bc files found in $ARTIFACTS_DIR"',
        '    exit 1',
        'fi',
        "",
        'BC_COUNT=$(echo "$BC_FILES" | wc -l)',
        'echo "Found $BC_COUNT .bc files in $ARTIFACTS_DIR"',
        "",
        '# Run KAMain',
        '"$KAMAIN" \\',
        '    $BC_FILES \\',
        '    --callgraph-json "$OUTPUT_JSON" \\',
    ]

    if allocator_file:
        lines.append(f'    --allocator-file "{allocator_file}" \\')
    if container_file:
        lines.append(f'    --container-file "{container_file}" \\')

    lines.append(f'    --verbose {verbose_level}')
    lines.append("")
    lines.append('echo "Call graph written to $OUTPUT_JSON"')
    lines.append("")

    script_content = "\n".join(lines)

    if output:
        output_path = Path(output)
        output_path.write_text(script_content)
        output_path.chmod(0o755)
        console.print(f"Script written to: {output_path}")
    else:
        console.print(script_content)


@main.command("import-callgraph")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True),
              help="Path to KAMain callgraph JSON")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--clear-edges", is_flag=True, help="Clear existing call_edges before import")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def import_callgraph(json_path, db_path, clear_edges, verbose):
    """Import a KAMain call graph JSON into the database.

    Parses the call graph, matches functions to existing DB entries,
    creates stubs for unmatched functions, and populates call_edges.

    Example:
        llm-summary import-callgraph --json /tmp/libpng_cg.json --db analysis.db --clear-edges -v
    """
    from .callgraph_import import CallGraphImporter

    json_path = Path(json_path)
    db = SummaryDB(db_path)

    try:
        # Show before stats
        before_stats = db.get_stats()
        console.print(f"Database: {db_path}")
        console.print(f"  Functions before: {before_stats['functions']}")
        console.print(f"  Call edges before: {before_stats['call_edges']}")

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
    summary_tables = [
        "allocation_summaries",
        "free_summaries",
        "init_summaries",
        "memsafe_summaries",
        "verification_summaries",
    ]

    target_db = SummaryDB(db_path)
    total_funcs = 0
    total_summaries = 0
    total_edges = 0

    try:
        for dep_path in dep_db_paths:
            dep_db = SummaryDB(dep_path)
            dep_tag = f"dep:{Path(dep_path).stem}"
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

                    if not src_summaries:
                        continue

                    # Find or create matching function in target DB
                    candidates = target_db.get_function_by_name(src_func.name)
                    tgt_func = candidates[0] if candidates else None
                    if tgt_func is None:
                        stub_id = target_db.insert_function_stub(
                            name=src_func.name,
                            file_path=src_func.file_path,
                            line_start=src_func.line_start,
                            line_end=src_func.line_end,
                        )
                        tgt_func = target_db.get_function(stub_id)

                    if tgt_func is None:
                        continue

                    func_imported = False
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

                    if func_imported:
                        copied_funcs += 1
                        if verbose:
                            kinds = [t.replace("_summaries", "") for t in src_summaries]
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

                    # Collect summaries from dep DB
                    src_summaries: dict[str, str] = {}
                    for table in summary_tables:
                        row = dep_db.conn.execute(
                            f"SELECT summary_json FROM {table} WHERE function_id = ?",
                            (dep_func.id,),
                        ).fetchone()
                        if row:
                            src_summaries[table] = row[0]

                    if not src_summaries:
                        if verbose:
                            console.print(f"  [dim]{func_name}: no summaries in {dep_path}[/dim]")
                        continue

                    if dry_run:
                        kinds = [t.replace("_summaries", "") for t in src_summaries]
                        console.print(f"  [dry-run] {func_name}: would import {', '.join(kinds)}")
                        copied_funcs += 1
                        copied_summaries += len(src_summaries)
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

                    if func_imported:
                        copied_funcs += 1
                        if verbose:
                            kinds = [t.replace("_summaries", "") for t in src_summaries]
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
            if not vsummary or not vsummary.issues:
                continue

            # Build fingerprint->review lookup
            fingerprints = [iss.fingerprint() for iss in vsummary.issues]
            reviews = db.get_issue_reviews_by_fingerprints(func.id, fingerprints)

            for idx, issue in enumerate(vsummary.issues):
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
def triage(
    db_path: str, backend: str, model: str | None,
    llm_host: str, llm_port: int | None, disable_thinking: bool,
    verbose: bool, function_names: tuple[str, ...],
    severity: str | None, issue_index: int | None,
    output: str | None,
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
    llm = create_backend(backend, model=model, **kwargs)
    db = SummaryDB(db_path)

    try:
        agent = TriageAgent(db, llm, verbose=verbose)

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
        all_results = []

        for func in functions:
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
                result = agent.triage_issue(func, issue, issue_index, vs)
                all_results.append(result)
            else:
                results = agent.triage_function(
                    func, severity_filter=severity_filter,
                )
                all_results.extend(results)

        # Output
        results_json = [r.to_dict() for r in all_results]

        if output:
            with open(output, "w") as f:
                json.dump(results_json, f, indent=2)
                f.write("\n")
            console.print(f"[green]Wrote {len(all_results)} results to {output}[/green]")
        else:
            # Print summary table
            safe_count = sum(1 for r in all_results if r.hypothesis == "safe")
            feasible_count = sum(1 for r in all_results if r.hypothesis == "feasible")
            console.print(
                f"\n[bold]Triage Results[/bold]: {len(all_results)} issues — "
                f"[green]{safe_count} safe[/green], "
                f"[red]{feasible_count} feasible[/red]"
            )
            for r in all_results:
                style = "green" if r.hypothesis == "safe" else "red"
                console.print(
                    f"  [{style}]{r.hypothesis}[/{style}] "
                    f"{r.function_name}#{r.issue_index}: "
                    f"{r.issue.issue_kind} — {r.reasoning[:120]}"
                )

            if verbose:
                console.print("\n[dim]Full JSON:[/dim]")
                console.print(json.dumps(results_json, indent=2))

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
def gen_harness(
    db_path, backend, model, llm_host, llm_port,
    disable_thinking, verbose, log_llm, output_dir, function_names,
    ko_clang_path, symsan_dir, compile_commands_path, project_path,
    build_dir, bc_file, plan, plan_only, validate,
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
        )

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
                if not relevant:
                    relevant = [func_name] if func_name else []
                if not relevant:
                    console.print("[red]No relevant_functions in verdict[/red]")
                    continue

                entries = _find_entry_functions(db, relevant)
                # Per-verdict output dir to avoid overwriting
                verdict_dir = str(
                    Path(output_dir) / func_name / f"v{issue_idx}"
                )
                Path(verdict_dir).mkdir(parents=True, exist_ok=True)
                console.print(
                    f"Validating {func_name}#{issue_idx} "
                    f"({v.get('hypothesis', '?')}): "
                    f"entries={entries} -> {verdict_dir}"
                )

                other_entries = set(entries)
                for entry in entries:
                    # Scope per entry: exclude other entry functions
                    entry_scope = [
                        f for f in relevant
                        if f == entry or f not in other_entries
                    ]
                    triage_ctx = {
                        "hypothesis": v.get("hypothesis", ""),
                        "reasoning": v.get("reasoning", ""),
                        "severity": v.get("issue", {}).get("severity", ""),
                        "issue_kind": v.get("issue", {}).get("issue_kind", ""),
                        "issue_description": v.get("issue", {}).get("description", ""),
                        "assumptions": v.get("assumptions", []),
                        "assertions": v.get("assertions", []),
                        "real_functions": entry_scope,
                    }

                    result = generator.validate_triage(
                        entry,
                        triage_context=triage_ctx,
                        output_dir=verdict_dir,
                        bc_file=bc_file,
                    )
                    if result:
                        console.print(f"[green]Harness generated: {entry}[/green]")

                        # Auto-build to get CFG dump, then generate
                        # validation plan with counter-example traces
                        out = Path(verdict_dir)
                        script = out / f"build_{entry}.sh"
                        if script.exists():
                            console.print(f"  Building {entry}...")
                            build_result = subprocess.run(
                                ["bash", str(script)],
                                capture_output=True, text=True,
                            )
                            if build_result.returncode == 0:
                                console.print(
                                    f"  [green]Built: {entry}.ucsan[/green]"
                                )
                                cfg_path = out / f"cfg_{entry}.txt"
                                if cfg_path.exists():
                                    plan = generator.generate_validation_plan(
                                        v, str(out),
                                        cfg_dump=str(cfg_path),
                                        entry_name=entry,
                                        scope_functions=entry_scope,
                                    )
                                    if plan:
                                        console.print(
                                            f"  [green]Validation plan: "
                                            f"{len(plan.get('traces', []))} "
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
                            else:
                                console.print(
                                    f"[red]  Build failed: "
                                    f"{build_result.stderr[:200]}[/red]"
                                )
                    else:
                        console.print(f"[red]Failed: {entry}[/red]")
            return

        # Determine which functions to process
        if function_names:
            targets = list(function_names)
        else:
            rows = db.conn.execute(
                "SELECT f.name FROM functions f "
                "JOIN memsafe_summaries m ON m.function_id = f.id"
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


if __name__ == "__main__":
    main()
