"""Command-line interface for LLM-based allocation summary analysis."""

import json
import sys
from pathlib import Path

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
@click.option("--db", "db_path", required=True, help="Database file path (must have functions + call_edges)")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-summarize even if summary exists")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--init-stdlib", is_flag=True, help="Auto-populate stdlib summaries before starting")
@click.option("--allocator-file", type=click.Path(exists=True), default=None,
              help="JSON file with custom allocator names (e.g. from find-allocator-candidates)")
@click.option("--type", "summary_types", multiple=True, type=click.Choice(["allocation", "free", "init", "memsafe", "verify"]),
              help="Summary pass(es) to run (default: allocation). Can be specified multiple times.")
@click.option("--deallocator-file", type=click.Path(exists=True), default=None,
              help="JSON file with custom deallocator names (for free pass)")
@click.option("--vsnap", type=click.Path(exists=True), default=None,
              help="V-snapshot (.vsnap) file for alias context in memsafe/verify passes")
@click.option("-j", "jobs", default=1, type=int, help="Parallel LLM queries (default: 1)")
@click.option("--cache-mode", type=click.Choice(["none", "instructions", "source"]), default="none",
              help="Prompt caching mode: none (default), instructions (cache task instructions), source (cache function source)")
def summarize(db_path, backend, model, llm_host, llm_port, disable_thinking, verbose, force, log_llm, init_stdlib, allocator_file, summary_types, deallocator_file, vsnap, jobs, cache_mode):
    """Generate allocation, free, init, memsafe, and/or verify summaries on a pre-populated database.

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
            console.print("[red]Error: No call edges in database. Run call graph import first.[/red]")
            sys.exit(1)

        # Prerequisite check for verify pass
        if "verify" in summary_types:
            missing = []
            for req_table in ["allocation_summaries", "free_summaries", "init_summaries", "memsafe_summaries"]:
                if stats.get(req_table, 0) == 0:
                    missing.append(req_table.replace("_summaries", "").replace("_", " "))
            if missing:
                console.print(f"[red]Error: --type verify requires all four prior passes. "
                              f"Missing: {', '.join(missing)}. "
                              f"Run --type allocation --type free --type init --type memsafe first.[/red]")
                return

        console.print(f"Database: {db_path}")
        console.print(f"  Functions: {func_count}")
        console.print(f"  Call edges: {edge_count}")

        # Show call graph stats
        edges = db.get_all_call_edges()
        if edges:
            graph = {}
            for edge in edges:
                if edge.caller_id not in graph:
                    graph[edge.caller_id] = []
                graph[edge.caller_id].append(edge.callee_id)

            orderer = ProcessingOrderer(graph)
            cg_stats = orderer.get_stats()
            console.print(f"  SCCs: {cg_stats['sccs']} ({cg_stats['recursive_sccs']} recursive, largest: {cg_stats['largest_scc']})")

        # Init stdlib if requested
        if init_stdlib:
            from .models import Function

            def _ensure_stdlib_func(name: str) -> Function:
                """Find or create a stdlib function stub."""
                existing = db.get_function_by_name(name)
                if existing:
                    func = existing[0]
                    # Update attributes if not already set
                    attrs = STDLIB_ATTRIBUTES.get(name, "")
                    if attrs and not func.attributes:
                        func.attributes = attrs
                        func.id = db.insert_function(func)
                    return func
                func = Function(
                    name=name,
                    file_path="<stdlib>",
                    line_start=0,
                    line_end=0,
                    source="",
                    signature=f"{name}(...)",
                    attributes=STDLIB_ATTRIBUTES.get(name, ""),
                )
                func.id = db.insert_function(func)
                return func

            # Allocation summaries
            alloc_summaries = get_all_stdlib_summaries()
            for name, summary in alloc_summaries.items():
                func = _ensure_stdlib_func(name)
                db.upsert_summary(func, summary, model_used="builtin")

            # Free summaries
            free_summaries = get_all_stdlib_free_summaries()
            for name, summary in free_summaries.items():
                func = _ensure_stdlib_func(name)
                db.upsert_free_summary(func, summary, model_used="builtin")

            # Init summaries
            init_summaries = get_all_stdlib_init_summaries()
            for name, summary in init_summaries.items():
                func = _ensure_stdlib_func(name)
                db.upsert_init_summary(func, summary, model_used="builtin")

            # Memsafe summaries
            memsafe_summaries = get_all_stdlib_memsafe_summaries()
            for name, summary in memsafe_summaries.items():
                func = _ensure_stdlib_func(name)
                db.upsert_memsafe_summary(func, summary, model_used="builtin")

            # Create stubs for attribute-only functions (e.g., exit, abort)
            for name in STDLIB_ATTRIBUTES:
                _ensure_stdlib_func(name)

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
        passes = []
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
                    console.print(f"Custom allocators: {len(allocators)} loaded from {allocator_file}")

                # Confirm candidates via vsnapshot alias analysis
                if allocators and alias_builder:
                    from .allocator import vsnapshot_confirm_allocators
                    confirmed, remaining = vsnapshot_confirm_allocators(alias_builder.snap, allocators)
                    console.print(f"  Vsnapshot alloc confirmation: {len(confirmed)} confirmed, "
                                  f"{len(remaining)} unconfirmed (dropped)")
                    allocators = confirmed

            alloc_summarizer = AllocationSummarizer(db, llm, verbose=verbose, log_file=log_llm, allocators=allocators, cache_mode=cache_mode)
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
                    console.print(f"Custom deallocators: {len(deallocators)} loaded from {deallocator_file}")

                # Confirm candidates via vsnapshot alias analysis
                if deallocators and alias_builder:
                    from .allocator import vsnapshot_confirm_deallocators
                    dconfirmed, dremaining = vsnapshot_confirm_deallocators(alias_builder.snap, deallocators)
                    console.print(f"  Vsnapshot dealloc confirmation: {len(dconfirmed)} confirmed, "
                                  f"{len(dremaining)} unconfirmed (dropped)")
                    deallocators = dconfirmed

            free_summarizer = FreeSummarizer(db, llm, verbose=verbose, log_file=log_llm, deallocators=deallocators, cache_mode=cache_mode)
            passes.append(FreePass(free_summarizer, db, llm.model))

        init_summarizer = None
        if "init" in summary_types:
            from .init_summarizer import InitSummarizer

            init_summarizer = InitSummarizer(db, llm, verbose=verbose, log_file=log_llm, cache_mode=cache_mode)
            passes.append(InitPass(init_summarizer, db, llm.model))

        memsafe_summarizer = None
        if "memsafe" in summary_types:
            from .memsafe_summarizer import MemsafeSummarizer

            memsafe_summarizer = MemsafeSummarizer(db, llm, verbose=verbose, log_file=log_llm, cache_mode=cache_mode)
            passes.append(MemsafePass(memsafe_summarizer, db, llm.model, alias_builder=alias_builder))

        verification_summarizer = None
        if "verify" in summary_types:
            from .verification_summarizer import VerificationSummarizer

            verification_summarizer = VerificationSummarizer(db, llm, verbose=verbose, log_file=log_llm, cache_mode=cache_mode)
            passes.append(VerificationPass(verification_summarizer, db, llm.model, alias_builder=alias_builder))

        pass_names = " + ".join(p.name for p in passes)
        console.print(f"\n[bold]Running passes: {pass_names}[/bold]")

        pool = None
        if jobs > 1:
            from .llm.pool import LLMPool
            pool = LLMPool(max_workers=jobs)
            if verbose:
                console.print(f"  Thread pool: {jobs} workers")

        driver = BottomUpDriver(db, verbose=verbose, pool=pool)
        try:
            results = driver.run(passes, force=force)
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
            if cache_mode != "none" and (s.get("cache_read_tokens") or s.get("cache_creation_tokens")):
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            allocating = sum(1 for sm in summaries.values() if sm.allocations)
            console.print(f"\nFunctions with allocations: {allocating}")

        if free_summarizer is not None:
            free_summaries = results["free"]
            s = free_summarizer.stats
            console.print("\nFree summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            if cache_mode != "none" and (s.get("cache_read_tokens") or s.get("cache_creation_tokens")):
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            freeing = sum(1 for sm in free_summaries.values() if sm.frees)
            console.print(f"\nFunctions with frees: {freeing}")

        if init_summarizer is not None:
            init_summaries = results["init"]
            s = init_summarizer.stats
            console.print("\nInit summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            if cache_mode != "none" and (s.get("cache_read_tokens") or s.get("cache_creation_tokens")):
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            initializing = sum(1 for sm in init_summaries.values() if sm.inits)
            console.print(f"\nFunctions with inits: {initializing}")

        if memsafe_summarizer is not None:
            memsafe_summaries = results["memsafe"]
            s = memsafe_summarizer.stats
            console.print("\nMemsafe summary generation complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            if cache_mode != "none" and (s.get("cache_read_tokens") or s.get("cache_creation_tokens")):
                console.print(f"  Cache read tokens: {s['cache_read_tokens']:,}")
                console.print(f"  Cache creation tokens: {s['cache_creation_tokens']:,}")
            if s["errors"] > 0:
                console.print(f"  [yellow]Errors: {s['errors']}[/yellow]")

            with_contracts = sum(1 for sm in memsafe_summaries.values() if sm.contracts)
            console.print(f"\nFunctions with safety contracts: {with_contracts}")

        if verification_summarizer is not None:
            verify_summaries = results["verify"]
            s = verification_summarizer.stats
            console.print("\nVerification complete:")
            console.print(f"  Functions processed: {s['functions_processed']}")
            console.print(f"  LLM calls: {s['llm_calls']}")
            console.print(f"  Cache hits: {s['cache_hits']}")
            if cache_mode != "none" and (s.get("cache_read_tokens") or s.get("cache_creation_tokens")):
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
@click.option("--path", "path_opt", type=click.Path(exists=True), default=None, help="Path to extract from")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), default=None,
              help="Path to compile_commands.json for proper macro/include handling")
@click.option("--project-path", "project_path", type=click.Path(), default=None,
              help="Host path to project source root. Required when compile_commands.json uses Docker "
                   "container paths (/workspace/src/...).")
@click.option("--recursive/--no-recursive", default=True)
@click.option("--preprocess", is_flag=True, help="Run clang -E to expand macros and store preprocessed source")
def extract(path_arg, path_opt, db_path, compile_commands_path, project_path, recursive, preprocess):
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

        func_ids = db.insert_functions_batch(all_functions)
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
def show(db_path, name, file_path, allocating_only, fmt):
    """Show stored summaries."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()

        if name:
            functions = [f for f in functions if name in f.name]

        if file_path:
            functions = [f for f in functions if file_path in f.file_path]

        if fmt == "json":
            output = []
            for func in functions:
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
                summary = db.get_summary_by_function_id(func.id)
                if summary:
                    if allocating_only and not summary.allocations:
                        continue

                    alloc_str = ", ".join(a.source for a in summary.allocations) or "-"
                    table.add_row(
                        func.name,
                        Path(func.file_path).name,
                        alloc_str,
                        summary.description[:50] + "..." if len(summary.description) > 50 else summary.description,
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
            graph = {}
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
@click.option("--db", "db_path", required=True, help="Database file path")
@click.option(
    "--abilist",
    "extra_abilists",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional abilist file(s) to merge with the bundled list (repeatable)",
)
@click.option(
    "--cache-db",
    default=None,
    help="Global stdlib cache path (default: ~/.llm-summary/stdlib_cache.db)",
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
@click.option("--log-llm", type=click.Path(), default=None, help="Log LLM prompts/responses to file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def init_stdlib(db_path, extra_abilists, cache_db, backend, model, llm_host, llm_port, log_llm, verbose):
    """Populate external-function summaries using a persistent global cache.

    For each function in the project DB that has no source body:

    \b
      1. If it is in the known-externals list (bundled abilist + --abilist):
           a. Cache hit  → copy summaries from the global cache into the project DB.
           b. Cache miss → generate summaries via LLM, save to cache, then copy.
      2. If not in the known-externals list → skip (let the main summarizer handle it).

    The global cache (~/.llm-summary/stdlib_cache.db) is pre-seeded with
    hand-crafted entries for common libc functions (malloc, free, memcpy, …).
    A --backend is only required when the cache is missing entries for
    functions that appear in the project.
    """
    from .external_summarizer import ExternalFunctionSummarizer
    from .models import Function, VerificationSummary
    from .stdlib_cache import StdlibCache, load_known_externals

    # 1. Load known-externals set
    known_externals = load_known_externals(list(extra_abilists) if extra_abilists else None)
    console.print(f"Known-externals registry: {len(known_externals):,} function names")

    # 2. Open and seed the global stdlib cache
    cache = StdlibCache(cache_db)
    seeded = cache.seed_builtins()
    if seeded:
        console.print(f"  Seeded {seeded} hand-crafted builtin entries into cache")

    db = SummaryDB(db_path)
    try:
        # 3. Identify functions that have no source and need summaries
        all_funcs = db.get_all_functions()
        sourceless = [f for f in all_funcs if not f.source]

        def _needs_summary(f: Function) -> bool:
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

        # 4. Require --backend if there are cache misses
        if cache_misses and not backend:
            names = sorted(f.name for f in cache_misses)
            console.print(
                f"\n[red]Error: {len(cache_misses)} known-external function(s) have no cached "
                f"summary and no --backend was given.[/red]"
            )
            console.print("  Re-run with --backend <claude|gemini|…> to generate them.")
            console.print(f"  Missing: {', '.join(names)}")
            sys.exit(1)

        # 5. Helper: get-or-create a function stub in the project DB
        def _get_or_create(name: str) -> Function:
            existing = db.get_function_by_name(name)
            if existing:
                return existing[0]
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

        # 6. Helper: apply a cache entry to the project DB
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

        # 7. Apply cache hits
        if cache_hits:
            console.print(f"\nApplying {len(cache_hits)} cached summaries...")
            for f in cache_hits:
                entry = cache.get(f.name)
                func = _get_or_create(f.name)
                _apply_entry(func, entry)
                if verbose:
                    console.print(f"  [cache] {f.name}")

        # 8. LLM-generate cache misses
        if cache_misses and backend:
            backend_kwargs = _build_backend_kwargs(backend, llm_host, llm_port, False)
            llm = create_backend(backend, model=model, **backend_kwargs)
            gen = ExternalFunctionSummarizer(llm, verbose=verbose, log_file=log_llm)
            console.print(
                f"\nGenerating summaries for {len(cache_misses)} uncached functions "
                f"via {backend} ({llm.model})..."
            )
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
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

        # 9. Report skipped unknowns
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
@click.option("--path", "path_opt", type=click.Path(exists=True), default=None, help="Path to analyze")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), default=None,
              help="Path to compile_commands.json for proper macro/include handling")
@click.option("--project-path", "project_path", type=click.Path(), default=None,
              help="Host path to project source root. Required when compile_commands.json uses Docker "
                   "container paths (/workspace/src/...).")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--recursive/--no-recursive", default=True, help="Scan directories recursively")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis (ignore cache)")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--pass1-only", is_flag=True, help="Only run Pass 1 (flow summarization)")
@click.option("--pass2-only", is_flag=True, help="Only run Pass 2 (resolution), requires Pass 1 already done")
def indirect_analyze(
    path_arg, path_opt, db_path, compile_commands_path, project_path, backend, model, llm_host, llm_port,
    disable_thinking, recursive, verbose, force, log_llm, pass1_only, pass2_only
):
    """
    Analyze indirect calls using LLM-based two-pass approach.

    Pass 1: Summarize where address-taken function pointers flow.
    Pass 2: Resolve indirect callsites using flow summaries.

    Example:
        llm-summary indirect-analyze --path src/ --db out.db --compile-commands compile_commands.json
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
            console.print(f"Using existing data: {atf_count} address-taken functions, {callsite_count} indirect callsites")

            if atf_count == 0:
                console.print("[red]No address-taken functions in database. Run without --pass2-only first.[/red]")
                return

            if callsite_count == 0:
                console.print("[red]No indirect callsites in database. Run without --pass2-only first.[/red]")
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
                console.print("[yellow]No address-taken functions found. Nothing to analyze.[/yellow]")
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

                flow_summaries = flow_summarizer.summarize_all(force=force)

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
                targets = db.get_indirect_call_targets(cs.id)
                caller = functions.get(cs.caller_function_id)

                target_list = []
                for t in targets:
                    target_func = functions.get(t.target_function_id)
                    target_list.append({
                        "function": target_func.name if target_func else f"ID:{t.target_function_id}",
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
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="ollama")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis (ignore cache)")
@click.option("--min-score", default=5, type=int, help="Minimum heuristic score for LLM confirmation (default: 5)")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
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
            results = detector.detect_all(force=force)
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
            console.print(f"  Tokens: {total_tok:,} ({stats['input_tokens']:,} in + {stats['output_tokens']:,} out)")
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
            console.print("[yellow]No container summaries found. Run 'container-analyze' first.[/yellow]")
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
@click.option("--output", "-o", "output_path", required=True, type=click.Path(), help="Output JSON path")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="ollama")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode")
@click.option("--min-score", default=5, type=int, help="Minimum heuristic score (default: 5)")
@click.option("--heuristic-only", is_flag=True, help="Skip LLM, output all candidates above threshold")
@click.option("--include-stdlib", is_flag=True, help="Include well-known stdlib allocators in confirmed list")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
def find_allocator_candidates(
    db_path, output_path, backend, model, llm_host, llm_port, disable_thinking,
    min_score, heuristic_only, include_stdlib, verbose, log_llm
):
    """Find allocator function candidates for KAMain's --allocator-file.

    Uses heuristic scoring to find candidates, optionally confirms with LLM.
    Outputs JSON with {candidates: [...], confirmed: [...]}.

    Example:
        llm-summary find-allocator-candidates --db functions.db -o alloc.json --heuristic-only -v
        llm-summary find-allocator-candidates --db functions.db -o alloc.json --backend ollama --model qwen3
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
                output = {"candidates": [], "confirmed": [],
                          "dealloc_candidates": [], "dealloc_confirmed": []}
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

        # All go into candidates for KAMain to verify; confirmed is left empty for KAMain to populate
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
            console.print(f"  Tokens: {total_tok:,} ({stats['input_tokens']:,} in + {stats['output_tokens']:,} out)")
        if stats["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats['errors']}[/yellow]")
        console.print(f"  Wrote {output_path}")

    finally:
        db.close()


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
                e["arguments"] = [_translate_arg(a, proj_dir, build_dir_path) for a in e["arguments"]]
            resolved.append(e)

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        _json.dump(resolved, tmp)
        tmp.flush()
        tmp.close()
        console.print(f"Remapped compile_commands.json: /workspace/src -> {proj_dir}")
        return CompileCommandsDB(tmp.name), tmp.name

    return CompileCommandsDB(compile_commands_path), None


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
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), required=True,
              help="Path to compile_commands.json")
@click.option("--link-units", "link_units_path", type=click.Path(exists=True), default=None,
              help="Path to link_units.json (from discover-link-units). Restricts scan to the named target.")
@click.option("--target", "target_name", default=None,
              help="Link-unit target name to scan (required when --link-units is given).")
@click.option("--project-path", "project_path", type=click.Path(), default=None,
              help="Host path to project source root. Required when compile_commands.json uses Docker "
                   "container paths (/workspace/src/...). Maps /workspace/src -> project_path and "
                   "/workspace/build -> build_dir.")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--preprocess", is_flag=True, help="Run clang -E to expand macros and store preprocessed source")
def scan(compile_commands_path, link_units_path, target_name, project_path, db_path, verbose, preprocess):
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
        bc_files_filter = set(targets_map[target_name].get("bc_files", []))
        lu_build_dir = lu_data.get("build_dir")
        console.print(f"Link-unit filter: target '{target_name}', {len(bc_files_filter)} bc files")

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
        source_files = [f for f in all_files if f in scoped and Path(f).suffix.lower() in c_extensions]
        asm_files = [f for f in all_files if f in scoped and Path(f).suffix in asm_extensions]
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

            for f in source_files:
                try:
                    functions = extractor.extract_from_file(f)
                    all_functions.extend(functions)
                    typedefs = extractor.extract_typedefs_from_file(f)
                    all_typedefs.extend(typedefs)
                    if verbose:
                        progress.console.print(f"  {Path(f).name}: {len(functions)} functions, {len(typedefs)} typedefs")
                except Exception as e:
                    extract_errors += 1
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
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

            console.print(f"\n[bold]Phase 1b: Extracting assembly functions[/bold]")

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
                            src_resolved = str(Path(src).resolve()) if Path(src).is_absolute() else src
                            asm_output_map[src_resolved] = out
                except (json.JSONDecodeError, OSError):
                    pass

            asm_functions = []
            asm_errors = 0
            for f in asm_files:
                try:
                    obj_path = asm_output_map.get(f)
                    funcs = extract_asm_functions(
                        Path(f),
                        Path(obj_path) if obj_path else None,
                    )
                    asm_functions.extend(funcs)
                    if verbose:
                        console.print(f"  {Path(f).name}: {len(funcs)} functions")
                except Exception as e:
                    asm_errors += 1
                    if verbose:
                        console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")

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

            for f in source_files:
                try:
                    scanner.scan_files([f])
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
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
            for f in source_files:
                try:
                    callsites = finder.find_in_files([f])
                    all_callsites.extend(callsites)
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
                progress.advance(task)

        console.print(f"  Indirect callsites: {len(all_callsites)}")

        # Summary
        console.print("\n[bold]Summary[/bold]")
        console.print(f"  Source files: {len(source_files)}")
        console.print(f"  Functions: {len(all_functions)}")
        console.print(f"  Indirect call targets: {len(atfs)}")
        console.print(f"  Indirect callsites: {len(all_callsites)}")
        console.print(f"  Database: {db_path}")

    finally:
        db.close()
        if _tmp_cc_file:
            Path(_tmp_cc_file).unlink(missing_ok=True)


@main.command("build-learn")
@click.option("--project-path", type=click.Path(exists=True), required=True, help="Path to the project to build")
@click.option("--build-dir", type=click.Path(), default=None, help="Custom build directory (default: <project-path>/build)")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude", help="LLM backend for incremental learning")
@click.option("--model", default=None, help="Model name (default depends on backend)")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--max-retries", default=3, help="Maximum build attempts")
@click.option("--container-image", default="llm-summary-builder:latest", help="Docker image to use")
@click.option("--enable-lto/--no-lto", default=True, help="Enable LLVM LTO")
@click.option("--prefer-static/--no-static", default=True, help="Prefer static linking")
@click.option("--generate-ir/--no-ir", default=True, help="Generate and save LLVM IR artifacts")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--ccache-dir", type=click.Path(), default="~/.cache/llm-summary-ccache",
              help="Host ccache directory (default: ~/.cache/llm-summary-ccache)")
@click.option("--no-ccache", is_flag=True, help="Disable ccache")
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
    )

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
    project_name = project_path.name
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
@click.option("--artifacts-dir", default=None, help="Artifacts directory (default: build-scripts/<project>/artifacts)")
@click.option("--output-json", required=True, help="Output path for KAMain JSON call graph")
@click.option("--kamain-bin", default="/home/csong/project/kanalyzer/release/lib/KAMain",
              help="Path to KAMain binary")
@click.option("--allocator-file", default=None, help="Path to allocator candidates JSON")
@click.option("--container-file", default=None, help="Path to container functions JSON")
@click.option("--verbose-level", default=1, type=int, help="KAMain verbose level (default: 1)")
@click.option("--output", "-o", default=None, help="Output script path (default: stdout)")
def generate_kanalyzer_script(project, artifacts_dir, output_json, kamain_bin, allocator_file, container_file, verbose_level, output):
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

        # Show after stats
        after_stats = db.get_stats()
        console.print("\nDatabase after:")
        console.print(f"  Functions: {after_stats['functions']}")
        console.print(f"  Call edges: {after_stats['call_edges']}")

    finally:
        db.close()


@main.command("discover-link-units")
@click.option("--project-name", default=None, help="Project name (default: inferred from project path)")
@click.option("--project-path", type=click.Path(exists=True), required=True, help="Path to the project source")
@click.option("--build-dir", type=click.Path(exists=True), required=True, help="Path to the build directory")
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), default=None,
              help="Path to compile_commands.json (auto-detected if not given)")
@click.option("--build-system", default=None, help="Build system hint (cmake, autotools, make)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output path (default: func-scans/<project>/link_units.json)")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude",
              help="LLM backend (only needed for non-Ninja builds)")
@click.option("--model", default=None, help="Model name")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking mode")
@click.option("--container-image", default="llm-summary-builder:latest",
              help="Docker image for sandboxed commands (agent mode only)")
@click.option("--log-llm", type=click.Path(), default=None, help="Log LLM prompts/responses to file")
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
    if not use_build_dir:
        build_dir = project_path
        if verbose:
            console.print(f"[yellow]In-tree build detected — using source dir as build dir[/yellow]")

    # Determine output path
    if output is None:
        output_dir = Path("func-scans") / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / "link_units.json")

    console.print(f"[bold]Link Unit Discovery[/bold]")
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
        console.print(f"[green]Deterministic discovery succeeded (no LLM needed)[/green]")
    else:
        # Try heuristic path (prescan + Makefile parsing)
        console.print("[yellow]No build.ninja — trying heuristic discovery...[/yellow]")

        from .link_units.skills import discover_heuristic

        heuristic_result, unresolved = discover_heuristic(
            build_dir, verbose=verbose, compile_commands_path=compile_commands_path,
        )
        heuristic_result["project"] = project_name

        if not unresolved:
            console.print(f"[green]Heuristic discovery fully resolved (no LLM needed)[/green]")
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

    console.print(f"\n[bold]Results[/bold]")
    console.print(f"  Link units: {len(link_units)} ({len(libs)} libraries, {len(exes)} executables)")
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
            finally:
                dep_db.close()

            total_funcs += copied_funcs
            total_summaries += copied_summaries
            console.print(
                f"  {Path(dep_path).name}: "
                f"{copied_funcs} functions, {copied_summaries} summaries imported"
            )

        console.print(
            f"\n[bold]Total:[/bold] {total_funcs} functions, "
            f"{total_summaries} summaries imported into {db_path}"
        )
    finally:
        target_db.close()


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
def review_issue(function_name, issue_index, db_path, review_status, reason, signature):
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
                console.print(f"[red]No function '{function_name}' with signature '{signature}'.[/red]")
                sys.exit(1)

        if len(functions) > 1:
            console.print(f"[yellow]Multiple functions named '{function_name}':[/yellow]")
            for f in functions:
                console.print(f"  {f.signature}  ({f.file_path}:{f.line_start})")
            console.print("Use --signature to disambiguate.")
            sys.exit(1)

        func = functions[0]

        # Load verification summary
        vsummary = db.get_verification_summary_by_function_id(func.id)
        if not vsummary:
            console.print(f"[red]No verification summary for '{function_name}'. Run verify pass first.[/red]")
            sys.exit(1)

        if not vsummary.issues:
            console.print(f"[yellow]'{function_name}' has no issues.[/yellow]")
            return

        if issue_index < 0 or issue_index >= len(vsummary.issues):
            console.print(
                f"[red]Invalid issue_index {issue_index}. "
                f"'{function_name}' has {len(vsummary.issues)} issue(s) (0-{len(vsummary.issues) - 1}).[/red]"
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
@click.option("--severity", default=None, type=click.Choice(["high", "medium", "low"]), help="Filter by severity")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_issues(db_path, name, filter_status, severity, fmt):
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
                sev_style = "red" if r["severity"] == "high" else ("yellow" if r["severity"] == "medium" else "dim")
                status_style = status_styles.get(r["status"], "")
                desc = r["description"]
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                table.add_row(
                    r["function"],
                    r["file"],
                    str(r["index"]),
                    f"[{sev_style}]{r['severity']}[/{sev_style}]",
                    r["kind"],
                    f"[{status_style}]{r['status']}[/{status_style}]" if status_style else r["status"],
                    desc,
                )

            console.print(table)

    finally:
        db.close()


if __name__ == "__main__":
    main()
