# Architecture Overview

This document describes the architecture of the LLM-based memory safety analysis tool.

## System Overview

The tool performs compositional, bottom-up analysis of C/C++ code to generate memory safety summaries across five passes (allocation, free, initialization, safety contracts, verification). It processes functions in dependency order (callees before callers) so that callee summaries are available when analyzing callers.

Analysis is **link-unit aware**: each build target (library, executable) gets its own database, and targets are processed in dependency order so that library summaries are available when analyzing executables that link them.

```
Phase 0         Phase 1            Phase 2          Phase 3
build-learn ──▶ discover-link- ──▶ scan (per ──▶ call graph
(Docker,LTO)    units              target)      (KAMain CFL)
                                   │                 │
                                   ▼                 ▼
                              functions.db      callgraph.json
                                   │            .cflcg, .vsnap
                                   │                 │
                              Phase 4            Phase 5
                              import-         ──▶ init-stdlib +
                              callgraph           import-dep-summaries
                                   │                 │
                                   └────────┬────────┘
                                            ▼
                                       Phase 6
                                       summarize (bottom-up)
                                       alloc → free → init → memsafe → verify
                                            │
                                            ▼
                                       functions.db
                                       (per link unit)
```

## Components

### 1. Function Extractor (`extractor.py`)

Uses libclang to parse C/C++ source files and extract function definitions.

**Key classes:**
- `FunctionExtractor`: Basic extraction with function signatures and source
- `FunctionExtractorWithBodies`: Full parsing including function bodies for call analysis

**Output:** List of `Function` objects containing:
- Function name and signature
- File path and line numbers
- Complete source code
- Source hash for change detection

### 2. Call Graph Builder (`callgraph.py`)

Builds the call graph by analyzing function bodies for call expressions.

**Features:**
- Extracts direct function calls
- Records callsite locations (file, line, column)
- Integrates with indirect call analysis

**Key class:** `CallGraphBuilder`

### 3. Indirect Call Analysis (`indirect/`)

Handles function pointer calls and virtual methods.

#### Scanner (`indirect/scanner.py`)
Finds functions whose addresses are taken:
- `&function` expressions
- Function passed as callback argument
- Function assigned to pointer variable

#### Callsite Finder (`indirect/callsites.py`)
Identifies indirect call expressions:
- `ptr->callback(args)`
- `handlers[i](args)`
- `(*fptr)(args)`

#### Resolver (`indirect/resolver.py`)
Uses LLM to determine likely targets for indirect calls based on:
- Matching function signatures
- Code context
- Address flow information

#### Flow Summarizer (`indirect/flow_summarizer.py`)
Uses LLM to analyze where function addresses flow to (struct fields, globals, callback registrations).

### 4. Topological Ordering (`ordering.py`)

Computes processing order using Tarjan's SCC algorithm.

**Features:**
- Identifies strongly connected components (recursive function groups)
- Orders SCCs topologically (callees before callers)
- Handles mutual recursion

**Key class:** `ProcessingOrderer`

### 5. LLM Backends (`llm/`)

Abstraction layer for different LLM providers.

**Base class:** `LLMBackend`
- `complete(prompt, system)`: Generate completion
- `complete_with_metadata(prompt, system)`: Get completion with token counts

**Implementations:**
- `ClaudeBackend`: Anthropic Claude API
- `OpenAIBackend`: OpenAI API (also works with compatible APIs)
- `OllamaBackend`: Local models via Ollama
- `LlamaCppBackend`: Local models via llama.cpp server
- `GeminiBackend`: Google Gemini API via Vertex AI

**Thread pool:** `LLMPool` (`llm/pool.py`) — `ThreadPoolExecutor` wrapper for parallel LLM queries, used by `BottomUpDriver` when `-j` > 1.

### 6. Graph Traversal Driver (`driver.py`)

Unified bottom-up traversal engine. Builds the call graph once, computes SCCs, and runs one or more summary passes over functions in topological order (callees first). All five passes (`allocation`, `free`, `init`, `memsafe`, `verify`) can run together. Sourceless stubs (stdlib functions without bodies) are skipped from all passes.

**Key classes:**
- `BottomUpDriver`: Owns graph building (cached) and SCC traversal. `run(passes, force, dirty_ids, pool)` executes all passes per function.
- `SummaryPass` (Protocol): Interface each pass implements — `get_cached()`, `summarize()`, `store()`
- `AllocationPass`: Adapter wrapping `AllocationSummarizer`
- `FreePass`: Adapter wrapping `FreeSummarizer`
- `InitPass`: Adapter wrapping `InitSummarizer`
- `MemsafePass`: Adapter wrapping `MemsafeSummarizer` (accepts `alias_builder`)
- `VerificationPass`: Adapter wrapping `VerificationSummarizer` (accepts `alias_builder`)

**Parallel execution:** When `-j N` is given (N > 1), the driver uses `LLMPool` and `orderer.get_parallel_levels()` to execute functions at the same depth in the SCC DAG concurrently. Synchronizes at level boundaries to ensure transitive dependencies are resolved.

**Incremental support:** When `dirty_ids` is provided, the driver computes the affected set (dirty functions + transitive callers via reverse edges) and only re-summarizes those; all others load from cache.

### 7. Summary Generators (`summarizer.py`, `free_summarizer.py`, `init_summarizer.py`, `memsafe_summarizer.py`, `verification_summarizer.py`)

Per-function LLM summarization logic. Each summarizer builds a prompt from the function source and callee summaries, queries the LLM, and parses the structured response.

**Key classes:**
- `AllocationSummarizer`: Allocation/buffer-size-pair analysis
- `FreeSummarizer`: Free/deallocation analysis
- `InitSummarizer`: Initialization post-condition analysis
- `MemsafeSummarizer`: Safety contract (pre-condition) analysis
- `VerificationSummarizer`: Cross-pass verification and contract simplification
- `IncrementalSummarizer`: Handles source-change invalidation, delegates re-summarization to `BottomUpDriver`
- `ExternalFunctionSummarizer` (`external_summarizer.py`): Generates summaries for functions without source code

### 8. Database (`db.py`)

SQLite storage for all analysis data.

**Tables:**
- `functions`: Function metadata, source, canonical signature, params JSON, callsites JSON
- `allocation_summaries`: Generated allocation summaries as JSON
- `free_summaries`: Generated free/deallocation summaries as JSON
- `init_summaries`: Generated initialization summaries as JSON
- `memsafe_summaries`: Generated safety contract summaries as JSON
- `verification_summaries`: Generated verification results as JSON
- `call_edges`: Call graph with callsite locations
- `address_taken_functions`: Functions whose addresses are taken
- `address_flows`: Where function addresses flow to
- `address_flow_summaries`: LLM-generated address flow analysis
- `indirect_callsites`: Indirect call expressions
- `indirect_call_targets`: Resolved indirect call targets with confidence
- `build_configs`: Project build system information
- `container_summaries`: Container/collection function detection results
- `typedefs`: Type declarations (typedef, using, struct/class/union)
- `issue_reviews`: Manual triage records for verification issues (status, reason, reviewer)

### 9. Standard Library (`stdlib.py`)

Pre-defined allocation, free, and initialization summaries for common C standard library functions.

**Allocation summaries:**
- Memory: `malloc`, `calloc`, `realloc`, `reallocarray`, `aligned_alloc`
- Strings: `strdup`, `strndup`, `asprintf`, `getline`
- Files: `fopen`, `fdopen`, `tmpfile`, `opendir`
- Memory mapping: `mmap`, `munmap`

**Free summaries:**
- `free`, `realloc`, `fclose`, `closedir`, `munmap`, `freeaddrinfo`

**Init summaries:**
- `calloc`, `memset`, `memcpy`, `memmove`, `strncpy`, `snprintf`, `strdup`, `strndup`

**Memsafe summaries:**
- `memcpy`, `memmove`, `memset`, `free`, `strlen`, `strcpy`, `strncpy`, `strcmp`, `snprintf`, `printf`, `fprintf`, `fwrite`, `fread`, `malloc`

### 10. V-Snapshot Alias Context (`vsnapshot.py`, `alias_context.py`)

Integrates whole-program pointer aliasing data from external CFL analysis (kanalyzer) into the memsafe and verification passes.

- **`VSnapshot`** (`vsnapshot.py`): Loads V-snapshot binary format — per-function alias sets showing which pointers may alias at each program point
- **`AliasContextBuilder`** (`alias_context.py`): Builds alias context sections for LLM prompts from V-snapshot data. Groups aliasing pointers and annotates which fields/parameters may point to the same memory

Used by `MemsafePass` and `VerificationPass` via the `alias_builder` parameter to improve precision of safety contract analysis.

### 11. Allocator & Container Detection (`allocator.py`, `container.py`)

Heuristic + LLM-based detection of project-specific patterns:

- **`AllocatorDetector`** (`allocator.py`): Identifies custom allocator/deallocator functions (e.g., `g_malloc`, `png_malloc`)
- **`ContainerDetector`** (`container.py`): Detects container/collection functions (e.g., list append, hash insert)

### 12. Harness Generator (`harness_generator.py`)

Generates C shim harnesses for contract-guided concolic execution via SymSan/ucsan. For a target function, the harness wraps it with a `test()` entry point and synthesizes `__dfsw_`-prefixed stubs for all callees so that SymSan's taint tracking can propagate through them.

**Key features:**
- LLM-generated shim: reads the function's memsafe contracts and post-conditions, asks the LLM to write a matching `test()` body and callee stubs
- Compilation loop: when `--ko-clang-path` is set, compiles the shim against project bitcode and iterates with LLM to fix errors (up to 3 attempts)
- Plan generation (`--plan` / `--plan-only`): instruments the target binary with basic-block IDs and asks the LLM to produce an exploration plan (sequence of BB targets) for the Thoroupy policy scheduler
- Issue assessment (`--assess-issue N`): injects a targeted assertion for verification issue N into an existing shim, rebuilds, and runs ucsan to confirm or refute the issue
- Outputs per-function `.c`, `.bc`, `.sh`, `.ucsan.cfg`, `.abilist`, and optionally `plan.json`

**Key class:** `HarnessGenerator`

See [ucsan-harness.md](ucsan-harness.md) for the full workflow.

### 13. Build-Learn System (`builder/`)

LLM-driven incremental build system that can configure, build, and learn from C/C++ projects.

**Key classes:**
- `Builder` (`builder.py`): ReAct-loop agent that iteratively configures and builds projects using LLM tool calls
- `AssemblyChecker` (`assembly_checker.py`): Detects standalone and inline assembly in build artifacts; supports iterative minimization
- `ErrorAnalyzer` (`error_analyzer.py`): Parses build failures for actionable diagnostics
- `ScriptGenerator` (`script_generator.py`): Auto-generates reproducible build scripts

Supports CMake, Autotools, Meson, Bazel, SCons, and custom build systems. Assembly detection scans `compile_commands.json`, source files, and LLVM IR. See [build-learn.md](build-learn.md) for details.

### 14. Link-Unit Pipeline (`link_units/`)

Batch analysis pipeline aware of build targets (executables, libraries). One DB per link unit; targets processed in dependency order.

- **`LinkUnitDiscoverer`** (`discoverer.py`): ReAct agent that explores build artifacts to identify all link units and their dependency relationships
- **`Pipeline`** (`pipeline.py`): Orchestrates per-target extract → summarize workflows
- **`batch_call_graph_gen.py`**: Runs two-phase compositional KAMain per target in topo order
- **`batch_summarize.py`**: Runs init-stdlib, import-dep-summaries, and summarization per target

See [link-unit-analysis.md](link-unit-analysis.md) for the full design.

### 15. CLI (`cli.py`)

Command-line interface using Click.

**Commands:**
- `summarize`: Generate summaries (`--type allocation|free|init|memsafe|verify`, `-j N` for parallel)
- `extract`: Function and call graph extraction only
- `callgraph`: Export call graph
- `show`: Display summaries
- `lookup`: Look up specific function
- `stats`: Database statistics
- `export`: Export to JSON
- `init-stdlib`: Add stdlib summaries
- `clear`: Clear database
- `indirect-analyze`: Resolve indirect calls via LLM
- `show-indirect`: Display indirect call analysis results
- `container-analyze`: Detect container/collection functions
- `show-containers`: Display container detection results
- `find-allocator-candidates`: Identify custom allocator functions
- `scan`: Comprehensive analysis using `compile_commands.json` (link-unit aware)
- `build-learn`: Incremental project builder with LLM-driven ReAct loop
- `generate-kanalyzer-script`: Generate kanalyzer analysis script
- `import-callgraph`: Import external call graph (e.g., from kanalyzer)
- `discover-link-units`: Detect build targets/link units
- `import-dep-summaries`: Import summaries from dependency databases
- `gen-harness`: Generate C shims for contract-guided concolic execution (SymSan/ucsan)
- `show-issues`: List verification issues with review status filters
- `review-issue`: Triage a verification issue (confirmed / false_positive / wontfix)

## End-to-End Pipeline

The full pipeline for analyzing a project has seven phases. Each phase corresponds to one or more CLI commands. For link-unit projects, phases 2–6 run per target in dependency order (libraries before executables).

See [link-unit-analysis.md](link-unit-analysis.md) for the full link-unit design and cross-project dependency handling.

### Phase 0: Build (`build-learn`)

LLM-driven ReAct agent that configures and builds the project inside a Docker container with LLVM 18. Produces `compile_commands.json` and `.bc` bitcode files (via `-flto=full -save-temps=obj`). Generates a reusable build script in `build-scripts/<project>/`.

See [build-learn.md](build-learn.md) for details.

**Output:** `compile_commands.json`, `.bc` files, `build.sh`, `config.json`

### Phase 1: Discover Link Units (`discover-link-units`)

ReAct agent that explores build artifacts to identify all link units (libraries, executables) and their intra-project dependency relationships. Parses `build.ninja`, `link.txt`, archives (`ar t`), and ELF headers.

**Output:** `func-scans/<project>/link_units.json`

### Phase 2: Scan Functions (`scan`)

Per-target function extraction using libclang. Parses source files from `compile_commands.json`, extracts function definitions, builds the AST-based call graph, and identifies address-taken functions and indirect callsites.

When `--link-units` and `--target` are given, restricts extraction to source files belonging to the named target.

**Output:** `functions.db` per target (functions, call_edges, address_taken_functions, indirect_callsites)

### Phase 3: Call Graph (`batch_call_graph_gen.py` / KAMain)

LLVM IR-based CFL-reachability points-to analysis via KAMain. The batch script locates `.bc` files using a three-tier strategy:

1. Look for `.bc` next to `.o` (from `-save-temps=obj`)
2. Use `.o` as bitcode directly if `-flto` was used
3. Recompile sources with `-emit-llvm` to produce `.bc`

Two KAMain sub-phases per target:

1. **Compress** — produces a compressed CFL constraint graph (`.cflcg`) from the target's `.bc` files
2. **Compose + solve** — composes the target's constraint graph with those of its transitive deps to produce the call graph (`.json`) and V-snapshot (`.vsnap`)

Processes targets in topological order (deps before dependents). Writes `db_path`, `callgraph_json`, `cflcg`, `vsnapshot` back into `link_units.json`. Also extracts allocator candidates from `functions.db` for KAMain.

See [indirect-call-analysis.md](indirect-call-analysis.md) for details on indirect call resolution.

**Output:** `callgraph.json`, `<target>.cflcg`, `<target>.vsnap` per target

### Phase 4: Import Call Graph (`import-callgraph`)

Imports KAMain JSON call graph into `functions.db`. Matches KAMain function entries to existing DB functions by name, file+name, or suffix. Creates stubs for unmatched external functions (libc, deps). Run automatically by `batch_call_graph_gen.py` after KAMain completes.

**Output:** `call_edges` table populated in `functions.db`

### Phase 5: Seed Summaries (`init-stdlib`, `import-dep-summaries`)

Before summarization, populate the target DB with pre-existing summaries so the bottom-up driver treats them as cache hits:

1. **`init-stdlib`** — inserts pre-defined summaries for C standard library functions (malloc, free, memcpy, etc.)
2. **`import-dep-summaries`** — copies function stubs and all summary types from dependency DBs (e.g., zlib summaries imported when analyzing libpng), tagged with `model_used="dep:<project>/<target>"`

**Output:** stdlib and dependency function stubs + summaries in `functions.db`

### Phase 6: Summarize (`summarize`)

Bottom-up LLM summarization via `BottomUpDriver`. Builds the call graph once, computes SCCs via Tarjan's algorithm, and traverses in topological order (callees first). Runs one or more summary passes per function:

| Pass | Type | Direction | What it captures |
|------|------|-----------|------------------|
| 1 | `allocation` | Post-condition | Allocations, buffer-size pairs, may-be-null |
| 2 | `free` | Post-condition | Deallocations, conditional frees, nulled-after |
| 3 | `init` | Post-condition | Guaranteed initializations (caller-visible) |
| 4 | `memsafe` | Pre-condition | Safety contracts (not-null, buffer-size, etc.) |
| 5 | `verify` | Cross-pass | Issues + simplified contracts |

Passes 1–4 are independent and can run together. Pass 5 requires passes 1–4 to exist. Optional `--vsnap` provides alias context for passes 4–5 (see [vsnapshot-alias-context.md](vsnapshot-alias-context.md)).

Functions with existing summaries (stdlib, deps, prior runs) are cache hits. Only new/dirty functions get LLM calls. Parallel execution across SCC levels with `-j N`.

**Output:** `allocation_summaries`, `free_summaries`, `init_summaries`, `memsafe_summaries`, `verification_summaries` in `functions.db`

### Phase 7: Issue Triage (`show-issues`, `review-issue`)

After verification, analysts review flagged issues. `show-issues` lists all `SafetyIssue` records from `verification_summaries` with their current review status. `review-issue` updates the `issue_reviews` table for a specific issue:

- **confirmed** — real bug, ready for downstream use
- **false_positive** — LLM hallucination or infeasible path
- **wontfix** — acknowledged but intentionally not fixed

**Output:** `issue_reviews` records in `functions.db`

### Phase 8 (optional): Harness Generation (`gen-harness`)

For issues of interest, generate a C shim harness to drive contract-guided concolic execution with SymSan/ucsan. The harness wraps the target function's contracts and can be used with the Thoroupy policy scheduler for path exploration.

See [ucsan-harness.md](ucsan-harness.md) for details.

**Output:** `harnesses/<project>/<func>.*` — shim, bitcode, build script, ucsan config

### Batch Processing

Batch scripts under `scripts/` orchestrate the pipeline across multiple projects and link-unit targets. Projects are read from `gpr_projects.json` and filtered by tier, name, or skip list.

| Script | Phase | What it does |
|--------|-------|-------------|
| `batch_build_learn.py` | 0 | Runs `build-learn` for each project sequentially |
| `batch_scan_targets.py` | 2 | Extracts functions, address-taken, indirect callsites (`-j` parallel) |
| `batch_call_graph_gen.py` | 3–4 | Runs KAMain (compositional CFL) + imports call graph per target |
| `batch_summarize.py` | 5–6 | Seeds stdlib/dep summaries, runs passes 1–3 then pass 4 separately |
| `batch_verify.py` | 6 | Runs pass 5 (verification) after passes 1–4 complete |
| `batch_container_detect.py` | aux | Detects container functions (heuristic + LLM) |
| `cgc_run.sh` | benchmark | Full CGC benchmark: extract GT, scan, verify, evaluate |

All link-unit-aware scripts read `link_units.json` and toposort targets by `link_deps`. `batch_summarize.py` runs `import-dep-summaries` from intra-project dep DBs before each target's summarization. Cross-project dependencies are tracked in `project_deps.json`.

See [cgc-benchmark.md](cgc-benchmark.md) for the CGC benchmark pipeline.

**Supporting scripts:**
- `gpr_utils.py` — shared utilities (Docker path translation, project discovery)
- `clone_gpr_projects.py` — clone projects from `gpr_projects.json` URLs
- `update_gpr_projects.py` — git pull and update project metadata
- `guess_language.py` — LLM-based language detection, auto-demotes non-C/C++ to tier 3
- `fix_compile_commands.py` — fix Docker/container paths in `compile_commands.json`
- `batch_rebuild.py` — re-run build scripts (e.g., with debug flags)

### Example: libpng depends on zlib

```
Step 1: Analyze zlib (no deps)
  build-learn              → compile_commands.json, .bc files
  discover-link-units      → zlib/link_units.json
  scan --target zlibstatic → zlib/zlibstatic/functions.db
  KAMain phase 1           → zlib/zlibstatic/zlibstatic.cflcg
  KAMain phase 2           → zlib/zlibstatic/callgraph.json, .vsnap
  import-callgraph         → call_edges in functions.db
  init-stdlib              → stdlib stubs
  summarize alloc+free+init → post-condition summaries
  summarize memsafe         → pre-condition contracts
  summarize verify          → cross-pass verification

Step 2: Analyze libpng (depends on zlib)
  build-learn              → compile_commands.json, .bc files
  discover-link-units      → libpng/link_units.json
  scan --target libpng16   → libpng/libpng16/functions.db
  KAMain phase 1           → libpng/libpng16/libpng16.cflcg
  KAMain phase 2 (+zlib)   → libpng/libpng16/callgraph.json, .vsnap
  import-callgraph         → call_edges
  import-dep-summaries     → zlib summaries copied in
  init-stdlib              → stdlib stubs
  summarize alloc+free+init → post-condition summaries (zlib → cache hit)
  summarize memsafe         → pre-condition contracts
  summarize verify          → cross-pass verification
```

### Directory Layout

```
func-scans/
  zlib/
    link_units.json
    allocator_candidates.json
    zlibstatic/
      functions.db
      callgraph.json
      zlibstatic.cflcg
      zlibstatic.vsnap
    zlib_static_example/
      functions.db
      callgraph.json
  libpng/
    link_units.json
    libpng16/
      functions.db
      callgraph.json
      libpng16.cflcg
      libpng16.vsnap
```

### Incremental Updates

When source files change:

1. Compute new source hash
2. Compare with stored hash
3. If changed:
   - Invalidate function's summary
   - Cascade invalidation to all callers (transitive)
4. Re-analyze only invalidated functions (dirty set + reverse-edge closure)

## Memory Safety Analysis Framework

The system uses a multi-pass, Hoare-logic-inspired approach to check memory safety. Post-condition passes (1-3) summarize what each function *produces*. The pre-condition pass (4) summarizes what each function *requires*. The verification pass (5) checks that post-conditions satisfy pre-conditions at each call site.

All passes are bottom-up (callees before callers) and independent of each other except where noted.

### Pass 1: Allocation Summary (post-condition) — existing

Captures memory allocations and buffer-size pairs produced by each function.

- What gets allocated (heap/stack/static), via which allocator, size expression
- Which parameters affect allocation size
- Buffer-size pairs established: `(buffer, size)` relationships produced by the function
- Supports project-specific allocators via `--allocator-file`

**Summarizer:** `AllocationSummarizer`

### Pass 2: Free Summary (post-condition) — implemented

Captures which buffers get freed by each function.

- **Target**: what gets freed (`ptr`, `info_ptr->palette`, `row_buf`)
- **Target kind**: `parameter`, `field`, `local`, or `return_value`
- **Deallocator**: `free`, `png_free`, `g_free`, or project-specific
- **Conditional**: whether the free is inside an if/error path
- **Nulled after**: whether the pointer is set to NULL after free
- Supports project-specific deallocators via `--deallocator-file`

Feeds temporal safety checks (use-after-free, double-free).

**Summarizer:** `FreeSummarizer` (`free_summarizer.py`)
**DB table:** `free_summaries`
**CLI:** `llm-summary summarize --type free`

### Pass 3: Initialization Summary (post-condition) — implemented

Captures what each function **always** initializes on all non-error exit paths (caller-visible only).

- **Target**: what gets initialized (`*out`, `ctx->data`, `return value`)
- **Target kind**: `parameter` (output param), `field` (struct field via param), or `return_value`
- **Initializer**: how it's initialized (`memset`, `assignment`, `calloc`, `callee:func_name`)
- **Byte count**: how many bytes (`n`, `sizeof(T)`, `full`, or null)

Only unconditional, guaranteed initializations visible to the caller. Local variables are excluded (not a post-condition). Feeds uninitialized-use checks (Pass 5).

**Summarizer:** `InitSummarizer` (`init_summarizer.py`)
**DB table:** `init_summaries`
**CLI:** `llm-summary summarize --type init`

### Pass 4: Safety Contracts (pre-condition) — implemented

Captures what contracts must hold for safe execution of each function. This is the *requirement* side — what callers must guarantee.

- **Not-null contracts** (`not_null`): pointer parameters that are dereferenced must not be NULL
- **Not-freed contracts** (`not_freed`): pointers passed to free/dealloc must point to live memory
- **Buffer-size contracts** (`buffer_size`): pointers used in memcpy/indexing must have sufficient capacity (includes `size_expr` and `relationship`)
- **Initialized contracts** (`initialized`): variables/fields used in deref, branch, or index must be initialized

Callee contracts that a function does NOT satisfy internally are propagated as the function's own contracts.

Note: uninitialized *read* into a variable is benign; uninitialized *use* (dereference, branch, index) is the safety issue.

**Summarizer:** `MemsafeSummarizer` (`memsafe_summarizer.py`)
**DB table:** `memsafe_summaries`
**CLI:** `llm-summary summarize --type memsafe`

### Pass 5: Verification & Contract Simplification — implemented

Cross-pass verification that checks post-conditions against pre-conditions at each call site. For each function, the verifier:

1. **Internal safety check** — does the function itself perform unsafe operations?
2. **Callee pre-condition satisfaction** — at each call site, are the callee's memsafe contracts satisfied?
3. **Contract simplification** — removes Pass 4 contracts that the function satisfies internally, keeping only contracts that must propagate to callers.
4. **Issue reporting** — unsatisfied pre-conditions become `SafetyIssue` findings with severity levels.

The verifier queries the DB directly for Passes 1-3 callee post-conditions and Pass 4 raw contracts (cross-pass data), while receiving `VerificationSummary` callee summaries from the driver for already-verified callees (simplified contracts).

| Safety class | Post-condition passes | Pre-condition (pass 4) |
|---|---|---|
| Buffer overflow | 1 (allocation size) | buffer-size contracts |
| Null dereference | 1 (may_be_null) | not-null contracts |
| Use-after-free | 2 (what's freed) | not-freed contracts |
| Double free | 2 (what's freed) | not-freed contracts |
| Uninitialized use | 3 (what's initialized) | must-be-initialized contracts |

Issue severity: **high** (definite violation), **medium** (depends on caller), **low** (unlikely/defensive).

**Summarizer:** `VerificationSummarizer` (`verification_summarizer.py`)
**DB table:** `verification_summaries`
**CLI:** `llm-summary summarize --type verify`

**Dependencies:** Passes 1-4 are independent and run together in a single `BottomUpDriver` traversal when multiple `--type` flags are given. Pass 5 requires all four prior passes to exist (prerequisite check in CLI).

## Design Decisions

### Why libclang?

- Accurate parsing of C/C++ including macros and templates
- Provides full AST access for call extraction
- Handles complex preprocessor directives
- Industry-standard tool

### Why SQLite?

- Single-file database, easy to manage
- Supports complex queries for lookups
- ACID transactions for data integrity
- No external server required

### Why bottom-up analysis?

- Callee summaries provide context for caller analysis
- Enables compositional reasoning
- Reduces redundant LLM calls
- Matches how humans understand code

### Why JSON for summaries?

- Flexible schema evolution
- Easy to parse and generate
- Human-readable for debugging
- LLMs handle JSON well
