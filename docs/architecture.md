# Architecture Overview

This document describes the architecture of the LLM-based memory allocation summary analysis tool.

## System Overview

The tool performs compositional, bottom-up analysis of C/C++ code to generate memory allocation summaries. It processes functions in dependency order (callees before callers) so that callee summaries are available when analyzing callers.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Source Files   │────▶│ Function        │────▶│ Call Graph      │
│  (.c/.cpp/.h)   │     │ Extractor       │     │ Builder         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Summary         │◀────│ LLM Summary     │◀────│ Topological     │
│ Database        │     │ Generator       │     │ Ordering (SCCs) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
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

### 12. Build-Learn System (`builder/`)

LLM-driven incremental build system that can configure, build, and learn from C/C++ projects.

**Key classes:**
- `Builder` (`builder.py`): ReAct-loop agent that iteratively configures and builds projects using LLM tool calls
- `AssemblyChecker` (`assembly_checker.py`): Detects standalone and inline assembly in build artifacts; supports iterative minimization
- `ErrorAnalyzer` (`error_analyzer.py`): Parses build failures for actionable diagnostics
- `ScriptGenerator` (`script_generator.py`): Auto-generates reproducible build scripts

Supports CMake and Autotools projects. Assembly detection scans `compile_commands.json`, source files, and LLVM IR.

### 13. Link-Unit Pipeline (`link_units/`)

Batch analysis pipeline aware of build targets (executables, libraries).

- **`LinkUnitDiscoverer`** (`discoverer.py`): Detects link units from `compile_commands.json` or build system
- **`Pipeline`** (`pipeline.py`): Orchestrates per-target extract → summarize workflows
- Enables dependency-aware analysis where shared libraries are analyzed before executables that link them

### 14. CLI (`cli.py`)

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

## Data Flow

### Analysis Pipeline

```
1. Source Files
   │
   ▼
2. Function Extraction (libclang)
   │
   ├──▶ Functions stored in DB
   │
   ▼
3. Call Graph Construction
   │
   ├──▶ Direct calls extracted from AST
   ├──▶ Indirect callsites identified
   ├──▶ Address-taken functions found
   │
   ▼
4. Indirect Call Resolution (LLM)
   │
   ├──▶ Candidates filtered by signature
   ├──▶ LLM determines likely targets
   │
   ▼
5. BottomUpDriver (driver.py)
   │
   ├──▶ Build call graph + compute SCCs (once)
   ├──▶ Traverse in topological order (callees first)
   ├──▶ Run all registered passes per function:
   │      AllocationPass, FreePass, InitPass, MemsafePass, VerificationPass
   ├──▶ Optional: parallel execution across SCC levels (-j N)
   │
   ▼
6. Summary Generation (LLM, per pass)
   │
   ├──▶ Gather callee summaries from prior results
   ├──▶ Build prompt, query LLM, parse response
   ├──▶ Store result in DB
   │
   ▼
7. Summary Database
```

### Incremental Updates

When source files change:

1. Compute new source hash
2. Compare with stored hash
3. If changed:
   - Invalidate function's summary
   - Cascade invalidation to all callers
4. Re-analyze invalidated functions

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
