# Bug Triage Agent

Five-stage pipeline that proves memory safety issues (from verification
Pass 5) are either **safe** (cannot manifest) or **feasible** (reachable
with concrete inputs from an entry function). Uses `complete_with_tools()`
with any LLM backend, fresh context per stage.

## Origin: pbfuzz Lessons

The agent descends from the pbfuzz bug triage agent (`~/fuzzing/pbfuzz/`).
Key takeaway: without strict phase enforcement, the agent loops on analysis
or jumps to conclusions. The multi-stage pipeline solved this.

### Reused from pbfuzz

- **Workflow state machine** — stage separation prevents agent from getting stuck
- **Structured memory** — persistent state across ReAct iterations
- **Prompt structure** — static context (project config) separated from
  dynamic state (workflow phase)

### Replaced

| pbfuzz | llm-summary |
|---|---|
| Claude Agent SDK | `complete_with_tools()` ReAct loop (any backend) |
| MCP call_graph/search/corpus servers | `functions.db` queries (in-process) |
| MCP workflow server | Three-stage pipeline orchestrator |
| MCP fuzzer server | ucsan / standalone harness runner |
| MCP protocol layer | Direct Python tool functions |
| cursor-cli | LLM backend abstraction |
| Single-stage local reasoning | Path-feasibility + doc audit + verdict |

## Architecture

```
Verification Pass (existing)
  | SafetyIssue[] (in functions.db)
  v
TriageAgent (triage.py) — five-stage pipeline
  |
  | Stage 1: REACHABILITY (fresh LLM, 30 turns)
  |   Is the bug reachable from an entry function?
  |   Are path constraints (guards, sanitizers) satisfiable?
  |   Tools: get_entry_functions, get_reachability_path,
  |          get_call_chain_contracts, read_function_source,
  |          get_callers, get_callees, get_contracts, get_issues
  |   → submit_reachability
  |
  |   unreachable/infeasible → SAFE (skip stages 2-5)
  |
  | Stage 2: DOC AUDIT (fresh LLM, 25 turns)
  |   Do project docs or inline comments mitigate the issue?
  |   Tools: git_show, git_grep, git_ls_tree,
  |          read_function_source, get_contracts
  |   → submit_doc_audit
  |
  |   mitigated → SAFE (skip stages 3-5)
  |
  | Stage 3: VERDICT (fresh LLM, 35 turns)
  |   Produce entry-level proof: safe, contract_gap, or feasible
  |   Tools: read_function_source, get_callers, get_callees,
  |          get_contracts, get_issues, verify_contract
  |   → submit_verdict
  v
TriageResult[]
  | Per-issue: hypothesis, reasoning, entry_function,
  | reachability_chain, path_constraints, data_flow_trace,
  | doc_audit_searched, feasible_path, relevant_functions
  |
  | hypothesis=safe/contract_gap → update contracts, mark FP
  | hypothesis=feasible → Stage 4 (if --build-script-dir set)
  v
Stage 4: WITNESS (optional, fresh LLM)
  | Generate self-contained ASan/UBSan unit test
  | Calls ENTRY function with bug-triggering inputs
  | Compile with clang -fsanitize=address,undefined in Docker
  |
  | ASan triggered → confirmed bug (done)
  | Ran clean → Stage 5
  v
Stage 5: DEBUG (optional, fresh LLM × 2)
  | Generate GDB batch script from path constraints
  | Run test under gdb -batch in Docker
  | Diagnose: path_blocked | not_reached | asan_limitation
  | If path_blocked → flip to safe, update contracts
  v
test_<func>_<idx>.c + _gdb_<test>.gdb + corrected verdict
```

## Three Stages

### Stage 1: REACHABILITY

Checks structural reachability AND path feasibility:

- **Structural**: Uses `get_entry_functions` to find interface functions
  (no callers in call graph), then `get_reachability_path` for BFS to
  the bug site.
- **Branch guards**: Reads source along the path, checks if conditionals
  (null checks, bounds checks, state guards) prevent the path.
- **Data flow**: Traces how the triggering parameter flows from entry
  input through each function. Checks if any function sanitizes/clamps
  the value.

**Short-circuit**: If unreachable or path constraints block the trigger,
returns `safe` immediately — stages 2-3 are skipped.

### Stage 2: DOC AUDIT

Only runs if stage 1 says reachable. Searches:

- **Project docs**: README, manual, header comments, examples, `docs/`
- **Inline comments**: Source code along the call chain for
  developer-documented invariants

Lean strict: vague mentions ("pass valid data") do NOT count. Only
clear constraints that prevent the bug qualify as mitigation.

**Short-circuit**: If mitigated, returns `safe` — stage 3 is skipped.

### Stage 3: ENTRY-LEVEL VERDICT

Only runs if not mitigated. Produces the final proof:

- **Safety proof**: Updated contracts showing the issue cannot manifest
- **Contract gap**: Strengthened requires (verified via `verify_contract`)
- **Feasibility proof**: Call chain from ENTRY function to bug site,
  with entry-level harness data

Key difference from prior design: `relevant_functions` = full call chain
from entry to bug site. `validation_plan` uses the entry function as
test entry point. Harness tests real-world triggering paths.

### Stage 4: WITNESS (optional)

Only runs for feasible verdicts when `--build-script-dir` is set.
Generates a self-contained C unit test that reproduces the bug:

- **Target**: Entry/interface function (from Stage 1), not the internal
  buggy function
- **Output**: `witness_<func>_<idx>.c` — standalone C program with `main()`
- **Build**: Docker-based, same container as project build, compiled with
  `clang -fsanitize=address,undefined`
- **Detection**: ASan/UBSan catches the memory safety violation at runtime
- **LLM-generated setup**: The LLM writes all state initialization
  (struct allocation, file creation, etc.) from the feasibility context
- **Compile-fix loop**: Up to 3 LLM attempts to fix compilation errors

### Stage 5: DEBUG (optional)

Only runs when Stage 4 produced a witness that compiled and ran **clean**
(no ASan/UBSan reports). A clean run of a "feasible" bug is suspicious
and triggers automated diagnosis:

1. **GDB script generation** (LLM call 1) — reads path constraints,
   data flow trace, and source code to produce a `gdb -batch` script
   with breakpoints at key control flow points
2. **Docker GDB run** — executes the test under GDB inside Docker
   (with `--cap-add=SYS_PTRACE`), capturing variable values at each
   breakpoint
3. **Diagnosis** (LLM call 2) — analyzes GDB output against the
   original reasoning to determine why ASan didn't trigger:
   - **path_blocked**: A guard/check in the caller prevents the bug
     condition (false positive) → flip verdict to safe, update contracts
   - **not_reached**: Test doesn't exercise the right path (bad test)
     → keep feasible, test needs improvement
   - **asan_limitation**: Bug IS reachable but ASan can't detect it
     (e.g., intra-object overflow) → keep feasible with note

When the diagnosis is `path_blocked`, the verdict flips from "feasible"
to "safe" and the caller's guard is recorded as a contract ensuring the
callee's precondition.

### Three Outcomes

**Safety proof** — the issue cannot manifest:
- Updated/new contracts that prove the property holds
- Path constraints or data-flow sanitization prevent the trigger

**Contract gap** — function's requires is too weak:
- Strengthened requires verified via trial verification
- Propagated from callee requirements

**Feasibility proof** — the issue is reachable:
- Concrete call chain from entry function to bug site
- Input assumptions at the entry interface that trigger it
- Entry-level harness for symbolic validation

## Data Models

```python
@dataclass
class TriageResult:
    function_name: str
    issue_index: int
    issue: SafetyIssue
    hypothesis: str              # "safe", "contract_gap", or "feasible"
    reasoning: str               # natural language proof

    # Safety proof
    updated_contracts: list[dict]

    # Feasibility proof
    feasible_path: list[str]     # call chain from ENTRY to bug site

    # For symbolic validation (ucsan)
    assumptions: list[str]
    assertions: list[str]
    relevant_functions: list[str]
    validation_plan: list[dict]

    # Enhanced: reachability and doc audit
    entry_function: str          # entry function name
    reachability_chain: list[str]  # full path from entry to bug
    path_constraints: list[str]  # guards/sanitizers found
    data_flow_trace: str         # how bug parameter flows
    doc_audit_searched: str      # audit trail
    doc_mitigated: bool          # True if docs mitigate

    # Stage 4-5
    witness_path: str            # path to generated witness C file
    debug_diagnosis: str         # path_blocked | not_reached | asan_limitation
    debug_explanation: str       # diagnosis detail

@dataclass
class ReachabilityVerdict:
    reachable: bool
    entry_function: str
    call_chain: list[str]
    path_constraints: list[str]
    data_flow_trace: str
    reasoning: str
    summarizer_gap: bool = False

@dataclass
class DocAuditVerdict:
    mitigated: bool
    doc_searched: str
    doc_quote: str = ""
    mitigating_constraint: str = ""
```

## Tools

### Stage 1 Tools

| Tool | Purpose |
|---|---|
| `get_entry_functions` | Find functions with no callers (interface boundary) |
| `get_reachability_path` | BFS shortest path from entry to bug site |
| `get_call_chain_contracts` | Batch-read contracts for a call chain |
| `read_function_source` | Read function source from DB |
| `get_callers` | Find caller functions |
| `get_callees` | Find callee functions |
| `get_contracts` | Read Hoare-style contracts |
| `get_issues` | Read verification issues |
| `submit_reachability` | Terminal: submit reachability verdict |

### Stage 2 Tools

| Tool | Purpose |
|---|---|
| `read_function_source` | Read source (for inline comments) |
| `get_contracts` | Re-confirm contracts |
| `git_show` | Read project files |
| `git_grep` | Search project files |
| `git_ls_tree` | List project directory |
| `submit_doc_audit` | Terminal: submit doc audit verdict |

### Stage 3 Tools

| Tool | Purpose |
|---|---|
| `read_function_source` | Read function source |
| `get_callers` / `get_callees` | Call graph navigation |
| `get_contracts` / `get_issues` | Contract and issue data |
| `verify_contract` | Trial-verify proposed contracts |
| `git_show` / `git_grep` / `git_ls_tree` | Source inspection |
| `submit_verdict` | Terminal: submit final verdict |

## Witness Generation & Debug (Stages 4-5)

When `--build-script-dir` is provided, triage automatically generates
ASan/UBSan witnesses for feasible verdicts and debugs false positives:

```
llm-summary triage --db <db> -f func --build-script-dir build-scripts/zlib/ -v
  ↓
Stages 1-3 produce TriageResult with hypothesis="feasible"
  ↓
Stage 4: _stage_witness()
  1. Read entry function + feasible path source from DB
  2. LLM generates standalone C test calling ENTRY function
  3. Write test_<func>_<idx>.c + build script
  4. Compile in Docker with clang -fsanitize=address,undefined
  5. If compile fails, feed errors to LLM for fix (up to 3 attempts)
  ↓ (if test ran clean — no ASan reports)
Stage 5: _stage_debug()
  1. LLM generates GDB batch script with breakpoints at path constraints
  2. Run test under gdb -batch in Docker
  3. LLM diagnoses from GDB output: path_blocked | not_reached | asan_limitation
  4. If path_blocked → flip verdict to safe, update caller contracts
```

### Legacy: ucsan symbolic validation

`gen-harness --validate` is still available for ucsan-based symbolic
validation with ko-clang. The witness approach (Stage 4) is simpler
and doesn't require ucsan infrastructure.

## Git Tools Integration

When `--project-path` is provided, stages 2 and 3 gain access to the
project repository via `GitTools`:

- `git_show` — read tracked files at any ref
- `git_grep` — search file contents
- `git_ls_tree` — list directory structure

These use git plumbing commands with `--` separators and input validation
for injection prevention. Shared with other agents via `git_tools.py`.

When `--project-path` is omitted, stage 2 (doc audit) is skipped entirely.

## CLI Usage

```bash
# Triage all high-severity issues
llm-summary triage --db func-scans/libpng/functions.db \
  --severity high --backend claude -v

# Triage specific function
llm-summary triage --db func-scans/zlib/functions.db \
  -f deflate -v

# Triage specific issue by index
llm-summary triage --db func-scans/zlib/functions.db \
  -f gz_write --issue-index 0 -v

# With git tools for doc audit + source inspection
llm-summary triage --db func-scans/libpng/functions.db \
  -f png_read_row --project-path /data/csong/opensource/libpng -v

# Save results for validation
llm-summary triage --db func-scans/zlib/functions.db \
  -f deflate -o verdict.json --backend gemini

# Triage with witness generation (Stage 4)
llm-summary triage --db func-scans/zlib/functions.db \
  -f deflate --build-script-dir build-scripts/zlib/ \
  -d harnesses/zlib/ -v

# Legacy: ucsan symbolic validation
llm-summary gen-harness --db func-scans/zlib/functions.db \
  --validate verdict.json --ko-clang-path ~/fuzzing/ucsan/ko-clang \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json
```

### CLI Options

```
llm-summary triage --db <path> [options]

Required:
  --db PATH                      Database file

Optional:
  -f, --function NAME...         Function(s) to triage (default: all with issues)
  --severity {high|medium|low}   Filter by issue severity
  --issue-index N                Triage specific issue (requires single -f)
  --backend {claude|openai|ollama|llamacpp|gemini}  (default: claude)
  --model STR                    Model override
  --project-path PATH            Enable doc audit + git tools
  -o, --output PATH              JSON output file (default: summary to stdout)
  -v, --verbose                  Print detailed logs
  --disable-thinking             Disable extended thinking
  --llm-host, --llm-port         For local backends
  --build-script-dir PATH        build-scripts/<project>/ dir (enables Stage 4 witness)
  -d, --harness-dir PATH         Output dir for witnesses (default: harnesses/<project>/)
```

## Key Files

| File | Purpose |
|---|---|
| `src/llm_summary/triage.py` | TriageAgent orchestrator, five stages (reachability, doc audit, verdict, witness, debug), system prompts |
| `src/llm_summary/agent_tools.py` | Tool definitions (DB read, triage, enhanced triage), ToolExecutor |
| `src/llm_summary/git_tools.py` | GitTools class, git_show/git_grep/git_ls_tree |
| `src/llm_summary/harness_generator.py` | validate_triage(), TRIAGE_VALIDATE_PROMPT |
| `src/llm_summary/models.py` | SafetyIssue, VerificationSummary |
| `src/llm_summary/code_contract/checker.py` | find_entry_functions(), check_entries() |
| `src/llm_summary/cli.py` | `triage` subcommand, `gen-harness --validate` |
