# Bug Triage Agent

Three-stage pipeline that proves memory safety issues (from verification
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
TriageAgent (triage.py) — three-stage pipeline
  |
  | Stage 1: REACHABILITY (fresh LLM, 30 turns)
  |   Is the bug reachable from an entry function?
  |   Are path constraints (guards, sanitizers) satisfiable?
  |   Tools: get_entry_functions, get_reachability_path,
  |          get_call_chain_contracts, read_function_source,
  |          get_callers, get_callees, get_contracts, get_issues
  |   → submit_reachability
  |
  |   unreachable/infeasible → SAFE (skip stages 2-3)
  |
  | Stage 2: DOC AUDIT (fresh LLM, 25 turns)
  |   Do project docs or inline comments mitigate the issue?
  |   Tools: git_show, git_grep, git_ls_tree,
  |          read_function_source, get_contracts
  |   → submit_doc_audit
  |
  |   mitigated → SAFE (skip stage 3)
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
  v
gen-harness --validate (existing, enhanced)
  | Reads verdict, generates ucsan harness per ENTRY function
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

## Validation Pipeline

Triage results feed into `gen-harness --validate` for symbolic confirmation:

```
llm-summary triage --db <db> -f func -o verdict.json
  ↓
llm-summary gen-harness --db <db> --validate verdict.json
  ↓
For each verdict:
  1. Extract relevant_functions, validation_plan
  2. Find entry functions (no callers within relevant set)
  3. Generate C harness via TRIAGE_VALIDATE_PROMPT
  4. Compile with ko-clang → .ucsan binary
  5. Run ucsan for symbolic validation
```

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

# Validate triage results with ucsan harness
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
```

## Key Files

| File | Purpose |
|---|---|
| `src/llm_summary/triage.py` | TriageAgent orchestrator, three stage agents, system prompts, ReachabilityVerdict, DocAuditVerdict |
| `src/llm_summary/agent_tools.py` | Tool definitions (DB read, triage, enhanced triage), ToolExecutor |
| `src/llm_summary/git_tools.py` | GitTools class, git_show/git_grep/git_ls_tree |
| `src/llm_summary/harness_generator.py` | validate_triage(), TRIAGE_VALIDATE_PROMPT |
| `src/llm_summary/models.py` | SafetyIssue, VerificationSummary |
| `src/llm_summary/code_contract/checker.py` | find_entry_functions(), check_entries() |
| `src/llm_summary/cli.py` | `triage` subcommand, `gen-harness --validate` |
