"""Bug triage agent: three-stage pipeline for path-feasibility-aware triage.

Three-stage decoupled pipeline, each with a fresh LLM context:

1. **Path feasibility** (REACHABILITY stage): Check if the bug is
   structurally reachable from an entry function AND if the triggering
   condition is satisfiable given path constraints (branch guards,
   data-flow sanitization).

2. **Doc audit** (DOC_AUDIT stage): Only if reachable. Search project
   docs AND inline source comments for constraints that mitigate the
   issue. Lean strict.

3. **Entry-level verdict** (VERDICT stage): Only if not mitigated.
   Produce a feasibility proof with entry-level harness data, or
   identify a contract gap / safety proof.

Short-circuits: unreachable/infeasible → safe (skip stages 2-3).
Doc-mitigated → safe (skip stage 3).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_tools import (
    ENHANCED_TRIAGE_TOOL_DEFINITIONS,
    GIT_TOOL_NAMES,
    READ_TOOL_DEFINITIONS,
    TRIAGE_ONLY_TOOL_DEFINITIONS,
    ToolExecutor,
)
from .db import SummaryDB
from .git_tools import GIT_TOOL_DEFINITIONS as _GIT_TOOL_DEFS
from .git_tools import GitTools
from .llm.base import LLMBackend
from .models import Function, SafetyIssue

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TriageResult:
    """Result of triaging a single SafetyIssue."""

    function_name: str
    issue_index: int
    issue: SafetyIssue
    hypothesis: str  # "safe", "contract_gap", "feasible", or "unknown"
    reasoning: str  # natural language proof

    # Safety proof: updated contracts that eliminate the issue
    updated_contracts: list[dict[str, Any]] = field(default_factory=list)

    # Feasibility proof: caller chain that can trigger the issue
    feasible_path: list[str] = field(default_factory=list)

    # For ucsan validation
    assumptions: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    relevant_functions: list[str] = field(default_factory=list)
    validation_plan: list[dict[str, Any]] = field(default_factory=list)

    # Enhanced triage: reachability and doc audit
    entry_function: str = ""
    reachability_chain: list[str] = field(default_factory=list)
    path_constraints: list[str] = field(default_factory=list)
    data_flow_trace: str = ""
    doc_audit_searched: str = ""
    doc_mitigated: bool = False

    # Stage 4: witness (ASan/UBSan unit test for feasible verdicts)
    witness_path: str = ""

    # Stage 5: debug (GDB diagnosis when witness runs clean)
    debug_diagnosis: str = ""  # path_blocked | not_reached | asan_limitation
    debug_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "function_name": self.function_name,
            "issue_index": self.issue_index,
            "issue": self.issue.to_dict(),
            "hypothesis": self.hypothesis,
            "reasoning": self.reasoning,
            "updated_contracts": self.updated_contracts,
            "feasible_path": self.feasible_path,
            "assumptions": self.assumptions,
            "assertions": self.assertions,
            "relevant_functions": self.relevant_functions,
        }
        if self.validation_plan:
            result["validation_plan"] = self.validation_plan
        if self.entry_function:
            result["entry_function"] = self.entry_function
        if self.reachability_chain:
            result["reachability_chain"] = self.reachability_chain
        if self.path_constraints:
            result["path_constraints"] = self.path_constraints
        if self.data_flow_trace:
            result["data_flow_trace"] = self.data_flow_trace
        if self.doc_audit_searched:
            result["doc_audit_searched"] = self.doc_audit_searched
        if self.doc_mitigated:
            result["doc_mitigated"] = self.doc_mitigated
        if self.witness_path:
            result["witness_path"] = self.witness_path
        if self.debug_diagnosis:
            result["debug_diagnosis"] = self.debug_diagnosis
        if self.debug_explanation:
            result["debug_explanation"] = self.debug_explanation
        return result


# ---------------------------------------------------------------------------
# Tool definitions — per-stage tool sets
# ---------------------------------------------------------------------------

_DB_TOOLS = {
    "read_function_source",
    "get_callers",
    "get_callees",
    "get_contracts",
    "get_issues",
}

_REACHABILITY_TOOLS = (
    _DB_TOOLS
    | {"get_entry_functions", "get_reachability_path",
       "get_call_chain_contracts", "submit_reachability"}
)

_DOC_AUDIT_TOOLS = (
    {"read_function_source", "get_contracts"}
    | GIT_TOOL_NAMES
    | {"submit_doc_audit"}
)

_VERDICT_TOOLS = (
    _DB_TOOLS
    | GIT_TOOL_NAMES
    | {"verify_contract", "submit_verdict"}
)

_ALL_TOOL_DEFS: list[dict[str, Any]] = [
    *READ_TOOL_DEFINITIONS,
    *TRIAGE_ONLY_TOOL_DEFINITIONS,
    *ENHANCED_TRIAGE_TOOL_DEFINITIONS,
    *_GIT_TOOL_DEFS,
]


def _filter_tools(
    names: set[str], *, has_git: bool = True,
) -> list[dict[str, Any]]:
    """Return tool definitions whose name is in *names*."""
    effective = names if has_git else names - GIT_TOOL_NAMES
    return [t for t in _ALL_TOOL_DEFS if t["name"] in effective]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REACHABILITY_SYSTEM_PROMPT = """\
You determine whether a safety issue in a C function is reachable AND
triggerable from an entry function (a function with no callers in the
call graph — directly callable by external code).

Your analysis has TWO parts:

## 1. Structural Reachability

Use get_entry_functions to find entry points. Use get_reachability_path
to find call chains from entries to the bug site. If no path exists,
the issue is unreachable — submit reachable=false.

## 2. Path Feasibility

For each viable path, read the source code at every function along the
chain (read_function_source). Read contracts for each function
(get_contracts or get_call_chain_contracts). Check:

a) BRANCH GUARDS: Are there conditionals (null checks, bounds checks,
   state guards, error-return checks) that prevent the path from
   being taken? E.g., "caller checks if(ptr==NULL) return before
   calling target, so null deref is dead."

b) DATA FLOW: Trace how the triggering parameter flows from the
   entry function's input through each call. Does any function along
   the chain sanitize, clamp, or constrain the value? E.g., "width
   is clamped to MAX_WIDTH in validate_header() before reaching
   alloc(width*height), so overflow is impossible."

   The requires/ensures contracts may already capture these
   constraints — check them.

## Verdicts

Submit reachable=true ONLY if: (1) a structural path exists AND
(2) no guards/sanitization along the path prevent the triggering
condition from being satisfied.

Set summarizer_gap=true if the verifier/summarizer missed important
details (e.g. a callee's ensures already guarantees the property but
was not recorded in contracts).

## Rules

- Always check ALL entry functions, not just the first one
- Read the source of every function in the call chain
- Report path_constraints: list every guard/sanitizer you found
- Report data_flow_trace: describe how the bug parameter flows
- If unsure whether guards block the path, lean toward reachable=true
"""

DOC_AUDIT_SYSTEM_PROMPT = """\
You check whether project documentation or inline source comments
enforce constraints that mitigate a reachable safety issue.

You are given the reachability path from entry to bug site.

## What to Search

1. PROJECT DOCS: README, manual, header comments, examples, docs/
   directory. Use git_ls_tree to find them, git_grep to search.

2. INLINE COMMENTS: Read source code along the call chain
   (read_function_source for each function in the path). Look for
   comments documenting invariants, preconditions, usage constraints.
   E.g., "/* caller must ensure buf_size >= header_len + padding */"

## Decision Criteria — LEAN STRICT

- If docs/comments CLEARLY state a constraint that prevents the bug,
  set mitigated=true and quote in doc_quote.
- If docs are silent or vague ("pass valid data"), set mitigated=false.
- Vague or partial mentions do NOT count as mitigation.
- Developer TODO/FIXME comments about the issue count as CONFIRMING
  the issue, not mitigating it.

## Rules

- Always populate doc_searched with your audit trail
- Aim for 3-8 searches; don't spiral with regex variations
- If the first 3 reasonable searches find nothing, docs are silent
"""

VERDICT_SYSTEM_PROMPT = """\
You are a memory safety bug triage agent for C/C++ code.

## Context

This issue has been confirmed as:
- **Reachable** from an entry function through a specific call chain
- **NOT mitigated** by project documentation or comments

Your job is to produce a FINAL VERDICT: either a safety proof (with
contract updates), a contract gap (with verified contract fixes), or
a feasibility proof (with entry-level harness data).

## Three Outcomes

### 1. Safety Proof (hypothesis: "safe")

Prove the issue cannot manifest despite being reachable. You MUST
provide updated_contracts, assumptions, and assertions.

### 2. Contract Gap (hypothesis: "contract_gap")

The function's requires is too weak for callee requires. Before
submitting, you MUST use verify_contract to trial-verify the fix.
Only submit if verify_contract reports resolved=true.

### 3. Feasibility Proof (hypothesis: "feasible")

The violation IS triggerable. You MUST provide:
- **feasible_path**: Call chain starting from the ENTRY FUNCTION
  (not the internal function) down to the bug site
- **assumptions**: Input conditions at the entry function's interface
  that trigger the bug (e.g., "attacker controls width via PNG header")
- **assertions**: The violation condition
- **relevant_functions**: The FULL call chain from entry to bug site —
  these will be kept as real code in symbolic validation
- **validation_plan**: Must use the entry function as the test entry
  point: [{"entries": ["<entry_function>"]}]

## Rules

- Read the function source and contracts before submitting
- For feasibility: the harness tests from the ENTRY function, not the
  internal buggy function — this catches real-world triggering paths
- Include assumptions and assertions for symbolic validation
- Do not guess — cite specific evidence
"""

WITNESS_SYSTEM_PROMPT = """\
You write a self-contained C unit test that reproduces a confirmed memory \
safety bug. The test will be compiled with `clang -fsanitize=address,undefined` \
and linked against the project's static library.

## Goal

Write a C program with a `main()` function that:
1. Sets up the minimal state required to call the ENTRY function
2. Calls the entry function with inputs that exercise the feasible bug path
3. When run under AddressSanitizer, triggers the memory safety violation

## Rules

- Include the project's public headers (not internal headers)
- Use standard libc for memory allocation and I/O
- Keep setup minimal — only create what the entry function needs
- Do NOT use assert.h assertions to detect the bug; ASan/UBSan will detect it
- If the entry function expects a file or stream, create a temporary file or \
use a buffer
- If the entry function expects a struct, allocate and initialize it with the \
minimum fields needed
- Output a single ```c fenced block with the complete program
- The program must be self-contained: no external test frameworks

## Runtime dependencies

The test runs in an Ubuntu 24.04 Docker container. If the bug path requires \
a library that is not part of the project itself (e.g., HarfBuzz, ICU, \
libxml2), you may request it by adding a `DEPS:` comment line at the top \
of the program listing the apt package names, for example:

```c
// DEPS: libharfbuzz-dev libxml2-dev
```

The build system will `apt-get install` these before compiling. \
Only request packages when the bug path genuinely requires them.
"""

DEFAULT_WITNESS_TURNS = 3
DEFAULT_WITNESS_DEBUG_ITERATIONS = 3

DEBUG_GDB_SYSTEM_PROMPT = """\
You write a GDB batch script to diagnose why an AddressSanitizer unit test \
did NOT trigger a memory safety violation that was predicted to be feasible.

## Goal

Write a GDB script that runs in `gdb -batch` mode and traces execution \
through the predicted bug path, printing key variables at each control \
flow point to determine:
1. Whether the target function was reached
2. What values the critical variables have at each branch point
3. Which guard/check blocks the predicted bug condition

## Rules

- Use `break <function>` or `break <file>:<line>` for breakpoints
- Use `commands ... end` blocks to print variables automatically
- Always include `set pagination off` and `set confirm off` at the top
- Print variables mentioned in the path constraints and data flow trace
- End with `run` followed by `quit`
- Output a single ```gdb fenced block
- Keep the script focused — only break at points along the feasible path
"""

DEBUG_DIAGNOSE_SYSTEM_PROMPT = """\
You are diagnosing why a unit test for a predicted memory safety bug ran \
clean under AddressSanitizer. You receive:
1. The original bug report and feasibility analysis
2. GDB trace output showing actual runtime values

## Diagnosis Categories

- **path_blocked**: The bug-triggering condition cannot hold because a \
guard/check in the caller prevents it (e.g., a length check ensures the \
denominator is never zero). This means the original verdict was wrong — \
the caller already guarantees the callee's precondition.
- **not_reached**: The target function was never called, or the test \
input doesn't exercise the right code path. The test needs fixing.
- **asan_limitation**: The bug condition IS reachable but ASan cannot \
detect it (e.g., intra-object overflow, uninitialized read). The \
verdict is correct but requires a different detection method.

## Output

Respond with a single JSON object (no markdown fencing):

{
  "diagnosis": "path_blocked|not_reached|asan_limitation",
  "explanation": "Detailed explanation of what the GDB trace shows...",
  "corrected_hypothesis": "safe|feasible",
  "updated_contracts": [
    {
      "function": "caller_function_name",
      "property": "overflow|memsafe|...",
      "clause_type": "ensures",
      "predicate": "C-like predicate that the caller guarantees"
    }
  ]
}

Rules:
- For path_blocked: set corrected_hypothesis to "safe" and emit \
updated_contracts showing what the caller guarantees
- For not_reached: set corrected_hypothesis to "feasible" (test is bad, \
not the analysis) with empty updated_contracts. In the explanation, \
describe WHY the target was not reached — missing runtime dependency \
(e.g., the code path requires HarfBuzz/ICU/etc. to be installed), \
wrong API usage, missing initialization, etc. If a library is needed, \
mention the apt package name (e.g., libharfbuzz-dev) so the next \
test iteration can add a `// DEPS: <pkg>` line
- For asan_limitation: set corrected_hypothesis to "feasible" with empty \
updated_contracts
- The predicate in updated_contracts should be a concise C-like \
expression describing the invariant the caller enforces
"""

DEFAULT_DEBUG_TURNS = 2


# ---------------------------------------------------------------------------
# Shared ReAct loop
# ---------------------------------------------------------------------------

DEFAULT_REACHABILITY_TURNS = 60
DEFAULT_DOC_AUDIT_TURNS = 50
DEFAULT_VERDICT_TURNS = 100


def _run_react_loop(
    *,
    llm: LLMBackend,
    db: SummaryDB,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    terminal_tool: str,
    max_turns: int,
    verbose: bool = False,
    log_prefix: str = "Triage",
    git_tools: GitTools | None = None,
    project_path: Path | None = None,
    llm_for_verify: LLMBackend | None = None,
) -> dict[str, Any] | None:
    """Run a ReAct loop until the terminal tool is called or turns run out.

    Returns the terminal tool's input dict if submitted, else None.
    """
    executor = ToolExecutor(
        db,
        verbose=verbose,
        git_tools=git_tools,
        project_path=project_path,
        llm=llm_for_verify,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    terminal_input: dict[str, Any] | None = None

    for turn in range(max_turns):
        response = llm.complete_with_tools(
            messages=messages, tools=tools, system=system_prompt,
        )

        stop = getattr(response, "stop_reason", None)
        if stop in ("end_turn", "stop"):
            if verbose:
                print(f"[{log_prefix}] LLM stopped at turn {turn + 1} ({stop})")
            break
        if stop != "tool_use":
            if verbose:
                print(f"[{log_prefix}] Unexpected stop_reason: {stop}")
            break

        assistant_content: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in response.content:
            if hasattr(block, "text") and block.type == "text":
                entry: dict[str, Any] = {"type": "text", "text": block.text}
                if getattr(block, "thought", False):
                    entry["thought"] = True
                sig = getattr(block, "thought_signature", None)
                if sig:
                    entry["thought_signature"] = sig
                assistant_content.append(entry)
                if verbose and not getattr(block, "thought", False):
                    print(f"  [LLM] {block.text[:300]}")

            elif block.type == "tool_use":
                tool_entry: dict[str, Any] = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                sig = getattr(block, "thought_signature", None)
                if sig:
                    tool_entry["thought_signature"] = sig
                assistant_content.append(tool_entry)

                result = executor.execute(block.name, block.input)

                if verbose:
                    err = result.get("error")
                    if err:
                        print(f"  [Tool] {block.name} -> ERROR: {err[:150]}")
                    elif block.name == terminal_tool:
                        print(f"  [Tool] {terminal_tool} -> accepted")
                    else:
                        arg = json.dumps(block.input)
                        if len(arg) > 80:
                            arg = arg[:80] + "..."
                        print(f"  [Tool] {block.name}({arg})")

                if block.name == terminal_tool and result.get("accepted"):
                    terminal_input = dict(block.input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

        if terminal_input is not None:
            break

    return terminal_input


def _extract_c_block(text: str) -> str | None:
    """Extract the first ```c ... ``` fenced block from LLM output."""
    m = re.search(r"```c\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_gdb_block(text: str) -> str | None:
    """Extract the first ```gdb ... ``` fenced block from LLM output."""
    m = re.search(r"```gdb\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


_ASAN_MARKERS = (
    "ERROR: AddressSanitizer",
    "ERROR: LeakSanitizer",
    "runtime error:",
    "SUMMARY: AddressSanitizer",
    "SUMMARY: UndefinedBehaviorSanitizer",
)


# ---------------------------------------------------------------------------
# Stage data models
# ---------------------------------------------------------------------------


@dataclass
class ReachabilityVerdict:
    """Output of stage 1: path feasibility check."""

    reachable: bool
    entry_function: str
    call_chain: list[str]
    path_constraints: list[str]
    data_flow_trace: str
    reasoning: str
    summarizer_gap: bool = False


@dataclass
class DocAuditVerdict:
    """Output of stage 2: documentation audit."""

    mitigated: bool
    doc_searched: str
    doc_quote: str = ""
    mitigating_constraint: str = ""


# ---------------------------------------------------------------------------
# Three-stage triage agent
# ---------------------------------------------------------------------------


class TriageAgent:
    """Three-stage triage: reachability → doc audit → verdict.

    Each stage uses a fresh LLM client via ``llm_factory`` to prevent
    context bleed between stages.
    """

    def __init__(
        self,
        db: SummaryDB,
        llm_factory: Callable[[], LLMBackend],
        verbose: bool = False,
        project_path: Path | None = None,
        output_path: Path | None = None,
        *,
        compile_commands: list[dict[str, Any]] | None = None,
        build_script_dir: Path | None = None,
        harness_output_dir: Path | None = None,
        build_dir: Path | None = None,
    ) -> None:
        self.db = db
        self.llm_factory = llm_factory
        self.verbose = verbose
        self.project_path = project_path
        self.output_path = output_path
        self.compile_commands = compile_commands
        self.build_script_dir = build_script_dir
        self.build_dir = build_dir
        self.harness_output_dir = harness_output_dir
        self._results: list[dict[str, Any]] = []

    def _rel_path(self, abs_path: str) -> str:
        if self.project_path is None:
            return abs_path
        try:
            return str(Path(abs_path).relative_to(self.project_path))
        except ValueError:
            return abs_path

    # -- DB writeback & incremental save --

    def _finalize_result(
        self, func: Function, result: TriageResult,
    ) -> TriageResult:
        """Apply DB side-effects and save incrementally."""
        if result.hypothesis in ("safe", "contract_gap"):
            self._apply_contract_updates(func, result)
            self._mark_issue_review(
                func, result, status="false_positive",
            )
        elif result.hypothesis == "unknown":
            self._mark_issue_review(
                func, result, status="unknown",
            )
        elif result.hypothesis == "feasible":
            self._mark_issue_review(
                func, result, status="confirmed",
            )
            if self.build_script_dir is not None and self.build_dir is not None:
                debug_feedback = ""
                for wd_iter in range(DEFAULT_WITNESS_DEBUG_ITERATIONS):
                    witness, ran_clean = self._stage_witness(
                        func, result, debug_feedback=debug_feedback,
                    )
                    if not witness:
                        break
                    result.witness_path = witness
                    if not ran_clean:
                        self._mark_issue_review(
                            func, result,
                            status="confirmed_witness",
                        )
                        break
                    # Witness ran clean → diagnose with GDB
                    result = self._stage_debug(func, result)
                    if result.debug_diagnosis != "not_reached":
                        break
                    # not_reached → retry witness with debug feedback
                    if wd_iter + 1 < DEFAULT_WITNESS_DEBUG_ITERATIONS:
                        debug_feedback = (
                            f"## Previous Attempt Failed\n\n"
                            f"The test did not reach the target "
                            f"function. GDB diagnosis:\n\n"
                            f"{result.debug_explanation}\n\n"
                            f"Generate a new test that addresses "
                            f"this issue."
                        )
                        if self.verbose:
                            print(
                                f"  [Witness↔Debug] iteration "
                                f"{wd_iter + 1}: not_reached, "
                                f"retrying witness",
                            )

        self._save_incremental(result)
        return result

    def _apply_contract_updates(
        self, func: Function, result: TriageResult,
    ) -> None:
        """Write updated_contracts from the verdict back to the DB."""
        if not result.updated_contracts:
            return

        from .code_contract.models import CodeContractSummary

        by_func: dict[str, list[dict[str, Any]]] = {}
        for upd in result.updated_contracts:
            fname = upd.get("function", "")
            if fname:
                by_func.setdefault(fname, []).append(upd)

        for fname, updates in by_func.items():
            funcs = self.db.get_function_by_name(fname)
            if not funcs or funcs[0].id is None:
                if self.verbose:
                    print(f"  [DB] skip contract update: {fname} not found")
                continue
            f = funcs[0]
            assert f.id is not None

            existing = self.db.get_code_contract_summary(f.id)
            if existing is None:
                data: dict[str, Any] = {
                    "function": fname,
                    "properties": [],
                    "requires": {},
                    "ensures": {},
                    "modifies": {},
                    "notes": {},
                    "origin": {},
                }
            else:
                data = existing.to_dict()
                data["function"] = fname

            for upd in updates:
                prop = upd.get("property", "")
                clause_type = upd.get("clause_type", "")
                predicate = upd.get("predicate", "")
                if not prop or not clause_type or not predicate:
                    continue

                if prop not in data.get("properties", []):
                    data.setdefault("properties", []).append(prop)

                clauses = data.setdefault(clause_type, {})
                existing_list = clauses.setdefault(prop, [])
                if predicate not in existing_list:
                    existing_list.append(predicate)

            try:
                summary = CodeContractSummary.from_dict(data)
                self.db.store_code_contract_summary(
                    f, summary, model_used="triage",
                )
                if self.verbose:
                    print(f"  [DB] updated contracts for {fname}")
            except Exception as e:
                if self.verbose:
                    print(f"  [DB] failed to update {fname}: {e}")

    def _mark_issue_review(
        self,
        func: Function,
        result: TriageResult,
        *,
        status: str,
    ) -> None:
        """Mark the issue as false_positive or confirmed in issue_reviews."""
        if func.id is None:
            return
        fp = result.issue.fingerprint()
        self.db.upsert_issue_review(
            function_id=func.id,
            issue_index=result.issue_index,
            fingerprint=fp,
            status=status,
            reason=result.reasoning[:500],
        )
        if self.verbose:
            print(f"  [DB] marked {func.name}#{result.issue_index} as {status}")

    def _save_incremental(self, result: TriageResult) -> None:
        """Append result to the output file atomically."""
        self._results.append(result.to_dict())
        if self.output_path is None:
            return
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_path.parent),
            suffix=".tmp",
        )
        try:
            with open(tmp_fd, "w") as f:
                json.dump(self._results, f, indent=2)
                f.write("\n")
            Path(tmp_path).replace(self.output_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    # -- public API (same shape as old TriageAgent) --

    def triage_issue(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
    ) -> TriageResult:
        """Triage a single issue via three-stage pipeline."""
        if self.verbose:
            print(
                f"\n[Triage] {func.name} issue #{issue_index}: "
                f"{issue.issue_kind} at {issue.location}",
            )
            print(f"  {issue.description}")

        # Stage 1: path feasibility
        rv = self._stage_reachability(func, issue, issue_index)

        if not rv.reachable:
            reason = "unreachable"
            if rv.summarizer_gap:
                reason = "summarizer_gap"
            if self.verbose:
                print(f"[Triage] Stage 1 → {reason}: {rv.reasoning[:200]}")
            return self._finalize_result(func, TriageResult(
                function_name=func.name,
                issue_index=issue_index,
                issue=issue,
                hypothesis="safe",
                reasoning=f"[{reason}] {rv.reasoning}",
                reachability_chain=rv.call_chain,
                path_constraints=rv.path_constraints,
                data_flow_trace=rv.data_flow_trace,
            ))

        if self.verbose:
            chain = " → ".join(rv.call_chain) if rv.call_chain else "(empty)"
            print(
                f"[Triage] Stage 1 → reachable via {rv.entry_function}: "
                f"{chain}",
            )

        # Stage 2: doc audit (only if project_path set)
        dv: DocAuditVerdict | None = None
        if self.project_path:
            dv = self._stage_doc_audit(func, issue, issue_index, rv)
            if dv.mitigated:
                if self.verbose:
                    print(
                        f"[Triage] Stage 2 → mitigated: "
                        f"{dv.mitigating_constraint[:200]}",
                    )
                return self._finalize_result(func, TriageResult(
                    function_name=func.name,
                    issue_index=issue_index,
                    issue=issue,
                    hypothesis="safe",
                    reasoning=(
                        f"[doc_mitigated] {dv.mitigating_constraint}. "
                        f"{dv.doc_quote}"
                    ),
                    entry_function=rv.entry_function,
                    reachability_chain=rv.call_chain,
                    path_constraints=rv.path_constraints,
                    data_flow_trace=rv.data_flow_trace,
                    doc_audit_searched=dv.doc_searched,
                    doc_mitigated=True,
                ))
            if self.verbose:
                print("[Triage] Stage 2 → not mitigated by docs")

        # Stage 3: entry-level verdict
        return self._finalize_result(
            func,
            self._stage_verdict(func, issue, issue_index, rv, dv),
        )

    def triage_function(
        self,
        func: Function,
        *,
        severity_filter: set[str] | None = None,
        force: bool = False,
    ) -> list[TriageResult]:
        """Triage all issues for a function, skipping already-reviewed ones."""
        if func.id is None:
            return []

        vs = self.db.get_verification_summary_by_function_id(func.id)
        if vs is None or not vs.issues:
            return []

        reviewed: dict[str, dict[str, Any]] = {}
        if not force:
            fingerprints = [iss.fingerprint() for iss in vs.issues]
            reviewed = self.db.get_issue_reviews_by_fingerprints(
                func.id, fingerprints,
            )

        results = []
        for i, issue in enumerate(vs.issues):
            if severity_filter and issue.severity not in severity_filter:
                continue
            if not force:
                review = reviewed.get(issue.fingerprint())
                if review and review.get("status") != "pending":
                    if self.verbose:
                        print(
                            f"[Triage] skip {func.name}#{i}: "
                            f"already {review['status']}",
                        )
                    continue
            result = self.triage_issue(func, issue, i)
            results.append(result)
        return results

    # -- stage 1: reachability --

    def _stage_reachability(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
    ) -> ReachabilityVerdict:
        has_git = self.project_path is not None
        tools = _filter_tools(_REACHABILITY_TOOLS, has_git=has_git)
        prompt = self._build_reachability_prompt(func, issue, issue_index)

        result = _run_react_loop(
            llm=self.llm_factory(),
            db=self.db,
            system_prompt=REACHABILITY_SYSTEM_PROMPT,
            user_prompt=prompt,
            tools=tools,
            terminal_tool="submit_reachability",
            max_turns=DEFAULT_REACHABILITY_TURNS,
            verbose=self.verbose,
            log_prefix="Reachability",
            git_tools=(
                GitTools(self.project_path) if self.project_path else None
            ),
            project_path=self.project_path,
        )

        if result is None:
            return ReachabilityVerdict(
                reachable=True,
                entry_function="",
                call_chain=[],
                path_constraints=[],
                data_flow_trace="",
                reasoning="Reachability stage hit turn limit; "
                          "defaulting to reachable.",
            )

        return ReachabilityVerdict(
            reachable=bool(result.get("reachable", True)),
            entry_function=str(result.get("entry_function", "")),
            call_chain=result.get("call_chain", []),
            path_constraints=result.get("path_constraints", []),
            data_flow_trace=str(result.get("data_flow_trace", "")),
            reasoning=str(result.get("reasoning", "")),
            summarizer_gap=bool(result.get("summarizer_gap", False)),
        )

    def _build_reachability_prompt(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
    ) -> str:
        lines = [
            "## Issue to Check for Reachability",
            "",
            f"Function: `{func.name}`",
            f"Signature: `{func.signature}`",
            f"File: {self._rel_path(func.file_path)}:{func.line_start}",
            "",
            f"### Issue #{issue_index}",
            f"- **Kind**: {issue.issue_kind}",
            f"- **Location**: {issue.location}",
            f"- **Severity**: {issue.severity}",
            f"- **Description**: {issue.description}",
        ]
        if issue.callee:
            lines.append(f"- **Callee involved**: {issue.callee}")

        lines.extend([
            "",
            "### Instructions",
            "1. Use get_entry_functions to find entry points.",
            "2. Use get_reachability_path from each entry to "
            f"`{func.name}` to find call chains.",
            "3. For each viable path, read function source along "
            "the chain and check branch guards + data flow.",
            "4. Submit your verdict via submit_reachability.",
        ])
        return "\n".join(lines)

    # -- stage 2: doc audit --

    def _stage_doc_audit(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        rv: ReachabilityVerdict,
    ) -> DocAuditVerdict:
        tools = _filter_tools(_DOC_AUDIT_TOOLS, has_git=True)
        prompt = self._build_doc_audit_prompt(func, issue, issue_index, rv)

        result = _run_react_loop(
            llm=self.llm_factory(),
            db=self.db,
            system_prompt=DOC_AUDIT_SYSTEM_PROMPT,
            user_prompt=prompt,
            tools=tools,
            terminal_tool="submit_doc_audit",
            max_turns=DEFAULT_DOC_AUDIT_TURNS,
            verbose=self.verbose,
            log_prefix="DocAudit",
            git_tools=GitTools(self.project_path) if self.project_path else None,
            project_path=self.project_path,
        )

        if result is None:
            return DocAuditVerdict(
                mitigated=False,
                doc_searched="(doc audit hit turn limit)",
            )

        return DocAuditVerdict(
            mitigated=bool(result.get("mitigated", False)),
            doc_searched=str(result.get("doc_searched", "")),
            doc_quote=str(result.get("doc_quote", "")),
            mitigating_constraint=str(
                result.get("mitigating_constraint", ""),
            ),
        )

    def _build_doc_audit_prompt(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        rv: ReachabilityVerdict,
    ) -> str:
        chain_str = " → ".join(rv.call_chain) if rv.call_chain else "(direct)"
        lines = [
            "## Doc Audit for Reachable Issue",
            "",
            f"Function: `{func.name}`",
            f"File: {self._rel_path(func.file_path)}",
            "",
            f"### Issue #{issue_index}",
            f"- **Kind**: {issue.issue_kind}",
            f"- **Description**: {issue.description}",
            "",
            "### Reachability (from stage 1)",
            f"- **Entry function**: `{rv.entry_function}`",
            f"- **Call chain**: {chain_str}",
            f"- **Data flow**: {rv.data_flow_trace}",
            "",
            "### Instructions",
            "Search project documentation AND inline source comments "
            "along the call chain for constraints that would mitigate "
            "this issue. Lean strict — vague mentions don't count.",
        ]
        return "\n".join(lines)

    # -- stage 3: verdict --

    def _stage_verdict(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        rv: ReachabilityVerdict,
        dv: DocAuditVerdict | None,
    ) -> TriageResult:
        has_git = self.project_path is not None
        tools = _filter_tools(_VERDICT_TOOLS, has_git=has_git)
        prompt = self._build_verdict_prompt(func, issue, issue_index, rv, dv)

        result = _run_react_loop(
            llm=self.llm_factory(),
            db=self.db,
            system_prompt=VERDICT_SYSTEM_PROMPT,
            user_prompt=prompt,
            tools=tools,
            terminal_tool="submit_verdict",
            max_turns=DEFAULT_VERDICT_TURNS,
            verbose=self.verbose,
            log_prefix="Verdict",
            git_tools=(
                GitTools(self.project_path) if self.project_path else None
            ),
            project_path=self.project_path,
            llm_for_verify=self.llm_factory(),
        )

        if result is None:
            return TriageResult(
                function_name=func.name,
                issue_index=issue_index,
                issue=issue,
                hypothesis="unknown",
                reasoning="Verdict stage hit turn limit.",
                entry_function=rv.entry_function,
                reachability_chain=rv.call_chain,
                path_constraints=rv.path_constraints,
                data_flow_trace=rv.data_flow_trace,
                doc_audit_searched=dv.doc_searched if dv else "",
            )

        hyp = str(result.get("hypothesis", "feasible"))
        if hyp not in ("safe", "contract_gap", "feasible"):
            hyp = "feasible"

        return TriageResult(
            function_name=func.name,
            issue_index=issue_index,
            issue=issue,
            hypothesis=hyp,
            reasoning=str(result.get("reasoning", "")),
            updated_contracts=result.get("updated_contracts", []),
            feasible_path=result.get("feasible_path", []),
            assumptions=result.get("assumptions", []),
            assertions=result.get("assertions", []),
            relevant_functions=result.get("relevant_functions", []),
            validation_plan=result.get("validation_plan", []),
            entry_function=rv.entry_function,
            reachability_chain=rv.call_chain,
            path_constraints=rv.path_constraints,
            data_flow_trace=rv.data_flow_trace,
            doc_audit_searched=dv.doc_searched if dv else "",
        )

    def _build_verdict_prompt(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        rv: ReachabilityVerdict,
        dv: DocAuditVerdict | None,
    ) -> str:
        chain_str = " → ".join(rv.call_chain) if rv.call_chain else "(direct)"
        lines = [
            "## Issue for Final Verdict",
            "",
            f"Function: `{func.name}`",
            f"Signature: `{func.signature}`",
            f"File: {self._rel_path(func.file_path)}:{func.line_start}",
            "",
            f"### Issue #{issue_index}",
            f"- **Kind**: {issue.issue_kind}",
            f"- **Location**: {issue.location}",
            f"- **Severity**: {issue.severity}",
            f"- **Description**: {issue.description}",
        ]
        if issue.callee:
            lines.append(f"- **Callee involved**: {issue.callee}")

        lines.extend([
            "",
            "### Reachability Context",
            f"- **Entry function**: `{rv.entry_function}`",
            f"- **Call chain**: {chain_str}",
            f"- **Data flow**: {rv.data_flow_trace}",
        ])
        if rv.path_constraints:
            lines.append("- **Path constraints found**:")
            for pc in rv.path_constraints:
                lines.append(f"  - {pc}")

        if dv:
            lines.extend([
                "",
                "### Doc Audit Result",
                f"- **Mitigated**: {dv.mitigated}",
                f"- **Searched**: {dv.doc_searched}",
            ])
            if dv.doc_quote:
                lines.append(f"- **Doc quote**: {dv.doc_quote}")

        lines.extend([
            "",
            "### Instructions",
            "Produce a final verdict: safe, contract_gap, or feasible.",
            f"For feasibility proofs, use `{rv.entry_function}` as the "
            "entry point in feasible_path and validation_plan.",
            "Read the function source and contracts before deciding.",
        ])
        return "\n".join(lines)

    # -- stage 4: witness (ASan/UBSan unit test for feasible verdicts) --

    def _stage_witness(
        self,
        func: Function,
        result: TriageResult,
        *,
        debug_feedback: str = "",
    ) -> tuple[str | None, bool]:
        """Generate a self-contained ASan/UBSan unit test for a feasible bug.

        Returns ``(witness_c_path, ran_clean)`` where *ran_clean* is
        ``True`` when the test compiled and ran without ASan reports.

        When *debug_feedback* is provided (from a prior Stage 5 debug
        iteration), it is appended to the witness prompt so the LLM can
        fix reachability issues in the generated test.
        """
        assert self.build_script_dir is not None
        assert self.build_dir is not None

        if self.verbose:
            print(
                f"[Triage] Stage 4 → generating witness for "
                f"{func.name}#{result.issue_index}",
            )

        # Determine output directory
        out_dir = self.harness_output_dir or Path("harnesses")
        out_dir.mkdir(parents=True, exist_ok=True)

        test_name = f"test_{func.name}_{result.issue_index}"
        witness_c = out_dir / f"{test_name}.c"
        build_sh = out_dir / f"build_{test_name}.sh"

        # Gather source code for the LLM
        prompt = self._build_witness_prompt(
            func, result, debug_feedback=debug_feedback,
        )

        llm = self.llm_factory()
        c_code: str | None = None
        compile_err = ""

        for attempt in range(DEFAULT_WITNESS_TURNS):
            if attempt == 0:
                response = llm.complete(
                    prompt, system=WITNESS_SYSTEM_PROMPT,
                )
            else:
                response = llm.complete(
                    f"{prompt}\n\n"
                    f"---\n\n"
                    f"The previous attempt:\n\n"
                    f"```c\n{c_code}\n```\n\n"
                    f"failed with:\n\n"
                    f"```\n{compile_err}\n```\n\n"
                    "Fix the code. Output a single ```c fenced block.",
                    system=WITNESS_SYSTEM_PROMPT,
                )

            text = (
                response.text
                if hasattr(response, "text")
                else str(response)
            )
            extracted = _extract_c_block(text)
            if not extracted:
                if self.verbose:
                    print(
                        f"  [Witness] attempt {attempt + 1}: "
                        f"no C block in response",
                    )
                continue

            c_code = extracted

            # Write and try to compile
            witness_c.write_text(c_code)
            build_script = self._build_witness_script(
                test_name, witness_c,
            )
            build_sh.write_text(build_script)
            build_sh.chmod(0o755)

            phase, stderr, _stdout = self._compile_witness(build_sh)

            if phase == "run_sanitizer":
                if self.verbose:
                    print(
                        f"  [Witness] sanitizer triggered: "
                        f"{witness_c}",
                    )
                return str(witness_c), False
            if phase == "run_clean":
                if self.verbose:
                    print(
                        f"  [Witness] ran clean (no reports): "
                        f"{witness_c}",
                    )
                return str(witness_c), True
            if phase == "run_crash":
                if self.verbose:
                    print(
                        f"  [Witness] crashed (no sanitizer): "
                        f"{witness_c}",
                    )
                return str(witness_c), False

            # compile_error / timeout / error → retry
            compile_err = stderr
            if self.verbose:
                print(
                    f"  [Witness] attempt {attempt + 1} "
                    f"{phase}: {compile_err[:200]}",
                )

        if c_code is not None:
            witness_c.write_text(c_code)
            if self.verbose:
                print(
                    f"  [Witness] saved (uncompiled): {witness_c}",
                )
            return str(witness_c), False

        return None, False

    def _build_witness_prompt(
        self,
        func: Function,
        result: TriageResult,
        *,
        debug_feedback: str = "",
    ) -> str:
        """Build the user prompt for witness generation."""
        lines = [
            "## Bug Report",
            "",
            f"Function: `{func.name}`",
            f"File: {self._rel_path(func.file_path)}",
            f"Issue: [{result.issue.severity}] {result.issue.issue_kind} "
            f"— {result.issue.description}",
            "",
            "## Feasibility Analysis",
            "",
            f"**Entry function**: `{result.entry_function}`",
        ]

        if result.feasible_path:
            chain = " → ".join(result.feasible_path)
            lines.append(f"**Feasible path**: {chain}")
        if result.data_flow_trace:
            lines.append(f"**Data flow**: {result.data_flow_trace}")
        if result.assumptions:
            lines.append("")
            lines.append("**Assumptions** (input conditions at entry):")
            for a in result.assumptions:
                lines.append(f"  - {a}")
        if result.assertions:
            lines.append("")
            lines.append("**Violation**:")
            for a in result.assertions:
                lines.append(f"  - {a}")

        lines.extend(["", f"**Reasoning**: {result.reasoning}", ""])

        # Include source code of entry function and functions along path
        funcs_to_show = []
        names_seen: set[str] = set()
        for name in [result.entry_function, *result.feasible_path,
                      func.name]:
            if not name or name in names_seen:
                continue
            names_seen.add(name)
            found = self.db.get_function_by_name(name)
            if found:
                funcs_to_show.append(found[0])

        if funcs_to_show:
            lines.append("## Source Code")
            lines.append("")
            for f in funcs_to_show:
                lines.append(f"### `{f.name}` ({self._rel_path(f.file_path)})")
                lines.append(f"```c\n// {f.signature}")
                if f.source:
                    lines.append(f.source)
                lines.append("```")
                lines.append("")

        # Build environment context
        if self.build_script_dir is not None:
            build_sh = self.build_script_dir / "build.sh"
            if build_sh.exists():
                lines.extend([
                    "## Project Build Script",
                    "",
                    "```bash",
                    build_sh.read_text().strip(),
                    "```",
                    "",
                ])

        # Available include directories
        if self.compile_commands:
            inc_dirs: set[str] = set()
            for entry in self.compile_commands:
                cmd = entry.get("command", "")
                for token in cmd.split():
                    if token.startswith("-I") and len(token) > 2:
                        inc_dirs.add(token[2:])
            if inc_dirs:
                lines.extend([
                    "## Available Include Directories",
                    "",
                    "The following `-I` paths are used when building "
                    "the project (Docker container paths):",
                    "",
                ])
                for d in sorted(inc_dirs):
                    lines.append(f"  - `{d}`")
                lines.append("")

        if debug_feedback:
            lines.extend(["", debug_feedback, ""])

        lines.extend([
            "## Instructions",
            "",
            f"Write a complete C program with main() that calls "
            f"`{result.entry_function}` with inputs that trigger the bug.",
            "The program will be compiled with "
            "`clang -fsanitize=address,undefined` and linked against the "
            "project's static library.",
        ])

        return "\n".join(lines)

    @staticmethod
    def _parse_witness_deps(test_c: Path) -> list[str]:
        """Extract ``// DEPS: pkg1 pkg2`` from the test source."""
        if not test_c.exists():
            return []
        pkgs: list[str] = []
        for line in test_c.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("// DEPS:"):
                for pkg in stripped[len("// DEPS:"):].split():
                    pkg = pkg.strip(",")
                    if pkg and re.fullmatch(r"[a-z0-9][a-z0-9.+\-]+", pkg):
                        pkgs.append(pkg)
        return pkgs

    def _build_witness_script(
        self,
        test_name: str,
        test_c: Path,
    ) -> str:
        """Generate a Docker-based build script for the unit test.

        Rebuilds the project with ASan/UBSan into ``--build-dir`` on
        the host (mounted at ``/workspace/build`` in Docker), then
        compiles the test and links against the sanitized library.
        The test binary is also run inside Docker.
        """
        assert self.build_script_dir is not None
        assert self.build_dir is not None

        config_path = self.build_script_dir / "config.json"
        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        project_path = config.get(
            "project_path", "/data/csong/opensource/unknown",
        )
        host_build_dir = str(self.build_dir)

        # Rewrite cmake flags: drop LTO, add sanitizers
        cmake_flags = list(config.get("cmake_flags", []))
        san_flags = (
            "-g -fsanitize=address,undefined -fno-omit-frame-pointer"
        )
        sanitized: list[str] = []
        for flag in cmake_flags:
            if flag.startswith("-DCMAKE_C_FLAGS="):
                sanitized.append(f"-DCMAKE_C_FLAGS={san_flags}")
            elif flag.startswith("-DCMAKE_CXX_FLAGS="):
                sanitized.append(f"-DCMAKE_CXX_FLAGS={san_flags}")
            elif flag.startswith(
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION",
            ):
                sanitized.append(
                    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
                )
            else:
                sanitized.append(flag)

        cmake_args = " ".join(f"'{f}'" for f in sanitized)

        # Parse // DEPS: lines from the test source for apt packages
        deps_pkgs = self._parse_witness_deps(test_c)

        # Write a small inner script to avoid bash -c quoting issues.
        # Steps 1-2 (build) use set -e; step 3 (run) captures the
        # exit code separately so the caller can distinguish compile
        # failure from a runtime sanitizer report.
        inner_lines = [
            "#!/bin/bash",
            "set -e",
            "",
        ]

        if deps_pkgs:
            pkgs = " ".join(deps_pkgs)
            inner_lines.extend([
                "# Step 0: install runtime dependencies",
                f"apt-get update -qq && apt-get install -y -qq"
                f" {pkgs} >/dev/null 2>&1",
                "",
            ])

        inner_lines.extend([
            "# Step 1: rebuild project with ASan/UBSan",
            "cd /workspace/build",
        ])
        if deps_pkgs:
            inner_lines.append("rm -rf CMakeCache.txt CMakeFiles")
        inner_lines.extend([
            f"cmake -G Ninja {cmake_args} /workspace/src",
            "ninja -j$(nproc)",
            'echo "--- project rebuilt with ASan ---"',
            "",
            "# Step 2: compile and link the test",
            "# Use ! -type l to skip symlinks and avoid duplicate symbols",
            "LIB=$(find /workspace/build -name '*.a' ! -type l | head -1)",
            "# Extract link libraries cmake discovered",
            "CMAKE_LIBS=$(grep -E 'LIBRAR(Y[^:]*|IES):FILEPATH=/' "
            "/workspace/build/CMakeCache.txt 2>/dev/null "
            "| grep -v NOTFOUND | sed 's/.*=//' "
            "| sort -u | tr '\\n' ' ')",
            "# Extract -I flags from the freshly-built compile_commands.json",
            "INC_FLAGS=$(grep -ohP '(?<= )-I[^ \"]+' "
            "/workspace/build/compile_commands.json 2>/dev/null "
            "| sort -u | tr '\\n' ' ')",
            'INC_FLAGS="${INC_FLAGS:--I/workspace/src -I/workspace/build}"',
            f"clang-18 -fsanitize=address,undefined -g "
            f"$INC_FLAGS "
            f"/test/{test_c.name} "
            '"$LIB" $CMAKE_LIBS '
            f"-lm -lz -lpthread -lstdc++ -ldl "
            f"-o /test/{test_name}",
            "",
            "# Compile succeeded — mark it so the caller knows",
            f'echo "COMPILE_OK" > /test/.{test_name}_compiled',
            "",
            "# Step 3: run the test (don't set -e; capture exit code)",
            'echo "--- running test ---"',
            "set +e",
            f"/test/{test_name}",
            "RC=$?",
            'echo "--- test exited with code $RC ---"',
            "exit $RC",
        ])
        inner_script = "\n".join(inner_lines) + "\n"

        # Write inner script next to the test
        inner_path = test_c.parent / f"_inner_{test_name}.sh"
        inner_path.write_text(inner_script)
        inner_path.chmod(0o755)

        lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Generated unit test — rebuilds project with ASan, "
            "then builds and runs the test",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            f'PROJECT_PATH="{project_path}"',
            f'BUILD_DIR="{host_build_dir}"',
            "",
            'mkdir -p "$BUILD_DIR"',
            "",
            f'echo "Building and running {test_name} (ASan/UBSan)..."',
            "docker run --rm \\",
            '  -v "$PROJECT_PATH":/workspace/src:ro \\',
            '  -v "$BUILD_DIR":/workspace/build \\',
            '  -v "$SCRIPT_DIR":/test \\',
            "  llm-summary-builder:latest \\",
            f'  bash /test/{inner_path.name}',
            "",
            'echo "Done."',
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compile_witness(
        build_sh: Path,
    ) -> tuple[str, str, str]:
        """Run the witness build+run script.

        Returns ``(phase, stderr, stdout)`` where *phase* is one of:

        - ``"compile_error"`` — build/compile failed
        - ``"run_clean"`` — test ran and exited 0 with no sanitizer reports
        - ``"run_sanitizer"`` — test ran and a sanitizer fired
        - ``"run_crash"`` — test ran and crashed (non-zero, no sanitizer)
        - ``"timeout"`` — overall timeout
        - ``"error"`` — unexpected exception
        """
        try:
            result = subprocess.run(
                ["bash", str(build_sh)],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "Build/run timed out after 180s", ""
        except Exception as e:
            return "error", str(e), ""

        stderr = result.stderr
        stdout = result.stdout
        combined = (stderr + stdout).strip()

        # Check for compile marker written by inner script
        # The marker file is next to the build script
        test_dir = build_sh.parent
        test_name = build_sh.name.removeprefix("build_")
        test_name = test_name.removesuffix(".sh")
        marker = test_dir / f".{test_name}_compiled"

        compiled = marker.exists()
        if compiled:
            marker.unlink(missing_ok=True)

        if not compiled:
            return "compile_error", combined, ""

        sanitizer_fired = any(m in combined for m in _ASAN_MARKERS)
        if sanitizer_fired:
            return "run_sanitizer", stderr, stdout
        if result.returncode == 0:
            return "run_clean", stderr, stdout
        return "run_crash", stderr, stdout

    # ------------------------------------------------------------------
    # Stage 5: DEBUG — GDB diagnosis when witness runs clean
    # ------------------------------------------------------------------

    def _stage_debug(
        self,
        func: Function,
        result: TriageResult,
    ) -> TriageResult:
        """Diagnose why a feasible-verdict witness ran clean under ASan.

        Generates a GDB script, runs the test under GDB inside Docker,
        then asks the LLM to diagnose.  If the diagnosis shows the bug
        path is blocked, updates the verdict and contracts.
        """
        assert self.build_script_dir is not None
        assert self.build_dir is not None

        if self.verbose:
            print(
                f"[Triage] Stage 5 → debugging witness for "
                f"{func.name}#{result.issue_index}",
            )

        out_dir = self.harness_output_dir or Path("harnesses")
        test_name = f"test_{func.name}_{result.issue_index}"

        # -- Step 1: LLM generates GDB script --
        gdb_prompt = self._build_gdb_prompt(func, result, test_name)
        llm = self.llm_factory()

        gdb_script: str | None = None
        for attempt in range(DEFAULT_DEBUG_TURNS):
            if attempt == 0:
                resp = llm.complete(
                    gdb_prompt, system=DEBUG_GDB_SYSTEM_PROMPT,
                )
            else:
                resp = llm.complete(
                    "The GDB script produced no useful output. "
                    "Revise the breakpoints and try again. "
                    "Output a single ```gdb fenced block.",
                    system=DEBUG_GDB_SYSTEM_PROMPT,
                )
            text = (
                resp.text if hasattr(resp, "text") else str(resp)
            )
            extracted = _extract_gdb_block(text)
            if extracted:
                gdb_script = extracted
                break
            if self.verbose:
                print(
                    f"  [Debug] attempt {attempt + 1}: "
                    f"no GDB block in response",
                )

        if gdb_script is None:
            if self.verbose:
                print("  [Debug] failed to generate GDB script")
            return result

        # Write GDB script
        gdb_path = out_dir / f"_gdb_{test_name}.gdb"
        gdb_path.write_text(gdb_script + "\n")

        # -- Step 2: run under GDB in Docker --
        gdb_output = self._run_gdb_in_docker(
            test_name, out_dir, gdb_path,
        )
        if gdb_output is None:
            if self.verbose:
                print("  [Debug] GDB run failed")
            return result

        if self.verbose:
            preview = gdb_output[:500]
            print(f"  [Debug] GDB output ({len(gdb_output)} chars):")
            print(f"    {preview}")

        # -- Step 3: LLM diagnoses from GDB output --
        diag_prompt = self._build_diagnose_prompt(
            func, result, gdb_output,
        )
        llm2 = self.llm_factory()
        diag_resp = llm2.complete(
            diag_prompt, system=DEBUG_DIAGNOSE_SYSTEM_PROMPT,
        )
        diag_text = (
            diag_resp.text
            if hasattr(diag_resp, "text")
            else str(diag_resp)
        )

        # Parse JSON from response
        diagnosis = self._parse_diagnosis(diag_text)
        if diagnosis is None:
            if self.verbose:
                print("  [Debug] could not parse diagnosis JSON")
            return result

        result.debug_diagnosis = diagnosis.get("diagnosis", "")
        result.debug_explanation = diagnosis.get("explanation", "")

        if self.verbose:
            print(
                f"  [Debug] diagnosis={result.debug_diagnosis}: "
                f"{result.debug_explanation[:200]}",
            )

        # -- Step 4: update verdict if path_blocked --
        corrected = diagnosis.get("corrected_hypothesis", "feasible")
        if corrected != "feasible":
            result.hypothesis = corrected
            contracts = diagnosis.get("updated_contracts", [])
            if contracts:
                result.updated_contracts = contracts
            result.reasoning += (
                f"\n\n## Stage 5 Debug Correction\n\n"
                f"**Diagnosis**: {result.debug_diagnosis}\n\n"
                f"{result.debug_explanation}"
            )
            self._apply_contract_updates(func, result)
            self._mark_issue_review(
                func, result, status="false_positive",
            )
            if self.verbose:
                print(
                    f"  [Debug] verdict corrected to "
                    f"{result.hypothesis}",
                )

        return result

    def _build_gdb_prompt(
        self,
        func: Function,
        result: TriageResult,
        test_name: str,
    ) -> str:
        """Build the user prompt for GDB script generation."""
        lines = [
            "## Bug Report",
            "",
            f"Function: `{func.name}`",
            f"Issue: [{result.issue.severity}] "
            f"{result.issue.issue_kind} — "
            f"{result.issue.description}",
            "",
        ]

        if result.feasible_path:
            chain = " → ".join(result.feasible_path)
            lines.append(f"**Feasible path**: {chain}")
        if result.path_constraints:
            lines.append("")
            lines.append("**Path constraints** (branch guards along path):")
            for pc in result.path_constraints:
                lines.append(f"  - {pc}")
        if result.data_flow_trace:
            lines.append("")
            lines.append(f"**Data flow**: {result.data_flow_trace}")
        if result.assumptions:
            lines.append("")
            lines.append("**Assumptions**:")
            for a in result.assumptions:
                lines.append(f"  - {a}")

        # Source code of functions along path
        names_seen: set[str] = set()
        funcs_to_show: list[Function] = []
        for name in [result.entry_function, *result.feasible_path,
                      func.name]:
            if not name or name in names_seen:
                continue
            names_seen.add(name)
            found = self.db.get_function_by_name(name)
            if found:
                funcs_to_show.append(found[0])

        if funcs_to_show:
            lines.extend(["", "## Source Code", ""])
            for f in funcs_to_show:
                lines.append(
                    f"### `{f.name}` "
                    f"({self._rel_path(f.file_path)})",
                )
                lines.append(f"```c\n// {f.signature}")
                if f.source:
                    lines.append(f.source)
                lines.append("```")
                lines.append("")

        # Witness test source
        out_dir = self.harness_output_dir or Path("harnesses")
        witness_c = out_dir / f"{test_name}.c"
        if witness_c.exists():
            lines.extend([
                "## Unit Test Source",
                "",
                f"```c\n{witness_c.read_text()}\n```",
                "",
            ])

        lines.extend([
            "## Task",
            "",
            "Write a GDB batch script that sets breakpoints at key "
            "points along the feasible path and prints the variables "
            "mentioned in the path constraints and data flow trace. "
            "The goal is to determine whether the bug condition "
            "actually holds at runtime.",
        ])

        return "\n".join(lines)

    def _build_diagnose_prompt(
        self,
        func: Function,
        result: TriageResult,
        gdb_output: str,
    ) -> str:
        """Build the prompt for diagnosis from GDB output."""
        lines = [
            "## Original Bug Report",
            "",
            f"Function: `{func.name}`",
            f"Issue: [{result.issue.severity}] "
            f"{result.issue.issue_kind} — "
            f"{result.issue.description}",
            "",
        ]

        if result.feasible_path:
            chain = " → ".join(result.feasible_path)
            lines.append(f"**Feasible path**: {chain}")
        if result.assumptions:
            lines.append("")
            lines.append("**Assumptions**:")
            for a in result.assumptions:
                lines.append(f"  - {a}")
        if result.assertions:
            lines.append("")
            lines.append("**Expected violation**:")
            for a in result.assertions:
                lines.append(f"  - {a}")

        lines.extend([
            "",
            f"**Original reasoning**: {result.reasoning}",
            "",
            "## GDB Trace Output",
            "",
            f"```\n{gdb_output}\n```",
            "",
            "## Task",
            "",
            "Analyze the GDB output against the original reasoning. "
            "Determine which diagnosis category applies "
            "(path_blocked, not_reached, or asan_limitation) and "
            "output the JSON object as specified.",
        ])

        return "\n".join(lines)

    def _run_gdb_in_docker(
        self,
        test_name: str,
        out_dir: Path,
        gdb_path: Path,
    ) -> str | None:
        """Run the test binary under GDB inside Docker.

        Returns GDB's combined stdout+stderr, or None on failure.
        """
        assert self.build_script_dir is not None
        assert self.build_dir is not None

        config_path = self.build_script_dir / "config.json"
        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        project_path = config.get(
            "project_path", "/data/csong/opensource/unknown",
        )
        host_build_dir = str(self.build_dir)

        # Parse deps from witness source for runtime libraries
        witness_c = out_dir / f"{test_name}.c"
        deps_pkgs = self._parse_witness_deps(witness_c)

        inner_lines = [
            "#!/bin/bash",
            "set -e",
            "",
        ]

        if deps_pkgs:
            pkgs = " ".join(deps_pkgs)
            inner_lines.extend([
                "# Install runtime deps",
                f"apt-get update -qq && apt-get install -y -qq"
                f" {pkgs} >/dev/null 2>&1",
                "",
            ])

        inner_lines.extend([
            "# Run test under GDB (binary already built by Stage 4)",
            'echo "--- running under GDB ---"',
            "ASAN_OPTIONS=detect_leaks=0 "
            f"gdb -batch -x /test/{gdb_path.name} "
            f"/test/{test_name}",
        ])
        inner_path = out_dir / f"_inner_gdb_{test_name}.sh"
        inner_path.write_text("\n".join(inner_lines) + "\n")
        inner_path.chmod(0o755)

        gdb_sh = out_dir / f"gdb_{test_name}.sh"
        lines = [
            "#!/bin/bash",
            "set -e",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" '
            '&& pwd)"',
            f'PROJECT_PATH="{project_path}"',
            f'BUILD_DIR="{host_build_dir}"',
            "",
            "docker run --rm \\",
            "  --cap-add=SYS_PTRACE \\",
            "  --security-opt seccomp=unconfined \\",
            '  -e ASAN_OPTIONS=detect_leaks=0 \\',
            '  -v "$PROJECT_PATH":/workspace/src:ro \\',
            '  -v "$BUILD_DIR":/workspace/build \\',
            '  -v "$SCRIPT_DIR":/test \\',
            "  llm-summary-builder:latest \\",
            f"  bash /test/{inner_path.name}",
        ]
        gdb_sh.write_text("\n".join(lines) + "\n")
        gdb_sh.chmod(0o755)

        try:
            proc = subprocess.run(
                ["bash", str(gdb_sh)],
                capture_output=True, text=True, timeout=120,
            )
            return (proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            if self.verbose:
                print("  [Debug] GDB run timed out")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  [Debug] GDB run error: {e}")
            return None

    @staticmethod
    def _parse_diagnosis(text: str) -> dict[str, Any] | None:
        """Parse diagnosis JSON from LLM response."""
        # Try raw JSON first
        text = text.strip()
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass
        # Try extracting from ```json block
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass
        # Try finding first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(  # type: ignore[no-any-return]
                    text[start:end + 1],
                )
            except json.JSONDecodeError:
                pass
        return None
