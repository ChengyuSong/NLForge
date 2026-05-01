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
    hypothesis: str  # "safe", "contract_gap", or "feasible"
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


# ---------------------------------------------------------------------------
# Shared ReAct loop
# ---------------------------------------------------------------------------

DEFAULT_REACHABILITY_TURNS = 30
DEFAULT_DOC_AUDIT_TURNS = 25
DEFAULT_VERDICT_TURNS = 35


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
    ) -> None:
        self.db = db
        self.llm_factory = llm_factory
        self.verbose = verbose
        self.project_path = project_path
        self.output_path = output_path
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
        elif result.hypothesis == "feasible":
            self._mark_issue_review(
                func, result, status="confirmed",
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
    ) -> list[TriageResult]:
        """Triage all issues for a function, skipping already-reviewed ones."""
        if func.id is None:
            return []

        vs = self.db.get_verification_summary_by_function_id(func.id)
        if vs is None or not vs.issues:
            return []

        # Look up existing reviews to skip already-triaged issues
        fingerprints = [iss.fingerprint() for iss in vs.issues]
        reviewed = self.db.get_issue_reviews_by_fingerprints(
            func.id, fingerprints,
        )

        results = []
        for i, issue in enumerate(vs.issues):
            if severity_filter and issue.severity not in severity_filter:
                continue
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
                hypothesis="feasible",
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
