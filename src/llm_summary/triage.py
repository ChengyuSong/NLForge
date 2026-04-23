"""Bug triage agent: prove safety or feasibility of verification issues.

The agent analyzes each SafetyIssue from the verification pass by tracing
caller/callee context through the DB. It produces one of two proofs:

1. **Safety proof**: Updated pre/post-conditions showing the issue cannot
   manifest. Key output: updated contracts for the DB so the verifier
   won't flag the same false positive again.

2. **Feasibility proof**: A concrete execution path showing the violation
   is reachable. Includes assumptions and assertions for future symbolic
   validation via ucsan.

Uses a phase-gated ReAct loop (ANALYZE → HYPOTHESIZE → VERDICT) to prevent
the agent from getting stuck.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_tools import (
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
    hypothesis: str  # "safe" or "feasible"
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

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        return result


# ---------------------------------------------------------------------------
# Workflow phases & gating
# ---------------------------------------------------------------------------

ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "ANALYZE": ["HYPOTHESIZE"],
    "HYPOTHESIZE": ["VERDICT"],
}

_DB_TOOLS = {
    "read_function_source",
    "get_callers",
    "get_callees",
    "get_contracts",
    "get_issues",
}

PHASE_TOOLS: dict[str, set[str]] = {
    "ANALYZE": _DB_TOOLS | GIT_TOOL_NAMES | {"transition_phase"},
    "HYPOTHESIZE": _DB_TOOLS | GIT_TOOL_NAMES | {"transition_phase"},
    "VERDICT": {
        "submit_verdict",
    },
}


# ---------------------------------------------------------------------------
# Tool definitions — assembled from shared + triage-specific
# ---------------------------------------------------------------------------

TRIAGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    *READ_TOOL_DEFINITIONS,
    *TRIAGE_ONLY_TOOL_DEFINITIONS,
    *_GIT_TOOL_DEFS,
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are a memory safety bug triage agent for C/C++ code.

## Task

You are given a potential safety issue found by verification. Each function
has a code-contract summary with Hoare-style requires/ensures per property
(memsafe, memleak, overflow). Your job is to produce a PROOF — either
proving the issue is safe (cannot manifest) or proving it is feasible (can
be triggered).

Use get_contracts to read any function's requires (preconditions) and
ensures (postconditions). Use get_issues to see verification issues.

## Two Outcomes

### 1. Safety Proof (hypothesis: "safe")

Prove the issue cannot manifest by showing that caller constraints prevent
the triggering condition. You MUST provide:
- **updated_contracts**: New or strengthened requires/ensures clauses
  that prove the issue away. Specify the function, property, clause_type
  (requires or ensures), and predicate (C expression).
- **assumptions**: What caller constraints make this safe.
  E.g., "all callers validate `size <= BUF_MAX` before calling this function"
- **assertions**: The violation condition that cannot be reached.
  E.g., "size * elem_size cannot overflow because size <= 1024 and elem_size <= 8"

### 2. Feasibility Proof (hypothesis: "feasible")

Prove the violation condition IS reachable by identifying a concrete
execution path. You MUST provide:
- **feasible_path**: The call chain from entry point to the bug site.
- **assumptions**: What input conditions trigger the path.
  E.g., "attacker controls width and height via PNG header fields"
- **assertions**: The violation condition that gets triggered.
  E.g., "width * height overflows uint32_t, causing undersized allocation"

## Key Principle

The verifier checks each function in isolation, assuming its requires
clauses hold. But callers may constrain arguments more tightly than the
declared contracts. Your job is to check the CALLER CONTEXT:

- An integer overflow in `alloc(width * height)` is real only if callers
  can pass values large enough to overflow
- A null deref is real only if callers can actually pass NULL
- A buffer overflow is real only if callers can pass a size exceeding bounds

IMPORTANT: If a function has zero callers in the call graph, it could be an
ENTRY FUNCTION — directly callable by external code with NO constraints on
its arguments. This means the function must be safe for ALL valid inputs.
Do NOT treat "no callers" as evidence of safety or unreachability.
Similarly, "deprecated" does NOT mean unused — existing code may still call
deprecated APIs. Treat deprecated functions as reachable entry points.

## Workflow (strictly enforced)

### ANALYZE phase
1. Read the flagged function's source code
2. Read its contracts (get_contracts) — understand its requires/ensures
3. Read caller functions — understand how the function is invoked
4. Check callee contracts — understand what guarantees callees provide
5. Trace the data flow: where do the problematic parameters come from?

### HYPOTHESIZE phase
1. Formulate your hypothesis: safe or feasible
2. Gather remaining evidence (you can still read functions/contracts)
3. Build your proof: identify specific constraints or paths

### VERDICT phase
1. Call submit_verdict with your proof
2. Include relevant_functions: all functions whose behavior matters for
   the proof (the target, constraining callers, relevant callees). These
   will be used to set up symbolic validation scope.
3. Include validation_plan: unit-test-style harnesses for validation.
   Each element has an "entries" list — functions test() will call in order.
   DO NOT include callees within relevant_functions that will be called
   by top-level functions — they are kept as real code automatically.
   Sequential: [{"entries": ["set_X", "destroy_X"]}].
   Independent: [{"entries": ["entry_a"]}, {"entries": ["entry_b"]}].

## Rules
- Always read the function source and at least its direct callers
- For safety proofs: you MUST show which callers constrain the parameters
- For feasibility proofs: you MUST show a concrete call chain
- Include assumptions and assertions — these will be used for symbolic validation
- Do not guess — if caller constraints are ambiguous, lean toward "feasible"
"""


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class TriageToolExecutor:
    """Phase-gated wrapper around the shared ToolExecutor for triage."""

    def __init__(
        self, db: SummaryDB, verbose: bool = False,
        git_tools: GitTools | None = None,
    ) -> None:
        self._executor = ToolExecutor(
            db, verbose=verbose, git_tools=git_tools,
        )
        self._current_phase = "ANALYZE"

    @property
    def phase(self) -> str:
        return self._current_phase

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call, enforcing phase gating."""
        allowed = PHASE_TOOLS.get(self._current_phase, set())
        if tool_name not in allowed:
            return {
                "error": (
                    f"Tool '{tool_name}' not allowed in {self._current_phase} "
                    f"phase. Allowed: {sorted(allowed)}. "
                    f"Use transition_phase to advance."
                ),
            }

        # Intercept transition_phase to manage local phase state
        if tool_name == "transition_phase":
            return self._transition_phase(tool_input)

        return self._executor.execute(tool_name, tool_input)

    def _transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        next_phase = inp["next_phase"]
        allowed_next = ALLOWED_TRANSITIONS.get(self._current_phase, [])
        if next_phase not in allowed_next:
            return {
                "error": (
                    f"Cannot transition from {self._current_phase} to "
                    f"{next_phase}. Allowed: {allowed_next}"
                ),
            }
        prev = self._current_phase
        self._current_phase = next_phase
        return {"previous_phase": prev, "current_phase": next_phase}


# ---------------------------------------------------------------------------
# Triage controller (ReAct loop)
# ---------------------------------------------------------------------------

MAX_TURNS = 50


class TriageAgent:
    """Triage verification issues using an LLM ReAct loop."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.project_path = project_path

    def _rel_path(self, abs_path: str) -> str:
        if self.project_path is None:
            return abs_path
        try:
            return str(Path(abs_path).relative_to(self.project_path))
        except ValueError:
            return abs_path

    def triage_issue(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
    ) -> TriageResult:
        """Triage a single issue via LLM ReAct loop."""
        git = (
            GitTools(self.project_path) if self.project_path else None
        )
        executor = TriageToolExecutor(
            self.db, verbose=self.verbose, git_tools=git,
        )
        user_prompt = self._build_issue_prompt(func, issue, issue_index)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        if self.verbose:
            print(f"\n[Triage] {func.name} issue #{issue_index}: "
                  f"{issue.issue_kind} at {issue.location}")
            print(f"  {issue.description}")

        verdict_result: dict[str, Any] | None = None

        for turn in range(MAX_TURNS):
            # Filter tools to only those allowed in current phase
            allowed = PHASE_TOOLS.get(executor.phase, set())
            # Hide git tools when no project_path was given
            if git is None:
                allowed = allowed - GIT_TOOL_NAMES
            tools = [t for t in TRIAGE_TOOL_DEFINITIONS if t["name"] in allowed]

            response = self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                system=TRIAGE_SYSTEM_PROMPT,
            )

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(f"[Triage] LLM stopped at turn {turn + 1} ({stop})")
                break

            if stop != "tool_use":
                if self.verbose:
                    print(f"[Triage] Unexpected stop_reason: {stop}")
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
                    if self.verbose and not getattr(block, "thought", False):
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

                    if self.verbose:
                        err = result.get("error")
                        if err:
                            print(f"  [Tool] {block.name} -> ERROR: {err[:150]}")
                        elif block.name == "submit_verdict":
                            print(f"  [Tool] submit_verdict -> "
                                  f"{result.get('hypothesis')}")
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"  [Tool] {block.name}({arg})")

                    if block.name == "submit_verdict" and result.get("accepted"):
                        verdict_result = result

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            if verdict_result is not None:
                break

        return self._build_result(func, issue, issue_index, verdict_result)

    def triage_function(
        self,
        func: Function,
        *,
        severity_filter: set[str] | None = None,
    ) -> list[TriageResult]:
        """Triage all issues for a function."""
        if func.id is None:
            return []

        vs = self.db.get_verification_summary_by_function_id(func.id)
        if vs is None or not vs.issues:
            return []

        results = []
        for i, issue in enumerate(vs.issues):
            if severity_filter and issue.severity not in severity_filter:
                continue
            result = self.triage_issue(func, issue, i)
            results.append(result)
        return results

    def _build_issue_prompt(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
    ) -> str:
        lines = [
            "## Issue to Triage",
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
            "Follow the workflow: ANALYZE -> HYPOTHESIZE -> VERDICT.",
            "Start by reading the function source and its contracts "
            "(get_contracts), then check its callers.",
            "Produce either a safety proof or a feasibility proof.",
        ])
        return "\n".join(lines)

    def _build_result(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        verdict: dict[str, Any] | None,
    ) -> TriageResult:
        if verdict is not None:
            hyp = verdict.get("hypothesis", "feasible")
            if hyp not in ("safe", "feasible"):
                hyp = "feasible"
            return TriageResult(
                function_name=func.name,
                issue_index=issue_index,
                issue=issue,
                hypothesis=hyp,
                reasoning=verdict.get("reasoning", ""),
                updated_contracts=verdict.get("updated_contracts", []),
                feasible_path=verdict.get("feasible_path", []),
                assumptions=verdict.get("assumptions", []),
                assertions=verdict.get("assertions", []),
                relevant_functions=verdict.get("relevant_functions", []),
                validation_plan=verdict.get("validation_plan", []),
            )

        return TriageResult(
            function_name=func.name,
            issue_index=issue_index,
            issue=issue,
            hypothesis="feasible",
            reasoning="Agent did not submit a verdict within turn limit.",
        )
