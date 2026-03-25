"""Reflect on validation outcomes via a ReAct agent.

Given the full validation context (verdict, plan, shim, annotated source,
and execution outcome), the reflection agent can:

1. Assess whether the original hypothesis is correct
2. Mark issues as false_positive or confirmed (upsert_review)
3. Correct wrong callee summaries that cause FPs (update_summary)
4. Read callee source/summaries to investigate root causes

The agent uses the same shared tools as triage (read_function_source,
get_summaries, etc.) plus write tools (upsert_review, update_summary).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agent_tools import (
    GIT_TOOL_NAMES,
    READ_TOOL_DEFINITIONS,
    REFLECTION_VERDICT_TOOL_DEFINITIONS,
    WRITE_TOOL_DEFINITIONS,
    ToolExecutor,
)
from .bbid_extractor import format_annotated_function, parse_cfg_dump
from .db import SummaryDB
from .git_tools import GIT_TOOL_DEFINITIONS as _GIT_TOOL_DEFS
from .git_tools import GitTools
from .llm.base import LLMBackend

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = """\
You are a validation analyst for a concolic execution system (ucsan).

## How the system works

- ucsan is a concolic executor: it starts with an initial input, runs the \
binary, and at each conditional branch the SMT solver can generate a new \
input ("seed") that flips the branch.
- The **plan** tells the scheduler which branches (BB IDs) to prioritize.
- The **shim** provides callee stubs (functions not compiled as real code). \
Stubs that return wrong values, miss post-conditions, or don't model side \
effects can cause spurious crashes that are harness artifacts, not real bugs.
- A trace status of **infeasible** means the solver proved it cannot reach \
that BB combination — this is strong evidence.
- A trace status of **missed** means the executor ran out of budget — \
weaker evidence, the path may still be reachable.
- ucsan exit codes: 150=ubi, 151=uaf, 152=oob, 153=null_deref, 171=panic.

## Your Task

Analyze the validation outcome and take action:

1. **Diagnose**: Is the issue a real bug or a false positive? Check the \
crash type, trace coverage, and whether shim stubs could have caused \
spurious results.

2. **Root-cause FPs**: If the issue is a false positive caused by a wrong \
callee summary (e.g., "png_malloc may return NULL" when it actually calls \
png_error on failure), investigate the callee:
   - Use `read_function_source` to read the callee's code
   - Use `get_summaries` to see its current summaries
   - Use `update_summary` to correct the wrong summary

3. **Record your findings**:
   - Use `upsert_review` to mark issues as false_positive or confirmed
   - Use `update_summary` to fix wrong callee summaries (so callers get \
re-verified with correct data on the next incremental run)

4. **Submit**: Call `submit_reflection` with your final verdict.

## Common FP Patterns

- **may_be_null=true but function never returns NULL**: wrapper allocators \
that call an error/abort handler on failure. Fix: update allocation summary \
to set may_be_null=false.
- **nullable contract but pointer is always valid**: function guarantees \
non-null via internal checks. Fix: update memsafe/verification summary.
- **Missing init post-condition**: function initializes output but init \
summary doesn't capture it. Fix: update init summary.
- **Wrong free post-condition**: function doesn't actually free in the \
flagged path. Fix: update free summary.

## Decision Guidelines

- **safe + high confidence**: Issue is definitely FP. Mark as false_positive. \
If caused by wrong callee summary, fix the summary too.
- **feasible + high confidence**: Issue is a real bug. Mark as confirmed.
- **Low confidence / ambiguous**: Submit with action=re-validate or re-triage.

## Rules
- Always investigate the callee if the issue has a callee field
- When updating a summary, use get_summaries first to see current state
- Only update summaries you are confident are wrong
- After all reviews and updates, call submit_reflection
"""

# Legacy single-shot prompt (kept for backward compat)
REFLECTION_PROMPT = """\
You are a validation analyst for a concolic execution system (ucsan).

You are given the full context of a validation run:

1. **Triage verdict**: the original hypothesis about a potential bug
2. **Validation plan**: what traces/paths the executor was told to explore
3. **Harness shim**: the C shim with callee stubs and test() entry point
4. **Annotated source**: real code with BB IDs showing branch structure
5. **Validation outcome**: what actually happened (traces, crashes, assertions)
6. **Caller context**: how actual callers in the codebase invoke the function

## How the system works

- ucsan is a concolic executor: it starts with an initial input, runs the \
binary, and at each conditional branch the SMT solver can generate a new \
input ("seed") that flips the branch.
- The **plan** tells the scheduler which branches (BB IDs) to prioritize.
- The **shim** provides callee stubs (functions not compiled as real code). \
Stubs that return wrong values, miss post-conditions, or don't model side \
effects can cause spurious crashes that are harness artifacts, not real bugs.
- A trace status of **infeasible** means the solver proved it cannot reach \
that BB combination — this is strong evidence.
- A trace status of **missed** means the executor ran out of budget — \
weaker evidence, the path may still be reachable.
- ucsan exit codes: 150=ubi, 151=uaf, 152=oob, 153=null_deref, 171=panic.

## Context

{context}

## Validation Plan

{plan_section}

## Harness Shim

{shim_section}

## Annotated Source Code

{annotated_sources}

## Caller Context

{caller_context}

## Validation Outcome

{outcome_section}

## Your Task

Analyze ALL the evidence and produce a revised verdict as JSON:

```json
{{
  "hypothesis": "safe | feasible",
  "confidence": "high | medium | low",
  "reasoning": "Why you believe this hypothesis, citing specific evidence",
  "action": "accept | re-validate | re-triage",
  "action_reason": "Why this action is needed",
  "original_correct": true | false,
  "practically_triggerable": true | false,
  "triggerability_analysis": "Can any real caller actually pass the violating \
input? Cite specific call sites and what arguments they use.",
  "crash_analysis": {{
    "is_real_bug": true | false | "unknown",
    "is_harness_artifact": true | false | "unknown",
    "explanation": "What caused the crash (if any)"
  }},
  "plan_analysis": {{
    "targeted_right_paths": true | false,
    "explanation": "Whether the plan explored the right code paths"
  }}
}}
```

Decision guidelines:
- **action=accept**: You are confident in the revised hypothesis. Evidence is \
strong enough (solver-proven infeasibility, matching crash type and path, \
or clear harness artifact). No further validation needed.
- **action=re-validate**: The hypothesis might be right but the plan was wrong \
or incomplete. Need a better plan to test the same hypothesis.
- **action=re-triage**: The hypothesis is wrong. Need fresh analysis with the \
new evidence (e.g., original said feasible but path is infeasible).

Key checks:
- If a crash type doesn't match the predicted issue, check if the shim stubs \
could have caused it (missing post-conditions, wrong return values, \
uninitialized struct fields).
- If a trace is infeasible, that's solver-proven — the path truly cannot be \
reached, which may confirm or contradict the hypothesis.
- If all assertions pass but no crash, the predicted bug may not exist.
- A crash from a stubbed function's missing side-effect is NOT a real bug.
- **Triggerability**: A contract violation that is technically feasible but \
never triggered by any real caller is a false positive in practice. Check \
the caller context — if all callers pass safe values (e.g., string literals, \
pre-validated pointers), the bug is not practically triggerable even if the \
contract allows it. Set `practically_triggerable: false` and consider \
revising the hypothesis to safe.

Output ONLY the JSON block, no other text.
"""


# ---------------------------------------------------------------------------
# Tool definitions for the reflection agent
# ---------------------------------------------------------------------------

REFLECTION_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    *READ_TOOL_DEFINITIONS,
    *WRITE_TOOL_DEFINITIONS,
    *REFLECTION_VERDICT_TOOL_DEFINITIONS,
    *_GIT_TOOL_DEFS,
]

_REFLECTION_TOOL_NAMES = {t["name"] for t in REFLECTION_TOOL_DEFINITIONS}

MAX_REFLECTION_TURNS = 30


# ---------------------------------------------------------------------------
# Context builders (shared between single-shot and agent)
# ---------------------------------------------------------------------------


def _read_file_if_exists(path: Path) -> str | None:
    """Read a file's contents, or return None if it doesn't exist."""
    if path.exists():
        return path.read_text()
    return None


def _build_caller_context(
    func_name: str,
    project_path: Path | None = None,
    context_lines: int = 2,
) -> str:
    """Build caller context by searching for actual call sites via git grep.

    Shows raw source lines where the function is called, so the LLM can see
    what arguments real callers pass (e.g., string literals vs variables).
    """
    if not project_path or not (project_path / ".git").exists():
        return "_No project path available for call site search._"

    git = GitTools(project_path)

    # Search for call sites: literal "func_name(" pattern
    grep_result = git.git_grep(
        f"{func_name}(",
        glob="*.c",
        max_results=20,
        context=context_lines,
    )

    if grep_result.get("error"):
        return f"_Git grep failed: {grep_result['error']}_"

    matches = grep_result.get("matches", [])
    if not matches:
        return (
            f"_No call sites found for `{func_name}` in the codebase. "
            f"It may only be called from external code._"
        )

    # Filter out the function definition itself (line with opening brace)
    call_sites = [
        m for m in matches
        if f"{func_name}(" in m and "{" not in m.split(func_name, 1)[-1]
    ]
    if not call_sites:
        call_sites = matches  # fallback: show all matches

    blocks: list[str] = [
        f"Actual call sites for `{func_name}` in the codebase "
        f"({len(call_sites)} found):\n",
        "```",
    ]
    for m in call_sites:
        blocks.append(m)
    blocks.append("```")

    return "\n".join(blocks)


def build_reflection_context(
    verdict: dict[str, Any],
    outcome: dict[str, Any],
    db: SummaryDB,
    cfg_dump_path: str | None = None,
    output_dir: str | None = None,
    entry_name: str | None = None,
    project_path: Path | None = None,
) -> dict[str, str]:
    """Build all context sections for the reflection prompt.

    Returns dict with keys: context, plan_section, shim_section,
    annotated_sources, caller_context, outcome_section.
    """
    func_name = verdict["function_name"]
    plan_name = entry_name or func_name
    hypothesis = verdict.get("hypothesis", "unknown")
    issue = verdict.get("issue", {})
    relevant = verdict.get("relevant_functions", [func_name])

    # -- Triage context --
    assumptions = verdict.get("assumptions", [])
    assertions = verdict.get("assertions", [])

    assumptions_text = "None."
    if assumptions:
        assumptions_text = "\n".join(
            f"  {i}. {a}" for i, a in enumerate(assumptions, 1)
        )

    assertions_text = "None."
    if assertions:
        assertions_text = "\n".join(
            f"  {i}. {a}" for i, a in enumerate(assertions, 1)
        )

    context = (
        f"### Triage Verdict\n\n"
        f"- Function: `{func_name}`\n"
        f"- Hypothesis: **{hypothesis}**\n"
        f"- Issue: [{issue.get('severity', '')}] "
        f"{issue.get('issue_kind', '')} — "
        f"{issue.get('description', '')}\n\n"
        f"### Reasoning\n\n"
        f"{verdict.get('reasoning', 'N/A')}\n\n"
        f"### Assumptions\n\n{assumptions_text}\n\n"
        f"### Assertions\n\n{assertions_text}"
    )

    # -- Plan section --
    plan_section = "_No plan available._"
    if output_dir:
        plan_path = Path(output_dir) / f"plan_{plan_name}_validation.json"
        plan_text = _read_file_if_exists(plan_path)
        if plan_text:
            plan_section = f"```json\n{plan_text}\n```"

    # -- Shim section --
    shim_section = "_No shim available._"
    if output_dir:
        shim_path = Path(output_dir) / f"shim_{plan_name}.c"
        shim_text = _read_file_if_exists(shim_path)
        if shim_text:
            shim_section = f"```c\n{shim_text}\n```"

    # -- Annotated sources --
    annotated_sources = "_No CFG dump available._"
    if cfg_dump_path is None and output_dir:
        cfg_candidate = Path(output_dir) / f"cfg_{plan_name}.txt"
        if cfg_candidate.exists():
            cfg_dump_path = str(cfg_candidate)

    if cfg_dump_path and Path(cfg_dump_path).exists():
        infos = parse_cfg_dump(cfg_dump_path)
        blocks = []
        for rname in relevant:
            funcs = db.get_function_by_name(rname)
            if not funcs:
                continue
            func = funcs[0]
            if not func.file_path or not func.line_start or not func.line_end:
                continue
            annotated = format_annotated_function(
                infos, func.file_path, func.line_start, func.line_end,
            )
            blocks.append(
                f"### `{func.name}` ({func.file_path}:"
                f"{func.line_start}-{func.line_end})\n\n"
                f"```c\n{annotated}\n```"
            )
        if blocks:
            annotated_sources = "\n\n".join(blocks)

    # -- Caller context --
    caller_context = _build_caller_context(
        func_name, project_path=project_path,
    )

    # -- Outcome section --
    crashes = outcome.get("crashes", [])
    crash_text = "None."
    if crashes:
        crash_text = ", ".join(
            f"exit {c['exit_code']} ({c['kind']})" for c in crashes
        )

    traces = []
    for t_key, label in [
        ("traces_covered", "COVERED"),
        ("traces_infeasible", "INFEASIBLE"),
        ("traces_missed", "MISSED"),
    ]:
        for t in outcome.get(t_key, []):
            goal = t if isinstance(t, str) else t.get("goal", str(t))
            traces.append(f"  - [{label}] {goal}")
    traces_text = "\n".join(traces) if traces else "No trace data."

    a_failures = outcome.get("assertion_failures", [])
    a_passes = outcome.get("assertion_passes", [])
    assertion_lines = []
    for a in a_failures:
        assertion_lines.append(
            f"  - FAIL: id={a['assertion_id']} types={a.get('types', [])}"
        )
    for a in a_passes:
        assertion_lines.append(
            f"  - PASS: id={a['assertion_id']} types={a.get('types', [])}"
        )
    assertions_result = (
        "\n".join(assertion_lines) if assertion_lines else "None."
    )

    outcome_section = (
        f"- **Classification**: {outcome.get('outcome', 'unknown')}\n"
        f"- **Crashes**: {crash_text}\n"
        f"- **Summary**: {outcome.get('summary', 'N/A')}\n\n"
        f"### Trace Coverage\n\n{traces_text}\n\n"
        f"### Assertion Results\n\n{assertions_result}"
    )

    return {
        "context": context,
        "plan_section": plan_section,
        "shim_section": shim_section,
        "annotated_sources": annotated_sources,
        "caller_context": caller_context,
        "outcome_section": outcome_section,
    }


# ---------------------------------------------------------------------------
# Legacy single-shot reflect (backward compat)
# ---------------------------------------------------------------------------


def reflect(
    verdict: dict[str, Any],
    outcome: dict[str, Any],
    db: SummaryDB,
    llm: LLMBackend,
    cfg_dump_path: str | None = None,
    output_dir: str | None = None,
    entry_name: str | None = None,
    project_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run single-shot reflection on a validation outcome (legacy).

    For the agent-based version with summary correction, use ReflectionAgent.
    """
    sections = build_reflection_context(
        verdict, outcome, db,
        cfg_dump_path=cfg_dump_path,
        output_dir=output_dir,
        entry_name=entry_name,
        project_path=project_path,
    )

    prompt = REFLECTION_PROMPT.format(**sections)

    if verbose:
        func_name = verdict["function_name"]
        print(f"[Reflect] {func_name}: {outcome.get('outcome', '?')}")

    response = llm.complete(prompt)

    # Parse JSON
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        result: dict[str, Any] = json.loads(json_match.group(1))
    else:
        result = json.loads(response.strip())

    if verbose:
        print(f"  hypothesis: {result.get('hypothesis')}")
        print(f"  confidence: {result.get('confidence')}")
        print(f"  action: {result.get('action')}")
        print(f"  original_correct: {result.get('original_correct')}")
        crash = result.get("crash_analysis", {})
        if crash:
            print(f"  crash: real={crash.get('is_real_bug')}, "
                  f"artifact={crash.get('is_harness_artifact')}")
        triggerable = result.get("practically_triggerable")
        if triggerable is not None:
            print(f"  practically_triggerable: {triggerable}")
        if result.get("triggerability_analysis"):
            print(
                f"  triggerability: "
                f"{result['triggerability_analysis'][:300]}",
            )
        if result.get("reasoning"):
            print(f"  reasoning: {result['reasoning'][:300]}")

    return result


# ---------------------------------------------------------------------------
# Agent-based reflection
# ---------------------------------------------------------------------------


class ReflectionAgent:
    """Reflect on validation outcomes via a ReAct agent loop.

    Unlike the legacy single-shot `reflect()`, the agent can:
    - Read callee source and summaries to investigate FP root causes
    - Correct wrong callee summaries via update_summary
    - Mark issues as FP/confirmed via upsert_review
    """

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

    def reflect(
        self,
        verdict: dict[str, Any],
        outcome: dict[str, Any],
        *,
        cfg_dump_path: str | None = None,
        output_dir: str | None = None,
        entry_name: str | None = None,
    ) -> dict[str, Any]:
        """Run agent-based reflection on a validation outcome.

        Returns the reflection verdict dict, including any summaries_updated
        and issues_reviewed fields from the agent's actions.
        """
        git = (
            GitTools(self.project_path) if self.project_path else None
        )
        executor = ToolExecutor(
            self.db,
            verbose=self.verbose,
            git_tools=git,
            model_used=self.llm.model,
        )

        # Build context and user prompt
        sections = build_reflection_context(
            verdict, outcome, self.db,
            cfg_dump_path=cfg_dump_path,
            output_dir=output_dir,
            entry_name=entry_name,
            project_path=self.project_path,
        )

        user_prompt = self._build_user_prompt(verdict, outcome, sections)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        func_name = verdict.get("function_name", "unknown")
        if self.verbose:
            print(f"[Reflect-Agent] {func_name}: "
                  f"{outcome.get('outcome', '?')}")

        verdict_result: dict[str, Any] | None = None

        # Compute allowed tools (hide git if no project_path)
        allowed = set(_REFLECTION_TOOL_NAMES)
        if git is None:
            allowed -= GIT_TOOL_NAMES

        tools = [
            t for t in REFLECTION_TOOL_DEFINITIONS
            if t["name"] in allowed
        ]

        for turn in range(MAX_REFLECTION_TURNS):
            response = self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                system=REFLECTION_SYSTEM_PROMPT,
            )

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(f"[Reflect-Agent] stopped at turn "
                          f"{turn + 1} ({stop})")
                break

            if stop != "tool_use":
                if self.verbose:
                    print(f"[Reflect-Agent] unexpected stop: {stop}")
                break

            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    entry: dict[str, Any] = {
                        "type": "text", "text": block.text,
                    }
                    if getattr(block, "thought", False):
                        entry["thought"] = True
                    sig = getattr(block, "thought_signature", None)
                    if sig:
                        entry["thought_signature"] = sig
                    assistant_content.append(entry)
                    if (self.verbose
                            and not getattr(block, "thought", False)):
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

                    result = executor.execute(
                        block.name, block.input, allowed=allowed,
                    )

                    if self.verbose:
                        err = result.get("error")
                        if err:
                            print(f"  [Tool] {block.name} -> "
                                  f"ERROR: {err[:150]}")
                        elif block.name == "submit_reflection":
                            hyp = result.get("hypothesis", "?")
                            print(f"  [Tool] submit_reflection -> "
                                  f"{hyp}")
                        elif block.name in (
                            "upsert_review", "update_summary",
                        ):
                            print(f"  [Tool] {block.name} -> "
                                  f"{json.dumps(result)[:200]}")
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"  [Tool] {block.name}({arg})")

                    if (block.name == "submit_reflection"
                            and result.get("accepted")):
                        verdict_result = result

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({
                "role": "assistant", "content": assistant_content,
            })
            messages.append({
                "role": "user", "content": tool_results,
            })

            if verdict_result is not None:
                break

        if verdict_result is None:
            # Agent didn't submit — return a default
            return {
                "hypothesis": outcome.get("outcome", "unknown"),
                "confidence": "low",
                "reasoning": (
                    "Reflection agent did not submit a verdict "
                    f"within {MAX_REFLECTION_TURNS} turns."
                ),
                "action": "re-triage",
                "original_correct": False,
            }

        return verdict_result

    def _build_user_prompt(
        self,
        verdict: dict[str, Any],
        outcome: dict[str, Any],
        sections: dict[str, str],
    ) -> str:
        """Build the initial user prompt with full validation context."""
        issue = verdict.get("issue", {})
        func_name = verdict.get("function_name", "unknown")
        issue_index = verdict.get("issue_index", 0)

        lines = [
            "## Validation Context",
            "",
            sections["context"],
            "",
            "## Validation Plan",
            "",
            sections["plan_section"],
            "",
            "## Harness Shim",
            "",
            sections["shim_section"],
            "",
            "## Annotated Source Code",
            "",
            sections["annotated_sources"],
            "",
            "## Caller Context",
            "",
            sections["caller_context"],
            "",
            "## Validation Outcome",
            "",
            sections["outcome_section"],
            "",
            "## Instructions",
            "",
            f"Function: `{func_name}`, issue #{issue_index}",
            f"Issue: [{issue.get('severity', '')}] "
            f"{issue.get('issue_kind', '')} — "
            f"{issue.get('description', '')}",
        ]

        callee = issue.get("callee")
        if callee:
            lines.append(f"Callee involved: `{callee}`")
            lines.append("")
            lines.append(
                f"This issue may be caused by a wrong summary of `{callee}`. "
                f"Investigate by reading its source and summaries. If the "
                f"summary is wrong, use update_summary to correct it."
            )

        lines.extend([
            "",
            "Analyze the outcome, review the issue, fix any wrong summaries, "
            "then call submit_reflection.",
        ])

        return "\n".join(lines)
