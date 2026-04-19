"""LLM-based memory leak detection pass.

Compares allocation summaries against free summaries to find unmatched
allocations (memory leaks).  Produces simplified allocation/free summaries
for compositional bottom-up analysis and reports ``memory_leak`` issues.
"""

from typing import Any

from .base_summarizer import BaseSummarizer
from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    Allocation,
    AllocationType,
    FreeOp,
    Function,
    LeakSummary,
    SafetyIssue,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

LEAK_SYSTEM_PROMPT = """\
You are analyzing C/C++ functions for memory leaks by comparing their \
allocation and free summaries.

A **memory leak** occurs when a heap allocation is:
- Not freed before the function returns, AND
- Not returned to the caller, AND
- Not stored to a caller-visible location (struct field, global, output parameter)

For entry-point functions like `main()`, storing to a global does NOT \
prevent a leak -- there is no caller to observe it.

Try to reason about path feasibility when matching allocations and frees, \
and when simplifying the summary (e.g., exclude allocations and frees on \
infeasible paths).

## Output

For each function, produce:
1. **leaks**: allocations that are never freed and not returned/stored -- these are bugs.
2. **simplified_allocations**: allocations NOT freed internally that ARE \
returned or stored to a caller-visible location. These are NOT leaks in this \
function, but callers must ensure they are eventually freed.
3. **simplified_frees**: frees of caller-provided pointers (parameters, \
struct fields passed in). These tell callers what this function frees.
"""

LEAK_USER_PROMPT = """\
## Function

Function: `{name}`
Signature: `{signature}`
File: {file_path}

```c
{source}
```

## This Function's Allocation Summary

{alloc_section}

## This Function's Free Summary

{free_section}

## Callee Post-conditions

{callee_section}

## Task

Match allocations against frees. For each heap allocation:
- If freed internally (directly or via callee) → resolved, omit from output
- If returned or stored to caller-visible location → add to `simplified_allocations`
- If neither freed, returned, nor stored → report as a leak

**Callee propagation**: If a callee has "unfreed allocations (caller must \
handle)", those allocations are now THIS function's responsibility. Match \
them against this function's frees. If still unresolved, propagate them \
as `simplified_allocations` or report as leaks.

{entry_note}

Respond in JSON:
```json
{{{{
  "function": "{name}",
  "description": "One-sentence summary of leak analysis",
  "leaks": [
    {{{{
      "allocation": "source expression (e.g., malloc(n))",
      "stored_to": "where it is stored, or null",
      "reason": "why it is not freed",
      "severity": "high|medium|low"
    }}}}
  ],
  "simplified_allocations": [
    {{{{
      "source": "malloc|calloc|realloc|...",
      "size_expr": "size expression or null",
      "returned": true,
      "stored_to": "field or variable, or null",
      "may_be_null": true
    }}}}
  ],
  "simplified_frees": [
    {{{{
      "target": "parameter or field being freed",
      "target_kind": "parameter|field",
      "deallocator": "free|custom_free|...",
      "conditional": false,
      "condition": "condition expression or null",
      "description": "optional description for transitive frees"
    }}}}
  ]
}}}}
```
\
"""

_LEAK_ITEM = {
    "type": "object",
    "properties": {
        "allocation": {"type": "string"},
        "stored_to": {"type": "string"},
        "reason": {"type": "string"},
        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["allocation", "reason", "severity"],
}

_SIMPLIFIED_ALLOC_ITEM = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "size_expr": {"type": "string"},
        "returned": {"type": "boolean"},
        "stored_to": {"type": "string"},
        "may_be_null": {"type": "boolean"},
    },
    "required": ["source"],
}

_SIMPLIFIED_FREE_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "target_kind": {"type": "string", "enum": ["parameter", "field"]},
        "deallocator": {"type": "string"},
        "conditional": {"type": "boolean"},
        "condition": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["target", "deallocator"],
}

LEAK_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "description": {"type": "string"},
        "leaks": {"type": "array", "items": _LEAK_ITEM},
        "simplified_allocations": {
            "type": "array",
            "items": _SIMPLIFIED_ALLOC_ITEM,
        },
        "simplified_frees": {
            "type": "array",
            "items": _SIMPLIFIED_FREE_ITEM,
        },
    },
    "required": ["function", "description", "leaks",
                  "simplified_allocations", "simplified_frees"],
})


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class LeakSummarizer(BaseSummarizer):
    """Detects memory leaks by comparing alloc vs free summaries.

    Returns a LeakSummary with simplified alloc/free for compositional
    bottom-up analysis plus memory_leak issues.
    """

    _extra_stats = {"leaks_found": 0}

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        entry_functions: set[str] | None = None,
    ):
        super().__init__(db, llm, verbose=verbose, log_file=log_file, pass_label="leak pass")
        self.entry_functions = entry_functions or {"main"}

    def should_skip(
        self,
        func: Function,
        callee_summaries: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Skip if no own allocations AND no callee leaves allocs unresolved."""
        if func.id is None:
            return (False, "")
        attrs_reason = self._attrs_skip_reason(func, "leak")
        if attrs_reason:
            return (True, attrs_reason)
        alloc_summary = self.db.get_summary_by_function_id(func.id)
        if alloc_summary and alloc_summary.allocations:
            return (False, "")
        if callee_summaries:
            for cs in callee_summaries.values():
                if getattr(cs, "simplified_allocations", None):
                    return (False, "")
        return (True, "no own allocations and no callee unresolved allocs")

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, LeakSummary] | None = None,
        **kwargs: Any,
    ) -> LeakSummary:
        """Detect leaks for a single function. Returns LeakSummary."""
        assert func.id is not None

        # Read alloc/free summaries from DB
        alloc_summary = self.db.get_summary_by_function_id(func.id)
        free_summary = self.db.get_free_summary_by_function_id(func.id)

        skip, _reason = self.should_skip(func, callee_summaries)
        if skip:
            self.record_skip()
            # Still propagate frees of caller-provided pointers so callers
            # can see what their inputs got freed.
            simplified_frees: list[FreeOp] = []
            if free_summary and free_summary.frees:
                simplified_frees = [
                    f for f in free_summary.frees
                    if f.target_kind in ("parameter", "field")
                ]
            return LeakSummary(
                function_name=func.name,
                description="No allocations, no leaks possible.",
                simplified_frees=simplified_frees,
            )

        # Build callee alloc/free sections, preferring simplified from leak summaries
        callee_section = self._build_callee_section(
            func, callee_summaries or {},
        )

        alloc_section = self._format_alloc_summary(alloc_summary)
        free_section = self._format_free_summary(free_summary)

        is_entry = func.name in self.entry_functions
        entry_note = (
            "**This is an entry-point function** (`main` or equivalent). "
            "Allocations stored to globals are still leaks -- there is no "
            "caller to observe or free them."
            if is_entry else ""
        )

        prompt = LEAK_USER_PROMPT.format(
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            source=func.llm_source,
            alloc_section=alloc_section,
            free_section=free_section,
            callee_section=callee_section,
            entry_note=entry_note,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    cur = self._progress_current
                    tot = self._progress_total
                    print(f"  ({cur}/{tot}) Leak check: {func.name}")
                else:
                    print(f"  Leak check: {func.name}")

            response = self.llm.complete(
                prompt,
                system=LEAK_SYSTEM_PROMPT,
                response_format=LEAK_RESPONSE_FORMAT,
            )
            self.record_call()

            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            summary = self._parse_response(response, func.name)
            with self._stats_lock:
                self._stats["functions_processed"] += 1
                self._stats["leaks_found"] += len(summary.issues)

            return summary

        except Exception as e:
            self.record_error()
            if self.verbose:
                print(f"  Error in leak check for {func.name}: {e}")
            return LeakSummary(
                function_name=func.name,
                description=f"Error during leak check: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_leak_summaries: dict[str, LeakSummary],
    ) -> str:
        """Build callee section, preferring simplified alloc/free from leak summaries."""
        if func.id is None:
            return "No callee information available."

        # Query call edges with is_indirect flag
        edge_rows = self.db.conn.execute(
            "SELECT DISTINCT callee_id, is_indirect "
            "FROM call_edges WHERE caller_id = ?",
            (func.id,),
        ).fetchall()
        if not edge_rows:
            return "No callees (leaf function)."

        lines = []
        for row in edge_rows:
            callee_id: int = row["callee_id"]
            is_indirect: bool = bool(row["is_indirect"])
            callee_func = self.db.get_function(callee_id)
            if callee_func is None:
                continue

            callee_name = callee_func.name
            parts = []

            # Mark indirect call targets so the model can connect
            # function pointer expressions in source to resolved callees
            if is_indirect:
                parts.append("(resolved indirect call target)")

            # Show noreturn attribute — critical for leak analysis
            if callee_func.attributes and "noreturn" in callee_func.attributes:
                parts.append("__attribute__((noreturn))")

            # Prefer simplified from leak summary if available
            leak_sum = callee_leak_summaries.get(callee_name)
            if leak_sum is not None:
                # Use simplified alloc/free
                if leak_sum.simplified_allocations:
                    alloc_descs = []
                    for a in leak_sum.simplified_allocations:
                        d = f"{a.source}({a.size_expr or '?'})"
                        extras = []
                        if a.returned:
                            extras.append("returned")
                        if a.stored_to:
                            extras.append(f"stored_to={a.stored_to}")
                        if not a.may_be_null:
                            extras.append("never null")
                        if extras:
                            d += f" [{', '.join(extras)}]"
                        alloc_descs.append(d)
                    parts.append(
                        f"Unfreed allocations (caller must handle): "
                        f"{', '.join(alloc_descs)}"
                    )
                else:
                    parts.append("All internal allocations freed")

                if leak_sum.simplified_frees:
                    free_descs = []
                    for f in leak_sum.simplified_frees:
                        d = f"{f.deallocator}({f.target})"
                        if f.description:
                            d += f" -- {f.description}"
                        free_descs.append(d)
                    parts.append(f"Frees: {', '.join(free_descs)}")
            else:
                # Fall back to raw alloc/free summaries from DB
                raw_alloc = self.db.get_summary_by_function_id(callee_id)
                if raw_alloc and raw_alloc.allocations:
                    alloc_descs = []
                    for a in raw_alloc.allocations:
                        d = f"{a.source}({a.size_expr or '?'})"
                        if a.stored_to:
                            d += f" -> {a.stored_to}"
                        if not a.may_be_null:
                            d += " [never null]"
                        alloc_descs.append(d)
                    parts.append(f"Allocates: {', '.join(alloc_descs)}")

                raw_free = self.db.get_free_summary_by_function_id(callee_id)
                if raw_free and raw_free.frees:
                    free_descs = []
                    for f in raw_free.frees:
                        d = f"{f.deallocator}({f.target})"
                        if f.description:
                            d += f" -- {f.description}"
                        free_descs.append(d)
                    parts.append(f"Frees: {', '.join(free_descs)}")

            if parts:
                lines.append(f"- `{callee_name}`: {'; '.join(parts)}")

        if not lines:
            return "No callee summaries."
        return "\n".join(lines)

    def _format_alloc_summary(self, summary: Any) -> str:
        if not summary or not summary.allocations:
            return "No allocations."
        lines = []
        for a in summary.allocations:
            desc = f"{a.source}({a.size_expr or 'unknown'})"
            extras = []
            if a.returned:
                extras.append("returned")
            if a.stored_to:
                extras.append(f"stored_to={a.stored_to}")
            if a.may_be_null:
                extras.append("may_be_null")
            else:
                extras.append("not_null")
            if extras:
                desc += f" [{', '.join(extras)}]"
            lines.append(f"- {desc}")
        return "\n".join(lines)

    def _format_free_summary(self, summary: Any) -> str:
        if not summary or not summary.frees:
            return "No frees."
        lines = []
        for f in summary.frees:
            desc = f"{f.deallocator}({f.target})"
            extras = []
            if f.conditional:
                cond_text = f"when {f.condition}" if f.condition else "conditional"
                extras.append(cond_text)
            if extras:
                desc += f" [{', '.join(extras)}]"
            if f.description:
                desc += f" -- {f.description}"
            lines.append(f"- {desc}")
        return "\n".join(lines)

    def _parse_response(self, response: str, func_name: str) -> LeakSummary:
        from .builder.json_utils import extract_json
        data = extract_json(response)

        # Parse leaks -> issues
        issues = []
        for leak in data.get("leaks", []):
            severity = leak.get("severity", "medium")
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            alloc_expr = leak.get("allocation", "unknown")
            stored_to = leak.get("stored_to")
            reason = leak.get("reason", "")

            desc = f"Allocation {alloc_expr}"
            if stored_to:
                desc += f" (stored to {stored_to})"
            desc += f" is never freed: {reason}"

            issues.append(SafetyIssue(
                location=f"allocation: {alloc_expr}",
                issue_kind="memory_leak",
                description=desc,
                severity=severity,
            ))

        # Parse simplified_allocations
        simplified_allocs = []
        for a in data.get("simplified_allocations", []):
            simplified_allocs.append(Allocation(
                alloc_type=AllocationType.HEAP,
                source=a.get("source", "unknown"),
                size_expr=a.get("size_expr"),
                returned=a.get("returned", False),
                stored_to=a.get("stored_to"),
                may_be_null=a.get("may_be_null", True),
            ))

        # Parse simplified_frees
        simplified_frees = []
        for f in data.get("simplified_frees", []):
            simplified_frees.append(FreeOp(
                target=f.get("target", ""),
                target_kind=f.get("target_kind", "parameter"),
                deallocator=f.get("deallocator", "free"),
                conditional=f.get("conditional", False),
                nulled_after=False,
                condition=f.get("condition"),
                description=f.get("description"),
            ))

        return LeakSummary(
            function_name=data.get("function", func_name),
            description=str(data.get("description", "Leak analysis complete.")),
            simplified_allocations=simplified_allocs,
            simplified_frees=simplified_frees,
            issues=issues,
        )

