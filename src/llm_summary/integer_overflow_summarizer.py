"""LLM-based integer UB detection pass.

Performs value-range analysis to detect signed integer overflow,
shift UB, and division-by-zero.  Produces compositional summaries
with constraints (pre-conditions on parameter ranges) and output
ranges (post-conditions) for bottom-up propagation.
"""

from typing import Any

from .base_summarizer import BaseSummarizer
from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    Function,
    IntegerConstraint,
    IntegerOverflowSummary,
    OutputRange,
    SafetyIssue,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

OVERFLOW_SYSTEM_PROMPT = """\
You are performing **value-range analysis** on a C/C++ function to \
detect integer-related undefined behaviour (UB).  Do NOT report \
memory-safety bugs (null deref, buffer overflow, use-after-free, \
etc.) — those are handled by a separate pass.

## Integer UB categories

1. **Signed overflow / underflow** (`integer_overflow`): arithmetic \
(`+`, `-`, `*`, `++`, `--`, unary `-`) on a *signed* integer type \
whose mathematical result falls outside the representable range.
2. **Division / modulo by zero** (`division_by_zero`): `a / 0`, \
`a % 0`.  Also `INT_MIN / -1` and `INT_MIN % -1` (signed overflow).
3. **Shift UB** (`shift_ub`): left or right shift where the shift \
amount is negative or ≥ the bit-width of the *promoted* left operand, \
or left-shifting a negative signed value (UB before C23).

## What is NOT UB — do NOT report

- **Unsigned arithmetic wrapping**: unsigned types wrap modulo 2^N by \
definition — this is well-defined, not UB.  `unsigned int x = UINT_MAX; \
x++` wraps to 0 — not a bug.
- **Integer promotions**: operands narrower than `int` (`char`, `short`, \
`_Bool`) are promoted to `int` before arithmetic (C11 §6.3.1.1).  The \
arithmetic is performed in `int`, not the original narrow type.  Only \
flag overflow if the *promoted-type* result overflows.  E.g. \
`short a = 32767; a + a + a` → computed as `int` 98301, no overflow.
- **Unsigned-to-signed conversion within range**: converting an \
`unsigned` value to `signed` when the value fits is well-defined.
- **Literal type widening**: an unsuffixed decimal constant too large \
for `int` is automatically `long` or `long long` (C11 §6.4.4.1).  \
E.g. `2147483648` is `long` on LP64, not `int`.
- **Well-defined edge values**: `INT_MIN` (−2147483648 for 32-bit \
`int`) IS representable.  `65536 * −32768 = −2147483648 = INT_MIN` → \
NOT overflow.
- **Guarded arithmetic**: if a preceding `if` / `assert` / \
`assume_abort_if_not` constrains the operand so UB cannot occur, do \
not report it.  Update the tracked range after the guard.
- **Dead / infeasible paths**: if a path is unreachable due to \
earlier conditions, do not report issues on it.

## Analysis method — value-range tracking

Walk the function statement by statement, maintaining an **interval** \
for each integer variable:

1. **Initialisation**: from constants, callee output ranges, or \
callee constraints on output parameters.
2. **Branch narrowing**: after `if (x > 0)`, narrow x to `[1, \
INT_MAX]` on the true branch and `[INT_MIN, 0]` on the false branch.
3. **Arithmetic**: compute the resulting interval.  If it exceeds the \
type's representable range → report.
4. **Callee constraints**: if a callee declares a constraint on a \
parameter (e.g., `n: [0, INT_MAX/4]`), check whether the actual \
argument's range satisfies it.  If not → report.

## Callee post-conditions

The **Callee Context** section provides two kinds of information \
for each callee:
- **Output ranges** (from the init pass): value ranges for the \
return value or out-parameters.  Use these to set the range of the \
variable receiving the call result.
- **Constraints** (from prior overflow analysis): pre-conditions on \
parameters.  Check actual arguments against these.

## Data model

Assume **LP64** (sizeof(int)=4, sizeof(long)=8, sizeof(long long)=8, \
sizeof(void*)=8) unless the source specifies ILP32.

Type ranges (LP64):
- `signed char`: [−128, 127]
- `short` / `int16_t`: [−32768, 32767]
- `int` / `int32_t`: [−2147483648, 2147483647]
- `long` / `int64_t`: [−2^63, 2^63−1]
- `unsigned int` / `uint32_t`: [0, 4294967295] (wrapping, not UB)

## Compositional output

Produce three things:

1. **constraints** — pre-conditions on this function's parameters.  \
If the function performs arithmetic on a parameter and the operation \
would overflow for some parameter values, declare the safe range the \
caller must provide.  Only include constraints that are NOT already \
internally guarded.
2. **output_ranges** — post-conditions on the return value and \
out-parameters.  Include the range even if it equals the full type \
range (callers need this to detect downstream overflow).
3. **issues** — concrete UB instances found in this function (where \
the range analysis shows the operation CAN overflow given the \
function's own pre-conditions and callee post-conditions).

Respond with JSON only.
"""

OVERFLOW_USER_PROMPT = """\
## Function

Function: `{name}`
Signature: `{signature}`
File: {file_path}

```c
{source}
```

## Callee Context

{callee_section}

## Task

Perform value-range analysis as described.  Track integer ranges \
through branches and arithmetic.  Report UB and produce compositional \
constraints / output ranges.

Respond in JSON:
```json
{{{{
  "function": "{name}",
  "description": "One-sentence summary of analysis result",
  "constraints": [
    {{{{
      "target": "parameter name",
      "range": "[lower, upper] or descriptive",
      "description": "why this constraint is needed"
    }}}}
  ],
  "output_ranges": [
    {{{{
      "target": "return value or *out_param",
      "range": "[lower, upper] or descriptive",
      "description": "brief context"
    }}}}
  ],
  "issues": [
    {{{{
      "location": "line or expression",
      "issue_kind": "integer_overflow|division_by_zero|shift_ub",
      "description": "which operation, operand ranges, why it is UB",
      "severity": "high|medium|low"
    }}}}
  ]
}}}}
```

If no issues, return empty issues / constraints lists with a summary.
"""

_VALID_OVERFLOW_KINDS = {
    "integer_overflow",
    "division_by_zero",
    "shift_ub",
}

_CONSTRAINT_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "range": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["target", "range", "description"],
}

_OUTPUT_RANGE_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "range": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["target", "range", "description"],
}

_ISSUE_ITEM = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "issue_kind": {
            "type": "string",
            "enum": sorted(_VALID_OVERFLOW_KINDS),
        },
        "description": {"type": "string"},
        "severity": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["location", "issue_kind", "description", "severity"],
}

OVERFLOW_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "description": {"type": "string"},
        "constraints": {"type": "array", "items": _CONSTRAINT_ITEM},
        "output_ranges": {"type": "array", "items": _OUTPUT_RANGE_ITEM},
        "issues": {"type": "array", "items": _ISSUE_ITEM},
    },
    "required": ["function", "description", "constraints",
                  "output_ranges", "issues"],
})


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class IntegerOverflowSummarizer(BaseSummarizer):
    """Detects integer UB via value-range analysis with compositional summaries."""

    _extra_stats = {"overflows_found": 0}

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
    ):
        super().__init__(
            db, llm, verbose=verbose, log_file=log_file,
            pass_label="overflow",
        )

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, IntegerOverflowSummary] | None = None,
        **kwargs: Any,
    ) -> IntegerOverflowSummary:
        """Analyse a single function for integer UB."""
        assert func.id is not None

        # Quick exit: no source to analyse
        if not func.source:
            with self._stats_lock:
                self._stats["functions_processed"] += 1
            return IntegerOverflowSummary(
                function_name=func.name,
                description="No source available.",
            )

        callee_section = self._build_callee_section(
            func, callee_summaries or {},
        )

        prompt = OVERFLOW_USER_PROMPT.format(
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            source=func.llm_source,
            callee_section=callee_section,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    cur = self._progress_current
                    tot = self._progress_total
                    print(f"  ({cur}/{tot}) Overflow check: {func.name}")
                else:
                    print(f"  Overflow check: {func.name}")

            response = self.llm.complete(
                prompt,
                system=OVERFLOW_SYSTEM_PROMPT,
                response_format=OVERFLOW_RESPONSE_FORMAT,
            )
            self.record_call()

            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            summary = self._parse_response(response, func.name)
            with self._stats_lock:
                self._stats["functions_processed"] += 1
                self._stats["overflows_found"] += len(summary.issues)

            return summary

        except Exception as e:
            self.record_error()
            if self.verbose:
                print(f"  Error in overflow check for {func.name}: {e}")
            return IntegerOverflowSummary(
                function_name=func.name,
                description=f"Error during overflow check: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_overflow_summaries: dict[str, IntegerOverflowSummary],
    ) -> str:
        """Build callee context from init output ranges + overflow constraints."""
        if func.id is None:
            return "No callee information available."

        edge_rows = self.db.conn.execute(
            "SELECT DISTINCT callee_id "
            "FROM call_edges WHERE caller_id = ?",
            (func.id,),
        ).fetchall()
        if not edge_rows:
            return "No callees (leaf function)."

        lines: list[str] = []
        for row in edge_rows:
            callee_id: int = row["callee_id"]
            callee_func = self.db.get_function(callee_id)
            if callee_func is None:
                continue

            parts: list[str] = []

            # Output ranges from init pass
            init_summary = self.db.get_init_summary_by_function_id(callee_id)
            if init_summary and init_summary.output_ranges:
                for o in init_summary.output_ranges:
                    parts.append(
                        f"Output range: {o.target} = {o.range}"
                        f" -- {o.description}"
                    )

            # Constraints and output ranges from prior overflow analysis
            ovf_sum = callee_overflow_summaries.get(callee_func.name)
            if ovf_sum is not None:
                for c in ovf_sum.constraints:
                    parts.append(
                        f"Constraint: {c.target} must be in {c.range}"
                        f" -- {c.description}"
                    )
                for o in ovf_sum.output_ranges:
                    # Prefer overflow pass ranges over init (more precise)
                    parts.append(
                        f"Output range: {o.target} = {o.range}"
                        f" -- {o.description}"
                    )

            if parts:
                lines.append(
                    f"### `{callee_func.name}`\n"
                    + "\n".join(f"- {p}" for p in parts)
                )

        if not lines:
            return "No callee context available."
        return "\n\n".join(lines)

    def _parse_response(
        self, response: str, func_name: str,
    ) -> IntegerOverflowSummary:
        from .builder.json_utils import extract_json
        data = extract_json(response)

        # Parse constraints
        constraints = []
        for item in data.get("constraints", []):
            constraints.append(IntegerConstraint(
                target=item.get("target", ""),
                range=item.get("range", ""),
                description=item.get("description", ""),
            ))

        # Parse output ranges
        output_ranges = []
        for item in data.get("output_ranges", []):
            output_ranges.append(OutputRange(
                target=item.get("target", ""),
                range=item.get("range", ""),
                description=item.get("description", ""),
            ))

        # Parse issues
        issues = []
        for item in data.get("issues", []):
            severity = item.get("severity", "medium")
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            kind = item.get("issue_kind", "integer_overflow")
            if kind not in _VALID_OVERFLOW_KINDS:
                kind = "integer_overflow"
            issues.append(SafetyIssue(
                location=item.get("location", ""),
                issue_kind=kind,
                description=item.get("description", ""),
                severity=severity,
            ))

        return IntegerOverflowSummary(
            function_name=data.get("function", func_name),
            constraints=constraints,
            output_ranges=output_ranges,
            issues=issues,
            description=str(
                data.get("description", "Integer UB analysis complete."),
            ),
        )
