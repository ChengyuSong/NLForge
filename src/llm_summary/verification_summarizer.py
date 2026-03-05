"""LLM-based verification and contract simplification (Pass 5)."""

import json
import re
import threading

from .db import SummaryDB


class IncompleteCalleeError(RuntimeError):
    """Raised when a callee's verification summary has simplified_contracts=None.

    The driver should catch this, re-run verification for the named callee,
    then retry the current function.
    """

    def __init__(self, callee_name: str):
        self.callee_name = callee_name
        super().__init__(
            f"Callee '{callee_name}' has simplified_contracts=None "
            f"(failed or incomplete prior verification)"
        )
from .llm.base import LLMBackend
from .models import (
    Function,
    FunctionBlock,
    MemsafeContract,
    SafetyIssue,
    VerificationSummary,
    build_skeleton,
)

VERIFICATION_PROMPT = """You are verifying memory safety of a C/C++ function using Hoare-logic-style reasoning.

## Reasoning Model

Assume this function's own pre-conditions (contracts from Pass 4) are ALREADY SATISFIED by its callers.
Your job is to check what the function does *given* those pre-conditions hold — not to re-flag them as bugs.

An issue is only real if it can occur even when all pre-conditions are satisfied.

## Function Under Verification

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## This Function's Pre-conditions (from Pass 4) — assume these hold

{own_contracts}

## Callee Information

{callee_section}

{alias_context}

## Verification Tasks

### Check 1: Internal Safety
Assuming all pre-conditions hold, does the function perform any unsafe operations
(null dereference, buffer overflow, use-after-free, double-free, use of uninitialized memory)?

**Do NOT report an issue if:**
- The unsafe operation is guarded by a runtime check (e.g., `if (ptr)` before deref)
- The condition is already covered by one of this function's own pre-conditions listed above
- The value is guaranteed safe by a callee's post-condition (e.g., successful malloc returns non-null)

### Check 2: Callee Pre-condition Satisfaction
For EACH call to a callee that has pre-conditions, determine whether this function
establishes those pre-conditions before the call — given its own pre-conditions hold.

A callee pre-condition is SATISFIED if:
- The argument is guarded before the call (e.g., null check)
- The value comes from an allocation that guarantees the property (e.g., calloc → initialized)
- The value flows from a parameter covered by this function's own pre-conditions (propagation)
- The callee has a `nullable` contract on that parameter (meaning it accepts NULL safely)

A callee pre-condition is VIOLATED if none of the above hold — this is a real bug.

**nullable contracts**: If a callee has `nullable` on a parameter, passing NULL to that parameter
is explicitly safe — the callee handles NULL internally. Do NOT report null_deref for such calls.

Also use callee post-conditions to detect internal issues:
- If a callee's post-condition says a returned pointer may_be_null, and the caller dereferences
  it without a null check, that is a real null_deref issue.
- If a callee's post-condition says it frees a pointer, and the caller uses it afterwards,
  that is a real use_after_free issue.

### Check 3: Sequential Call Chain Consistency
Track the state of each value across the sequence of calls in the function body.
After each call, apply its post-conditions to update the state of affected values,
then check whether the next call's pre-conditions are satisfied given that updated state.

Examples of violations to detect:
- `foo(x)` post-condition: x is freed → `bar(x)` pre-condition: x must not be freed
  → use_after_free at the call to bar
- `foo(x)` post-condition: x->field is initialized → `bar(x)` pre-condition: x must be initialized
  → satisfied (no issue)
- `p = alloc()` post-condition: may_be_null → `foo(p)` pre-condition: p not_null
  → null_deref at the call to foo if no null check between alloc and foo

### Check 4: Contract Simplification
For each of this function's own pre-conditions: is it actually needed, or is it satisfied
internally before first use? Keep only those that callers must genuinely satisfy.

## Severity Guidelines

- **high**: Definite violation unconditionally reachable given valid inputs
- **medium**: Violation reachable on a specific code path with valid inputs
- **low**: Violation only reachable on error paths or under highly unusual conditions

## Output

Respond with JSON:
```json
{{
  "function": "{name}",
  "simplified_contracts": [
    {{
      "target": "parameter or expression",
      "contract_kind": "not_null|nullable|not_freed|initialized|buffer_size",
      "description": "brief description",
      "size_expr": "n (buffer_size only, omit otherwise)",
      "relationship": "byte_count (buffer_size only, omit otherwise)"
    }}
  ],
  "issues": [
    {{
      "location": "line 42 or call to foo at line 42",
      "issue_kind": "null_deref|buffer_overflow|use_after_free|double_free|uninitialized_use",
      "description": "what the problem is",
      "severity": "high|medium|low",
      "callee": "callee_name (if callee contract violation, omit if internal)",
      "contract_kind": "which contract kind was violated (omit if internal)"
    }}
  ],
  "description": "One-sentence verification summary"
}}
```

If no issues and no contracts remain:
```json
{{
  "function": "{name}",
  "simplified_contracts": [],
  "issues": [],
  "description": "Function is internally safe with no propagated contracts"
}}
```
"""

BLOCK_VERIFICATION_PROMPT = """You are verifying memory safety of a code block from a large C/C++ function.

## Context

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## This Function's Pre-conditions — assume these hold

{own_contracts}

## Code Block

```c
{block_source}
```

## Task

Verify this code block for memory safety issues. Check for null dereferences, buffer overflows,
use-after-free, double-free, and uninitialized use. Also suggest a descriptive pseudo-function
name and signature.

Respond in JSON:
```json
{{{{
  "suggested_name": "descriptive_name_for_this_case",
  "suggested_signature": "void descriptive_name(args)",
  "issues": [
    {{{{
      "location": "line N or description",
      "issue_kind": "null_deref|buffer_overflow|use_after_free|double_free|uninitialized_use",
      "description": "what the problem is",
      "severity": "high|medium|low"
    }}}}
  ],
  "summary": "One-sentence verification summary for this block"
}}}}
```

If no issues, return empty issues list with a summary.
"""

_VALID_ISSUE_KINDS = {
    "null_deref",
    "buffer_overflow",
    "use_after_free",
    "double_free",
    "uninitialized_use",
}
_VALID_SEVERITIES = {"high", "medium", "low"}
_VALID_CONTRACT_KINDS = {"not_null", "nullable", "not_freed", "initialized", "buffer_size"}

class VerificationSummarizer:
    """Verifies memory safety and simplifies contracts using cross-pass data."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        cache_mode: str = "none",
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.cache_mode = cache_mode
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "issues_found": 0,
            "contracts_simplified": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        self._stats_lock = threading.Lock()
        self._progress_current = 0
        self._progress_total = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return self._stats.copy()

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary] | None = None,
        alias_context: str | None = None,
    ) -> VerificationSummary:
        """Verify a function and simplify its contracts."""
        if callee_summaries is None:
            callee_summaries = {}

        # Check for large function with blocks
        blocks = self.db.get_function_blocks(func.id) if func.id else []
        if blocks and len(func.llm_source) > 40000:
            return self._summarize_large_function(
                func, callee_summaries, blocks, alias_context
            )

        callee_section = self._build_callee_section(func, callee_summaries)
        own_contracts = self._build_own_contracts_section(func)

        prompt, system, cache_system = self._build_prompt_and_system(
            func.llm_source, func, own_contracts, callee_section, alias_context,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(
                        f"  ({self._progress_current}/{self._progress_total}) "
                        f"Verifying: {func.name}"
                    )
                else:
                    print(f"  Verifying: {func.name}")

            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
            )
            with self._stats_lock:
                self._stats["llm_calls"] += 1
                if llm_response.cached:
                    self._stats["cache_hits"] += 1
                self._stats["cache_read_tokens"] += llm_response.cache_read_tokens
                self._stats["cache_creation_tokens"] += llm_response.cache_creation_tokens

            if self.log_file:
                self._log_interaction(func.name, prompt, llm_response.content)

            summary = self._parse_response(llm_response.content, func.name)
            with self._stats_lock:
                self._stats["functions_processed"] += 1
                self._stats["issues_found"] += len(summary.issues)

            # Count simplified contracts: raw - remaining
            raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
            if raw_memsafe and summary.simplified_contracts is not None:
                raw_count = len(raw_memsafe.contracts)
                remaining_count = len(summary.simplified_contracts)
                with self._stats_lock:
                    self._stats["contracts_simplified"] += max(
                        0, raw_count - remaining_count
                    )

            return summary

        except Exception as e:
            with self._stats_lock:
                self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error verifying {func.name}: {e}")

            return VerificationSummary(
                function_name=func.name,
                description=f"Error during verification: {e}",
            )

    def _summarize_large_function(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary],
        blocks: list[FunctionBlock],
        alias_context: str | None = None,
    ) -> VerificationSummary:
        """Chunked verification for large functions."""
        import json

        if self.verbose:
            print(f"  Large function ({len(func.llm_source)} chars, {len(blocks)} blocks): {func.name}")

        own_contracts = self._build_own_contracts_section(func)
        block_summaries: dict[int, str] = {}
        all_block_issues: list[SafetyIssue] = []

        for i, block in enumerate(blocks):
            if block.summary_json:
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    for issue in data.get("issues", []):
                        issue_kind = issue.get("issue_kind", "null_deref")
                        if issue_kind not in _VALID_ISSUE_KINDS:
                            issue_kind = "null_deref"
                        severity = issue.get("severity", "medium")
                        if severity not in _VALID_SEVERITIES:
                            severity = "medium"
                        all_block_issues.append(SafetyIssue(
                            location=issue.get("location", ""),
                            issue_kind=issue_kind,
                            description=issue.get("description", ""),
                            severity=severity,
                            callee=issue.get("callee"),
                            contract_kind=issue.get("contract_kind"),
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            prompt = BLOCK_VERIFICATION_PROMPT.format(
                name=func.name, signature=func.signature,
                file_path=func.file_path, own_contracts=own_contracts,
                block_source=block.source,
            )

            try:
                if self.verbose:
                    print(f"    Block {i+1}/{len(blocks)}: {block.label[:60]}")
                response = self.llm.complete(prompt)
                with self._stats_lock:
                    self._stats["llm_calls"] += 1

                json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    json_str = json_match.group(0) if json_match else "{}"

                data = json.loads(json_str)
                block_summaries[block.id] = data.get("summary", "no summary")
                self.db.update_function_block_summary(
                    block.id, json.dumps(data),
                    data.get("suggested_name"), data.get("suggested_signature"),
                )

                for issue in data.get("issues", []):
                    issue_kind = issue.get("issue_kind", "null_deref")
                    if issue_kind not in _VALID_ISSUE_KINDS:
                        issue_kind = "null_deref"
                    severity = issue.get("severity", "medium")
                    if severity not in _VALID_SEVERITIES:
                        severity = "medium"
                    all_block_issues.append(SafetyIssue(
                        location=issue.get("location", ""),
                        issue_kind=issue_kind,
                        description=issue.get("description", ""),
                        severity=severity,
                        callee=issue.get("callee"),
                        contract_kind=issue.get("contract_kind"),
                    ))
            except Exception as e:
                if self.verbose:
                    print(f"    Error verifying block {block.label}: {e}")
                block_summaries[block.id] = f"(error: {e})"

        # Phase B: skeleton verification
        skeleton = build_skeleton(func.llm_source, func.line_start, blocks, block_summaries)
        callee_section = self._build_callee_section(func, callee_summaries)

        prompt, system, cache_system = self._build_prompt_and_system(
            skeleton, func, own_contracts, callee_section, alias_context,
        )

        try:
            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
            )
            with self._stats_lock:
                self._stats["llm_calls"] += 1
                if llm_response.cached:
                    self._stats["cache_hits"] += 1
                self._stats["cache_read_tokens"] += llm_response.cache_read_tokens
                self._stats["cache_creation_tokens"] += llm_response.cache_creation_tokens
            skeleton_summary = self._parse_response(llm_response.content, func.name)
        except Exception as e:
            with self._stats_lock:
                self._stats["errors"] += 1
            skeleton_summary = VerificationSummary(
                function_name=func.name, description=f"Error verifying skeleton: {e}",
            )

        # Phase C: merge issues
        skeleton_summary.issues = list(skeleton_summary.issues) + all_block_issues

        # Count simplified contracts
        raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if raw_memsafe and skeleton_summary.simplified_contracts is not None:
            raw_count = len(raw_memsafe.contracts)
            remaining_count = len(skeleton_summary.simplified_contracts)
            with self._stats_lock:
                self._stats["contracts_simplified"] += max(0, raw_count - remaining_count)

        with self._stats_lock:
            self._stats["functions_processed"] += 1
            self._stats["issues_found"] += len(skeleton_summary.issues)
        return skeleton_summary

    def _build_prompt_and_system(
        self, source: str, func: Function, own_contracts: str,
        callee_section: str, alias_context: str | None,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system).

        Verification always uses the monolithic prompt regardless of cache_mode.
        The complex Hoare-logic reasoning requires source, contracts, and callee
        context tightly coupled in a single prompt.
        """
        prompt = VERIFICATION_PROMPT.format(
            source=source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            own_contracts=own_contracts,
            callee_section=callee_section,
            alias_context=alias_context or "",
        )
        return prompt, None, False

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary],
    ) -> str:
        """Build cross-pass callee context for the prompt."""
        if func.id is None:
            return "No callee information available."

        callee_ids = self.db.get_callees(func.id)
        if not callee_ids:
            return "No callees (leaf function)."

        lines = []
        for callee_id in callee_ids:
            callee_func = self.db.get_function(callee_id)
            if callee_func is None:
                continue

            callee_name = callee_func.name
            attr_text = f" {callee_func.attributes}" if callee_func.attributes else ""
            section_lines = [f"### `{callee_name}`{attr_text}"]

            # Pre-conditions: from verified summary (simplified_contracts)
            if callee_name in callee_summaries:
                verified = callee_summaries[callee_name]
                if verified.simplified_contracts is None:
                    raise IncompleteCalleeError(callee_name)
                if verified.simplified_contracts:
                    section_lines.append("**Pre-conditions (simplified):**")
                    for c in verified.simplified_contracts:
                        if c.contract_kind == "buffer_size" and c.size_expr:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind}({c.size_expr}) "
                                f"-- {c.description}"
                            )
                        else:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind} -- {c.description}"
                            )
                else:
                    section_lines.append(
                        "**Pre-conditions:** None (all satisfied internally)"
                    )
            else:
                # Callee not yet verified — no summary available
                section_lines.append("**Pre-conditions:** None")

            # Post-conditions from Passes 1-3
            post_parts = []

            # Pass 1: Allocations
            alloc_summary = self.db.get_summary_by_function_id(callee_id)
            if alloc_summary and alloc_summary.allocations:
                alloc_descs = []
                for a in alloc_summary.allocations:
                    desc = f"{a.source}"
                    if a.size_expr:
                        desc += f"({a.size_expr})"
                    extras = []
                    if a.may_be_null:
                        extras.append("may_be_null")
                    if a.returned:
                        extras.append("returned")
                    if extras:
                        desc += f" [{', '.join(extras)}]"
                    alloc_descs.append(desc)
                post_parts.append(f"  Allocations: {'; '.join(alloc_descs)}")
                if alloc_summary.buffer_size_pairs:
                    for bsp in alloc_summary.buffer_size_pairs:
                        post_parts.append(
                            f"  Buffer-size pair: ({bsp.buffer}, {bsp.size}) "
                            f"{bsp.relationship}"
                        )

            # Pass 2: Frees
            free_summary = self.db.get_free_summary_by_function_id(callee_id)
            if free_summary and free_summary.frees:
                free_descs = []
                for f in free_summary.frees:
                    desc = f"{f.deallocator}({f.target})"
                    extras = []
                    if f.conditional:
                        extras.append("conditional")
                    if f.nulled_after:
                        extras.append("nulled_after")
                    if extras:
                        desc += f" [{', '.join(extras)}]"
                    free_descs.append(desc)
                post_parts.append(f"  Frees: {'; '.join(free_descs)}")

            # Pass 3: Initializations
            init_summary = self.db.get_init_summary_by_function_id(callee_id)
            if init_summary and init_summary.inits:
                init_descs = []
                for i in init_summary.inits:
                    desc = f"{i.initializer}({i.target})"
                    if i.byte_count:
                        desc += f" [{i.byte_count} bytes]"
                    init_descs.append(desc)
                post_parts.append(f"  Initializations: {'; '.join(init_descs)}")

            if post_parts:
                section_lines.append("**Post-conditions:**")
                section_lines.extend(post_parts)
            else:
                section_lines.append("**Post-conditions:** None available")

            lines.append("\n".join(section_lines))

        if not lines:
            return "No callee information available."

        return "\n\n".join(lines)

    def _build_own_contracts_section(self, func: Function) -> str:
        """Format this function's raw Pass 4 contracts."""
        if func.id is None:
            return "No raw contracts available."

        raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if not raw_memsafe or not raw_memsafe.contracts:
            return "No raw safety contracts (Pass 4 found no pre-conditions)."

        lines = []
        for c in raw_memsafe.contracts:
            if c.contract_kind == "buffer_size" and c.size_expr:
                lines.append(
                    f"- {c.target}: {c.contract_kind}({c.size_expr}) -- {c.description}"
                )
            else:
                lines.append(f"- {c.target}: {c.contract_kind} -- {c.description}")

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [verification pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> VerificationSummary:
        """Parse LLM response into VerificationSummary."""
        from .builder.json_utils import extract_json

        data = extract_json(response)

        # Parse simplified contracts
        contracts = []
        for c in data.get("simplified_contracts", []):
            contract_kind = c.get("contract_kind", "not_null")
            if contract_kind not in _VALID_CONTRACT_KINDS:
                contract_kind = "not_null"

            size_expr = None
            relationship = None
            if contract_kind == "buffer_size":
                size_expr = c.get("size_expr")
                relationship = c.get("relationship")

            contracts.append(
                MemsafeContract(
                    target=c.get("target", ""),
                    contract_kind=contract_kind,
                    description=c.get("description", ""),
                    size_expr=size_expr,
                    relationship=relationship,
                )
            )

        # Parse issues
        issues = []
        for i in data.get("issues", []):
            issue_kind = i.get("issue_kind", "null_deref")
            if issue_kind not in _VALID_ISSUE_KINDS:
                issue_kind = "null_deref"

            severity = i.get("severity", "medium")
            if severity not in _VALID_SEVERITIES:
                severity = "medium"

            issues.append(
                SafetyIssue(
                    location=i.get("location", ""),
                    issue_kind=issue_kind,
                    description=i.get("description", ""),
                    severity=severity,
                    callee=i.get("callee"),
                    contract_kind=i.get("contract_kind"),
                )
            )

        return VerificationSummary(
            function_name=data.get("function", func_name),
            simplified_contracts=contracts,
            issues=issues,
            description=data.get("description", ""),
        )
