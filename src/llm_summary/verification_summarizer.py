"""LLM-based verification and contract simplification (Pass 5)."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    Function,
    FunctionBlock,
    MemsafeContract,
    SafetyIssue,
    VerificationSummary,
    build_skeleton,
)


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

VERIFICATION_PROMPT = """\
You are verifying memory safety of a C/C++ function \
using Hoare-logic-style reasoning.

## Reasoning Model

Assume this function's own pre-conditions (contracts from the memory safety pass) \
are ALREADY SATISFIED by its callers.
Your job is to check what the function does *given* those \
pre-conditions hold — not to re-flag them as bugs.
An issue is only real if it can occur even when all pre-conditions are satisfied.

For C++ member functions, `this` is an implicit pointer parameter. \
Track its state (freed, null) like any other pointer.

Walk the function statement by statement, tracking two kinds of state:
- **Pointer/buffer properties**: null, non-null, freed, initialized, allocated size
- **Integer value ranges**: bounds from checks, assignments, return values

At each statement or call:
1. **Check**: Do the tracked pre-conditions satisfy this statement's memory safety \
requirements and each callee's pre-conditions? If not → report an issue.
2. **Update**: Update post-conditions from the statement's effect \
or the callee's post-conditions (allocations, frees, nullability, etc.).

## Function Under Verification

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

{type_defs_section}## Pre-conditions (assume these hold)

{own_contracts}

## Callee Information

{callee_section}

{alias_context}

## Verification

Report ALL issues — do not omit or consolidate.
Calling through a NULL function pointer (indirect call via a null pointer) is a `null_deref`.
A `nullable` callee parameter accepts NULL safely — not a bug.
Unchecked `may_be_null` return dereferenced → `null_deref`.
Use after callee frees → `use_after_free`.

Pay special attention to these commonly missed patterns:
- **Integer overflow in size calculations**: narrow integer in multiplication for allocation/VLA
- **Off-by-one in bounds checks**: loop bounds, size comparisons, fence-post errors
- **Struct field overflow**: write length exceeds field's declared size — check type defs
- **Wrong size variable**: total vs remaining length, container vs payload size
- **Type confusion**: pointer cast to incompatible type, wrong offset/size
- **Format string misuse**: non-literal format arg to printf-family → `buffer_overflow`

**Not a bug if:**
- Guarded by runtime check (e.g., `if (ptr)` before deref)
- Covered by this function's own pre-conditions above
- Guaranteed by callee post-condition
- Static/global variable (C guarantees zero-init before startup)

**Contract simplification**: Drop pre-conditions satisfied internally. \
Keep only those callers must provide.

**Severity**: high = unconditional, medium = specific path, low = error path

## Output

Respond with JSON:
```json
{{
  "function": "{name}",
  "description": "One-sentence summary",
  "simplified_contracts": [
    {{"target": "param", "contract_kind": "not_null|nullable|not_freed|initialized|buffer_size",
      "description": "brief", "size_expr": "buffer_size only", "relationship": "buffer_size only"}}
  ],
  "issues": [
    {{"location": "line N",
      "issue_kind": "null_deref|buffer_overflow|use_after_free|double_free|uninitialized_use",
      "description": "the problem", "severity": "high|medium|low",
      "callee": "if contract violation",
      "contract_kind": "if contract violation"}}
  ]
}}
```
"""

BLOCK_VERIFICATION_PROMPT = """\
You are verifying memory safety of a code block from a \
large C/C++ function.

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
  "summary": "One-sentence verification summary for this block",
  "issues": [
    {{{{
      "location": "line N or description",
      "issue_kind": "null_deref|buffer_overflow|use_after_free|double_free|uninitialized_use",
      "description": "what the problem is",
      "severity": "high|medium|low"
    }}}}
  ]
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

_CONTRACT_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "contract_kind": {"type": "string"},
        "description": {"type": "string"},
        "size_expr": {"type": "string"},
        "relationship": {"type": "string"},
    },
    "required": ["target", "contract_kind", "description"],
}

_ISSUE_ITEM = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "issue_kind": {"type": "string"},
        "description": {"type": "string"},
        "severity": {"type": "string"},
        "callee": {"type": "string"},
        "contract_kind": {"type": "string"},
    },
    "required": ["location", "issue_kind", "description", "severity"],
}

VERIFY_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "description": {"type": "string"},
        "simplified_contracts": {"type": "array", "items": _CONTRACT_ITEM},
        "issues": {"type": "array", "items": _ISSUE_ITEM},
    },
    "required": ["function", "description", "simplified_contracts", "issues"],
})

VERIFY_BLOCK_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "suggested_name": {"type": "string"},
        "suggested_signature": {"type": "string"},
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": _ISSUE_ITEM},
    },
    "required": ["suggested_name", "suggested_signature", "summary", "issues"],
})


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
                response_format=VERIFY_RESPONSE_FORMAT,
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
            if func.id is not None:
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

        if self.verbose:
            n_chars = len(func.llm_source)
            n_blocks = len(blocks)
            print(
                f"  Large function ({n_chars} chars, "
                f"{n_blocks} blocks): {func.name}"
            )

        own_contracts = self._build_own_contracts_section(func)
        block_summaries: dict[int, str] = {}
        all_block_issues: list[SafetyIssue] = []

        for i, block in enumerate(blocks):
            assert block.id is not None
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
                response = self.llm.complete(
                    prompt, response_format=VERIFY_BLOCK_RESPONSE_FORMAT,
                )
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
                response_format=VERIFY_RESPONSE_FORMAT,
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
        if func.id is None:
            raw_memsafe = None
        else:
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

    def _build_type_defs_section(self, source: str, file_path: str = "") -> str:
        """Build a section with struct/union/typedef definitions referenced in source,
        plus file-scope static variable declarations from the same source file.


        Extracts type names from the source text, looks them up in the DB, and
        returns a formatted section string (empty string if no definitions found).
        """
        import re

        # Find referenced type names: struct X, union X, typedef names in casts/params
        names: set[str] = set()
        for m in re.finditer(r'\b(?:struct|union|enum)\s+(\w+)', source):
            names.add(m.group(1))

        rows = self.db.get_typedefs_by_names(list(names)) if names else []

        # Also include file-scope static variables from the same source file that
        # are actually referenced by name in the function source.
        all_identifiers = set(re.findall(r'\b([A-Za-z_]\w*)\b', source))
        static_rows = [
            r for r in (self.db.get_static_vars_by_file(file_path) if file_path else [])
            if r["name"] in all_identifiers
        ]

        # Resolve types referenced in static var declarations
        # e.g., "static engine_t *engine;" → look up engine_t typedef
        static_type_names: set[str] = set()
        for srow in static_rows:
            defn = srow.get("definition") or ""
            # Extract type name: skip 'static', 'const', 'volatile', 'unsigned', etc.
            for tok in re.findall(r'\b([A-Za-z_]\w*)\b', defn):
                if tok not in ("static", "const", "volatile", "unsigned",
                               "signed", "char", "int", "long", "short",
                               "float", "double", "void", "bool",
                               srow["name"]):
                    static_type_names.add(tok)
        new_names = static_type_names - {r["name"] for r in rows} - names
        if new_names:
            extra = self.db.get_typedefs_by_names(list(new_names))
            rows.extend(extra)
            names.update(new_names)

        # Deduplicate by name: same-file definition wins; among cross-file,
        # prefer shortest (least likely to be the wrong variant).
        # pp_definition stores the annotated macro-expanded form (// (macro) lines)
        # produced at scan time; use it when available so the LLM sees concrete values.
        seen: dict[str, str] = {}
        seen_from_same_file: set[str] = set()
        for row in rows + static_rows:
            name = row["name"]
            defn = row.get("pp_definition") or row.get("definition") or ""
            if not defn:
                continue
            same_file = row.get("file_path") == file_path
            if same_file:
                seen[name] = defn
                seen_from_same_file.add(name)
            elif name not in seen_from_same_file:
                if name not in seen or len(defn) < len(seen[name]):
                    seen[name] = defn

        if not seen:
            return ""

        # Emit each unique definition block only once (multiple vars may share a block)
        emitted: set[str] = set()
        lines = ["## Referenced Type Definitions\n", "```c"]
        for defn in seen.values():
            if defn not in emitted:
                emitted.add(defn)
                lines.append(defn)
                lines.append("")
        lines.append("```\n\n")
        return "\n".join(lines)

    def _build_prompt_and_system(
        self, source: str, func: Function, own_contracts: str,
        callee_section: str, alias_context: str | None,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system).

        Verification always uses the monolithic prompt regardless of cache_mode.
        The complex Hoare-logic reasoning requires source, contracts, and callee
        context tightly coupled in a single prompt.
        """
        type_defs_section = self._build_type_defs_section(source, func.file_path)
        prompt = VERIFICATION_PROMPT.format(
            source=source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            type_defs_section=type_defs_section,
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
            sig_text = f" — `{callee_func.signature}`" if callee_func.signature else ""
            section_lines = [f"### `{callee_name}`{attr_text}{sig_text}"]

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
            if free_summary and (free_summary.frees or free_summary.resource_releases):
                def _fmt_ops(ops: list) -> list[str]:
                    descs = []
                    for f in ops:
                        desc = f"{f.deallocator}({f.target})"
                        extras = []
                        if f.conditional:
                            cond_text = f"when {f.condition}" if f.condition else "conditional"
                            extras.append(cond_text)
                        if f.nulled_after:
                            extras.append("nulled_after")
                        if extras:
                            desc += f" [{', '.join(extras)}]"
                        descs.append(desc)
                    return descs
                if free_summary.frees:
                    post_parts.append(f"  Frees: {'; '.join(_fmt_ops(free_summary.frees))}")
                if free_summary.resource_releases:
                    releases = _fmt_ops(free_summary.resource_releases)
                    post_parts.append(
                        f"  Releases: {'; '.join(releases)}"
                    )

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
        """Format this function's raw memory safety contracts."""
        if func.id is None:
            return "No raw contracts available."

        raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if not raw_memsafe or not raw_memsafe.contracts:
            return "No raw safety contracts (memory safety analysis found no pre-conditions)."

        lines = []
        if raw_memsafe.description:
            lines.append(f"Memory safety assessment: {raw_memsafe.description}")
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
        if not self.log_file:
            return
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
