"""LLM-based allocation summary generator."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    BufferSizePair,
    Function,
    FunctionBlock,
    ParameterInfo,
    build_skeleton,
)

# --- Shared allocation task instructions (single source of truth) ---
# Use _ALLOC_INSTRUCTIONS in all prompt templates. The JSON schema uses
# {func_placeholder} so callers can substitute the actual function name
# or a generic "<function_name>" marker.

_ALLOC_INSTRUCTIONS = """\
1. **Allocations and returned pointers**: Report every pointer that is \
**returned** or **stored to a caller-visible location**. This includes:
   - **Heap allocations** (malloc, calloc, realloc, mmap, new, etc.) or \
allocations via wrapper/helper functions that ultimately call heap allocators.
   - **Non-heap returned pointers**: If the function returns a pointer to \
a static variable, global variable, or a parameter (including pointer \
arithmetic on these), report it so callers know the provenance.

   Fields:
   - Type: "heap" for heap allocations. \
"static" if returning a pointer to a static or global variable. \
"parameter_derived" if returning a parameter or a pointer derived from \
a parameter (e.g., `return p;`, `return &p->field;`). \
Use "escaped_stack" ONLY if a stack-allocated local escapes — this is a bug.
   - Source: The allocating function/operator, or "static", "global", \
"parameter" for non-heap pointers
   - Size expression: How size is computed (heap only, null for non-heap)
   - Size parameters: Which function parameters affect size
   - Returned: Is the pointer returned?
   - Stored to: Is it stored to a field/global?
   - May be null: Can it be null? (always false for static/global/parameter)

   **IMPORTANT**: Enumerate EVERY distinct allocation site individually. \
Do NOT collapse multiple allocations into a single entry. Each call to \
malloc/calloc/realloc (direct or via wrapper) is a separate entry.

   **Do NOT report**: ordinary local variables, fixed-size stack arrays, \
compound literals, static const tables, struct declarations on the stack, \
or assembly push instructions — UNLESS they are returned or stored to a \
caller-visible location.

2. **Parameters**: Role of each parameter
   - Role: size_indicator, buffer, count, pointer_out, etc.
   - Used in allocation: Does it affect allocation size?

3. **Buffer-size pairs** (post-condition only): Identify (buffer, size) \
pairs that this function **creates or establishes**. Only include pairs \
where this function is the one that sets up the relationship (e.g., \
allocates the buffer with the given size, or assigns both fields of a \
struct). Do NOT include pairs that the function merely reads, accesses, \
or requires as input — those are pre-conditions and belong to a \
separate analysis.
   - "param_pair": function allocates a buffer and stores it alongside \
its size (e.g., `buf = malloc(n); *out_buf = buf; *out_len = n;`)
   - "struct_field": function sets both buffer and size fields on a \
struct (e.g., `s->data = malloc(n); s->len = n;`)
   - "flexible_array": function allocates a struct with a trailing \
flexible array and sets the length field
   - Kind must be exactly one of: "param_pair", "struct_field", "flexible_array"
   - Both buffer and size must be non-null — omit entries where the size is unknown
   - Relationship: how the size relates to the buffer (e.g., "byte count", "element count")

4. **Description**: One-sentence summary of what this function allocates

Consider:
- Wrapper functions that call allocators
- Size calculations (n + 1, n * sizeof(T), etc.)
- Conditional allocations
- Allocations stored to struct fields or output parameters\
"""


def _alloc_json_schema(func_name: str, *, brace: str = "{{") -> str:
    """Build the JSON response schema section.

    *brace* controls escaping depth: ``"{{"`` for single-format templates,
    ``"{{{{"```` for double-format templates (Approach B).
    """
    ob = brace          # opening brace  e.g. {{ or {{{{
    cb = "}}" if brace == "{{" else "}}}}"  # closing brace
    empty = "{{}}" if brace == "{{" else "{{{{}}}}"

    return f"""\
Respond in JSON format:
```json
{ob}
  "function": "{func_name}",
  "description": "One-sentence summary of what this function allocates",
  "parameters": {ob}
    "param_name": {ob}
      "role": "role description",
      "used_in_allocation": true|false
    {cb}
  {cb},
  "allocations": [
    {ob}
      "type": "heap|static|parameter_derived|escaped_stack",
      "source": "allocator or provenance",
      "size_expr": "size expression or null",
      "size_params": ["parameter names affecting size"],
      "returned": true|false,
      "stored_to": "field/variable name or null",
      "may_be_null": true|false
    {cb}
  ],
  "buffer_size_pairs": [
    {ob}
      "buffer": "buffer variable/field",
      "size": "size variable/field",
      "kind": "param_pair|struct_field|flexible_array",
      "relationship": "byte count|element count|max capacity"
    {cb}
  ]
{cb}
```

If the function does not allocate memory or return/store pointers, return:
```json
{ob}
  "function": "{func_name}",
  "description": "Does not allocate memory",
  "parameters": {empty},
  "allocations": [],
  "buffer_size_pairs": []
{cb}
```"""


# --- Single-message prompt (no caching) ---

ALLOCATION_SUMMARY_PROMPT = (
    "You are analyzing C/C++ code to generate memory allocation summaries.\n\n"
    "## Function to Analyze\n\n"
    "```c\n{source}\n```\n\n"
    "Function: `{name}`\n"
    "Signature: `{signature}`\n"
    "File: {file_path}\n\n"
    "## Callee Summaries\n\n"
    "{callee_summaries}\n\n"
    "## Task\n\n"
    "Generate a memory allocation summary for this function. Identify:\n\n"
    + _ALLOC_INSTRUCTIONS + "\n\n"
    + _alloc_json_schema("{name}") + "\n"
)

# --- Approach A templates (cache_mode="instructions") ---
# System: static task instructions (cached across all functions in a pass)
# User: function source + callee summaries (varies per function)

ALLOCATION_SYSTEM_PROMPT = (
    "You are analyzing C/C++ code to generate memory allocation summaries.\n\n"
    "## Task\n\n"
    "Generate a memory allocation summary for the function provided "
    "in the user message. Identify:\n\n"
    + _ALLOC_INSTRUCTIONS + "\n\n"
    + _alloc_json_schema("<function_name>")
)

ALLOCATION_USER_PROMPT = """\
## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Summaries

{callee_summaries}\
"""

# --- Approach B template (cache_mode="source") ---
# System: function source (cached across passes for same function, via prompts.py)
# User: task instructions + callee summaries

ALLOC_TASK_PROMPT = (
    "## Task\n\n"
    "Generate a memory allocation summary for the function "
    "in the system message. Identify:\n\n"
    + _ALLOC_INSTRUCTIONS + "\n\n"
    "## Callee Summaries\n\n"
    "{callee_summaries}\n\n"
    + _alloc_json_schema("{name}", brace="{{{{")
)


# --- Block prompt for chunked summarization of large functions ---

BLOCK_ALLOCATION_PROMPT = (
    "You are analyzing a code block from a large C/C++ function.\n\n"
    "## Context\n\n"
    "Function: `{name}`\n"
    "Signature: `{signature}`\n"
    "File: {file_path}\n\n"
    "## Code Block\n\n"
    "```c\n{block_source}\n```\n\n"
    "## Task\n\n"
    "Analyze this code block for **heap** memory allocations (malloc, calloc, "
    "realloc, mmap, new, etc.). Also suggest a descriptive pseudo-function name "
    "and signature for this block (as if it were extracted into its own function).\n\n"
    "Do NOT report ordinary local variables, fixed-size stack arrays, compound "
    "literals, static const tables, or struct declarations on the stack. Only "
    'report "escaped_stack" if a stack buffer escapes (returned or stored to a '
    "caller-visible location) — this is a bug.\n\n"
    "Respond in JSON:\n"
    "```json\n"
    "{{{{\n"
    '  "suggested_name": "descriptive_name_for_this_case",\n'
    '  "suggested_signature": "void descriptive_name(args)",\n'
    '  "summary": "One-sentence description of what this case block does '
    'regarding allocation",\n'
    '  "allocations": [\n'
    "    {{{{\n"
    '      "type": "heap",\n'
    '      "source": "allocator function name",\n'
    '      "size_expr": "size expr (prefer persistent: '
    'param/field/global; prefix local vars with local:)",\n'
    '      "size_params": [],\n'
    '      "returned": false,\n'
    '      "stored_to": "field or null",\n'
    '      "may_be_null": true\n'
    "    }}}}\n"
    "  ]\n"
    "}}}}\n"
    "```\n\n"
    "If no allocations, return empty allocations list with a summary of "
    "what the block does.\n"
)


_ALLOC_ITEM = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "source": {"type": "string"},
        "size_expr": {"type": "string"},
        "size_params": {"type": "array", "items": {"type": "string"}},
        "returned": {"type": "boolean"},
        "stored_to": {"type": "string"},
        "may_be_null": {"type": "boolean"},
    },
    "required": ["type", "source"],
}

_BSP_ITEM = {
    "type": "object",
    "properties": {
        "buffer": {"type": "string"},
        "size": {"type": "string"},
        "kind": {"type": "string"},
        "relationship": {"type": "string"},
    },
    "required": ["buffer", "size", "kind", "relationship"],
}

ALLOC_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "allocations": {"type": "array", "items": _ALLOC_ITEM},
        "parameters": {"type": "object"},
        "buffer_size_pairs": {"type": "array", "items": _BSP_ITEM},
        "description": {"type": "string"},
    },
    "required": ["function", "allocations", "parameters",
                  "buffer_size_pairs", "description"],
})

ALLOC_BLOCK_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "suggested_name": {"type": "string"},
        "suggested_signature": {"type": "string"},
        "allocations": {"type": "array", "items": _ALLOC_ITEM},
        "summary": {"type": "string"},
    },
    "required": ["suggested_name", "suggested_signature",
                  "allocations", "summary"],
})


class AllocationSummarizer:
    """Generates allocation summaries for functions using LLM."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        allocators: list[str] | None = None,
        cache_mode: str = "none",
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.allocators = allocators or []
        self.cache_mode = cache_mode
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
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
        callee_summaries: dict[str, AllocationSummary] | None = None,
        previous_summary_json: str | None = None,
    ) -> AllocationSummary:
        """
        Generate allocation summary for a single function.

        Args:
            func: The function to summarize
            callee_summaries: Pre-computed summaries for callees

        Returns:
            AllocationSummary for the function
        """
        if callee_summaries is None:
            callee_summaries = {}

        # Check for large function with blocks
        blocks = self.db.get_function_blocks(func.id) if func.id else []
        if blocks and len(func.llm_source) > 40000:
            return self._summarize_large_function(func, callee_summaries, blocks)

        # Build callee summaries section
        callee_section = self._build_callee_section(func, callee_summaries)

        # Build prompt (with optional system message for caching)
        prompt, system, cache_system = self._build_prompt_and_system(
            func.llm_source, func, callee_section,
        )

        # On SCC re-iterations, append previous summary for convergence check
        if previous_summary_json is not None:
            from .driver import SCC_PREVIOUS_SUMMARY_SECTION
            prompt += SCC_PREVIOUS_SUMMARY_SECTION.format(
                previous_json=previous_summary_json,
            )

        # Query LLM
        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(
                        f"  ({self._progress_current}/"
                        f"{self._progress_total})"
                        f" Summarizing: {func.name}"
                    )
                else:
                    print(f"  Summarizing: {func.name}")

            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=ALLOC_RESPONSE_FORMAT,
            )
            with self._stats_lock:
                self._stats["llm_calls"] += 1
                if llm_response.cached:
                    self._stats["cache_hits"] += 1
                self._stats["cache_read_tokens"] += llm_response.cache_read_tokens
                self._stats["cache_creation_tokens"] += llm_response.cache_creation_tokens

            # Log prompt and response if requested
            if self.log_file:
                self._log_interaction(func.name, prompt, llm_response.content)

            summary = self._parse_response(llm_response.content, func.name)
            with self._stats_lock:
                self._stats["functions_processed"] += 1

            # SCC convergence: extract "changed" from parsed JSON
            if previous_summary_json is not None:
                from .builder.json_utils import extract_json as _ej
                from .driver import extract_scc_changed
                summary._scc_changed = extract_scc_changed(  # type: ignore[attr-defined]
                    _ej(llm_response.content),
                )

            return summary

        except Exception as e:
            with self._stats_lock:
                self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error summarizing {func.name}: {e}")

            # Return empty summary on error
            return AllocationSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _summarize_large_function(
        self,
        func: Function,
        callee_summaries: dict[str, AllocationSummary],
        blocks: list[FunctionBlock],
    ) -> AllocationSummary:
        """Chunked summarization for functions too large for a single prompt.

        Phase A: Summarize each switch-case block individually.
        Phase B: Build a skeleton with block summaries, then summarize the skeleton.
        Phase C: Merge block-level and skeleton-level results.
        """
        if self.verbose:
            src_len = len(func.llm_source)
            n_blocks = len(blocks)
            print(
                f"  Large function ({src_len} chars,"
                f" {n_blocks} blocks): {func.name}"
            )

        # Phase A: Summarize each block
        block_summaries: dict[int, str] = {}
        all_block_allocations: list[Allocation] = []

        for i, block in enumerate(blocks):
            assert block.id is not None
            if block.summary_json:
                # Already summarized (cached)
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    for a in data.get("allocations", []):
                        alloc_type_str = a.get("type", "unknown").lower()
                        try:
                            alloc_type = AllocationType(alloc_type_str)
                        except ValueError:
                            alloc_type = AllocationType.UNKNOWN
                        all_block_allocations.append(Allocation(
                            alloc_type=alloc_type,
                            source=a.get("source", ""),
                            size_expr=a.get("size_expr"),
                            size_params=a.get("size_params", []),
                            returned=a.get("returned", False),
                            stored_to=a.get("stored_to"),
                            may_be_null=a.get("may_be_null", True),
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            prompt = BLOCK_ALLOCATION_PROMPT.format(
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                block_source=block.source,
            )

            try:
                if self.verbose:
                    print(f"    Block {i+1}/{len(blocks)}: {block.label[:60]}")

                response = self.llm.complete(
                    prompt, response_format=ALLOC_BLOCK_RESPONSE_FORMAT,
                )
                with self._stats_lock:
                    self._stats["llm_calls"] += 1

                # Parse block response
                json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    json_str = json_match.group(0) if json_match else "{}"

                data = json.loads(json_str)
                summary_text = data.get("summary", "no summary")
                block_summaries[block.id] = summary_text

                # Store block summary in DB
                self.db.update_function_block_summary(
                    block.id,
                    json.dumps(data),
                    data.get("suggested_name"),
                    data.get("suggested_signature"),
                )

                # Collect allocations from block
                for a in data.get("allocations", []):
                    alloc_type_str = a.get("type", "unknown").lower()
                    try:
                        alloc_type = AllocationType(alloc_type_str)
                    except ValueError:
                        alloc_type = AllocationType.UNKNOWN
                    all_block_allocations.append(Allocation(
                        alloc_type=alloc_type,
                        source=a.get("source", ""),
                        size_expr=a.get("size_expr"),
                        size_params=a.get("size_params", []),
                        returned=a.get("returned", False),
                        stored_to=a.get("stored_to"),
                        may_be_null=a.get("may_be_null", True),
                    ))

            except Exception as e:
                if self.verbose:
                    print(f"    Error summarizing block {block.label}: {e}")
                block_summaries[block.id] = f"(error: {e})"

        # Phase B: Build skeleton and summarize
        skeleton = build_skeleton(func.llm_source, func.line_start, blocks, block_summaries)

        callee_section = self._build_callee_section(func, callee_summaries)
        prompt, system, cache_system = self._build_prompt_and_system(
            skeleton, func, callee_section,
        )

        try:
            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=ALLOC_RESPONSE_FORMAT,
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
            skeleton_summary = AllocationSummary(
                function_name=func.name,
                description=f"Error summarizing skeleton: {e}",
            )

        # Phase C: Merge — combine block allocations with skeleton allocations
        merged_allocations = list(skeleton_summary.allocations) + all_block_allocations
        skeleton_summary.allocations = merged_allocations

        with self._stats_lock:
            self._stats["functions_processed"] += 1

        return skeleton_summary

    def _build_prompt_and_system(
        self, source: str, func: Function, callee_section: str,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system) based on self.cache_mode."""
        if self.cache_mode == "instructions":
            prompt = ALLOCATION_USER_PROMPT.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                callee_summaries=callee_section,
            )
            return prompt, ALLOCATION_SYSTEM_PROMPT, True
        elif self.cache_mode == "source":
            from .prompts import FUNCTION_CONTEXT_SYSTEM
            system = FUNCTION_CONTEXT_SYSTEM.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
            )
            prompt = ALLOC_TASK_PROMPT.format(
                name=func.name,
                callee_summaries=callee_section,
            )
            return prompt, system, True
        else:
            prompt = ALLOCATION_SUMMARY_PROMPT.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                callee_summaries=callee_section,
            )
            return prompt, None, False

    def _get_callee_attributes(self, callee_names: list[str]) -> dict[str, str]:
        """Look up attributes for callee functions."""
        attrs = {}
        for name in callee_names:
            funcs = self.db.get_function_by_name(name)
            if funcs and funcs[0].attributes:
                attrs[name] = funcs[0].attributes
        return attrs

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, AllocationSummary],
    ) -> str:
        """Build the callee summaries section for the prompt."""
        if not callee_summaries and not self.allocators:
            return "No callee summaries available (leaf function or external calls only)."

        callee_attrs = self._get_callee_attributes(list(callee_summaries.keys()))

        lines = []
        for name, summary in callee_summaries.items():
            attr_suffix = f" {callee_attrs[name]}" if name in callee_attrs else ""
            if summary.allocations:
                parts = []
                for a in summary.allocations:
                    s = f"{a.source}({a.size_expr or 'unknown size'})"
                    if a.stored_to:
                        s += f" → {a.stored_to}"
                    if not a.may_be_null:
                        s += " [never null]"
                    parts.append(s)
                alloc_desc = ", ".join(parts)
                lines.append(f"- `{name}`: Allocates via {alloc_desc}{attr_suffix}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'No allocations'}{attr_suffix}")
            if summary.buffer_size_pairs:
                pairs_desc = ", ".join(
                    f"({p.buffer}, {p.size})" for p in summary.buffer_size_pairs
                )
                lines.append(f"  Buffer-size pairs: {pairs_desc}")

        # Append project-specific allocators not already covered
        covered_names = set(callee_summaries.keys())
        for alloc_name in self.allocators:
            if alloc_name not in covered_names:
                lines.append(f"- `{alloc_name}`: Known heap allocator (project-specific)")

        if not lines:
            return "No callee summaries available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        if not self.log_file:
            return
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> AllocationSummary:
        """Parse LLM response into AllocationSummary."""
        from .builder.json_utils import extract_json

        data = extract_json(response)

        # Parse allocations
        allocations = []
        for a in data.get("allocations", []):
            alloc_type_str = a.get("type", "unknown").lower()
            try:
                alloc_type = AllocationType(alloc_type_str)
            except ValueError:
                alloc_type = AllocationType.UNKNOWN

            allocations.append(
                Allocation(
                    alloc_type=alloc_type,
                    source=a.get("source", ""),
                    size_expr=a.get("size_expr"),
                    size_params=a.get("size_params", []),
                    returned=a.get("returned", False),
                    stored_to=a.get("stored_to"),
                    may_be_null=a.get("may_be_null", True),
                )
            )

        # Parse parameters
        parameters = {}
        for name, info in data.get("parameters", {}).items():
            parameters[name] = ParameterInfo(
                role=info.get("role", ""),
                used_in_allocation=info.get("used_in_allocation", False),
            )

        # Parse buffer-size pairs (validate and filter)
        valid_kinds = {"param_pair", "struct_field", "flexible_array"}
        buffer_size_pairs = []
        for p in data.get("buffer_size_pairs", []):
            buf = p.get("buffer")
            size = p.get("size")
            if not buf or not size or size == "None":
                continue
            kind = p.get("kind", "param_pair")
            if kind not in valid_kinds:
                continue
            buffer_size_pairs.append(
                BufferSizePair(
                    buffer=buf,
                    size=str(size),
                    kind=kind,
                    relationship=p.get("relationship", ""),
                )
            )

        return AllocationSummary(
            function_name=data.get("function", func_name),
            allocations=allocations,
            parameters=parameters,
            buffer_size_pairs=buffer_size_pairs,
            description=data.get("description", ""),
        )


class IncrementalSummarizer:
    """Handles incremental updates when source files change."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.summarizer = AllocationSummarizer(db, llm, verbose, log_file)

    def update_function(self, func: Function) -> list[int]:
        """
        Update summary for a function and invalidate dependents.

        Returns list of invalidated function IDs.
        """
        if func.id is None:
            func.id = self.db.insert_function(func)

        # Invalidate this function and all its callers
        invalidated = self.db.invalidate_and_cascade(func.id)

        if self.verbose:
            print(f"Invalidated {len(invalidated)} functions")

        return invalidated

    def resync_invalidated(self) -> dict[int, AllocationSummary]:
        """
        Re-summarize all functions that need updating.

        Returns mapping of function ID to new summary.
        """
        from .driver import AllocationPass, BottomUpDriver

        driver = BottomUpDriver(self.db, verbose=self.verbose)
        alloc_pass = AllocationPass(self.summarizer, self.db, self.llm.model)
        results = driver.run([alloc_pass], force=False)
        return results["allocation"]

    def update_file(
        self, file_path: str, new_functions: list[Function],
    ) -> dict[int, AllocationSummary]:
        """
        Update summaries for all functions in a file.

        Args:
            file_path: Path to the updated file
            new_functions: Newly extracted functions from the file

        Returns:
            Updated summaries
        """
        # Get existing functions in this file
        existing = self.db.get_functions_by_file(file_path)
        existing_by_sig = {(f.name, f.signature): f for f in existing}

        # Find changed and new functions
        for func in new_functions:
            key = (func.name, func.signature)
            if key in existing_by_sig:
                old_func = existing_by_sig[key]
                if self.db.needs_update(func):
                    func.id = old_func.id
                    self.update_function(func)
            else:
                # New function
                self.db.insert_function(func)

        # Re-sync all invalidated
        return self.resync_invalidated()
