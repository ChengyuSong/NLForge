"""LLM-based free/deallocation summary generator."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    FreeOp,
    FreeSummary,
    Function,
    FunctionBlock,
    build_skeleton,
)

# --- Shared free/deallocation task instructions (single source of truth) ---

_FREE_INSTRUCTIONS = """\
Separate heap deallocations from non-heap resource cleanup:

- **frees**: Heap memory deallocations — free, realloc, munmap, custom wrappers \
(xfree, g_free, etc.), fclose (frees FILE*), closedir (frees DIR*).
- **resource_releases**: Non-heap resource cleanup — close(fd), sem_destroy, \
pthread_join, pthread_cancel, and similar. Do NOT include internal stdio \
buffer management (e.g. fprintf/fwrite/putc may internally realloc a \
buffer — that is an implementation detail, not a free or resource release).

For C++ member functions, `this` is an implicit pointer parameter. \
Frees of `this` itself (e.g., `delete this`) or fields accessed via `this` \
(e.g., `delete m_data` which is `this->m_data`) should be reported accordingly.

For each operation (in either list), identify:

1. **target**: What gets freed — the expression (e.g., "ptr", "info_ptr->palette", "row_buf")
2. **target_kind**: One of:
   - "parameter" — a function parameter (including `this`) is freed
   - "field" — a struct field (accessed via parameter, `this`, or global) is freed
   - "local" — a local variable is freed
   - "return_value" — the freed pointer is also returned (rare)
3. **deallocator**: The function that performs the free (e.g., "free", "close", "sem_destroy")
4. **conditional**: true if the free is inside an if-block, error path, or conditional
5. **condition**: If conditional is true, express the condition in terms of \
**caller-observable quantities only** — the function's own parameters \
and/or return value. The caller cannot see internal locals or callee results. \
Over-approximation is fine: soundness matters more than precision. \
Omit if conditional is false.
6. **nulled_after**: true if the pointer is set to NULL after the free
7. **description**: For loop-based or transitive frees, describe what is \
freed (e.g., "frees all elements in a linked list", \
"frees all entries in a hash table"). Omit for simple single-pointer frees.

**Caller-visible abstraction**: Only report frees at the abstraction level of \
THIS function's code. If this function calls `cleanup(ctx)` and the callee \
summary says `cleanup` frees `ctx->buf`, `ctx->name`, and `ctx->data`, \
report ONE entry: target="ctx", deallocator="cleanup", target_kind="parameter". \
Do NOT enumerate the callee's internal frees — the callee's own summary \
already captures those details. \
Exception: if this function directly accesses and frees a field \
(e.g., `free(ctx->buf); cleanup(ctx);`), report the direct `free(ctx->buf)` \
AND the `cleanup(ctx)` call separately.

**IMPORTANT**: Enumerate EVERY distinct free/release site individually. \
Do NOT collapse multiple frees into a single entry.

Other considerations:
- Direct calls to free/deallocator functions
- Wrapper functions that free (use callee summaries)
- Conditional frees (inside if-blocks, error paths)
- Whether the pointer is NULLed after free (defensive pattern)
- If a callee has a conditional free, check whether this function always \
satisfies or never satisfies that condition, and adjust accordingly: \
if the condition is always false at the call site, omit the free entirely; \
if always true, mark it as unconditional. \
Example: if callee `cleanup(p, int do_free)` has `free(p->data) [when do_free != 0]`, \
and this function calls `cleanup(x, 0)`, then do_free=0 so the condition \
`do_free != 0` is ALWAYS FALSE — **do NOT include** that free in the output. \
Conversely, if this function calls `cleanup(x, 1)`, the condition is ALWAYS TRUE — \
include the free with `conditional: false`.\
"""


def _free_json_schema(func_name: str, *, brace: str = "{{") -> str:
    """Build the JSON response schema section for free summaries."""
    ob = brace
    cb = "}}" if brace == "{{" else "}}}}"

    return f"""\
Respond in JSON format:
```json
{ob}
  "function": "{func_name}",
  "description": "One-sentence description of what this function frees/releases",
  "frees": [
    {ob}
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false,
      "description": "for loop/transitive frees only (omit for simple frees)"
    {cb}
  ],
  "resource_releases": [
    {ob}
      "target": "resource being released",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "close|sem_destroy|etc",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false
    {cb}
  ]
{cb}
```

If the function does not free any memory or release resources, return:
```json
{ob}
  "function": "{func_name}",
  "description": "Does not free memory",
  "frees": [],
  "resource_releases": []
{cb}
```"""


# --- Single-message prompt (no caching) ---

FREE_SUMMARY_PROMPT = (
    "You are analyzing C/C++ code to generate deallocation (free) summaries.\n\n"
    "## Function to Analyze\n\n"
    "```c\n{source}\n```\n\n"
    "Function: `{name}`\n"
    "Signature: `{signature}`\n"
    "File: {file_path}\n\n"
    "## Callee Free Summaries\n\n"
    "{callee_summaries}\n\n"
    "## Task\n\n"
    "Generate a deallocation summary for this function. Identify every buffer or resource\n"
    "that this function frees (directly or via callees).\n\n"
    + _FREE_INSTRUCTIONS + "\n\n"
    + _free_json_schema("{name}") + "\n"
)

# --- Approach A templates (cache_mode="instructions") ---

FREE_SYSTEM_PROMPT = (
    "You are analyzing C/C++ code to generate deallocation (free) summaries.\n\n"
    "## Task\n\n"
    "Generate a deallocation summary for the function provided in the "
    "user message. Identify every buffer or resource that this function "
    "frees (directly or via callees).\n\n"
    + _FREE_INSTRUCTIONS + "\n\n"
    + _free_json_schema("<function_name>")
)

FREE_USER_PROMPT = """\
## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Free Summaries

{callee_summaries}\
"""

# --- Approach B template (cache_mode="source") ---

FREE_TASK_PROMPT = (
    "## Task\n\n"
    "Generate a deallocation summary for the function in the system "
    "message. Identify every buffer or resource that this function "
    "frees (directly or via callees).\n\n"
    + _FREE_INSTRUCTIONS + "\n\n"
    "## Callee Free Summaries\n\n"
    "{callee_summaries}\n\n"
    + _free_json_schema("{name}", brace="{{{{")
)


# --- Block prompt for chunked summarization of large functions ---

BLOCK_FREE_PROMPT = (
    "You are analyzing a code block from a large C/C++ function.\n\n"
    "## Context\n\n"
    "Function: `{name}`\n"
    "Signature: `{signature}`\n"
    "File: {file_path}\n\n"
    "## Code Block\n\n"
    "```c\n{block_source}\n```\n\n"
    "## Task\n\n"
    "Analyze this code block for free/deallocation operations. Also suggest a descriptive\n"
    "pseudo-function name and signature for this block.\n\n"
    "Respond in JSON:\n"
    "```json\n"
    "{{{{\n"
    '  "suggested_name": "descriptive_name_for_this_case",\n'
    '  "suggested_signature": "void descriptive_name(args)",\n'
    '  "summary": "One-sentence description of what this case block does '
    'regarding deallocation",\n'
    '  "frees": [\n'
    "    {{{{\n"
    '      "target": "expression being freed",\n'
    '      "target_kind": "parameter|field|local|return_value",\n'
    '      "deallocator": "free function name",\n'
    '      "conditional": true|false,\n'
    '      "condition": "guard expression (omit if unconditional)",\n'
    '      "nulled_after": true|false\n'
    "    }}}}\n"
    "  ]\n"
    "}}}}\n"
    "```\n\n"
    "If no frees, return empty frees list with a summary of what the block does.\n"
)


_FREE_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "target_kind": {"type": "string"},
        "deallocator": {"type": "string"},
        "conditional": {"type": "boolean"},
        "condition": {"type": "string"},
        "nulled_after": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["target", "target_kind", "deallocator"],
}

FREE_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "description": {"type": "string"},
        "frees": {"type": "array", "items": _FREE_ITEM},
        "resource_releases": {"type": "array", "items": _FREE_ITEM},
    },
    "required": ["function", "description", "frees", "resource_releases"],
})

FREE_BLOCK_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "suggested_name": {"type": "string"},
        "suggested_signature": {"type": "string"},
        "summary": {"type": "string"},
        "frees": {"type": "array", "items": _FREE_ITEM},
    },
    "required": ["suggested_name", "suggested_signature", "summary", "frees"],
})


class FreeSummarizer:
    """Generates free/deallocation summaries for functions using LLM."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        deallocators: list[str] | None = None,
        cache_mode: str = "none",
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.deallocators = deallocators or []
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
        callee_summaries: dict[str, FreeSummary] | None = None,
        previous_summary_json: str | None = None,
    ) -> FreeSummary:
        """Generate free summary for a single function."""
        if callee_summaries is None:
            callee_summaries = {}

        # Check for large function with blocks
        blocks = self.db.get_function_blocks(func.id) if func.id else []
        if blocks and len(func.llm_source) > 40000:
            return self._summarize_large_function(func, callee_summaries, blocks)

        callee_section = self._build_callee_section(func, callee_summaries)

        prompt, system, cache_system = self._build_prompt_and_system(
            func.llm_source, func, callee_section,
        )

        if previous_summary_json is not None:
            from .driver import SCC_PREVIOUS_SUMMARY_SECTION
            prompt += SCC_PREVIOUS_SUMMARY_SECTION.format(
                previous_json=previous_summary_json,
            )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    cur = self._progress_current
                    tot = self._progress_total
                    print(
                        f"  ({cur}/{tot}) "
                        f"Summarizing (free): {func.name}"
                    )
                else:
                    print(f"  Summarizing (free): {func.name}")

            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=FREE_RESPONSE_FORMAT,
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
                print(f"  Error summarizing (free) {func.name}: {e}")

            return FreeSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _summarize_large_function(
        self,
        func: Function,
        callee_summaries: dict[str, FreeSummary],
        blocks: list[FunctionBlock],
    ) -> FreeSummary:
        """Chunked summarization for large functions (free pass)."""
        if self.verbose:
            n_chars = len(func.llm_source)
            n_blocks = len(blocks)
            print(
                f"  Large function ({n_chars} chars, "
                f"{n_blocks} blocks): {func.name}"
            )

        block_summaries: dict[int, str] = {}
        all_block_frees: list[FreeOp] = []
        all_block_releases: list[FreeOp] = []

        def _parse_block_ops(data: dict) -> None:
            for f in data.get("frees", []):
                cond = f.get("conditional", False)
                all_block_frees.append(FreeOp(
                    target=f.get("target", ""),
                    target_kind=f.get("target_kind", "local"),
                    deallocator=f.get("deallocator", "free"),
                    conditional=cond,
                    nulled_after=f.get("nulled_after", False),
                    condition=f.get("condition") if cond else None,
                    description=f.get("description"),
                ))
            for f in data.get("resource_releases", []):
                cond = f.get("conditional", False)
                all_block_releases.append(FreeOp(
                    target=f.get("target", ""),
                    target_kind=f.get("target_kind", "local"),
                    deallocator=f.get("deallocator", "close"),
                    conditional=cond,
                    nulled_after=f.get("nulled_after", False),
                    condition=f.get("condition") if cond else None,
                    description=f.get("description"),
                ))

        for i, block in enumerate(blocks):
            assert block.id is not None
            if block.summary_json:
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    _parse_block_ops(data)
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            prompt = BLOCK_FREE_PROMPT.format(
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                block_source=block.source,
            )

            try:
                if self.verbose:
                    print(f"    Block {i+1}/{len(blocks)}: {block.label[:60]}")
                response = self.llm.complete(
                    prompt, response_format=FREE_BLOCK_RESPONSE_FORMAT,
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
                _parse_block_ops(data)
            except Exception as e:
                if self.verbose:
                    print(f"    Error summarizing block {block.label}: {e}")
                block_summaries[block.id] = f"(error: {e})"

        # Phase B: skeleton
        skeleton = build_skeleton(func.llm_source, func.line_start, blocks, block_summaries)
        callee_section = self._build_callee_section(func, callee_summaries)
        prompt, system, cache_system = self._build_prompt_and_system(
            skeleton, func, callee_section,
        )

        try:
            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=FREE_RESPONSE_FORMAT,
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
            skeleton_summary = FreeSummary(
                function_name=func.name, description=f"Error summarizing skeleton: {e}",
            )

        # Phase C: merge
        skeleton_summary.frees = list(skeleton_summary.frees) + all_block_frees
        skeleton_summary.resource_releases = (
            list(skeleton_summary.resource_releases) + all_block_releases
        )
        with self._stats_lock:
            self._stats["functions_processed"] += 1
        return skeleton_summary

    def _build_prompt_and_system(
        self, source: str, func: Function, callee_section: str,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system) based on self.cache_mode."""
        if self.cache_mode == "instructions":
            prompt = FREE_USER_PROMPT.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                callee_summaries=callee_section,
            )
            return prompt, FREE_SYSTEM_PROMPT, True
        elif self.cache_mode == "source":
            from .prompts import FUNCTION_CONTEXT_SYSTEM
            system = FUNCTION_CONTEXT_SYSTEM.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
            )
            prompt = FREE_TASK_PROMPT.format(
                name=func.name,
                callee_summaries=callee_section,
            )
            return prompt, system, True
        else:
            prompt = FREE_SUMMARY_PROMPT.format(
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
        callee_summaries: dict[str, FreeSummary],
    ) -> str:
        """Build the callee free summaries section for the prompt."""
        if not callee_summaries and not self.deallocators:
            return "No callee free summaries available (leaf function or external calls only)."

        callee_attrs = self._get_callee_attributes(list(callee_summaries.keys()))

        def _format_ops(ops: list[FreeOp]) -> str:
            parts = []
            for f in ops:
                part = f"{f.deallocator}({f.target})"
                extras = []
                if f.conditional:
                    cond_text = f"when {f.condition}" if f.condition else "conditional"
                    extras.append(cond_text)
                if f.nulled_after:
                    extras.append("nulled_after")
                if extras:
                    part += f" [{', '.join(extras)}]"
                if f.description:
                    part += f" — {f.description}"
                parts.append(part)
            return ", ".join(parts)

        lines = []
        for name, summary in callee_summaries.items():
            attr_suffix = f" {callee_attrs[name]}" if name in callee_attrs else ""
            parts = []
            if summary.frees:
                parts.append(f"Frees {_format_ops(summary.frees)}")
            if summary.resource_releases:
                parts.append(f"Releases {_format_ops(summary.resource_releases)}")
            if parts:
                lines.append(f"- `{name}`: {'; '.join(parts)}{attr_suffix}")
            else:
                desc = summary.description or 'Does not free memory'
                lines.append(
                    f"- `{name}`: {desc}{attr_suffix}"
                )

        # Append project-specific deallocators not already covered
        covered_names = set(callee_summaries.keys())
        for dealloc_name in self.deallocators:
            if dealloc_name not in covered_names:
                lines.append(f"- `{dealloc_name}`: Known deallocator (project-specific)")

        if not lines:
            return "No callee free summaries available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        if not self.log_file:
            return
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [free pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> FreeSummary:
        """Parse LLM response into FreeSummary."""
        from .builder.json_utils import extract_json

        data = extract_json(response)

        valid_kinds = {"parameter", "field", "local", "return_value"}

        def _parse_ops(items: list[dict]) -> list[FreeOp]:
            ops = []
            for f in items:
                target_kind = f.get("target_kind", "local")
                if target_kind not in valid_kinds:
                    target_kind = "local"
                conditional = f.get("conditional", False)
                condition = f.get("condition") if conditional else None
                ops.append(
                    FreeOp(
                        target=f.get("target", ""),
                        target_kind=target_kind,
                        deallocator=f.get("deallocator", "free"),
                        conditional=conditional,
                        nulled_after=f.get("nulled_after", False),
                        condition=condition,
                        description=f.get("description"),
                    )
                )
            return ops

        return FreeSummary(
            function_name=data.get("function", func_name),
            frees=_parse_ops(data.get("frees", [])),
            resource_releases=_parse_ops(data.get("resource_releases", [])),
            description=data.get("description", ""),
        )
