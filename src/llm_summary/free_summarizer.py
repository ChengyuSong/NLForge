"""LLM-based free/deallocation summary generator."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    FreeOp,
    FreeSummary,
    Function,
    FunctionBlock,
    build_skeleton,
)

FREE_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate deallocation (free) summaries.

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Free Summaries

{callee_summaries}

## Task

Generate a deallocation summary for this function. Identify every buffer or resource
that this function frees (directly or via callees).

For each free operation, identify:

1. **target**: What gets freed — the expression (e.g., "ptr", "info_ptr->palette", "row_buf")
2. **target_kind**: One of:
   - "parameter" — a function parameter is freed
   - "field" — a struct field (accessed via parameter or global) is freed
   - "local" — a local variable is freed
   - "return_value" — the freed pointer is also returned (rare)
3. **deallocator**: The function that performs the free (e.g., "free", "fclose", "closedir")
4. **conditional**: true if the free is inside an if-block, error path, or conditional
5. **condition**: If conditional is true, the C expression that guards the free \
(e.g., "do_close != 0", "ptr != NULL", "error path"). \
If a callee's free is conditional on one of this function's parameters, \
record the condition in terms of this function's parameters. Omit if conditional is false.
6. **nulled_after**: true if the pointer is set to NULL after the free

Consider:
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
include the free with `conditional: false`.

Respond in JSON format:
```json
{{
  "function": "{name}",
  "frees": [
    {{
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false
    }}
  ],
  "description": "One-sentence description of what this function frees"
}}
```

If the function does not free any memory (directly or via callees), return:
```json
{{
  "function": "{name}",
  "frees": [],
  "description": "Does not free memory"
}}
```
"""


BLOCK_FREE_PROMPT = """You are analyzing a code block from a large C/C++ function.

## Context

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Code Block

```c
{block_source}
```

## Task

Analyze this code block for free/deallocation operations. Also suggest a descriptive
pseudo-function name and signature for this block.

Respond in JSON:
```json
{{{{
  "suggested_name": "descriptive_name_for_this_case",
  "suggested_signature": "void descriptive_name(args)",
  "frees": [
    {{{{
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false
    }}}}
  ],
  "summary": "One-sentence description of what this case block does regarding deallocation"
}}}}
```

If no frees, return empty frees list with a summary of what the block does.
"""


# --- Approach A templates (cache_mode="instructions") ---

FREE_SYSTEM_PROMPT = """\
You are analyzing C/C++ code to generate deallocation (free) summaries.

## Task

Generate a deallocation summary for the function provided in the \
user message. Identify every buffer or resource that this function \
frees (directly or via callees).

For each free operation, identify:

1. **target**: What gets freed — the expression (e.g., "ptr", "info_ptr->palette", "row_buf")
2. **target_kind**: One of:
   - "parameter" — a function parameter is freed
   - "field" — a struct field (accessed via parameter or global) is freed
   - "local" — a local variable is freed
   - "return_value" — the freed pointer is also returned (rare)
3. **deallocator**: The function that performs the free (e.g., "free", "fclose", "closedir")
4. **conditional**: true if the free is inside an if-block, error path, or conditional
5. **condition**: If conditional is true, the C expression that guards the free \
(e.g., "do_close != 0", "ptr != NULL", "error path"). \
If a callee's free is conditional on one of this function's parameters, \
record the condition in terms of this function's parameters. Omit if conditional is false.
6. **nulled_after**: true if the pointer is set to NULL after the free

Consider:
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
include the free with `conditional: false`.

Respond in JSON format:
```json
{{
  "function": "<function_name>",
  "frees": [
    {{
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false
    }}
  ],
  "description": "One-sentence description of what this function frees"
}}
```

If the function does not free any memory (directly or via callees), return:
```json
{{
  "function": "<function_name>",
  "frees": [],
  "description": "Does not free memory"
}}
```\
"""

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

FREE_TASK_PROMPT = """\
## Task

Generate a deallocation summary for the function in the system \
message. Identify every buffer or resource that this function \
frees (directly or via callees).

For each free operation, identify:

1. **target**: What gets freed — the expression
2. **target_kind**: One of: "parameter", "field", "local", "return_value"
3. **deallocator**: The function that performs the free
4. **conditional**: true if the free is conditional
5. **condition**: If conditional is true, the C expression that guards the free. \
If a callee's free is conditional on a parameter, record it in terms of \
this function's parameters. Omit if unconditional.
6. **nulled_after**: true if the pointer is set to NULL after the free

Consider:
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
include the free with `conditional: false`.

## Callee Free Summaries

{callee_summaries}

Respond in JSON format:
```json
{{{{
  "function": "{name}",
  "frees": [
    {{{{
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "condition": "guard expression (omit if unconditional)",
      "nulled_after": true|false
    }}}}
  ],
  "description": "One-sentence description of what this function frees"
}}}}
```

If the function does not free any memory (directly or via callees), return:
```json
{{{{
  "function": "{name}",
  "frees": [],
  "description": "Does not free memory"
}}}}
```\
"""


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

        for i, block in enumerate(blocks):
            assert block.id is not None
            if block.summary_json:
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    for f in data.get("frees", []):
                        cond = f.get("conditional", False)
                        all_block_frees.append(FreeOp(
                            target=f.get("target", ""),
                            target_kind=f.get("target_kind", "local"),
                            deallocator=f.get("deallocator", "free"),
                            conditional=cond,
                            nulled_after=f.get("nulled_after", False),
                            condition=f.get("condition") if cond else None,
                        ))
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

                for f in data.get("frees", []):
                    cond = f.get("conditional", False)
                    all_block_frees.append(FreeOp(
                        target=f.get("target", ""),
                        target_kind=f.get("target_kind", "local"),
                        deallocator=f.get("deallocator", "free"),
                        conditional=cond,
                        nulled_after=f.get("nulled_after", False),
                        condition=f.get("condition") if cond else None,
                    ))
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

        lines = []
        for name, summary in callee_summaries.items():
            attr_suffix = f" {callee_attrs[name]}" if name in callee_attrs else ""
            if summary.frees:
                free_parts = []
                for f in summary.frees:
                    part = f"{f.deallocator}({f.target})"
                    extras = []
                    if f.conditional:
                        cond_text = f"when {f.condition}" if f.condition else "conditional"
                        extras.append(cond_text)
                    if f.nulled_after:
                        extras.append("nulled_after")
                    if extras:
                        part += f" [{', '.join(extras)}]"
                    free_parts.append(part)
                free_desc = ", ".join(free_parts)
                lines.append(f"- `{name}`: Frees {free_desc}{attr_suffix}")
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

        # Parse frees
        valid_kinds = {"parameter", "field", "local", "return_value"}
        frees = []
        for f in data.get("frees", []):
            target_kind = f.get("target_kind", "local")
            if target_kind not in valid_kinds:
                target_kind = "local"
            conditional = f.get("conditional", False)
            condition = f.get("condition") if conditional else None
            frees.append(
                FreeOp(
                    target=f.get("target", ""),
                    target_kind=target_kind,
                    deallocator=f.get("deallocator", "free"),
                    conditional=conditional,
                    nulled_after=f.get("nulled_after", False),
                    condition=condition,
                )
            )

        return FreeSummary(
            function_name=data.get("function", func_name),
            frees=frees,
            description=data.get("description", ""),
        )
