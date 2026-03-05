"""LLM-based initialization summary generator."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Function,
    FunctionBlock,
    InitOp,
    InitSummary,
    build_skeleton,
)

INIT_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate initialization summaries (post-conditions).

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Initialization Summaries

{callee_summaries}

## Task

Generate an initialization summary for this function. Identify what this function
**always** initializes on ALL exit paths — only guaranteed, unconditional
initializations. This is a post-condition: only things visible to the CALLER matter.

Only include caller-visible initializations:
- **Output parameters**: memory written via pointer parameters (e.g., `*out = value`)
- **Struct fields**: fields written via a parameter (e.g., `ctx->data = ...`)
- **Return values**: the function's return value itself

Do NOT include local variables — they are not visible to the caller after return.

**Return value rule**: If every exit path returns a value (including NULL, 0, or error codes),
the return value IS unconditionally initialized. A function that returns NULL on error and a
pointer on success still always initializes its return value.

For each initialization, identify:

1. **target**: What gets initialized — the expression (e.g., "*out", "ctx->data", "return value")
2. **target_kind**: One of:
   - "parameter" — an output parameter is written via pointer dereference
   - "field" — a struct field is written via a parameter
   - "return_value" — the return value is always set (including NULL/error returns)
3. **initializer**: How it's initialized (e.g., "memset", "assignment", "calloc", "callee:func_name")
4. **byte_count**: How many bytes are initialized — "n", "sizeof(T)", "full", or null if unknown

Consider:
- Direct assignments to output parameters and struct fields
- Calls to memset, memcpy, calloc, etc. (use callee summaries)
- Only include initializations that happen on ALL exit paths
- If a field is only initialized on some paths, do NOT include it

Respond in JSON format:
```json
{{
  "function": "{name}",
  "inits": [
    {{
      "target": "expression being initialized",
      "target_kind": "parameter|field|return_value",
      "initializer": "how it is initialized",
      "byte_count": "n|sizeof(T)|full|null"
    }}
  ],
  "description": "One-sentence description of what this function always initializes"
}}
```

If the function does not unconditionally initialize any caller-visible state, return:
```json
{{
  "function": "{name}",
  "inits": [],
  "description": "Does not unconditionally initialize caller-visible state"
}}
```
"""


BLOCK_INIT_PROMPT = """You are analyzing a code block from a large C/C++ function.

## Context

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Code Block

```c
{block_source}
```

## Task

Analyze this code block for initialization operations (caller-visible only).
Also suggest a descriptive pseudo-function name and signature.

Respond in JSON:
```json
{{{{
  "suggested_name": "descriptive_name_for_this_case",
  "suggested_signature": "void descriptive_name(args)",
  "inits": [
    {{{{
      "target": "expression being initialized",
      "target_kind": "parameter|field|return_value",
      "initializer": "how it is initialized",
      "byte_count": "n|sizeof(T)|full|null"
    }}}}
  ],
  "summary": "One-sentence description of what this case block does regarding initialization"
}}}}
```

If no caller-visible initializations, return empty inits list with a summary.
"""


# --- Approach A templates (cache_mode="instructions") ---

INIT_SYSTEM_PROMPT = """\
You are analyzing C/C++ code to generate initialization summaries (post-conditions).

## Task

Generate an initialization summary for the function provided in the user message. Identify what this function
**always** initializes on ALL exit paths — only guaranteed, unconditional
initializations. This is a post-condition: only things visible to the CALLER matter.

Only include caller-visible initializations:
- **Output parameters**: memory written via pointer parameters (e.g., `*out = value`)
- **Struct fields**: fields written via a parameter (e.g., `ctx->data = ...`)
- **Return values**: the function's return value itself

Do NOT include local variables — they are not visible to the caller after return.

**Return value rule**: If every exit path returns a value (including NULL, 0, or error codes),
the return value IS unconditionally initialized. A function that returns NULL on error and a
pointer on success still always initializes its return value.

For each initialization, identify:

1. **target**: What gets initialized — the expression (e.g., "*out", "ctx->data", "return value")
2. **target_kind**: One of:
   - "parameter" — an output parameter is written via pointer dereference
   - "field" — a struct field is written via a parameter
   - "return_value" — the return value is always set (including NULL/error returns)
3. **initializer**: How it's initialized (e.g., "memset", "assignment", "calloc", "callee:func_name")
4. **byte_count**: How many bytes are initialized — "n", "sizeof(T)", "full", or null if unknown

Consider:
- Direct assignments to output parameters and struct fields
- Calls to memset, memcpy, calloc, etc. (use callee summaries)
- Only include initializations that happen on ALL exit paths
- If a field is only initialized on some paths, do NOT include it

Respond in JSON format:
```json
{{
  "function": "<function_name>",
  "inits": [
    {{
      "target": "expression being initialized",
      "target_kind": "parameter|field|return_value",
      "initializer": "how it is initialized",
      "byte_count": "n|sizeof(T)|full|null"
    }}
  ],
  "description": "One-sentence description of what this function always initializes"
}}
```

If the function does not unconditionally initialize any caller-visible state, return:
```json
{{
  "function": "<function_name>",
  "inits": [],
  "description": "Does not unconditionally initialize caller-visible state"
}}
```\
"""

INIT_USER_PROMPT = """\
## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Initialization Summaries

{callee_summaries}\
"""

# --- Approach B template (cache_mode="source") ---

INIT_TASK_PROMPT = """\
## Task

Generate an initialization summary for the function in the system message. Identify what this function
**always** initializes on ALL exit paths — only guaranteed, unconditional
initializations. This is a post-condition: only things visible to the CALLER matter.

Only include caller-visible initializations:
- **Output parameters**: memory written via pointer parameters
- **Struct fields**: fields written via a parameter
- **Return values**: the function's return value itself

Do NOT include local variables.

**Return value rule**: If every exit path returns a value (including NULL, 0, or error codes),
the return value IS unconditionally initialized.

For each initialization, identify:

1. **target**: What gets initialized
2. **target_kind**: One of: "parameter", "field", "return_value"
3. **initializer**: How it's initialized
4. **byte_count**: How many bytes — "n", "sizeof(T)", "full", or null

## Callee Initialization Summaries

{callee_summaries}

Respond in JSON format:
```json
{{{{
  "function": "{name}",
  "inits": [
    {{{{
      "target": "expression being initialized",
      "target_kind": "parameter|field|return_value",
      "initializer": "how it is initialized",
      "byte_count": "n|sizeof(T)|full|null"
    }}}}
  ],
  "description": "One-sentence description of what this function always initializes"
}}}}
```

If the function does not unconditionally initialize any caller-visible state, return:
```json
{{{{
  "function": "{name}",
  "inits": [],
  "description": "Does not unconditionally initialize caller-visible state"
}}}}
```\
"""


class InitSummarizer:
    """Generates initialization summaries for functions using LLM."""

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
        callee_summaries: dict[str, InitSummary] | None = None,
    ) -> InitSummary:
        """Generate init summary for a single function."""
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
                    print(f"  ({self._progress_current}/{self._progress_total}) Summarizing (init): {func.name}")
                else:
                    print(f"  Summarizing (init): {func.name}")

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
                print(f"  Error summarizing (init) {func.name}: {e}")

            return InitSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _summarize_large_function(
        self,
        func: Function,
        callee_summaries: dict[str, InitSummary],
        blocks: list[FunctionBlock],
    ) -> InitSummary:
        """Chunked summarization for large functions (init pass)."""
        if self.verbose:
            print(f"  Large function ({len(func.llm_source)} chars, {len(blocks)} blocks): {func.name}")

        block_summaries: dict[int, str] = {}
        all_block_inits: list[InitOp] = []

        for i, block in enumerate(blocks):
            if block.summary_json:
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    for init in data.get("inits", []):
                        all_block_inits.append(InitOp(
                            target=init.get("target", ""),
                            target_kind=init.get("target_kind", "parameter"),
                            initializer=init.get("initializer", "assignment"),
                            byte_count=init.get("byte_count"),
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            prompt = BLOCK_INIT_PROMPT.format(
                name=func.name, signature=func.signature,
                file_path=func.file_path, block_source=block.source,
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

                for init in data.get("inits", []):
                    all_block_inits.append(InitOp(
                        target=init.get("target", ""),
                        target_kind=init.get("target_kind", "parameter"),
                        initializer=init.get("initializer", "assignment"),
                        byte_count=init.get("byte_count"),
                    ))
            except Exception as e:
                if self.verbose:
                    print(f"    Error summarizing block {block.label}: {e}")
                block_summaries[block.id] = f"(error: {e})"

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
            skeleton_summary = InitSummary(
                function_name=func.name, description=f"Error summarizing skeleton: {e}",
            )

        skeleton_summary.inits = list(skeleton_summary.inits) + all_block_inits
        with self._stats_lock:
            self._stats["functions_processed"] += 1
        return skeleton_summary

    def _build_prompt_and_system(
        self, source: str, func: Function, callee_section: str,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system) based on self.cache_mode."""
        if self.cache_mode == "instructions":
            prompt = INIT_USER_PROMPT.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
                callee_summaries=callee_section,
            )
            return prompt, INIT_SYSTEM_PROMPT, True
        elif self.cache_mode == "source":
            from .prompts import FUNCTION_CONTEXT_SYSTEM
            system = FUNCTION_CONTEXT_SYSTEM.format(
                source=source,
                name=func.name,
                signature=func.signature,
                file_path=func.file_path,
            )
            prompt = INIT_TASK_PROMPT.format(
                name=func.name,
                callee_summaries=callee_section,
            )
            return prompt, system, True
        else:
            prompt = INIT_SUMMARY_PROMPT.format(
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
        callee_summaries: dict[str, InitSummary],
    ) -> str:
        """Build the callee init summaries section for the prompt."""
        if not callee_summaries:
            return "No callee initialization summaries available (leaf function or external calls only)."

        callee_attrs = self._get_callee_attributes(list(callee_summaries.keys()))

        lines = []
        for name, summary in callee_summaries.items():
            attr_suffix = f" {callee_attrs[name]}" if name in callee_attrs else ""
            if summary.inits:
                init_desc = ", ".join(
                    f"{i.initializer}({i.target})"
                    for i in summary.inits
                )
                lines.append(f"- `{name}`: Initializes {init_desc}{attr_suffix}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'Does not initialize caller-visible state'}{attr_suffix}")

        if not lines:
            return "No callee initialization summaries available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [init pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> InitSummary:
        """Parse LLM response into InitSummary."""
        from .builder.json_utils import extract_json

        data = extract_json(response)

        # Parse inits
        _VALID_KINDS = {"parameter", "field", "return_value"}
        inits = []
        for i in data.get("inits", []):
            target_kind = i.get("target_kind", "parameter")
            if target_kind not in _VALID_KINDS:
                target_kind = "parameter"
            inits.append(
                InitOp(
                    target=i.get("target", ""),
                    target_kind=target_kind,
                    initializer=i.get("initializer", "assignment"),
                    byte_count=i.get("byte_count"),
                )
            )

        return InitSummary(
            function_name=data.get("function", func_name),
            inits=inits,
            description=data.get("description", ""),
        )
