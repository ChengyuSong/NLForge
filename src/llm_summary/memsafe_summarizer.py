"""LLM-based safety contract (pre-condition) summary generator."""

import json
import re
import threading

from .db import SummaryDB
from .llm.base import LLMBackend, make_json_response_format
from .models import (
    Function,
    FunctionBlock,
    MemsafeContract,
    MemsafeSummary,
    build_skeleton,
)


def _substitute(expr: str, formals: list[str], actuals: list[str]) -> str:
    """Replace formal parameter names with actual argument texts in an expression.

    Performs whole-word replacement in declaration order so longer formals are
    processed first to avoid partial matches (e.g., 'buf' before 'buf_size').
    """
    if not formals or not actuals:
        return expr
    pairs = sorted(zip(formals, actuals, strict=False), key=lambda p: -len(p[0]))
    for formal, actual in pairs:
        if formal and actual and formal != actual:
            expr = re.sub(r"\b" + re.escape(formal) + r"\b", lambda _: actual, expr)
    return expr

MEMSAFE_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate safety pre-condition contracts.

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

{callee_note}

{alias_context}

## Task

Generate safety contracts (pre-conditions) for this function. Identify what the
**caller must guarantee** for this function to execute in a memory-safe manner.

Callee pre-conditions are annotated inline in the source as `/* PRE[callee(actual_args)]: */`
comments immediately before each call. Use these to propagate unsatisfied requirements
upward — if a callee requires a buffer size that this function does not internally guarantee,
include it in this function's own contracts. Apply formal→actual argument substitution: when the
annotation lists actual arguments (e.g., `PRE[foo(s->buf, n)]`), the contract targets named after
the callee's formals should be read with those actuals substituted in.

For each contract, identify:

1. **target**: The parameter or expression the contract applies to (e.g., "ptr", "buf", "ctx->data")
2. **contract_kind**: One of:
   - "not_null" — pointer parameter that is dereferenced and must not be NULL
   - "nullable" — pointer parameter that is explicitly checked \
for NULL before use (caller MAY pass NULL)
   - "not_freed" — pointer passed to free/dealloc that must \
point to live memory
   - "buffer_size" — pointer used with memcpy/memset/indexing \
that must have sufficient capacity
   - "initialized" — variable/field used in dereference, branch, or index that must be initialized
3. **description**: Brief description of the requirement
4. **size_expr**: (buffer_size only) The size expression required, \
e.g., "n", "sizeof(T)", "strlen(src)+1"
5. **relationship**: (buffer_size only) One of "byte_count" or "element_count"
6. **condition** (optional): A C expression under which this contract applies, \
e.g., "n > 0" for a buffer_size that only applies when n is nonzero, \
or "n == 0" for a nullable that is only safe when n is zero

Rules:
- Pointer params that are **dereferenced** (read/write through \
`*p`, `p->field`, `p[i]`) without a NULL check → `not_null`
- Pointer params that are **checked for NULL** before any \
dereference (e.g., `if (p == NULL) return`) → `nullable`
- Pointer params nullable only under a specific condition (e.g., when n==0) → \
`nullable` with `condition`
- Params passed to `free()` or deallocators → `not_freed`
- Params used in memcpy/memset/array indexing with a size → \
`buffer_size` (include size_expr + relationship); add `condition` if only applies conditionally
- Params/fields used in dereference, branch, or index before \
being set → `initialized`
- If a callee PRE annotation lists a requirement this function \
does NOT satisfy internally, propagate it
- Only include size_expr and relationship for buffer_size contracts
- Only include condition when the contract is not unconditional

Respond in JSON format:
```json
{{
  "function": "{name}",
  "contracts": [
    {{
      "target": "parameter or expression",
      "contract_kind": "not_null|nullable|not_freed|initialized|buffer_size",
      "description": "brief description of the requirement",
      "size_expr": "n (buffer_size only, omit otherwise)",
      "relationship": "byte_count (buffer_size only, omit otherwise)",
      "condition": "C expression (omit if unconditional)"
    }}
  ],
  "description": "One-sentence summary of this function's safety requirements"
}}
```

If the function has no safety pre-conditions (e.g., all pointers are checked before use), return:
```json
{{
  "function": "{name}",
  "contracts": [],
  "description": "No safety pre-conditions required"
}}
```
"""

BLOCK_MEMSAFE_PROMPT = """You are analyzing a code block from a large C/C++ function.

## Context

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Code Block

```c
{block_source}
```

## Task

Generate safety contracts (pre-conditions) for this code block. What must the
caller guarantee for memory-safe execution of this block? Also suggest a descriptive
pseudo-function name and signature.

Respond in JSON:
```json
{{{{
  "suggested_name": "descriptive_name_for_this_case",
  "suggested_signature": "void descriptive_name(args)",
  "contracts": [
    {{{{
      "target": "parameter or expression",
      "contract_kind": "not_null|nullable|not_freed|initialized|buffer_size",
      "description": "brief description",
      "size_expr": "n (buffer_size only)",
      "relationship": "byte_count (buffer_size only)",
      "condition": "C expression (omit if unconditional)"
    }}}}
  ],
  "summary": "One-sentence description of this block's safety requirements"
}}}}
```

If no safety pre-conditions, return empty contracts list with a summary.
"""

_CALLEE_NOTE_WITH_ANNOTATIONS = """\
## Callee Safety Contracts

Callee contracts are embedded as `/* PRE[...] */` comments in the source above.\
"""

_CALLEE_NOTE_FLAT = """\
## Callee Safety Contracts

{flat_list}\
"""

_MEMSAFE_CONTRACT_ITEM = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "contract_kind": {"type": "string"},
        "description": {"type": "string"},
        "size_expr": {"type": "string"},
        "relationship": {"type": "string"},
        "condition": {"type": "string"},
    },
    "required": ["target", "contract_kind", "description"],
}

MEMSAFE_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "contracts": {"type": "array", "items": _MEMSAFE_CONTRACT_ITEM},
        "description": {"type": "string"},
    },
    "required": ["function", "contracts", "description"],
})

MEMSAFE_BLOCK_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "suggested_name": {"type": "string"},
        "suggested_signature": {"type": "string"},
        "contracts": {"type": "array", "items": _MEMSAFE_CONTRACT_ITEM},
        "summary": {"type": "string"},
    },
    "required": ["suggested_name", "suggested_signature",
                  "contracts", "summary"],
})


class MemsafeSummarizer:
    """Generates safety contract summaries for functions using LLM."""

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
        callee_summaries: dict[str, MemsafeSummary] | None = None,
        callee_params: dict[str, list[str]] | None = None,
        alias_context: str | None = None,
    ) -> MemsafeSummary:
        """Generate safety contract summary for a single function.

        Args:
            func: The function to summarize (must have .callsites and .params populated).
            callee_summaries: Memsafe summaries keyed by callee name.
            callee_params: Formal parameter names keyed by callee name, used for
                formal→actual substitution in inline annotations. When omitted,
                substitution is skipped and only actual args are shown in the header.
        """
        if callee_summaries is None:
            callee_summaries = {}
        if callee_params is None:
            callee_params = {}

        # Check for large function with blocks
        blocks = self.db.get_function_blocks(func.id) if func.id else []
        if blocks and len(func.llm_source) > 40000:
            return self._summarize_large_function(
                func, callee_summaries, callee_params, blocks, alias_context
            )

        annotated_source, used_inline = self._annotate_source(func, callee_summaries, callee_params)

        if used_inline:
            callee_note = _CALLEE_NOTE_WITH_ANNOTATIONS
        else:
            flat = self._build_flat_callee_list(callee_summaries)
            callee_note = _CALLEE_NOTE_FLAT.format(flat_list=flat)

        prompt, system, cache_system = self._build_prompt_and_system(
            annotated_source, func, callee_note, alias_context, used_inline,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    cur = self._progress_current
                    tot = self._progress_total
                    print(
                        f"  ({cur}/{tot}) "
                        f"Summarizing (memsafe): {func.name}"
                    )
                else:
                    print(f"  Summarizing (memsafe): {func.name}")

            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=MEMSAFE_RESPONSE_FORMAT,
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
                print(f"  Error summarizing (memsafe) {func.name}: {e}")

            return MemsafeSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _summarize_large_function(
        self,
        func: Function,
        callee_summaries: dict[str, MemsafeSummary],
        callee_params: dict[str, list[str]],
        blocks: list[FunctionBlock],
        alias_context: str | None = None,
    ) -> MemsafeSummary:
        """Chunked summarization for large functions (memsafe pass)."""
        if self.verbose:
            n_chars = len(func.llm_source)
            n_blocks = len(blocks)
            print(
                f"  Large function ({n_chars} chars, "
                f"{n_blocks} blocks): {func.name}"
            )

        block_summaries: dict[int, str] = {}
        all_block_contracts: list[MemsafeContract] = []

        for i, block in enumerate(blocks):
            assert block.id is not None
            if block.summary_json:
                try:
                    data = json.loads(block.summary_json)
                    block_summaries[block.id] = data.get("summary", "")
                    for c in data.get("contracts", []):
                        all_block_contracts.append(MemsafeContract(
                            target=c.get("target", ""),
                            contract_kind=c.get("contract_kind", "not_null"),
                            description=c.get("description", ""),
                            size_expr=c.get("size_expr"),
                            relationship=c.get("relationship"),
                            condition=c.get("condition"),
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            prompt = BLOCK_MEMSAFE_PROMPT.format(
                name=func.name, signature=func.signature,
                file_path=func.file_path, block_source=block.source,
            )

            try:
                if self.verbose:
                    print(f"    Block {i+1}/{len(blocks)}: {block.label[:60]}")
                response = self.llm.complete(
                    prompt, response_format=MEMSAFE_BLOCK_RESPONSE_FORMAT,
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

                for c in data.get("contracts", []):
                    all_block_contracts.append(MemsafeContract(
                        target=c.get("target", ""),
                        contract_kind=c.get("contract_kind", "not_null"),
                        description=c.get("description", ""),
                        size_expr=c.get("size_expr"),
                        relationship=c.get("relationship"),
                        condition=c.get("condition"),
                    ))
            except Exception as e:
                if self.verbose:
                    print(f"    Error summarizing block {block.label}: {e}")
                block_summaries[block.id] = f"(error: {e})"

        # Phase B: Build skeleton from llm_source (preprocessed when available)
        skeleton = build_skeleton(func.llm_source, func.line_start, blocks, block_summaries)

        # Create a temporary Function-like object with skeleton source for annotation
        # We need to filter callsites to only those outside removed blocks
        block_line_ranges: set[int] = set()
        for block in blocks:
            if block.id in block_summaries:
                for line in range(block.line_start + 1, block.line_end + 1):
                    block_line_ranges.add(line - func.line_start)  # 0-based line_in_body

        skeleton_callsites = [
            cs for cs in func.callsites
            if cs["line_in_body"] not in block_line_ranges
        ]

        skeleton_func = Function(
            name=func.name, file_path=func.file_path,
            line_start=func.line_start, line_end=func.line_end,
            source=skeleton, signature=func.signature,
            params=func.params, callsites=skeleton_callsites,
        )

        annotated_source, used_inline = self._annotate_source(
            skeleton_func, callee_summaries, callee_params
        )

        if used_inline:
            callee_note = _CALLEE_NOTE_WITH_ANNOTATIONS
        else:
            flat = self._build_flat_callee_list(callee_summaries)
            callee_note = _CALLEE_NOTE_FLAT.format(flat_list=flat)

        prompt, system, cache_system = self._build_prompt_and_system(
            annotated_source, func, callee_note, alias_context, used_inline,
        )

        try:
            llm_response = self.llm.complete_with_metadata(
                prompt, system=system, cache_system=cache_system,
                response_format=MEMSAFE_RESPONSE_FORMAT,
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
            skeleton_summary = MemsafeSummary(
                function_name=func.name, description=f"Error summarizing skeleton: {e}",
            )

        # Phase C: merge — deduplicate contracts by (target, contract_kind)
        seen = {(c.target, c.contract_kind) for c in skeleton_summary.contracts}
        for c in all_block_contracts:
            key = (c.target, c.contract_kind)
            if key not in seen:
                seen.add(key)
                skeleton_summary.contracts.append(c)

        with self._stats_lock:
            self._stats["functions_processed"] += 1
        return skeleton_summary

    def _build_prompt_and_system(
        self, source: str, func: Function, callee_note: str,
        alias_context: str | None, used_inline: bool,
    ) -> tuple[str, str | None, bool]:
        """Return (prompt, system, cache_system).

        Memsafe always uses the monolithic prompt regardless of cache_mode.
        Splitting source from task instructions degrades contract quality
        (39% agreement vs baseline in A/B testing).
        """
        prompt = MEMSAFE_SUMMARY_PROMPT.format(
            source=source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            callee_note=callee_note,
            alias_context=alias_context or "",
        )
        return prompt, None, False

    def _get_callee_attributes(self, callee_names: list[str]) -> dict[str, str]:
        """Look up attributes for callee functions."""
        if not getattr(self, "db", None):
            return {}
        attrs = {}
        for name in callee_names:
            funcs = self.db.get_function_by_name(name)
            if funcs and funcs[0].attributes:
                attrs[name] = funcs[0].attributes
        return attrs

    def _annotate_source(
        self,
        func: Function,
        callee_summaries: dict[str, MemsafeSummary],
        callee_params: dict[str, list[str]],
    ) -> tuple[str, bool]:
        """Return (annotated_source, used_inline).

        Injects `/* PRE[callee(actual_args)]: ... */` comments immediately before
        each callsite whose callee has a memsafe summary with contracts. Applies
        formal→actual substitution in contract targets/size_exprs when callee_params
        is provided. Returns used_inline=True when at least one annotation was added.
        """
        if not func.callsites or not callee_summaries:
            return func.llm_source, False

        # Look up attributes for all callees referenced in callsites
        all_callee_names = {cs["callee"] for cs in func.callsites}
        callee_attrs = self._get_callee_attributes(list(all_callee_names))

        # Group callsites by line_in_body (0-based offset from function start line).
        # line_in_body offsets are relative to the *original* source, so annotation
        # must always operate on func.source, not pp_source.
        # Include callees with contracts OR attributes (e.g., noreturn).
        by_line: dict[int, list[dict]] = {}
        for cs in func.callsites:
            callee = cs["callee"]
            if callee in callee_summaries or callee in callee_attrs:
                by_line.setdefault(cs["line_in_body"], []).append(cs)

        if not by_line:
            return func.llm_source, False

        lines = func.source.splitlines()
        result: list[str] = []
        for i, line in enumerate(lines):
            for cs in by_line.get(i, []):
                callee = cs["callee"]
                summary = callee_summaries.get(callee)
                has_contracts = summary and summary.contracts
                has_attrs = callee in callee_attrs

                if not has_contracts and not has_attrs:
                    continue

                actual_args: list[str] = cs.get("args", [])
                formal_params: list[str] = callee_params.get(callee, [])
                via_macro = cs.get("via_macro", False)
                macro_name = cs.get("macro_name")

                # For macro-hidden calls the expanded arg tokens are messy; omit them.
                if via_macro:
                    header = f"{callee}  [via macro {macro_name or '?'}]"
                    actual_args = []  # skip substitution — formals don't map cleanly
                else:
                    args_str = ", ".join(actual_args)
                    header = f"{callee}({args_str})"

                indent = " " * (len(line) - len(line.lstrip()))

                # Add attribute annotation (e.g., noreturn)
                if has_attrs:
                    result.append(f"{indent}/* {callee}: {callee_attrs[callee]} */")

                # Add contract annotations
                if has_contracts and summary is not None:
                    result.append(f"{indent}/* PRE[{header}]:")
                    for c in summary.contracts:
                        target = _substitute(c.target, formal_params, actual_args)
                        if c.contract_kind == "buffer_size" and c.size_expr:
                            size = _substitute(c.size_expr, formal_params, actual_args)
                            result.append(f"{indent} *   {target}: {c.contract_kind}({size})")
                        else:
                            result.append(f"{indent} *   {target}: {c.contract_kind}")
                    result.append(f"{indent} */")

            result.append(line)

        return "\n".join(result), True

    def _build_flat_callee_list(
        self,
        callee_summaries: dict[str, MemsafeSummary],
    ) -> str:
        """Fallback: flat list of callee contracts (used when no callsite metadata)."""
        if not callee_summaries:
            return "No callee safety contracts available (leaf function or external calls only)."

        callee_attrs = self._get_callee_attributes(list(callee_summaries.keys()))

        lines = []
        for name, summary in callee_summaries.items():
            attr_suffix = f" {callee_attrs[name]}" if name in callee_attrs else ""
            if summary.contracts:
                contract_descs = []
                for c in summary.contracts:
                    if c.contract_kind == "buffer_size" and c.size_expr:
                        contract_descs.append(f"{c.target}: {c.contract_kind}({c.size_expr})")
                    else:
                        contract_descs.append(f"{c.target}: {c.contract_kind}")
                lines.append(f"- `{name}`: Requires {', '.join(contract_descs)}{attr_suffix}")
            else:
                desc = summary.description or 'No safety pre-conditions'
                lines.append(
                    f"- `{name}`: {desc}{attr_suffix}"
                )

        return "\n".join(lines) if lines else "No callee safety contracts available."

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        if not self.log_file:
            return
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [memsafe pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> MemsafeSummary:
        """Parse LLM response into MemsafeSummary."""
        from .builder.json_utils import extract_json

        data = extract_json(response)

        # Parse contracts
        valid_kinds = {"not_null", "nullable", "not_freed", "initialized", "buffer_size"}
        contracts = []
        for c in data.get("contracts", []):
            contract_kind = c.get("contract_kind", "not_null")
            if contract_kind not in valid_kinds:
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
                    condition=c.get("condition"),
                )
            )

        return MemsafeSummary(
            function_name=data.get("function", func_name),
            contracts=contracts,
            description=data.get("description", ""),
        )
