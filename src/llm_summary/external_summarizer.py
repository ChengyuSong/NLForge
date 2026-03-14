"""LLM-based summarizer for external (libc/syscall) functions.

Unlike the main summarizers which work from C source code, this class
generates all four summary types (allocation, free, init, memsafe) from the
function name alone, relying on the LLM's training knowledge of standard
library semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm.base import make_json_response_format

EXTERNAL_SUMMARY_PROMPT = """\
You are an expert in C memory safety analysis. Given the name of a function
from libc or a common POSIX/system library, provide memory-safety summaries
based on the function's well-known documented behaviour.

Function name: {name}

Return a single JSON object with exactly these four keys:

1. "allocation": AllocationSummary or null
   null  → function does not heap-allocate anything visible to the caller.
   When non-null:
   {{
     "function": "<name>",
     "allocations": [
       {{
         "type": "heap",
         "source": "<allocating primitive, e.g. malloc>",
         "size_expr": "<expression for byte count, e.g. size or nmemb*size>",
         "size_params": ["<param names that influence size>"],
         "returned": <true if the pointer is returned>,
         "stored_to": <null or "param_name" if stored into a parameter>,
         "may_be_null": <true if allocation can fail and return NULL>
       }}
     ],
     "parameters": {{
       "<param>": {{"role": "<role description>", "used_in_allocation": <bool>}}
     }},
     "description": "<one sentence>"
   }}

2. "free": FreeSummary or null
   null  → function does not release heap memory.
   When non-null:
   {{
     "function": "<name>",
     "frees": [
       {{
         "target": "<param name or '*param'>",
         "target_kind": "parameter",
         "deallocator": "free",
         "conditional": <true if freeing is conditional>,
         "nulled_after": <true if pointer is set to NULL after freeing>
       }}
     ],
     "description": "<one sentence>"
   }}

3. "init": InitSummary or null
   null  → function makes no guaranteed initialisation of caller-visible
   memory (i.e. output buffers, return values, or struct fields written on
   every execution path).
   When non-null:
   {{
     "function": "<name>",
     "inits": [
       {{
         "target": "<param name or '*param' or 'return'>",
         "target_kind": "parameter|return_value",
         "initializer": "<name of the primitive that writes the bytes>",
         "byte_count": "<expression for bytes written, e.g. n or strlen(src)+1>"
       }}
     ],
     "description": "<one sentence>"
   }}

4. "memsafe": MemsafeSummary  (always required; use empty contracts list when there are none)
   {{
     "function": "<name>",
     "contracts": [
       {{
         "target": "<param name>",
         "contract_kind": "not_null|not_freed|buffer_size|initialized|non_negative",
         "description": "<human-readable precondition>",
         "size_expr": "<expression>",        (only for buffer_size)
         "relationship": "byte_count|element_count"  (only for buffer_size)
       }}
     ],
     "description": "<one sentence summarising overall preconditions>"
   }}

Rules:
- Use null (JSON null, not the string "null") for unused summary fields.
- Only describe behaviour that is guaranteed on every call path (no "may").
- For allocation: include only heap allocations returned to or stored in the
  caller (do not include internal temporary allocations).
- For init: only include buffers/fields that are ALWAYS written (e.g. memset,
  strcpy, snprintf destinations), not ones written only on success paths.
- Return ONLY the JSON object, with no markdown fences or extra commentary.
"""


@dataclass
class ExternalSummaryResult:
    """Raw JSON strings for each summary type; None means not applicable."""
    allocation_json: str | None
    free_json: str | None
    init_json: str | None
    memsafe_json: str | None


_EXTERNAL_ALLOC_SCHEMA = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "allocations": {"type": "array", "items": {"type": "object"}},
        "parameters": {"type": "object"},
        "description": {"type": "string"},
    },
    "required": ["function", "allocations", "parameters", "description"],
}

_EXTERNAL_FREE_SCHEMA = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "frees": {"type": "array", "items": {"type": "object"}},
        "description": {"type": "string"},
    },
    "required": ["function", "frees", "description"],
}

_EXTERNAL_INIT_SCHEMA = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "inits": {"type": "array", "items": {"type": "object"}},
        "description": {"type": "string"},
    },
    "required": ["function", "inits", "description"],
}

_EXTERNAL_MEMSAFE_SCHEMA = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "contracts": {"type": "array", "items": {"type": "object"}},
        "description": {"type": "string"},
    },
    "required": ["function", "contracts", "description"],
}

EXTERNAL_RESPONSE_FORMAT = make_json_response_format({
    "type": "object",
    "properties": {
        "allocation": _EXTERNAL_ALLOC_SCHEMA,
        "free": _EXTERNAL_FREE_SCHEMA,
        "init": _EXTERNAL_INIT_SCHEMA,
        "memsafe": _EXTERNAL_MEMSAFE_SCHEMA,
    },
    "required": ["memsafe"],
})


class ExternalFunctionSummarizer:
    """Generates stdlib/external function summaries from function name only."""

    def __init__(self, llm, verbose: bool = False, log_file: str | None = None) -> None:
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self._stats: dict[str, int] = {
            "llm_calls": 0,
            "errors": 0,
            "functions_processed": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def generate(self, name: str) -> ExternalSummaryResult:
        """Generate all four summary types for the named external function."""
        prompt = EXTERNAL_SUMMARY_PROMPT.format(name=name)
        try:
            response = self.llm.complete(
                prompt, response_format=EXTERNAL_RESPONSE_FORMAT,
            )
            self._stats["llm_calls"] += 1
            if self.log_file:
                self._log(name, prompt, response)
            result = self._parse(name, response)
            self._stats["functions_processed"] += 1
            return result
        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  [external_summarizer] error for {name}: {e}")
            # Return a safe empty result so the caller can still cache something
            empty_memsafe = json.dumps({"function": name, "contracts": [], "description": ""})
            return ExternalSummaryResult(
                allocation_json=None,
                free_json=None,
                init_json=None,
                memsafe_json=empty_memsafe,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, name: str, response: str) -> ExternalSummaryResult:
        data = _extract_json(response)

        alloc = data.get("allocation")
        free = data.get("free")
        init = data.get("init")
        memsafe = data.get("memsafe")

        # Ensure function name is set correctly in each summary
        for d in [alloc, free, init, memsafe]:
            if isinstance(d, dict):
                d["function"] = name

        # memsafe is always required; supply empty default if missing/null
        if not isinstance(memsafe, dict):
            memsafe = {"function": name, "contracts": [], "description": ""}

        return ExternalSummaryResult(
            allocation_json=json.dumps(alloc) if isinstance(alloc, dict) else None,
            free_json=json.dumps(free) if isinstance(free, dict) else None,
            init_json=json.dumps(init) if isinstance(init, dict) else None,
            memsafe_json=json.dumps(memsafe),
        )

    def _log(self, name: str, prompt: str, response: str) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*60}\n")
            fh.write(f"EXTERNAL SUMMARIZER: {name}\n")
            fh.write(f"{'='*60}\nPROMPT:\n{prompt}\n")
            fh.write(f"RESPONSE:\n{response}\n")


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response string."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try to find a top-level { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            result: dict = json.loads(match.group(0))
            return result
        except json.JSONDecodeError:
            pass

    # Fall back: try parsing the whole cleaned string
    try:
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        return {}
