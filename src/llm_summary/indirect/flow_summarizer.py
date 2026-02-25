"""LLM-based flow summarization for address-taken functions (Pass 1)."""

import json
import re

from ..db import SummaryDB
from ..llm.base import LLMBackend
from ..models import AddressFlowSummary, FlowDestination

FLOW_SUMMARY_PROMPT = """You are analyzing where a function pointer flows in C/C++ code.

## Key Insight

A function can only access memory through two roots:
1. **Arguments** - pointers/values passed as parameters
2. **Globals** - global variables accessible from any function

When a function pointer is stored, trace the full access path back to these roots.
This determines which callers can invoke the pointer.

## Function Being Analyzed

Function `{func_name}` has its address taken at the following locations:

{address_flows_section}

## Function Source

```c
{func_source}
```

File: {file_path}
Signature: {signature}

## Task

Analyze where this function pointer flows, focusing on:
1. **Access paths**: Trace through nested structs and dereference chains
2. **Root identification**: Determine if the root is an argument or global
3. **Caller requirements**: What must a caller have access to in order to invoke this?

## Examples

**Example 1: Nested struct via argument**
```c
void register_handler(struct context *ctx, handler_fn fn) {{
    ctx->events->on_click = fn;  // fn flows here
}}
```
Analysis:
- Immediate destination: `events_t.on_click` (struct field)
- Access path: `ctx->events->on_click`
- Root: argument `ctx` (type `struct context *`)
- Callers need: access to a `struct context *` to call `ptr->events->on_click(...)`

**Example 2: Global variable**
```c
static handler_fn g_error_handler;
void set_error_handler(handler_fn fn) {{
    g_error_handler = fn;
}}
```
Analysis:
- Immediate destination: `g_error_handler` (global)
- Access path: `g_error_handler` (direct)
- Root: global `g_error_handler`
- Callers need: visibility of the global (same TU or extern)

**Example 3: Array in struct**
```c
void add_callback(struct dispatcher *d, int idx, callback_fn cb) {{
    d->callbacks[idx] = cb;
}}
```
Analysis:
- Immediate destination: `dispatcher.callbacks[]` (array in struct)
- Access path: `d->callbacks[idx]`
- Root: argument `d` (type `struct dispatcher *`)
- Callers need: access to a `struct dispatcher *`

## Response Format

Respond in JSON:
```json
{{
  "flow_destinations": [
    {{
      "type": "struct_field|global_var|parameter|array",
      "name": "type_name.field_name or global_name",
      "access_path": "full dereference chain (e.g., ctx->events->handler)",
      "root_type": "arg|global",
      "root_name": "parameter name or global variable name",
      "file_path": "source file where this flow occurs",
      "line_number": 123,
      "confidence": "high|medium|low"
    }}
  ],
  "semantic_role": "Brief description (e.g., 'error handler callback', 'comparator for sorting')",
  "likely_callers": ["functions that have access to the root and would invoke this pointer"]
}}
```

Focus on the access path and root - these determine which functions can invoke the pointer.
"""


class FlowSummarizer:
    """
    Uses LLM to summarize where address-taken function pointers flow.

    This is Pass 1 of the two-pass indirect call resolution approach.
    For each address-taken function, we make one LLM call to understand
    where its pointer flows and its semantic role.
    """

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
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def summarize_function(self, function_id: int) -> AddressFlowSummary | None:
        """
        Generate a flow summary for a single address-taken function.

        Args:
            function_id: The database ID of the function to summarize

        Returns:
            AddressFlowSummary or None if function not found
        """
        # Get function details
        func = self.db.get_function(function_id)
        if not func:
            return None

        # Get address flow information
        flows = self.db.get_address_flows(function_id)
        if not flows:
            # No flow information - still generate a basic summary
            pass

        # Build address flows section
        address_flows_section = self._build_flows_section(flows)

        # Get signature from address-taken info
        atf_list = self.db.get_address_taken_functions()
        signature = func.signature
        for atf in atf_list:
            if atf.function_id == function_id:
                signature = atf.signature
                break

        # Build prompt
        prompt = FLOW_SUMMARY_PROMPT.format(
            func_name=func.name,
            address_flows_section=address_flows_section,
            func_source=func.llm_source or "(source not available)",
            file_path=func.file_path,
            signature=signature,
        )

        # Query LLM
        try:
            if self.verbose:
                print(f"  Flow summarizing: {func.name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            # Log if requested
            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            summary = self._parse_response(response, function_id)
            summary.model_used = self.llm.model
            self._stats["functions_processed"] += 1

            # Store in database
            self.db.add_flow_summary(summary)

            return summary

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error flow summarizing {func.name}: {e}")

            # Return a minimal summary on error
            return AddressFlowSummary(
                function_id=function_id,
                semantic_role=f"Error: {e}",
            )

    def summarize_all(self, force: bool = False) -> dict[int, AddressFlowSummary]:
        """
        Generate flow summaries for all address-taken functions.

        Args:
            force: If True, re-summarize even if summary exists

        Returns:
            Mapping of function ID to flow summary
        """
        # Get all address-taken functions
        address_taken = self.db.get_address_taken_functions()

        if self.verbose:
            print(f"Pass 1: Summarizing flows for {len(address_taken)} address-taken functions")

        summaries: dict[int, AddressFlowSummary] = {}

        for atf in address_taken:
            function_id = atf.function_id

            # Check if summary exists
            if not force and self.db.has_flow_summary(function_id):
                existing = self.db.get_flow_summary(function_id)
                if existing:
                    summaries[function_id] = existing
                    self._stats["cache_hits"] += 1
                    if self.verbose:
                        func = self.db.get_function(function_id)
                        func_name = func.name if func else f"ID:{function_id}"
                        print(f"  Cached: {func_name}")
                    continue

            # Generate summary
            summary = self.summarize_function(function_id)
            if summary:
                summaries[function_id] = summary

        return summaries

    def _build_flows_section(self, flows: list) -> str:
        """Build the address flows section for the prompt."""
        if not flows:
            return "No specific flow information available."

        lines = []
        for i, flow in enumerate(flows, 1):
            context = flow.context_snippet or "(no context)"
            lines.append(
                f"{i}. {flow.file_path}:{flow.line_number}\n"
                f"   Flow target: {flow.flow_target}\n"
                f"   Context:\n   ```c\n   {context}\n   ```"
            )

        return "\n\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[FLOW SUMMARY] Function: {func_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, function_id: int) -> AddressFlowSummary:
        """Parse LLM response into AddressFlowSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return AddressFlowSummary(
                    function_id=function_id,
                    semantic_role="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return AddressFlowSummary(
                function_id=function_id,
                semantic_role=f"JSON parse error: {e}",
            )

        # Parse flow destinations
        flow_destinations = []
        for fd in data.get("flow_destinations", []):
            flow_destinations.append(
                FlowDestination(
                    dest_type=fd.get("type", "unknown"),
                    name=fd.get("name", ""),
                    confidence=fd.get("confidence", "low"),
                    access_path=fd.get("access_path", ""),
                    root_type=fd.get("root_type", ""),
                    root_name=fd.get("root_name", ""),
                    file_path=fd.get("file_path", ""),
                    line_number=fd.get("line_number", 0),
                )
            )

        # Parse likely callers
        likely_callers = data.get("likely_callers", [])
        if not isinstance(likely_callers, list):
            likely_callers = []

        return AddressFlowSummary(
            function_id=function_id,
            flow_destinations=flow_destinations,
            semantic_role=data.get("semantic_role", ""),
            likely_callers=likely_callers,
        )
