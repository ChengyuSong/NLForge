"""LLM-based indirect call resolution."""

from ..db import SummaryDB
from ..llm.base import LLMBackend
from ..models import AddressFlowSummary, IndirectCallsite, IndirectCallTarget

INDIRECT_CALL_PROMPT = """\
You are analyzing indirect function calls in C/C++ code \
to determine which functions could be called.

## Indirect Call Site

Caller function: `{caller_name}`
Expression: `{callee_expr}`
Expected signature: `{signature}`
Location: {file_path}:{line_number}

Context:
```c
{context_snippet}
```

## Candidate Functions ({num_candidates} candidates with compatible signature)

{candidates_section}

## Task

Match the indirect call to the most likely target function(s).

Consider:
1. Does the flow summary indicate this function reaches the callsite context?
2. Do naming conventions suggest a match (e.g., on_click -> click handler)?
3. Is the struct field or variable name consistent with the callee expression?
4. What is the semantic role of each candidate - does it fit this call context?

Respond in JSON format:
```json
{{
  "targets": [
    {{
      "function": "function_name",
      "confidence": "high|medium|low",
      "reasoning": "Brief explanation"
    }}
  ]
}}
```

Only include functions that could realistically be called. \
If no candidates seem likely, return an empty targets array.
"""


class IndirectCallResolver:
    """
    Uses LLM to resolve indirect call targets (Pass 2).

    Given an indirect call site and candidate functions (address-taken with
    matching signatures), asks the LLM to determine likely targets.

    This version uses flow summaries from Pass 1 to provide better context.
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
            "callsites_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def resolve_callsite(self, callsite: IndirectCallsite) -> list[IndirectCallTarget]:
        """Resolve a single indirect call site."""
        # Get caller function name
        caller = self.db.get_function(callsite.caller_function_id)
        caller_name = caller.name if caller else "unknown"

        # Get candidate functions - use signature-only filtering (no limit)
        candidates = self._get_signature_compatible_candidates(callsite.signature)

        if not candidates:
            return []

        # Build candidate section with flow summaries
        candidates_section = self._build_candidates_section_with_flows(candidates)

        # Build prompt
        prompt = INDIRECT_CALL_PROMPT.format(
            caller_name=caller_name,
            context_snippet=callsite.context_snippet or "",
            callee_expr=callsite.callee_expr,
            signature=callsite.signature,
            file_path=callsite.file_path,
            line_number=callsite.line_number,
            num_candidates=len(candidates),
            candidates_section=candidates_section,
        )

        # Query LLM
        try:
            if self.verbose:
                print(f"  Resolving: {callsite.callee_expr} in {caller_name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            # Log if requested
            if self.log_file:
                self._log_interaction(callsite, prompt, response)

            targets = self._parse_response(response, callsite, candidates)
            self._stats["callsites_processed"] += 1

            # Store resolved targets
            for target in targets:
                self.db.add_indirect_call_target(target)

            return targets

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error resolving callsite: {e}")
            return []

    def _get_signature_compatible_candidates(self, signature: str) -> list:
        """
        Get all address-taken functions with compatible signatures.

        Currently uses exact signature match. Future enhancement could
        use more flexible signature compatibility checking.
        """
        # First try exact signature match
        candidates = self.db.get_address_taken_functions(signature)

        if not candidates:
            # Fall back to all address-taken functions (let LLM filter)
            candidates = self.db.get_address_taken_functions()

        return candidates

    def _log_interaction(
        self, callsite: IndirectCallsite, prompt: str, response: str
    ) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[RESOLUTION] Callsite: {callsite.callee_expr}\n")
            f.write(f"Location: {callsite.file_path}:{callsite.line_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def resolve_all_callsites(self, force: bool = False) -> dict[int, list[IndirectCallTarget]]:
        """
        Resolve all indirect call sites.

        Args:
            force: If True, re-resolve even if already resolved

        Returns:
            Mapping of callsite ID to list of resolved targets
        """
        callsites = self.db.get_indirect_callsites()

        if self.verbose:
            print(f"Pass 2: Resolving {len(callsites)} indirect call sites")

        results = {}

        for callsite in callsites:
            if callsite.id is None:
                continue

            # Check if already resolved
            if not force:
                existing = self.db.get_indirect_call_targets(callsite.id)
                if existing:
                    results[callsite.id] = existing
                    self._stats["cache_hits"] += 1
                    if self.verbose:
                        print(f"  Cached: {callsite.callee_expr}")
                    continue

            targets = self.resolve_callsite(callsite)
            results[callsite.id] = targets

        return results

    def _build_candidates_section_with_flows(self, candidates: list) -> str:
        """Build the candidates section as structured JSON with flow summaries."""
        import json

        candidates_data = []
        for atf in candidates:
            func = self.db.get_function(atf.function_id)
            if not func:
                continue

            # Get first 10 lines of source
            source_lines = func.llm_source.split("\n")[:10] if func.source else []
            source_snippet = "\n".join(source_lines)

            # Get flow summary from Pass 1
            flow_summary = self.db.get_flow_summary(atf.function_id)

            candidate_info = {
                "name": func.name,
                "signature": atf.signature,
                "source_location": {
                    "file": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                },
                "source_snippet": source_snippet,
                "flow_summary": self._format_flow_summary_json(flow_summary),
            }
            candidates_data.append(candidate_info)

        if not candidates_data:
            return "No candidates found."

        return "```json\n" + json.dumps(candidates_data, indent=2) + "\n```"

    def _format_flow_summary_json(self, summary: AddressFlowSummary | None) -> dict:
        """Format a flow summary as a JSON-serializable dict."""
        if not summary:
            return {"available": False}

        result = {"available": True}

        if summary.semantic_role:
            result["semantic_role"] = summary.semantic_role

        if summary.flow_destinations:
            result["flow_destinations"] = [
                {
                    "type": fd.dest_type,
                    "name": fd.name,
                    "access_path": fd.access_path,
                    "root_type": fd.root_type,
                    "root_name": fd.root_name,
                    "file": fd.file_path,
                    "line": fd.line_number,
                    "confidence": fd.confidence,
                }
                for fd in summary.flow_destinations
            ]

        if summary.likely_callers:
            result["likely_callers"] = summary.likely_callers[:5]  # Limit to 5

        return result

    def _format_flow_summary(self, summary: AddressFlowSummary | None) -> str:
        """Format a flow summary for inclusion in the prompt (legacy text format)."""
        if not summary:
            return "Flow summary: Not available"

        parts = []

        if summary.semantic_role:
            parts.append(f"**Semantic role**: {summary.semantic_role}")

        if summary.flow_destinations:
            dests = []
            for fd in summary.flow_destinations:
                dest_info = f"{fd.dest_type}: {fd.name}"
                if fd.access_path:
                    dest_info += f" (path: {fd.access_path})"
                if fd.root_type and fd.root_name:
                    dest_info += f" [root: {fd.root_type} '{fd.root_name}']"
                dest_info += f" ({fd.confidence} confidence)"
                dests.append(f"  - {dest_info}")
            parts.append("**Flow destinations**:\n" + "\n".join(dests))

        if summary.likely_callers:
            callers = ", ".join(summary.likely_callers[:5])  # Limit to 5
            parts.append(f"**Likely callers**: {callers}")

        if parts:
            return "Flow summary:\n" + "\n".join(parts)
        else:
            return "Flow summary: No detailed information"

    def _build_candidates_section(self, candidates: list) -> str:
        """Build the candidates section for the prompt (legacy method)."""
        return self._build_candidates_section_with_flows(candidates)

    def _parse_response(
        self,
        response: str,
        callsite: IndirectCallsite,
        candidates: list,
    ) -> list[IndirectCallTarget]:
        """Parse LLM response into IndirectCallTarget objects."""
        import json
        import re

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
                return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        targets = []
        target_list = data.get("targets", [])

        # Build function name to ID mapping
        name_to_id = {}
        for atf in candidates:
            func = self.db.get_function(atf.function_id)
            if func:
                name_to_id[func.name] = atf.function_id

        for item in target_list:
            func_name = item.get("function", "")
            confidence = item.get("confidence", "low")
            reasoning = item.get("reasoning", "")

            if func_name in name_to_id and callsite.id is not None:
                targets.append(
                    IndirectCallTarget(
                        callsite_id=callsite.id,
                        target_function_id=name_to_id[func_name],
                        confidence=confidence,
                        llm_reasoning=reasoning,
                    )
                )

        return targets
