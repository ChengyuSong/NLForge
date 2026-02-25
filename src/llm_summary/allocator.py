"""Allocator function detection via heuristic pre-filter + LLM confirmation."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import Function

# Keywords suggesting allocator functions
ALLOC_KEYWORDS = {
    "alloc", "malloc", "calloc", "realloc", "zalloc", "memalign",
    "strdup", "strndup",
}

# Well-known stdlib allocator names (including C++ mangled)
STDLIB_ALLOCATORS = {
    "malloc", "calloc", "realloc", "aligned_alloc",
    "_Znwm", "_Znam",  # C++ operator new / new[]
}

# Known allocator functions whose calls indicate allocator wrapping
KNOWN_ALLOC_CALLS = {
    "malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign",
    "mmap", "new",
}

# Size-related parameter names
SIZE_PARAM_NAMES = {"size", "len", "count", "n", "num", "bytes"}

# Pre-compiled regexes
_RE_PTR_RETURN = re.compile(r'\*\s*$')
_RE_RETURN_TYPE = re.compile(r'^(.*?)\b\w+\s*\(')
_RE_SIZE_T_PARAM = re.compile(r'\bsize_t\b')
_RE_PAREN_CONTENT = re.compile(r'\(([^)]*)\)')
_RE_RETURN_ALLOC = re.compile(
    r'return\s+(?:\([^)]*\)\s*)?(?:' +
    '|'.join(re.escape(f) for f in KNOWN_ALLOC_CALLS) +
    r')\s*\('
)
_RE_CALL_ALLOC = re.compile(
    r'\b(?:' + '|'.join(re.escape(f) for f in KNOWN_ALLOC_CALLS) + r')\s*\('
)
_RE_NULL_CHECK = re.compile(
    r'(?:==\s*NULL|!=\s*NULL|==\s*0\b|!=\s*0\b|!\s*\w+\s*\)|if\s*\(\s*!\s*\w+\s*\))'
)

ALLOCATOR_PROMPT = """You are analyzing a C/C++ function to determine if it is a memory allocator \
that returns newly allocated memory.

## Function Source

```c
{source_with_lines}
```

Function: `{name}`
Project: `{project_name}`
File: `{file_path}`
Signature: `{signature}`

## Detected Evidence

The following patterns were detected in this function:
{evidence_section}

## What IS an allocator function

- Wrapper around malloc/calloc/realloc/mmap that returns the allocated pointer
- Project-specific allocation routines (e.g., png_malloc, g_malloc, apr_palloc)
- Pool allocators that return memory from a pre-allocated pool
- Any function whose primary purpose is to allocate and return new memory

## What is NOT an allocator function

- Functions that allocate memory internally but don't return it
- Constructors that allocate AND initialize a complex object (e.g., create_context)
- Functions that just call realloc to resize an existing buffer in-place
- Free/dealloc functions
- Functions that return pointers to existing (non-newly-allocated) memory

## Task

Based on the source code and evidence above, is this function a memory allocator \
that returns newly allocated memory?

Respond in JSON:
```json
{{"is_allocator": true, "reasoning": "brief explanation"}}
```

If this is NOT an allocator function, return:
```json
{{"is_allocator": false, "reasoning": "brief explanation why not"}}
```
"""


class AllocatorDetector:
    """
    Detects allocator functions using heuristic pre-filter + LLM confirmation.

    Phase 1: Score each function based on source code patterns (no LLM).
    Phase 2: For candidates above threshold, use LLM to confirm.
    """

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend | None = None,
        verbose: bool = False,
        log_file: str | None = None,
        min_score: int = 5,
        project_name: str = "",
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.min_score = min_score
        self.project_name = project_name
        self._stats = {
            "functions_scanned": 0,
            "candidates": 0,
            "llm_calls": 0,
            "confirmed": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def detect_all(
        self, include_stdlib: bool = False
    ) -> tuple[list[str], list[str]]:
        """
        Run full detection: heuristic pre-filter + LLM confirmation.

        Returns:
            (candidates, confirmed) — unconfirmed candidate names and
            LLM-confirmed allocator names.
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        if self.verbose:
            print(f"Scanning {len(functions)} functions for allocator patterns...")

        # Phase 1: Heuristic pre-filter
        scored = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self.heuristic_score(func)
            if score >= self.min_score:
                scored.append((func, score, signals))

        self._stats["candidates"] = len(scored)

        if self.verbose:
            print(f"Found {len(scored)} candidates (score >= {self.min_score})")

        # Phase 2: LLM confirmation
        candidates: list[str] = []
        confirmed: list[str] = []

        for func, score, signals in scored:
            if self.llm is None:
                # No LLM — everything goes to candidates
                candidates.append(func.name)
                continue

            is_alloc = self._confirm_with_llm(func, score, signals)
            if is_alloc:
                confirmed.append(func.name)
                self._stats["confirmed"] += 1
            else:
                candidates.append(func.name)

        if include_stdlib:
            for name in sorted(STDLIB_ALLOCATORS):
                if name not in confirmed:
                    confirmed.append(name)

        return candidates, confirmed

    def heuristic_only(self) -> list[tuple[Function, int, list[str]]]:
        """
        Run heuristic scoring only (no LLM calls).

        Returns:
            List of (function, score, signals) tuples for candidates above threshold.
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        results = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self.heuristic_score(func)
            if score >= self.min_score:
                results.append((func, score, signals))

        self._stats["candidates"] = len(results)
        return results

    def heuristic_score(self, func: Function) -> tuple[int, list[str]]:
        """
        Score a function based on allocator-related patterns.

        Returns:
            (score, evidence_signals) where each signal is a descriptive string.
        """
        score = 0
        signals: list[str] = []
        name_lower = func.name.lower()
        source = func.source or ""
        sig = func.canonical_signature or func.signature or ""

        # --- Name keyword (+3) ---
        for kw in ALLOC_KEYWORDS:
            if kw in name_lower:
                score += 3
                signals.append(f"name_keyword: name contains '{kw}'")
                break

        # --- Pointer return (+3) ---
        if self._has_pointer_return(sig):
            score += 3
            signals.append("pointer_return: signature returns a pointer type")

        # --- Size parameter (+2) ---
        param_names = self._extract_param_names(sig)
        has_size_param = any(
            p.lower() in SIZE_PARAM_NAMES for p in param_names if p
        )
        has_size_t = bool(_RE_SIZE_T_PARAM.search(sig))
        if has_size_param or has_size_t:
            score += 2
            detail = []
            if has_size_param:
                matched = [p for p in param_names if p and p.lower() in SIZE_PARAM_NAMES]
                detail.append(f"param named '{matched[0]}'")
            if has_size_t:
                detail.append("size_t type")
            signals.append(f"size_param: {', '.join(detail)}")

        # --- Body-based signals (need source) ---
        if not source:
            return score, signals

        # Calls known allocator (+2)
        if _RE_CALL_ALLOC.search(source):
            score += 2
            # Find which one
            for alloc_fn in KNOWN_ALLOC_CALLS:
                if re.search(rf'\b{re.escape(alloc_fn)}\s*\(', source):
                    signals.append(f"calls_allocator: body calls {alloc_fn}()")
                    break

        # Returns allocator result (+2)
        if _RE_RETURN_ALLOC.search(source):
            score += 2
            signals.append("returns_alloc_result: returns result of allocator call")

        # NULL check pattern (+1)
        # Only count if there's also an allocator call
        if _RE_CALL_ALLOC.search(source) and _RE_NULL_CHECK.search(source):
            score += 1
            signals.append("null_check: checks allocator return for NULL")

        return score, signals

    def _has_pointer_return(self, sig: str) -> bool:
        """Check if a function signature returns a pointer type."""
        # Extract return type (everything before the function name and parens)
        m = _RE_RETURN_TYPE.match(sig)
        if m:
            ret_type = m.group(1).strip()
            return '*' in ret_type
        # Fallback: check if there's a * before the opening paren
        paren_idx = sig.find('(')
        if paren_idx > 0:
            before_paren = sig[:paren_idx]
            return '*' in before_paren
        return False

    def _extract_param_names(self, signature: str) -> list[str]:
        """Extract parameter names from a C function signature."""
        m = _RE_PAREN_CONTENT.search(signature)
        if not m:
            return []

        params_str = m.group(1).strip()
        if not params_str or params_str == 'void':
            return []

        names = []
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            # Remove array brackets
            param = re.sub(r'\[.*?\]', '', param)
            # Get the last word (parameter name), ignoring pointer stars
            tokens = param.replace('*', ' ').split()
            if tokens:
                name = tokens[-1]
                if name not in ('void', 'int', 'char', 'long', 'short', 'unsigned',
                                'signed', 'float', 'double', 'const', 'struct',
                                'enum', 'union', 'size_t', 'uint8_t', 'uint16_t',
                                'uint32_t', 'uint64_t', 'int8_t', 'int16_t',
                                'int32_t', 'int64_t', 'bool', '...'):
                    names.append(name)
                else:
                    names.append('')
            else:
                names.append('')

        return names

    def _build_prompt(self, func: Function, signals: list[str]) -> str:
        """Build the evidence-grounded LLM prompt for a function."""
        lines = (func.llm_source or "").split('\n')
        numbered = []
        for i, line in enumerate(lines, func.line_start or 1):
            numbered.append(f"{i:4d} | {line}")
        source_with_lines = '\n'.join(numbered)

        evidence_lines = [f"- {s}" for s in signals]
        evidence_section = '\n'.join(evidence_lines) if evidence_lines else "(no specific evidence)"

        return ALLOCATOR_PROMPT.format(
            source_with_lines=source_with_lines,
            name=func.name,
            project_name=self.project_name or "(unknown)",
            file_path=func.file_path or "(unknown)",
            signature=func.signature or "(unknown)",
            evidence_section=evidence_section,
        )

    def _confirm_with_llm(
        self, func: Function, score: int, signals: list[str]
    ) -> bool:
        """Use the LLM to confirm whether a candidate is an allocator.

        Returns True if confirmed as allocator."""
        prompt = self._build_prompt(func, signals)

        try:
            if self.verbose:
                print(f"  Analyzing: {func.name} (score={score})")

            llm_response = self.llm.complete_with_metadata(prompt)
            self._stats["llm_calls"] += 1
            self._stats["input_tokens"] += llm_response.input_tokens
            self._stats["output_tokens"] += llm_response.output_tokens

            if self.log_file:
                self._log_interaction(func.name, prompt, llm_response.content)

            return self._parse_response(llm_response.content, func.name)

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error analyzing {func.name}: {e}")
            return False

    def _parse_response(self, response: str, func_name: str) -> bool:
        """Parse LLM response. Returns True if is_allocator=true."""
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                if self.verbose:
                    print(f"  Failed to extract JSON from response for {func_name}")
                return False

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  JSON parse error for {func_name}: {e}")
            return False

        is_alloc = data.get("is_allocator", False)
        if self.verbose:
            reasoning = data.get("reasoning", "no reason given")
            status = "Allocator" if is_alloc else "Not allocator"
            print(f"  {status}: {func_name} - {reasoning}")

        return bool(is_alloc)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[ALLOCATOR DETECTION] Function: {func_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model if self.llm else 'N/A'}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")
