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

# Keywords suggesting deallocator functions
DEALLOC_KEYWORDS = {
    "free", "dealloc", "destroy", "release", "dispose", "delete",
}

# Well-known stdlib deallocator names (including C++ mangled)
STDLIB_DEALLOCATORS = {
    "free", "munmap",
    "_ZdlPv", "_ZdaPv",  # C++ operator delete / delete[]
}

# Known free functions whose calls indicate deallocator wrapping
KNOWN_FREE_CALLS = {
    "free", "munmap", "_ZdlPv", "_ZdaPv",
}

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
_RE_CALL_FREE = re.compile(
    r'\b(?:' + '|'.join(re.escape(f) for f in KNOWN_FREE_CALLS) + r')\s*\('
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


DEALLOCATOR_PROMPT = """You are analyzing a C/C++ function to determine if it is a memory deallocator \
that frees or releases memory.

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

## What IS a deallocator function

- Wrapper around free/munmap that deallocates memory
- Project-specific deallocation routines (e.g., png_free, g_free, apr_pfree)
- Pool deallocators that return memory to a pool
- Any function whose primary purpose is to free or release allocated memory

## What is NOT a deallocator function

- Functions that free memory internally as a side effect
- Destructors that do complex cleanup beyond just freeing
- Functions that release locks or other non-memory resources
- Allocator functions

## Task

Based on the source code and evidence above, is this function a memory deallocator \
that frees or releases memory?

Respond in JSON:
```json
{{"is_deallocator": true, "reasoning": "brief explanation"}}
```

If this is NOT a deallocator function, return:
```json
{{"is_deallocator": false, "reasoning": "brief explanation why not"}}
```
"""


def vsnapshot_confirm_allocators(snap, candidates: list[str]) -> tuple[list[str], list[str]]:
    """Confirm allocator candidates via vsnapshot alias analysis.

    A candidate is confirmed if ret:candidate may-alias ret:k for any known
    allocator k.  Confirmed names are added to the known set so transitive
    chains are recognised (fixed-point).

    Returns (confirmed, remaining).
    """
    known = set(STDLIB_ALLOCATORS)
    remaining = list(candidates)
    confirmed: list[str] = []
    changed = True
    while changed:
        changed = False
        still_remaining: list[str] = []
        for cand in remaining:
            cand_name = f"ret:{cand}"
            found = False
            for k in known:
                if snap.may_alias_name(cand_name, f"ret:{k}"):
                    found = True
                    break
            if found:
                confirmed.append(cand)
                known.add(cand)
                changed = True
            else:
                still_remaining.append(cand)
        remaining = still_remaining
    return confirmed, remaining


def vsnapshot_confirm_deallocators(snap, candidates: list[str]) -> tuple[list[str], list[str]]:
    """Confirm deallocator candidates via vsnapshot alias analysis.

    A candidate is confirmed if candidate::arg#0 may-alias k::arg#0 for any
    known deallocator k.  Fixed-point like the allocator version.

    Returns (confirmed, remaining).
    """
    known = set(STDLIB_DEALLOCATORS)
    remaining = list(candidates)
    confirmed: list[str] = []
    changed = True
    while changed:
        changed = False
        still_remaining: list[str] = []
        for cand in remaining:
            cand_name = f"{cand}::arg#0"
            found = False
            for k in known:
                if snap.may_alias_name(cand_name, f"{k}::arg#0"):
                    found = True
                    break
            if found:
                confirmed.append(cand)
                known.add(cand)
                changed = True
            else:
                still_remaining.append(cand)
        remaining = still_remaining
    return confirmed, remaining


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
            "dealloc_candidates": 0,
            "dealloc_confirmed": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def detect_all(
        self, include_stdlib: bool = False
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Run full detection: heuristic pre-filter + LLM confirmation.

        Returns:
            (alloc_candidates, alloc_confirmed, dealloc_candidates, dealloc_confirmed)
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        if self.verbose:
            print(f"Scanning {len(functions)} functions for allocator/deallocator patterns...")

        # Phase 1: Heuristic pre-filter
        alloc_scored = []
        dealloc_scored = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self.heuristic_score(func)
            if score >= self.min_score:
                alloc_scored.append((func, score, signals))
            dscore, dsignals = self.heuristic_score_dealloc(func)
            if dscore >= self.min_score:
                dealloc_scored.append((func, dscore, dsignals))

        self._stats["candidates"] = len(alloc_scored)
        self._stats["dealloc_candidates"] = len(dealloc_scored)

        if self.verbose:
            print(f"Found {len(alloc_scored)} alloc candidates, "
                  f"{len(dealloc_scored)} dealloc candidates (score >= {self.min_score})")

        # Phase 2: LLM confirmation — allocators
        alloc_candidates: list[str] = []
        alloc_confirmed: list[str] = []

        for func, score, signals in alloc_scored:
            if self.llm is None:
                alloc_candidates.append(func.name)
                continue

            if self._confirm_with_llm(func, score, signals):
                alloc_confirmed.append(func.name)
                self._stats["confirmed"] += 1
            else:
                alloc_candidates.append(func.name)

        # Phase 2: LLM confirmation — deallocators
        dealloc_candidates: list[str] = []
        dealloc_confirmed: list[str] = []

        for func, score, signals in dealloc_scored:
            if self.llm is None:
                dealloc_candidates.append(func.name)
                continue

            if self._confirm_dealloc_with_llm(func, score, signals):
                dealloc_confirmed.append(func.name)
                self._stats["dealloc_confirmed"] += 1
            else:
                dealloc_candidates.append(func.name)

        if include_stdlib:
            for name in sorted(STDLIB_ALLOCATORS):
                if name not in alloc_confirmed:
                    alloc_confirmed.append(name)
            for name in sorted(STDLIB_DEALLOCATORS):
                if name not in dealloc_confirmed:
                    dealloc_confirmed.append(name)

        return alloc_candidates, alloc_confirmed, dealloc_candidates, dealloc_confirmed

    def heuristic_only(
        self,
    ) -> tuple[list[tuple[Function, int, list[str]]], list[tuple[Function, int, list[str]]]]:
        """
        Run heuristic scoring only (no LLM calls).

        Returns:
            (alloc_scored, dealloc_scored) — each a list of
            (function, score, signals) tuples for candidates above threshold.
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        alloc_results = []
        dealloc_results = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self.heuristic_score(func)
            if score >= self.min_score:
                alloc_results.append((func, score, signals))
            dscore, dsignals = self.heuristic_score_dealloc(func)
            if dscore >= self.min_score:
                dealloc_results.append((func, dscore, dsignals))

        self._stats["candidates"] = len(alloc_results)
        self._stats["dealloc_candidates"] = len(dealloc_results)
        return alloc_results, dealloc_results

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

    def heuristic_score_dealloc(self, func: Function) -> tuple[int, list[str]]:
        """
        Score a function based on deallocator-related patterns.

        Returns:
            (score, evidence_signals) where each signal is a descriptive string.
        """
        score = 0
        signals: list[str] = []
        name_lower = func.name.lower()
        source = func.source or ""
        sig = func.canonical_signature or func.signature or ""

        # --- Name keyword (+3) ---
        for kw in DEALLOC_KEYWORDS:
            if kw in name_lower:
                score += 3
                signals.append(f"name_keyword: name contains '{kw}'")
                break

        # --- Void return type (+2) ---
        if self._has_void_return(sig):
            score += 2
            signals.append("void_return: function returns void")

        # --- Pointer parameter (+2) ---
        if self._has_pointer_param(sig):
            score += 2
            signals.append("pointer_param: has pointer parameter")

        # --- Body-based signals (need source) ---
        if not source:
            return score, signals

        # Calls known free function (+2)
        if _RE_CALL_FREE.search(source):
            score += 2
            for free_fn in KNOWN_FREE_CALLS:
                if re.search(rf'\b{re.escape(free_fn)}\s*\(', source):
                    signals.append(f"calls_free: body calls {free_fn}()")
                    break

        # Frees a parameter directly (+2)
        param_names = self._extract_param_names(sig)
        for pname in param_names:
            if not pname:
                continue
            for free_fn in KNOWN_FREE_CALLS:
                if re.search(rf'\b{re.escape(free_fn)}\s*\(\s*{re.escape(pname)}\b', source):
                    score += 2
                    signals.append(f"frees_param: {free_fn}({pname})")
                    break
            else:
                continue
            break

        return score, signals

    def _has_void_return(self, sig: str) -> bool:
        """Check if a function signature returns void."""
        m = _RE_RETURN_TYPE.match(sig)
        if m:
            ret_type = m.group(1).strip()
            return ret_type == 'void'
        paren_idx = sig.find('(')
        if paren_idx > 0:
            before_paren = sig[:paren_idx].strip()
            tokens = before_paren.replace('*', ' ').split()
            return len(tokens) >= 2 and tokens[0] == 'void'
        return False

    def _has_pointer_param(self, sig: str) -> bool:
        """Check if any parameter is a pointer type."""
        m = _RE_PAREN_CONTENT.search(sig)
        if not m:
            return False
        params_str = m.group(1).strip()
        if not params_str or params_str == 'void':
            return False
        return '*' in params_str

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

    def _build_prompt(
        self, func: Function, signals: list[str], template: str = ALLOCATOR_PROMPT
    ) -> str:
        """Build the evidence-grounded LLM prompt for a function."""
        lines = (func.llm_source or "").split('\n')
        numbered = []
        for i, line in enumerate(lines, func.line_start or 1):
            numbered.append(f"{i:4d} | {line}")
        source_with_lines = '\n'.join(numbered)

        evidence_lines = [f"- {s}" for s in signals]
        evidence_section = '\n'.join(evidence_lines) if evidence_lines else "(no specific evidence)"

        return template.format(
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

    def _confirm_dealloc_with_llm(
        self, func: Function, score: int, signals: list[str]
    ) -> bool:
        """Use the LLM to confirm whether a candidate is a deallocator.

        Returns True if confirmed as deallocator."""
        prompt = self._build_prompt(func, signals, DEALLOCATOR_PROMPT)

        try:
            if self.verbose:
                print(f"  Analyzing (dealloc): {func.name} (score={score})")

            llm_response = self.llm.complete_with_metadata(prompt)
            self._stats["llm_calls"] += 1
            self._stats["input_tokens"] += llm_response.input_tokens
            self._stats["output_tokens"] += llm_response.output_tokens

            if self.log_file:
                self._log_interaction(func.name, prompt, llm_response.content)

            return self._parse_response(llm_response.content, func.name, "is_deallocator")

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error analyzing {func.name}: {e}")
            return False

    def _parse_response(self, response: str, func_name: str, key: str = "is_allocator") -> bool:
        """Parse LLM response. Returns True if the given key is true."""
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

        is_match = data.get(key, False)
        if self.verbose:
            reasoning = data.get("reasoning", "no reason given")
            label = key.replace("is_", "").capitalize()
            status = label if is_match else f"Not {label.lower()}"
            print(f"  {status}: {func_name} - {reasoning}")

        return bool(is_match)

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
