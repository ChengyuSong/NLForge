"""Container/collection function detection via heuristic pre-filter + LLM confirmation."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import ContainerSummary, Function

# Container-related keywords in function names
CONTAINER_KEYWORDS = {
    "hash", "map", "list", "queue", "tree", "cache", "table", "set",
    "btree", "rbtree", "heap", "pool", "trie", "bucket", "ring",
    "dict", "slist", "dlist", "deque", "stack", "avl", "skiplist",
    "lru", "fifo", "lifo",
}

# Action keywords suggesting container operations
ACTION_KEYWORDS = {
    "insert", "find", "lookup", "add", "remove", "get", "put",
    "push", "pop", "enqueue", "dequeue", "search", "store",
    "retrieve", "fetch", "delete", "erase", "append", "prepend",
    "emplace", "contains", "has",
}

# Pre-compiled regexes for heuristic scoring
_RE_ACTION_KEYWORDS = {
    kw: re.compile(rf'(?:^|_){kw}(?:_|$)')
    for kw in ACTION_KEYWORDS
}
_RE_VOID_STAR_PARAM = re.compile(r'void\s*\*(?!\*)\s*\w*')
_RE_VOID_STAR_RETURN = re.compile(r'\s*void\s*\*(?!\*)')
_RE_VOID_DSTAR_PARAM = re.compile(r'void\s*\*\*\s*\w*')
_RE_STRUCT_ASSIGN = re.compile(r'(\w+(?:->|\.)(?:\w+(?:->|\.))*\w+)\s*=\s*(.+?)\s*;')
_RE_CAST_STRIP = re.compile(r'\([^)]*\)\s*')
_RE_RETURN_FIELD = re.compile(r'return\s+(\w+(?:->|\.)(?:\w+(?:->|\.))*\w+)\s*;')
_RE_LINKED_PTR = re.compile(r'\b\w+(?:->|\.)(?:next|prev|left|right|child|parent|sibling|flink|blink)\b')
_RE_HASH_MOD = re.compile(r'%\s*\w*(?:size|cap|len|count|num|buckets|slots)\b')
_RE_HASH_CALL = re.compile(r'\b(?:hash|bucket)\s*[=(]', re.IGNORECASE)
_RE_VOID_CAST = re.compile(r'\(\s*void\s*\*\s*\)')
_RE_KEY_CMP = re.compile(r'\b(?:strcmp|memcmp|strncmp|wcscmp)\s*\(')
_RE_PAREN_CONTENT = re.compile(r'\(([^)]*)\)')

VALID_CONTAINER_TYPES = {
    "hash_table", "linked_list", "tree", "array", "queue",
    "stack", "cache", "set", "heap", "pool", "other",
}

# Map common LLM variants to canonical types
_TYPE_NORMALIZATION = {
    # linked_list variants
    "list": "linked_list",
    "singly_linked_list": "linked_list",
    "doubly_linked_list": "linked_list",
    "slist": "linked_list",
    "dlist": "linked_list",
    "skiplist": "linked_list",
    "skip_list": "linked_list",
    "circular_linked_list": "linked_list",
    "circular_doubly_linked_list": "linked_list",
    # tree variants
    "red_black_tree": "tree",
    "rb_tree": "tree",
    "rbtree": "tree",
    "rbt": "tree",
    "avl_tree": "tree",
    "avl": "tree",
    "splay_tree": "tree",
    "btree": "tree",
    "b_tree": "tree",
    "trie": "tree",
    "radix_tree": "tree",
    "binary_tree": "tree",
    "binary_search_tree": "tree",
    "balanced_binary_search_tree": "tree",
    "treap": "tree",
    # array variants
    "dynamic_array": "array",
    "array_list": "array",
    "vector": "array",
    "ring_buffer": "array",
    "circular_buffer": "array",
    "sorted_array": "array",
    # heap variants
    "priority_queue": "heap",
    "fibonacci_heap": "heap",
    # queue variants
    "fifo": "queue",
    "fifo_queue": "queue",
    "deque": "queue",
    # stack variants
    "dynamic_array_stack": "stack",
    "array_stack": "stack",
    "array_based_stack": "stack",
    # cache variants
    "lru": "cache",
    "lru_cache": "cache",
    # pool variants
    "memory_pool": "pool",
    "freelist": "pool",
    "free_list": "pool",
    "buffer_pool": "pool",
    "slab_allocator": "pool",
    "obstack": "pool",
}


def _normalize_container_type(raw: str) -> str:
    """Normalize LLM-generated container type to canonical enum value."""
    t = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if t in VALID_CONTAINER_TYPES:
        return t
    if t in _TYPE_NORMALIZATION:
        return _TYPE_NORMALIZATION[t]
    # Fuzzy fallback: check if any canonical type is a substring
    for canonical in VALID_CONTAINER_TYPES:
        if canonical in t:
            return canonical
    return "other"


CONTAINER_PROMPT = """You are analyzing a C/C++ function to determine if it is a container/collection \
operation (hash table, linked list, tree, map, cache, queue, etc.) that stores \
or retrieves pointer values.

## Function Source

```c
{source_with_lines}
```

Function: `{name}`
Project: `{project_name}`
File: `{file_path}`
Signature: `{signature}`
Parameters (0-indexed): {param_list}

## Detected Evidence

The following patterns were detected in this function:
{evidence_section}

## What is NOT a container function

- Struct initializers that set fields once during setup (not dynamic add/remove)
- Simple setters like set_callback(ctx, fn) that store a single value, not a collection
- Callback registration functions (one slot, not a dynamic collection)
- Functions that just read struct fields without collection semantics
- Memory allocators (malloc wrappers, pool allocators that return raw memory)
- String manipulation functions (even if they use linked structures internally)

## Task

Based on the source code and evidence above, is this function a container \
operation that stores or retrieves pointer values in a dynamic collection?

Respond in JSON:
```json
{{
  "is_container": true,
  "name": "function name",
  "container_arg": 0,
  "store_args": [2],
  "load_return": false,
  "container_type": "hash_table",
  "confidence": "high",
  "reasoning": "brief explanation"
}}
```

`container_type` MUST be one of: `hash_table`, `linked_list`, `tree`, `array`, \
`queue`, `stack`, `cache`, `set`, `heap`, `pool`, `other`.
Use `linked_list` for all list variants (singly/doubly linked, skip lists). \
Use `tree` for all tree variants (red-black, AVL, splay, B-tree, trie, radix). \
Use `array` for dynamic arrays, vectors, and ring buffers. \
Use `heap` for priority queues and Fibonacci heaps.

If this is NOT a container function, return:
```json
{{"is_container": false, "reasoning": "brief explanation why not"}}
```
"""


class ContainerDetector:
    """
    Detects container/collection functions using heuristic pre-filter + LLM confirmation.

    Phase 1: Score each function based on source code patterns (no LLM).
    Phase 2: For candidates above threshold, use LLM to confirm and extract details.
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
            "containers_found": 0,
            "cache_hits": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def detect_all(self, force: bool = False) -> dict[int, ContainerSummary]:
        """
        Run full detection: heuristic pre-filter + LLM confirmation.

        Args:
            force: If True, re-analyze even if summary exists

        Returns:
            Mapping of function_id to ContainerSummary
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        if self.verbose:
            print(f"Scanning {len(functions)} functions for container patterns...")

        # Phase 1: Heuristic pre-filter
        candidates = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self._heuristic_score(func)
            if score >= self.min_score:
                candidates.append((func, score, signals))

        self._stats["candidates"] = len(candidates)

        # Clean up stale summaries for functions no longer above threshold
        candidate_ids = {func.id for func, _, _ in candidates}
        existing = self.db.get_all_container_summaries()
        stale_ids = [s.function_id for s in existing if s.function_id not in candidate_ids]
        if stale_ids:
            removed = self.db.delete_container_summaries(stale_ids)
            if self.verbose:
                print(f"Removed {removed} stale container summaries")

        if self.verbose:
            print(f"Found {len(candidates)} candidates (score >= {self.min_score})")

        # Phase 2: LLM confirmation
        results: dict[int, ContainerSummary] = {}

        for func, score, signals in candidates:
            # Check cache
            if not force and self.db.has_container_summary(func.id):
                existing = self.db.get_container_summary(func.id)
                if existing:
                    results[func.id] = existing
                    self._stats["cache_hits"] += 1
                    if self.verbose:
                        print(f"  Cached: {func.name}")
                    continue

            if self.llm is None:
                # No LLM - store based on heuristic only
                summary = ContainerSummary(
                    function_id=func.id,
                    container_arg=0,
                    store_args=[],
                    load_return=False,
                    container_type="other",
                    confidence="low",
                    heuristic_score=score,
                    heuristic_signals=[s.split(": ", 1)[0] if ": " in s else s for s in signals],
                    model_used="heuristic_only",
                )
                self.db.add_container_summary(summary)
                results[func.id] = summary
                self._stats["containers_found"] += 1
                continue

            # LLM confirmation
            summary = self._analyze_function(func, score, signals)
            if summary:
                self.db.add_container_summary(summary)
                results[func.id] = summary
                self._stats["containers_found"] += 1

        return results

    def heuristic_only(self) -> list[tuple[Function, int, list[str]]]:
        """
        Run heuristic scoring only (no LLM calls).

        Returns:
            List of (function, score, signals) tuples for candidates above threshold.
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        candidates = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self._heuristic_score(func)
            if score >= self.min_score:
                candidates.append((func, score, signals))

        self._stats["candidates"] = len(candidates)
        return candidates

    def _heuristic_score(self, func: Function) -> tuple[int, list[str]]:
        """
        Score a function based on container-related patterns.

        Returns:
            (score, evidence_signals) where each signal is a descriptive string.
        """
        score = 0
        signals: list[str] = []
        name_lower = func.name.lower()
        source = func.source or ""

        # --- Name-based signals ---

        # Container keyword in name
        for kw in CONTAINER_KEYWORDS:
            if kw in name_lower:
                score += 3
                signals.append(f"container_keyword: name contains '{kw}'")
                break  # Only count once

        # Action keyword in name
        for kw, pat in _RE_ACTION_KEYWORDS.items():
            if pat.search(name_lower):
                score += 2
                signals.append(f"action_keyword: name contains '{kw}'")
                break

        # --- Signature-based signals ---
        sig = func.signature or ""

        # void* parameter (generic value storage)
        void_star_params = _RE_VOID_STAR_PARAM.findall(sig)
        if void_star_params:
            score += 2
            signals.append(f"void_ptr_param: {void_star_params[0].strip()}")

        # void* return type
        if _RE_VOID_STAR_RETURN.match(sig):
            score += 2
            signals.append("void_ptr_return: returns void*")

        # void** parameter (output param)
        void_dstar_params = _RE_VOID_DSTAR_PARAM.findall(sig)
        if void_dstar_params:
            score += 2
            signals.append(f"void_dblptr_param: {void_dstar_params[0].strip()}")

        # --- Body-based signals ---
        if not source:
            return score, signals

        # Whole-source signals (no per-line loop needed)
        if _RE_VOID_CAST.search(source):
            score += 1
            signals.append("void_cast: body contains (void*) cast")

        if _RE_KEY_CMP.search(source):
            score += 1
            signals.append("key_comparison: body calls strcmp/memcmp")

        # Parse parameter names for tracking stores
        param_names = self._extract_param_names(sig)
        # Pre-compile param name patterns
        param_pats = []
        for pname in param_names:
            if pname:
                param_pats.append((pname, re.compile(rf'\b{re.escape(pname)}\b')))

        # Single pass over lines for all per-line signals
        found_store = False
        found_return = False
        found_linked = False
        found_hash = False

        for i, line in enumerate(source.split('\n'), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Param stored to struct field
            if not found_store and param_pats:
                m = _RE_STRUCT_ASSIGN.match(stripped)
                if m:
                    rhs = m.group(2)
                    rhs_clean = _RE_CAST_STRIP.sub('', rhs)
                    for pname, ppat in param_pats:
                        if ppat.search(rhs_clean):
                            score += 4
                            signals.append(f"param_stored_to_field: Line {i}: `{stripped}` stores param '{pname}' into struct field")
                            found_store = True
                            break

            # Return of struct field
            if not found_return:
                m = _RE_RETURN_FIELD.match(stripped)
                if m:
                    score += 1
                    signals.append(f"return_struct_field: Line {i}: `{stripped}` returns struct field")
                    found_return = True

            # next/prev pointer manipulation
            if not found_linked:
                if _RE_LINKED_PTR.search(stripped):
                    score += 3
                    signals.append(f"linked_ptr_manipulation: Line {i}: `{stripped[:60]}`")
                    found_linked = True

            # Hash/bucket computation
            if not found_hash:
                if _RE_HASH_MOD.search(stripped) or _RE_HASH_CALL.search(stripped):
                    score += 2
                    signals.append(f"hash_computation: Line {i}: `{stripped[:60]}`")
                    found_hash = True

            # Early exit if all per-line signals found
            if found_store and found_return and found_linked and found_hash:
                break

        return score, signals

    def _extract_param_names(self, signature: str) -> list[str]:
        """Extract parameter names from a C function signature."""
        # Find content between parentheses
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
                # Filter out type-only params (e.g., "void" or "int")
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
        # Add line numbers to source
        lines = (func.llm_source or "").split('\n')
        numbered = []
        for i, line in enumerate(lines, func.line_start or 1):
            numbered.append(f"{i:4d} | {line}")
        source_with_lines = '\n'.join(numbered)

        # Build parameter list
        param_names = self._extract_param_names(func.signature or "")
        sig = func.signature or ""
        m = re.search(r'\(([^)]*)\)', sig)
        param_list_str = ""
        if m:
            params_raw = m.group(1).strip()
            if params_raw and params_raw != 'void':
                parts = [p.strip() for p in params_raw.split(',')]
                param_entries = []
                for idx, part in enumerate(parts):
                    pname = param_names[idx] if idx < len(param_names) else "?"
                    param_entries.append(f"  [{idx}] {part}")
                param_list_str = '\n'.join(param_entries)

        if not param_list_str:
            param_list_str = "(none)"

        # Build evidence section
        evidence_lines = []
        for s in signals:
            evidence_lines.append(f"- {s}")
        evidence_section = '\n'.join(evidence_lines) if evidence_lines else "(no specific evidence)"

        return CONTAINER_PROMPT.format(
            source_with_lines=source_with_lines,
            name=func.name,
            project_name=self.project_name or "(unknown)",
            file_path=func.file_path or "(unknown)",
            signature=func.signature or "(unknown)",
            param_list=param_list_str,
            evidence_section=evidence_section,
        )

    def _analyze_function(
        self, func: Function, score: int, signals: list[str]
    ) -> ContainerSummary | None:
        """
        Analyze a single candidate function using the LLM.

        Returns ContainerSummary if confirmed as container, None otherwise.
        """
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

            return self._parse_response(llm_response.content, func, score, signals)

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error analyzing {func.name}: {e}")
            return None

    def _parse_response(
        self, response: str, func: Function, score: int, signals: list[str]
    ) -> ContainerSummary | None:
        """Parse LLM response. Returns None if is_container=false."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                if self.verbose:
                    print(f"  Failed to extract JSON from response for {func.name}")
                return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  JSON parse error for {func.name}: {e}")
            return None

        if not data.get("is_container", False):
            if self.verbose:
                reason = data.get("reasoning", "no reason given")
                print(f"  Not a container: {func.name} - {reason}")
            return None

        container_arg = data.get("container_arg", 0)
        store_args = data.get("store_args", [])
        if not isinstance(store_args, list):
            store_args = []

        container_type = _normalize_container_type(
            data.get("container_type", "other")
        )

        return ContainerSummary(
            function_id=func.id,
            container_arg=container_arg,
            store_args=store_args,
            load_return=bool(data.get("load_return", False)),
            container_type=container_type,
            confidence=data.get("confidence", "medium"),
            heuristic_score=score,
            heuristic_signals=[s.split(": ", 1)[0] if ": " in s else s for s in signals],
            model_used=self.llm.model if self.llm else "",
        )

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[CONTAINER DETECTION] Function: {func_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model if self.llm else 'N/A'}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")
