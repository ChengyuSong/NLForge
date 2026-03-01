"""Builds alias context sections from V-snapshot data for LLM prompts.

Uses whole-program pointer aliasing (V-relation) from CFL-reachability analysis
to identify functions whose source reveals field-level aliasing relevant to
contract propagation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .db import SummaryDB
from .models import Function
from .vsnapshot import NamedEntry, VSnapshot


@dataclass
class AliasGroup:
    """A set of named entries connected through V-relation aliasing."""

    entries: list[NamedEntry] = field(default_factory=list)
    functions: set[str] = field(default_factory=set)
    globals_: set[str] = field(default_factory=set)
    returns: set[str] = field(default_factory=set)


def _entry_func_name(entry: NamedEntry) -> str | None:
    """Extract function name from a named entry, or None for globals."""
    if entry.kind == 3:  # arg: "funcname::arg#N"
        return entry.name.split("::")[0]
    elif entry.kind == 6:  # return: "ret:funcname"
        return entry.name.split(":", 1)[1]
    elif entry.kind == 7:  # vararg: "funcname::vararg"
        return entry.name.split("::")[0]
    return None


def _word_in_line(word: str, line: str) -> bool:
    """Check if *word* appears as a whole word in *line* (no regex)."""
    wlen = len(word)
    start = 0
    while True:
        idx = line.find(word, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_")
        end = idx + wlen
        after_ok = end >= len(line) or not (line[end].isalnum() or line[end] == "_")
        if before_ok and after_ok:
            return True
        start = idx + 1


def _aliased_param_names(func: Function, group: AliasGroup) -> list[str]:
    """Return parameter names of *func* that participate in the alias group."""
    names: list[str] = []
    for entry in group.entries:
        if entry.kind != 3:  # only args
            continue
        if _entry_func_name(entry) != func.name:
            continue
        # entry.name is like "funcname::arg#2"
        suffix = entry.name.rsplit("#", 1)[-1]
        try:
            idx = int(suffix)
        except ValueError:
            continue
        if idx < len(func.params) and func.params[idx] not in names:
            names.append(func.params[idx])
    return names


def _extract_relevant_lines(
    source: str, param_names: list[str], context: int = 2
) -> str:
    """Extract function signature + lines referencing aliased params.

    Always includes the signature lines (everything up to and including the
    opening '{'). Then includes lines mentioning any of *param_names* with
    ±context surrounding lines.

    Falls back to full source if param_names is empty or the heuristic
    matches almost everything (>80%).
    """
    if not param_names:
        return source

    # Skip very short names (1 char) — too noisy
    usable = [p for p in param_names if len(p) >= 2]
    if not usable:
        return source

    lines = source.splitlines()
    matched: set[int] = set()

    # Always include signature: lines until we see '{'
    sig_end = 0
    for i, line in enumerate(lines):
        matched.add(i)
        sig_end = i
        if "{" in line:
            break

    # Scan remaining lines for param references
    for i in range(sig_end + 1, len(lines)):
        for pname in usable:
            if _word_in_line(pname, lines[i]):
                matched.add(i)
                break

    if len(matched) >= len(lines) * 0.8:
        return source  # almost everything matched, just use full source

    # Expand with context
    expanded: set[int] = set()
    for i in matched:
        for j in range(max(0, i - context), min(len(lines), i + context + 1)):
            expanded.add(j)

    # Build output with ... separators for gaps
    result: list[str] = []
    prev = -2
    for i in sorted(expanded):
        if i > prev + 1 and result:
            result.append("    ...")
        result.append(lines[i])
        prev = i

    return "\n".join(result)


class AliasContextBuilder:
    """Builds alias context sections from V-snapshot data."""

    def __init__(
        self,
        vsnap_path: str,
        db: SummaryDB,
        max_candidates: int = 3,
        alloc_candidates_path: str | None = None,
    ):
        self.snap = VSnapshot.load(vsnap_path)
        self.db = db
        self.max_candidates = max_candidates
        self._alloc_candidates_path = alloc_candidates_path
        # Index: rep → list of named entries with that rep
        self._rep_to_entries: dict[int, list[NamedEntry]] | None = None
        # Index: function name → set of reps for its entries
        self._func_to_reps: dict[str, set[int]] | None = None
        # Reps to exclude from alias expansion (allocator returns, deallocator args)
        self._heap_reps: set[int] | None = None

    def _build_index(self) -> None:
        """Build indexes from named entries (kinds 2/3/6/7)."""
        self._rep_to_entries = defaultdict(list)
        self._func_to_reps = defaultdict(set)

        for entry in self.snap.named_entries:
            if entry.kind not in (2, 3, 6, 7):
                continue
            rep = self.snap.rep(entry.node)
            self._rep_to_entries[rep].append(entry)

            func_name = _entry_func_name(entry)
            if func_name:
                self._func_to_reps[func_name].add(rep)

        self._heap_reps = self._find_heap_reps()

    def _find_heap_reps(self) -> set[int]:
        """Identify reps for allocator return values and deallocator args.

        These alias with every heap object in the program and produce
        useless noise in the alias context.
        """
        import json

        allocator_names: set[str] = set()
        deallocator_names: set[str] = set()

        # Query DB for functions with returned allocations → allocators
        rows = self.db.conn.execute(
            "SELECT f.name, a.summary_json "
            "FROM allocation_summaries a "
            "JOIN functions f ON f.id = a.function_id"
        ).fetchall()
        for name, summary_json in rows:
            data = json.loads(summary_json)
            if any(a.get("returned") for a in data.get("allocations", [])):
                allocator_names.add(name)

        # Query DB for functions with free operations → deallocators
        rows = self.db.conn.execute(
            "SELECT f.name, a.summary_json "
            "FROM free_summaries a "
            "JOIN functions f ON f.id = a.function_id"
        ).fetchall()
        for name, summary_json in rows:
            data = json.loads(summary_json)
            if data.get("frees"):
                deallocator_names.add(name)

        # Also load from allocator candidates JSON if available
        if self._alloc_candidates_path:
            try:
                with open(self._alloc_candidates_path) as f:
                    cands = json.load(f)
                for name in cands.get("candidates", []) + cands.get("confirmed", []):
                    allocator_names.add(name)
                for name in cands.get("dealloc_candidates", []) + cands.get("dealloc_confirmed", []):
                    deallocator_names.add(name)
            except (OSError, json.JSONDecodeError):
                pass

        # Collect reps to exclude
        heap_reps: set[int] = set()
        for entry in self.snap.named_entries:
            func_name = _entry_func_name(entry)
            if func_name is None:
                continue
            # Allocator return values (kind 6 = ret:funcname)
            if entry.kind == 6 and func_name in allocator_names:
                heap_reps.add(self.snap.rep(entry.node))
            # Deallocator first arg (kind 3 = funcname::arg#0)
            if entry.kind == 3 and entry.name.endswith("::arg#0") and func_name in deallocator_names:
                heap_reps.add(self.snap.rep(entry.node))

        return heap_reps

    def _find_aliased_functions(
        self, query_names: set[str]
    ) -> AliasGroup:
        """Find all named entries aliased with the query functions.

        Uses rep_rows adjacency to find aliased reps, then collects all
        named entries at those reps.
        """
        if self._rep_to_entries is None or self._func_to_reps is None:
            self._build_index()

        # Collect all reps for query functions, excluding heap-level reps
        # (allocator returns, deallocator args) that alias with everything.
        query_reps: set[int] = set()
        for name in query_names:
            reps = self._func_to_reps.get(name, set())
            query_reps |= (reps - self._heap_reps)

        if not query_reps:
            return AliasGroup()

        # Expand through rep_rows to find all aliased reps that have named entries
        aliased_reps: set[int] = set()
        named_rep_set = set(self._rep_to_entries.keys())
        for qr in query_reps:
            for dst_rep in self.snap.rep_rows[qr]:
                if dst_rep in named_rep_set:
                    aliased_reps.add(dst_rep)

        # Also include the query reps themselves
        aliased_reps |= (query_reps & named_rep_set)

        # Collect all named entries at aliased reps
        group = AliasGroup()
        for rep in aliased_reps:
            for entry in self._rep_to_entries[rep]:
                group.entries.append(entry)
                func_name = _entry_func_name(entry)
                if func_name:
                    group.functions.add(func_name)
                    if entry.kind == 6:
                        group.returns.add(func_name)
                elif entry.kind == 2:
                    group.globals_.add(entry.name)

        return group

    def _is_context_provider(self, func: Function) -> bool:
        """Check if function is likely to reveal aliasing structure."""
        name = func.name.lower()
        if any(
            p in name
            for p in ("init", "create", "setup", "new", "open", "alloc")
        ):
            return True
        # Contains field store patterns
        if func.source and "->" in func.source and "=" in func.source:
            return True
        return False

    def _relevance_score(
        self, func_name: str, query_names: set[str]
    ) -> float:
        """Score candidate by alias overlap + naming heuristics.

        Higher is better. Initializer-like names get a bonus.
        """
        if self._func_to_reps is None:
            return 0.0
        cand_reps = self._func_to_reps.get(func_name, set())
        if not cand_reps:
            return 0.0

        score = 0.0
        for qname in query_names:
            for qrep in self._func_to_reps.get(qname, set()):
                for crep in cand_reps:
                    if self.snap.may_alias(
                        self.snap.rep_to_node[qrep],
                        self.snap.rep_to_node[crep],
                    ):
                        score += 1.0
                        break

        # Bonus for initializer/factory name patterns
        name_lower = func_name.lower()
        if any(p in name_lower for p in ("init", "create", "setup", "new", "alloc")):
            score += 0.5

        # Bonus for sharing a name prefix with any query function
        # (e.g., _tr_init shares prefix with _tr_flush_block)
        for qname in query_names:
            # Check common prefix of at least 3 chars
            prefix_len = 0
            for a, b in zip(func_name, qname):
                if a == b:
                    prefix_len += 1
                else:
                    break
            if prefix_len >= 3:
                score += 0.3
                break

        return score

    def _format_alias_annotation(
        self,
        func_name: str,
        target_func: str,
        callee_names: set[str],
        group: AliasGroup,
    ) -> list[str]:
        """Format the specific alias relationships for a candidate function."""
        lines: list[str] = []
        seen: set[str] = set()

        # Collect entries for the candidate function
        cand_entries = [
            e for e in group.entries
            if e.kind in (3, 6, 7) and _entry_func_name(e) == func_name
        ]

        # Collect entries for target/callees (exclude globals — too noisy)
        query_set = {target_func} | callee_names
        related_entries = [
            e for e in group.entries
            if e.kind in (3, 6, 7) and _entry_func_name(e) in query_set
        ]

        for ce in cand_entries:
            ce_rep = self.snap.rep(ce.node)
            for re_ in related_entries:
                re_rep = self.snap.rep(re_.node)
                # Check actual aliasing through rep_rows
                if not self.snap.may_alias(
                    self.snap.rep_to_node[ce_rep],
                    self.snap.rep_to_node[re_rep],
                ):
                    continue

                pair_key = f"{ce.name}|{re_.name}"
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                rel_func = _entry_func_name(re_)

                note = ""
                if rel_func and rel_func in callee_names and rel_func != target_func:
                    note = "  (callee)"

                lines.append(f"  - {ce.name}  \u2194  {re_.name}{note}")

        return lines

    def _format_context(
        self,
        target: Function,
        candidates: list[Function],
        callee_names: set[str],
        group: AliasGroup,
    ) -> str:
        """Format alias context section with alias annotations and source."""
        parts: list[str] = []
        parts.append(
            "## Alias Context (from whole-program analysis)\n\n"
            "The following function(s) operate on pointers that alias with this "
            "function's parameters, return value, or relevant globals. They may "
            "establish field-level relationships relevant for understanding buffer "
            "sizes and pointer validity.\n"
        )

        for cand in candidates:
            annotations = self._format_alias_annotation(
                cand.name, target.name, callee_names, group
            )
            parts.append(f"### `{cand.name}` \u2014 aliases with target function:")
            if annotations:
                parts.extend(annotations)
            parts.append("")
            if cand.source:
                pnames = _aliased_param_names(cand, group)
                src = _extract_relevant_lines(cand.source, pnames)
                parts.append(f"```c\n{src}\n```\n")

        parts.append(
            "When a callee's contract references a field path (e.g., "
            "`bl_desc.dyn_tree`), check whether an aliased function establishes "
            "it as an alias for another field (e.g., `bl_tree`). If so, propagate "
            "the contract to the aliased field.\n\n"
            "Also consider return-value and global aliasing: if a factory's return "
            "value aliases a parameter here, contracts on the return apply to that "
            "parameter."
        )

        return "\n".join(parts)

    def build_context(
        self, func: Function, callee_names: list[str]
    ) -> str | None:
        """Build alias context for func + its callees.

        Returns formatted prompt section or None if no relevant aliases found.
        """
        if self._rep_to_entries is None:
            self._build_index()

        # Skip alias context for mega-functions — too many callees make
        # the annotation cross-product useless noise.
        if len(callee_names) > 30:
            return None

        query_names = {func.name} | set(callee_names)
        group = self._find_aliased_functions(query_names)

        if not group.functions:
            return None

        # Find candidate functions (exclude self and direct callees)
        candidates = group.functions - query_names

        # Filter and rank candidates
        scored: list[tuple[int, Function]] = []
        for cand_name in candidates:
            cand_funcs = self.db.get_function_by_name(cand_name)
            if cand_funcs and self._is_context_provider(cand_funcs[0]):
                score = self._relevance_score(cand_name, query_names)
                scored.append((score, cand_funcs[0]))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        selected = [f for _, f in scored[: self.max_candidates]]

        return self._format_context(
            target=func,
            candidates=selected,
            callee_names=set(callee_names),
            group=group,
        )
