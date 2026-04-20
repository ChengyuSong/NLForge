"""Inline callee `requires`/`ensures`/`modifies` into the function source.

Two outputs per (function, property) call:
  - `build_callee_block(...)` — the "=== CALLEE SUMMARIES ===" header block
    listing each in-scope callee's contract for the property.
  - `inline_callee_contracts(...)` — the function source with each callsite
    preceded by `// >>> callee contract for P:` hint lines.

Both lifted from `scripts/contract_pipeline.py:703-840`.
"""

from __future__ import annotations

from ..models import Function, _annotate_macro_diff
from .models import CodeContractSummary, is_nontrivial


def _format_callee_for_property(
    s: CodeContractSummary, prop: str,
) -> list[str]:
    if s.inline_body:
        # Inline-body callees are pasted at the callsite by
        # `inline_callee_contracts`; skipping them here keeps the callee
        # block from duplicating their source.
        return []
    reqs = s.requires.get(prop, [])
    ens = s.ensures.get(prop, [])
    mods = s.modifies.get(prop, [])
    note = s.notes.get(prop, "").strip()
    lines = [f"  {s.function}:"]
    if s.noreturn:
        lines.append("    noreturn: true")
    lines.append(
        f"    requires[{prop}]: " + ("; ".join(reqs) if reqs else "true")
    )
    lines.append(
        f"    ensures[{prop}]:  "
        + ("; ".join(ens) if ens else "(no observable effect)")
    )
    if mods:
        lines.append("    modifies: " + ", ".join(mods))
    if note:
        lines.append(f"    notes:    {note}")
    return lines


def build_inline_body(
    func: Function,
    summaries: dict[str, CodeContractSummary],
) -> str:
    """Build F.inline_body: F's raw source with each inline-body callee's
    already-expanded body pasted as `// >>> body of <name>:` above the
    callsite line.

    Bottom-up topo guarantees `summaries[<callee>].inline_body` is already
    transitively expanded — one pass suffices, no recursion here.
    """
    raw_lines = func.source.splitlines()
    insertions: dict[int, list[str]] = {}
    for cs in func.callsites:
        callee_name = cs.get("callee")
        if not callee_name or callee_name not in summaries:
            continue
        callee = summaries[callee_name]
        if not callee.inline_body:
            continue
        line_in_body = cs.get("line_in_body")
        if line_in_body is None or line_in_body < 0 or line_in_body >= len(raw_lines):
            continue
        body_line = raw_lines[line_in_body]
        indent = " " * (len(body_line) - len(body_line.lstrip()))
        block = [f"{indent}// >>> body of {callee_name}:"]
        for src_line in callee.inline_body.splitlines():
            block.append(f"{indent}// >>>   {src_line}")
        insertions.setdefault(line_in_body, []).extend(block)

    if not insertions:
        return func.llm_source

    out: list[str] = []
    for i, line in enumerate(raw_lines):
        if i in insertions:
            out.extend(insertions[i])
        out.append(line)
    return "\n".join(out)


def build_callee_block(
    func: Function,
    summaries: dict[str, CodeContractSummary],
    prop: str,
    callee_names: list[str],
) -> str:
    """Build the '=== CALLEE SUMMARIES ===' block injected into the prompt.

    Includes only callees whose summary is present in `summaries` (i.e.
    project-internal callees produced bottom-up + stdlib seeds).
    """
    in_scope = [n for n in callee_names if n in summaries]

    if not in_scope:
        return "=== CALLEE SUMMARIES ===\n(no in-scope callees)"

    parts = ["=== CALLEE SUMMARIES ==="]
    for name in in_scope:
        parts.extend(_format_callee_for_property(summaries[name], prop))
    if len(parts) == 1:
        # Every callee was skipped (e.g. all inline-body; their bodies
        # appear at the callsites instead).
        parts.append("(no contract-form callees; bodies inlined at callsites)")
    return "\n".join(parts)


def ordered_callee_names(
    func: Function,
    edges: dict[str, set[str]],
    summaries: dict[str, CodeContractSummary] | None = None,
) -> list[str]:
    """Stable order: first by recorded callsite order, then any DB-only
    callees (e.g., resolved indirect targets) appended in name order.

    Includes external callees that have a seeded summary (stdlib /
    harness contracts) so they appear in the prompt's callee block
    alongside project-internal callees.
    """
    proj = edges.get(func.name, set())
    extern: set[str] = set()
    if summaries is not None:
        for cs in func.callsites:
            name = cs.get("callee")
            if name and name not in proj and name in summaries:
                extern.add(name)
    in_scope = proj | extern

    seen: set[str] = set()
    out: list[str] = []
    for cs in func.callsites:
        name = cs.get("callee")
        if name and name in in_scope and name not in seen:
            seen.add(name)
            out.append(name)
    for name in sorted(in_scope):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def inline_callee_contracts(
    func: Function,
    summaries: dict[str, CodeContractSummary],
    edges: dict[str, set[str]],
    prop: str,
) -> str:
    """Build the source the LLM sees for `prop`: raw lines with each
    callsite preceded by `// >>> callee.requires/ensures/modifies` hint
    lines, then layered through macro/sizeof annotation.

    Hints inserted at `cs['line_in_body']` (0-based offset into raw
    `func.source`). Skipped for callsites whose callee has no recorded
    summary or has nothing to say about this property (keeps output
    terse for the boilerplate cases).
    """
    raw_lines = func.source.splitlines()
    insertions: dict[int, list[str]] = {}
    # In-scope = anything we have a summary for. That covers both
    # project-internal callees (added during the topo walk) and stdlib /
    # harness contracts (seeded before the walk). The `edges` map only
    # tracks intra-project edges, so we cannot use it as the gate.
    for cs in func.callsites:
        callee_name = cs.get("callee")
        if not callee_name or callee_name not in summaries:
            continue
        line_in_body = cs.get("line_in_body")
        if line_in_body is None or line_in_body < 0 or line_in_body >= len(raw_lines):
            continue
        s = summaries[callee_name]
        body_line = raw_lines[line_in_body]
        indent = " " * (len(body_line) - len(body_line.lstrip()))
        # Inline-body callees: paste the (already-expanded) source above
        # the callsite instead of the contract.
        if s.inline_body:
            block = [f"{indent}// >>> body of {callee_name}:"]
            for src_line in s.inline_body.splitlines():
                block.append(f"{indent}// >>>   {src_line}")
            insertions.setdefault(line_in_body, []).extend(block)
            continue
        reqs = [r for r in s.requires.get(prop, []) if is_nontrivial(r)]
        ens = [e for e in s.ensures.get(prop, []) if is_nontrivial(e)]
        mods = s.modifies.get(prop, [])
        note = s.notes.get(prop, "").strip()
        if not reqs and not ens and not mods and not note and not s.noreturn:
            continue
        hint: list[str] = [
            f"{indent}// >>> {callee_name} contract for {prop}:"
        ]
        if s.noreturn:
            hint.append(f"{indent}// >>>   noreturn: true")
        if reqs:
            hint.append(f"{indent}// >>>   requires: " + "; ".join(reqs))
        if ens:
            hint.append(f"{indent}// >>>   ensures:  " + "; ".join(ens))
        if mods:
            hint.append(f"{indent}// >>>   modifies: " + ", ".join(mods))
        if note:
            hint.append(f"{indent}// >>>   notes:    " + note)
        insertions.setdefault(line_in_body, []).extend(hint)

    if not insertions:
        return func.llm_source

    out: list[str] = []
    for i, line in enumerate(raw_lines):
        if i in insertions:
            out.extend(insertions[i])
        out.append(line)
    annotated = "\n".join(out)
    if func.pp_source and func.pp_source != func.source:
        return _annotate_macro_diff(annotated, func.pp_source)
    return annotated
