"""Phase 4 entry-point check (pure Python; no LLM).

Walks the call graph from each entry function, reads its
`code_contract_summaries.requires`, and reports each non-trivial entry
`requires` clause as an `Obligation`. The witness chain is built by
following each clause's `origin` field back through the callees that
propagated it, terminating at `"local"` leaves.

Used by `llm-summary check --db ... [--entry FUNC]`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..db import SummaryDB
from .models import PROPERTIES, CodeContractSummary, is_nontrivial


@dataclass
class WitnessStep:
    """One link in a witness chain.

    Reads as: at function `function`, clause `requires[property][index]` is
    the predicate `predicate`; its origin is `origin` (either "local" or
    "<callee>:<idx>" naming the next step).
    """

    function: str
    property: str
    index: int
    predicate: str
    origin: str


@dataclass
class Obligation:
    """One un-discharged precondition surfaced at an entry function."""

    entry_function: str
    property: str
    predicate: str
    witness_chain: list[WitnessStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_function": self.entry_function,
            "property": self.property,
            "predicate": self.predicate,
            "witness_chain": [
                {
                    "function": s.function,
                    "property": s.property,
                    "index": s.index,
                    "predicate": s.predicate,
                    "origin": s.origin,
                }
                for s in self.witness_chain
            ],
        }

    def fingerprint(self) -> str:
        key = f"{self.property}|{self.entry_function}|{self.predicate[:80]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_witness_chain(
    db: SummaryDB,
    function_name: str,
    prop: str,
    index: int,
    summary: CodeContractSummary,
    visited: set[tuple[str, str, int]] | None = None,
) -> list[WitnessStep]:
    """Walk `origin` links back to a leaf; cycle-safe."""
    visited = visited if visited is not None else set()
    key = (function_name, prop, index)
    if key in visited:
        return []
    visited.add(key)

    reqs = summary.requires.get(prop, [])
    origins = summary.origin.get(prop, [])
    if index < 0 or index >= len(reqs):
        return []

    pred = reqs[index]
    origin = origins[index] if index < len(origins) else "local"
    step = WitnessStep(
        function=function_name, property=prop, index=index,
        predicate=pred, origin=origin,
    )
    chain: list[WitnessStep] = [step]

    if origin == "local" or ":" not in origin:
        return chain

    callee_name, _, idx_str = origin.partition(":")
    try:
        callee_idx = int(idx_str)
    except ValueError:
        return chain

    callee_func = db.find_function_by_name(callee_name)
    if callee_func is None or callee_func.id is None:
        return chain
    callee_summary = db.get_code_contract_summary(callee_func.id)
    if callee_summary is None:
        return chain

    chain.extend(_build_witness_chain(
        db, callee_name, prop, callee_idx, callee_summary, visited,
    ))
    return chain


def find_entry_functions(
    db: SummaryDB, restrict_to: list[str] | None = None,
) -> list[str]:
    """Return functions that have no callers within the function set.

    Mirrors the existing `cli._find_entry_functions` shape: a function is
    an entry iff no edge in `call_edges` (or resolved indirect targets)
    targets it. `restrict_to` (when given) limits both the entry set and
    the caller-set used for the check.
    """
    funcs = db.get_all_functions()
    if restrict_to is not None:
        wanted = set(restrict_to)
        funcs = [f for f in funcs if f.name in wanted]

    func_ids = {f.id for f in funcs if f.id is not None}
    id_to_name = {f.id: f.name for f in funcs if f.id is not None}

    has_caller: set[int] = set()
    for edge in db.get_all_call_edges():
        if edge.callee_id in func_ids and edge.caller_id in func_ids:
            has_caller.add(edge.callee_id)

    return [id_to_name[fid] for fid in func_ids if fid not in has_caller]


def check_entries(
    db: SummaryDB,
    entries: list[str] | None = None,
) -> list[Obligation]:
    """Run the entry-point check and return one Obligation per non-trivial
    entry `requires` clause across all in-scope properties.

    `entries`: optional list of function names to check. When None, uses
    `find_entry_functions(db)` over the full function set.
    """
    if entries is None:
        entries = find_entry_functions(db)

    obligations: list[Obligation] = []
    for entry_name in entries:
        func = db.find_function_by_name(entry_name)
        if func is None or func.id is None:
            continue
        summary = db.get_code_contract_summary(func.id)
        if summary is None:
            continue
        for prop in PROPERTIES:
            reqs = summary.requires.get(prop, [])
            for idx, pred in enumerate(reqs):
                if not is_nontrivial(pred):
                    continue
                chain = _build_witness_chain(
                    db, entry_name, prop, idx, summary,
                )
                obligations.append(Obligation(
                    entry_function=entry_name,
                    property=prop,
                    predicate=pred,
                    witness_chain=chain,
                ))
    return obligations
