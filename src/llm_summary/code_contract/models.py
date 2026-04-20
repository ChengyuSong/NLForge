"""Data model for code-contract summaries.

Lifted from `scripts/contract_pipeline.py:67-113` (FunctionSummary,
PROPERTY_SCHEMA, _is_nontrivial). Renamed `FunctionSummary` →
`CodeContractSummary` to avoid colliding with the existing
`MemsafeSummary` / `LeakSummary` etc. in `models.py`.

Per-property fields are dicts keyed by property name (`"memsafe"` /
`"memleak"` / `"overflow"`). `noreturn` is property-independent (any
per-property call may set it; OR-merge across properties).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROPERTIES: list[str] = ["memsafe", "memleak", "overflow"]


PROPERTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requires": {"type": "array", "items": {"type": "string"}},
        "ensures":  {"type": "array", "items": {"type": "string"}},
        "modifies": {"type": "array", "items": {"type": "string"}},
        "notes":    {"type": "string"},
        # Property-independent: set true iff this function has NO returning
        # path (e.g. body unconditionally aborts/exits/longjmps). The same
        # answer is expected from every per-property call; we OR them.
        "noreturn": {"type": "boolean"},
    },
    "required": ["requires", "ensures"],
}


def is_nontrivial(predicate: str) -> bool:
    """True iff `predicate` is a real obligation/effect (not a no-op stand-in)."""
    s = predicate.strip().lower()
    return s not in (
        "", "true", "1", "tt", "(no observable effect)",
        "no resource acquired", "nothing acquired",
    )


@dataclass
class CodeContractSummary:
    """Hoare-style per-function summary. NO verdict field by design.

    Per-property maps (`requires`, `ensures`, `modifies`, `notes`) carry the
    output of the per-property summarization call for each property in
    `properties`. `noreturn` is OR-ed across property calls.

    `origin` records, for each requires entry per property, where the clause
    came from — `"local"` for body-derived, `"<callee>:<idx>"` for one
    propagated verbatim from a callee's `requires[P][idx]`. The Phase 4
    entry checker walks `origin` chains back to leaf operations to build
    a witness chain.
    """

    function: str
    properties: list[str] = field(default_factory=list)
    requires: dict[str, list[str]] = field(default_factory=dict)
    ensures: dict[str, list[str]] = field(default_factory=dict)
    modifies: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    origin: dict[str, list[str]] = field(default_factory=dict)
    # Property-independent: callsites to a noreturn callee cut the path.
    # Sources: explicit `__attribute__((noreturn))` / `_Noreturn` on the
    # declaration, stdlib seed (abort/exit/...), or LLM-detected
    # body-always-aborts. Used by callers for path narrowing.
    noreturn: bool = False
    # Non-empty iff the function was small enough that its raw body is
    # cheaper to paste at every callsite than to summarize. The body text
    # (raw `func.llm_source`) is what the inliner pastes at each callsite
    # under `// >>> inlined body of <name>`. When set, `properties` and
    # the per-property maps are empty — there is no contract.
    inline_body: str = ""

    def has_requires(self, prop: str) -> bool:
        return any(is_nontrivial(r) for r in self.requires.get(prop, []))

    def has_ensures(self, prop: str) -> bool:
        return any(is_nontrivial(e) for e in self.ensures.get(prop, []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "properties": list(self.properties),
            "requires": {k: list(v) for k, v in self.requires.items()},
            "ensures":  {k: list(v) for k, v in self.ensures.items()},
            "modifies": {k: list(v) for k, v in self.modifies.items()},
            "notes":    dict(self.notes),
            "origin":   {k: list(v) for k, v in self.origin.items()},
            "noreturn": bool(self.noreturn),
            "inline_body": self.inline_body,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeContractSummary:
        return cls(
            function=str(d["function"]),
            properties=list(d.get("properties") or []),
            requires={k: list(v) for k, v in (d.get("requires") or {}).items()},
            ensures={k: list(v) for k, v in (d.get("ensures") or {}).items()},
            modifies={k: list(v) for k, v in (d.get("modifies") or {}).items()},
            notes=dict(d.get("notes") or {}),
            origin={k: list(v) for k, v in (d.get("origin") or {}).items()},
            noreturn=bool(d.get("noreturn", False)),
            inline_body=str(d.get("inline_body") or ""),
        )

    def to_annotated_source(self, function_source: str) -> str:
        """Return `function_source` with the contract pinned above as a
        block of `// @requires[P]: ...` / `// @ensures[P]: ...` comments,
        per the design doc's code-as-summary form.

        Used for downstream consumption (gen-harness, juliet eval debugging,
        triage agent). The block lists every property in `self.properties`,
        with non-trivial requires/ensures only; trivial entries (`true`,
        `(no observable effect)`, etc.) are skipped to keep the header
        short.
        """
        header: list[str] = []
        if self.noreturn:
            header.append("// @noreturn: true")
        for prop in self.properties:
            reqs = [r for r in self.requires.get(prop, []) if is_nontrivial(r)]
            ens = [e for e in self.ensures.get(prop, []) if is_nontrivial(e)]
            mods = self.modifies.get(prop, [])
            note = self.notes.get(prop, "").strip()
            if not (reqs or ens or mods or note):
                continue
            for r in reqs:
                header.append(f"// @requires[{prop}]: {r}")
            for e in ens:
                header.append(f"// @ensures[{prop}]: {e}")
            if mods:
                header.append(f"// @modifies[{prop}]: " + ", ".join(mods))
            if note:
                header.append(f"// @notes[{prop}]: {note}")
        if not header:
            return function_source
        return "\n".join(header) + "\n" + function_source
