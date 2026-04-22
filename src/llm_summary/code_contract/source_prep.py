"""Per-function source-prep for the code-contract pipeline.

Two-step source-prep API:
  - `build_type_defs_section(db, source, file_path)` — typedef / struct /
    static-var definitions referenced by the source. Property-independent
    (cacheable across the per-property calls for one function).
  - `prepare_source(func, summaries, edges, prop)` — macro-annotated
    function source with callee contracts inlined for `prop`. Per-property
    (the `{source}` placeholder in PROPERTY_PROMPT).

`build_type_defs_section` is a free function lifted from the legacy
`VerificationSummarizer._build_type_defs_section` — single implementation,
both pipelines import from here.
"""

from __future__ import annotations

import re

from ..db import SummaryDB
from ..models import Function
from .inliner import inline_callee_contracts
from .models import CodeContractSummary


def build_type_defs_section(
    db: SummaryDB, source: str, file_path: str = "",
) -> str:
    """Build a section with struct/union/typedef definitions referenced in
    `source`, plus file-scope static variable declarations from the same file.

    Lifted from `verification_summarizer.py:521-594`. Both the legacy
    verification summarizer and the new code-contract pipeline call this
    single implementation.
    """
    # Find all identifiers in source, look up any that match a typedef.
    # Catches struct/union/enum tags AND plain typedefs (cgc_size_t, pmeta,
    # uint16_t, etc.).
    all_identifiers = set(re.findall(r"\b([A-Za-z_]\w*)\b", source))
    rows = db.get_typedefs_by_names(list(all_identifiers)) if all_identifiers else []
    names = {r["name"] for r in rows}

    # Also include file-scope static variables from the same source file
    # that are actually referenced by name in the function source.
    static_rows = [
        r for r in (db.get_static_vars_by_file(file_path) if file_path else [])
        if r["name"] in all_identifiers
    ]

    # Resolve types referenced in static var declarations
    # e.g., "static engine_t *engine;" → look up engine_t typedef
    static_type_names: set[str] = set()
    for srow in static_rows:
        defn = srow.get("definition") or ""
        for tok in re.findall(r"\b([A-Za-z_]\w*)\b", defn):
            if tok not in (
                "static", "const", "volatile", "unsigned", "signed",
                "char", "int", "long", "short", "float", "double",
                "void", "bool", srow["name"],
            ):
                static_type_names.add(tok)
    new_names = static_type_names - {r["name"] for r in rows} - names
    if new_names:
        extra = db.get_typedefs_by_names(list(new_names))
        rows.extend(extra)
        names.update(new_names)

    # Deduplicate by name: same-file definition wins; among cross-file,
    # prefer shortest (least likely to be the wrong variant).
    # pp_definition stores the annotated macro-expanded form (// (macro)
    # lines) produced at scan time; use it when available.
    seen: dict[str, str] = {}
    seen_canonical: dict[str, str] = {}
    seen_kind: dict[str, str] = {}
    seen_from_same_file: set[str] = set()
    for row in rows + static_rows:
        name = row["name"]
        defn = row.get("pp_definition") or row.get("definition") or ""
        if not defn:
            continue
        canonical = row.get("canonical_type") or ""
        kind = row.get("kind") or ""
        same_file = row.get("file_path") == file_path
        if same_file:
            seen[name] = defn
            seen_canonical[name] = canonical
            seen_kind[name] = kind
            seen_from_same_file.add(name)
        elif name not in seen_from_same_file:
            if name not in seen or len(defn) < len(seen[name]):
                seen[name] = defn
                seen_canonical[name] = canonical
                seen_kind[name] = kind

    if not seen:
        return ""

    # Emit each unique definition block only once.
    # When the definition text doesn't reveal the primitive type
    # (e.g. "typedef Z_U4 z_crc_t;"), append a canonical-type comment.
    emitted: set[str] = set()
    lines = ["## Referenced Type Definitions\n", "```c"]
    for name, defn in seen.items():
        if defn not in emitted:
            emitted.add(defn)
            canonical = seen_canonical.get(name, "")
            if (
                canonical
                and canonical not in defn
                and seen_kind.get(name) == "typedef"
            ):
                lines.append(f"{defn}  // canonical: {canonical}")
            else:
                lines.append(defn)
            lines.append("")
    lines.append("```\n\n")
    return "\n".join(lines)


def prepare_source(
    func: Function,
    summaries: dict[str, CodeContractSummary],
    edges: dict[str, set[str]],
    prop: str,
) -> str:
    """Macro-annotated function source with callee contracts inlined for
    `prop`. Goes into the `{source}` placeholder of `PROPERTY_PROMPT[prop]`.

    Wrapper around `inliner.inline_callee_contracts` so callers don't need
    to import from two modules.
    """
    return inline_callee_contracts(func, summaries, edges, prop)
