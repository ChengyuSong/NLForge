"""Importer + helpers for KAMain's IR fact sidecar files.

KAMain emits one ``<bc>.facts.json`` per bitcode when invoked with
``--ir-sidecar-dir <dir>``. Each file is::

    {
      "metadata": {"bc_path": "...", "total_functions": N, "version": 1},
      "functions": {
        "<func_name>": {
          "function": "...",
          "ir_hash": "...",
          "cg_hash": "...",
          "effects": [...],
          "branches": [...],
          "ranges": [...],
          "int_ops": [...],
          "features": {...}
        },
        ...
      }
    }

This module joins those per-function blobs to ``functions`` rows by name
and stores them in the ``function_ir_facts`` table. KAMain itself never
touches the DB (per ``docs/todo-kamain-ir-sidecar.md``); this is the
llm-summary side of that contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .db import SummaryDB

log = logging.getLogger("ir_sidecar")


@dataclass
class ImportStats:
    files_read: int = 0
    functions_in_sidecar: int = 0
    functions_imported: int = 0
    functions_unmatched: int = 0  # in sidecar but not in DB


def import_sidecar_dir(
    db: SummaryDB,
    sidecar_dir: Path | str,
) -> ImportStats:
    """Read every ``*.facts.json`` under *sidecar_dir* and upsert into DB.

    Match key is the function name. Sidecar entries that don't match a
    DB row (e.g., functions inlined away or scoped differently) are
    counted as ``functions_unmatched`` and skipped — not an error.
    """
    sidecar_dir = Path(sidecar_dir)
    stats = ImportStats()
    if not sidecar_dir.is_dir():
        log.debug("Sidecar dir %s does not exist", sidecar_dir)
        return stats

    for path in sorted(sidecar_dir.glob("*.facts.json")):
        stats.files_read += 1
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Skipping malformed sidecar %s: %s", path, e)
            continue

        funcs = payload.get("functions") or {}
        stats.functions_in_sidecar += len(funcs)
        for fname, fdata in funcs.items():
            if not isinstance(fdata, dict):
                continue
            db_funcs = db.get_function_by_name(fname)
            if not db_funcs:
                stats.functions_unmatched += 1
                continue
            blob = json.dumps(fdata, sort_keys=True)
            ir_hash = fdata.get("ir_hash")
            cg_hash = fdata.get("cg_hash")
            for f in db_funcs:
                if f.id is None:
                    continue
                db.upsert_ir_facts(f.id, ir_hash, cg_hash, blob)
                stats.functions_imported += 1

    db.conn.commit()
    return stats


# ---------------------------------------------------------------------------
# Source annotation
# ---------------------------------------------------------------------------

_SAFE_FLAGS = ("wraps_legally", "src_fits_dst", "rhs_nonzero", "amt_in_range")
_ARITH_OPS = {"add", "sub", "mul"}
_SHIFT_OPS = {"shl", "lshr", "ashr"}
_DIV_OPS = {"sdiv", "udiv", "srem", "urem"}
_CAST_OPS = {"sext", "zext", "trunc", "fptosi", "fptoui", "sitofp", "uitofp"}


def _format_int_op(op: dict) -> str | None:
    """Short natural-language hint for the LLM.

    Returns ``"safe"`` if the IR discharged the op, or ``"check ..."``
    naming the hazard. Returns ``None`` for ops we have no opinion on
    (so the line is left unannotated).
    """
    if any(op.get(f) is True for f in _SAFE_FLAGS):
        return "safe"
    kind = op.get("op", "")
    if kind in _ARITH_OPS:
        return "check overflow"
    if kind in _SHIFT_OPS:
        return "check shift"
    if kind in _DIV_OPS:
        return "check divisor"
    if kind in _CAST_OPS:
        return "check cast"
    return None


def annotate_source_with_int_ops(
    source: str,
    line_start: int,
    int_ops: list[dict],
) -> str:
    """Append ``// Facts: <op summary>`` to each source line carrying an int_op.

    *line_start* is the absolute file line of *source*'s first line (i.e.,
    ``Function.line_start``). Sidecar ``loc`` is ``"<file>:<line>"`` —
    only the line number is used; the path is assumed to match.

    Multiple ops on the same line are joined with ``; ``. Lines without
    matching ops pass through unchanged.
    """
    by_line: dict[int, set[str]] = {}
    for op in int_ops:
        loc = op.get("loc", "")
        if not loc or ":" not in loc:
            continue
        try:
            abs_line = int(loc.rsplit(":", 1)[1])
        except ValueError:
            continue
        rel = abs_line - line_start
        if rel < 0:
            continue
        hint = _format_int_op(op)
        if hint is not None:
            by_line.setdefault(rel, set()).add(hint)

    if not by_line:
        return source

    lines = source.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in by_line:
            hints = by_line[i]
            # If any op on this line is unsafe, the line is unsafe overall.
            non_safe = sorted(h for h in hints if h != "safe")
            ann = ", ".join(non_safe) if non_safe else "safe"
            out.append(f"{line.rstrip()}  // {ann}")
        else:
            out.append(line)
    return "\n".join(out)
