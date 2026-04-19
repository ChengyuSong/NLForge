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
          "effects": [...],         # alloc/free/read/write/call/return/
                                    # assume/atomicrmw/cmpxchg
          "branches": [...],
          "ranges": [...],
          "int_ops": [...],
          "features": {...},
          "attrs": {                # LLVM-inferred attributes
            "function": {...},      # memory, willreturn, ...
            "params":   [{...}],    # nocapture, nonnull, dereferenceable(N), ...
            "ret":      {...},
            "callsites": {"csN": {...}}
          }
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
from collections.abc import Callable
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
    """Append ``// <op summary>`` to each source line carrying an int_op.

    Back-compat wrapper — equivalent to
    ``annotate_source_with_ir_facts(source, line_start, {"int_ops": int_ops})``.
    """
    return annotate_source_with_ir_facts(
        source, line_start, {"int_ops": int_ops},
    )


# ---------------------------------------------------------------------------
# Memory-effect hints (load/store/atomicrmw/cmpxchg)
# ---------------------------------------------------------------------------

# Atomic orderings worth surfacing: anything beyond plain unordered/monotonic
# carries memory-model implications the LLM should see.
_ATOMIC_NOTABLE = {"acquire", "release", "acq_rel", "seq_cst"}


def _format_load_md(load_md: dict) -> list[str]:
    parts: list[str] = []
    if load_md.get("nonnull") is True:
        parts.append("nonnull")
    deref = load_md.get("dereferenceable")
    if isinstance(deref, int) and deref > 0:
        parts.append(f"dereferenceable({deref})")
    rng = load_md.get("range")
    if isinstance(rng, list) and rng:
        # Compact range: e.g. range[0,256] for a single interval.
        try:
            if len(rng) == 1 and len(rng[0]) == 2:
                lo, hi = rng[0]
                parts.append(f"range[{lo},{hi}]")
            else:
                parts.append(f"range×{len(rng)}")
        except (TypeError, ValueError):
            pass
    if load_md.get("invariant") is True:
        parts.append("invariant")
    return parts


def _format_effect_hint(eff: dict) -> str | None:
    """Per-effect hint string, or None if the effect carries no notable signal.

    Default-aligned, non-volatile, non-atomic load/store with no ``load_md``
    produces nothing — we don't want to annotate every memory op.
    """
    kind = eff.get("kind")
    if kind not in {"read", "write", "atomicrmw", "cmpxchg"}:
        return None

    parts: list[str] = []

    if eff.get("volatile") is True:
        parts.append("volatile")

    # Atomic ordering — single value for read/write/atomicrmw, success+failure
    # for cmpxchg. Skip plain monotonic/unordered as low-signal.
    ordering = eff.get("atomic")
    if isinstance(ordering, str) and ordering in _ATOMIC_NOTABLE:
        parts.append(f"atomic {ordering}")
    if kind == "cmpxchg":
        succ = eff.get("atomic_success")
        fail = eff.get("atomic_failure")
        if isinstance(succ, str) and succ in _ATOMIC_NOTABLE:
            parts.append(f"cmpxchg {succ}/{fail or 'monotonic'}")

    if kind == "atomicrmw":
        op = eff.get("op")
        if isinstance(op, str):
            parts.append(f"rmw {op}")

    if kind == "read":
        load_md = eff.get("load_md")
        if isinstance(load_md, dict):
            parts.extend(_format_load_md(load_md))

    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# LLVM attribute formatting
# ---------------------------------------------------------------------------

# Function-scope attrs that imply skipping entire property passes.
_FUNC_MEMORY_NONE = "readnone"
_FUNC_MEMORY_READONLY = "readonly"
_FUNC_MEMORY_WRITEONLY = "writeonly"

# Low-signal LLVM attrs that don't help the LLM reason about behavior —
# strip them from the preamble so the signal-to-noise stays high.
_FN_ATTR_NOISE = frozenset({
    "noinline", "optnone", "uwtable", "alwaysinline", "minsize",
    "optsize", "ssp", "sspreq", "sspstrong", "stackprotect", "naked",
    "no-builtins", "norecurse", "nofree",
})


def _scope_attrs_to_strs(scope: object) -> list[str]:
    """Normalize a scope's attrs to a flat ``list[str]``.

    KAMain emits flat ``list[str]`` per scope. The doc spec used a
    value-dict (``{"memory": "readonly", "dereferenceable": 32, ...}``);
    we accept both — dict entries collapse to either the bare key (for
    True flags) or ``key(value)`` strings.
    """
    if isinstance(scope, list):
        return [a for a in scope if isinstance(a, str)]
    if isinstance(scope, dict):
        out: list[str] = []
        for k, v in scope.items():
            if k == "i":  # param index, not an attribute
                continue
            if v is True:
                out.append(k)
            elif isinstance(v, int) and v > 0:
                out.append(f"{k}({v})")
            elif isinstance(v, str):
                out.append(v)
        return out
    return []


def _filter_signal(attrs: list[str]) -> list[str]:
    return [a for a in attrs if a not in _FN_ATTR_NOISE]


def format_attrs_preamble(attrs: dict | None) -> str | None:
    """Build a multi-line `// LLVM: ...` preamble from the attrs block.

    Returns None when *attrs* is empty or contributes nothing notable
    after noise filtering. KAMain emission shape (flat string lists per
    scope; ``return`` key for return value) and the doc-spec value-dict
    shape are both accepted.
    """
    if not attrs:
        return None
    lines: list[str] = []

    fn_strs = _filter_signal(_scope_attrs_to_strs(attrs.get("function")))
    if fn_strs:
        lines.append(f"// LLVM fn: {' '.join(fn_strs)}")

    params = attrs.get("params")
    if isinstance(params, list):
        for i, p in enumerate(params):
            p_strs = _filter_signal(_scope_attrs_to_strs(p))
            if p_strs:
                # Allow per-param dict to override the index (doc spec).
                idx = p.get("i", i) if isinstance(p, dict) else i
                lines.append(f"// LLVM arg{idx}: {' '.join(p_strs)}")

    # KAMain uses "return"; doc spec used "ret". Accept both.
    ret_block = attrs.get("return")
    if ret_block is None:
        ret_block = attrs.get("ret")
    ret_strs = _filter_signal(_scope_attrs_to_strs(ret_block))
    if ret_strs:
        lines.append(f"// LLVM ret: {' '.join(ret_strs)}")

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Combined source annotation
# ---------------------------------------------------------------------------

def _collect_line_hints(
    items: list[dict],
    line_start: int,
    formatter: Callable[[dict], str | None],
) -> dict[int, set[str]]:
    by_line: dict[int, set[str]] = {}
    for item in items:
        loc = item.get("loc", "")
        if not loc or ":" not in loc:
            continue
        try:
            abs_line = int(loc.rsplit(":", 1)[1])
        except ValueError:
            continue
        rel = abs_line - line_start
        if rel < 0:
            continue
        hint = formatter(item)
        if hint is not None:
            by_line.setdefault(rel, set()).add(hint)
    return by_line


def annotate_source_with_ir_facts(
    source: str,
    line_start: int,
    ir_facts: dict,
    *,
    include_int_ops: bool = True,
    include_effects: bool = False,
    include_attrs_preamble: bool = False,
) -> str:
    """Inline IR-derived hints into *source*.

    - ``include_int_ops``: per-line ``// safe`` / ``// check overflow`` etc.
    - ``include_effects``: per-line ``// volatile``, ``// atomic seq_cst``,
      ``// nonnull dereferenceable(N)`` for memory-touching effects.
    - ``include_attrs_preamble``: prepend a few ``// LLVM ...`` lines from
      the ``attrs`` block. Does not shift source line numbers since the
      preamble sits above the original first line.

    Multiple hints on the same line collapse with ``, `` and the int_op
    "safe" rule still applies (any non-safe int_op overrides "safe").
    """
    by_line: dict[int, set[str]] = {}

    if include_int_ops:
        int_ops = ir_facts.get("int_ops") or []
        merged = _collect_line_hints(int_ops, line_start, _format_int_op)
        for k, v in merged.items():
            by_line.setdefault(k, set()).update(v)

    if include_effects:
        effects = ir_facts.get("effects") or []
        merged = _collect_line_hints(effects, line_start, _format_effect_hint)
        for k, v in merged.items():
            by_line.setdefault(k, set()).update(v)

    if by_line:
        lines = source.split("\n")
        out: list[str] = []
        for i, line in enumerate(lines):
            hints = by_line.get(i)
            if not hints:
                out.append(line)
                continue
            non_safe = sorted(h for h in hints if h != "safe")
            ann = ", ".join(non_safe) if non_safe else "safe"
            out.append(f"{line.rstrip()}  // {ann}")
        body = "\n".join(out)
    else:
        body = source

    if include_attrs_preamble:
        preamble = format_attrs_preamble(ir_facts.get("attrs"))
        if preamble:
            body = f"{preamble}\n{body}"

    return body
