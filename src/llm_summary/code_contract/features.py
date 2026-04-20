"""Static-analysis feature gating for per-function property scope.

Decides which subset of {memsafe, memleak, overflow} applies to each
function. A property is in scope iff (a) the function body shows behavior
that could touch it, or (b) any callee published a non-trivial
requires/ensures for it.

Adapter contract:
  - `features_for(func, db=None) -> Features`
  - `property_set(features, callee_summaries) -> list[str]`
  - `attrs_drops(facts) -> set[str]` (post-filter from LLVM attrs)

When the KAMain JSON sidecar is available (`db.get_ir_facts(func.id)`
returns a blob), `features_for` reads the `features` block directly:
load_count / store_count / ptr_params / alloc_count / free_count /
signed_arith_count / div_count / shift_count map to the dataclass
fields. Without sidecar data we fall back to the regex helpers lifted
from `scripts/contract_pipeline.py:262-303`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..models import Function
from .models import PROPERTIES, CodeContractSummary

if TYPE_CHECKING:
    from ..db import SummaryDB

log = logging.getLogger("code_contract.features")


@dataclass
class Features:
    has_deref: bool
    has_alloc: bool
    has_free: bool
    has_index: bool
    has_arith: bool   # +, -, *, ++, --  (signed-overflow risk)
    has_div: bool     # /, %             (division_by_zero / INT_MIN/-1)
    has_shift: bool   # <<, >>           (shift_ub)

    @property
    def memsafe_relevant(self) -> bool:
        return self.has_deref or self.has_index or self.has_alloc or self.has_free

    @property
    def memleak_relevant(self) -> bool:
        return self.has_alloc or self.has_free

    @property
    def overflow_relevant(self) -> bool:
        return self.has_arith or self.has_div or self.has_shift


_DEREF_RE = re.compile(r"(\*\s*\w)|(\w\s*->\s*\w)")
_INDEX_RE = re.compile(r"\w\s*\[")
_ALLOC_RE = re.compile(
    r"\b(malloc|calloc|realloc|alloca|kmalloc|kzalloc|kcalloc|"
    r"vmalloc|vzalloc|xmalloc|xzalloc|ldv_malloc|ldv_calloc|ldv_realloc|"
    r"ldv_zalloc|ldv_xmalloc|ldv_xzalloc)\s*\("
)
_FREE_RE = re.compile(r"\b(free|kfree|vfree|kvfree|ldv_free)\s*\(")
# Arithmetic: any +, -, *, ++, -- not part of a comment/pointer-deref/etc.
# Crude over-approximation is fine — adaptive scoping favors recall.
_ARITH_RE = re.compile(r"(\+\+|--|[^/*+\-]\s*[+\-*]\s*\w)")
_DIV_RE = re.compile(r"\w\s*[/%]\s*\w")
_SHIFT_RE = re.compile(r"<<|>>")


def _extract_features_regex(source: str) -> Features:
    return Features(
        has_deref=bool(_DEREF_RE.search(source)),
        has_alloc=bool(_ALLOC_RE.search(source)),
        has_free=bool(_FREE_RE.search(source)),
        has_index=bool(_INDEX_RE.search(source)),
        has_arith=bool(_ARITH_RE.search(source)),
        has_div=bool(_DIV_RE.search(source)),
        has_shift=bool(_SHIFT_RE.search(source)),
    )


def _features_from_sidecar(feats: dict[str, Any]) -> Features:
    """Map KAMain `features` block → Features dataclass.

    has_deref ← load_count > 0 OR store_count > 0  (any actual memory op)
    has_alloc / has_free ← alloc_count / free_count
    has_index ← ptr_params > 0  (proxy: pointer params imply indexing risk;
        plain loads alone don't, since an indirect call site may load a fn
        pointer with no element access.)
    has_arith ← signed_arith_count > 0 OR sign_changing_cast_count > 0
    has_div / has_shift ← div_count / shift_count
    """
    return Features(
        has_deref=int(feats.get("load_count") or 0) > 0
                  or int(feats.get("store_count") or 0) > 0,
        has_alloc=int(feats.get("alloc_count") or 0) > 0,
        has_free=int(feats.get("free_count") or 0) > 0,
        has_index=int(feats.get("ptr_params") or 0) > 0,
        has_arith=int(feats.get("signed_arith_count") or 0) > 0
                  or int(feats.get("sign_changing_cast_count") or 0) > 0,
        has_div=int(feats.get("div_count") or 0) > 0,
        has_shift=int(feats.get("shift_count") or 0) > 0,
    )


def features_for(func: Function, db: SummaryDB | None = None) -> Features:
    """Return the SA feature set for `func`.

    When `db` is provided AND the KAMain sidecar has a `features` block for
    this function, read from the sidecar. Otherwise fall back to regex over
    `func.llm_source`. Both paths produce the same dataclass shape so callers
    don't need to branch.
    """
    if db is not None and func.id is not None:
        facts = db.get_ir_facts(func.id)
        if facts:
            feats = facts.get("features")
            if isinstance(feats, dict) and feats:
                return _features_from_sidecar(feats)
    log.debug("features_for %s using regex fallback", func.name)
    return _extract_features_regex(func.llm_source)


# LLVM function-attribute → set of properties that can be dropped.
# `readnone` ⇒ no memory effects of any kind ⇒ no memsafe/memleak work.
#   Overflow can still arise from purely-register arithmetic, so we keep it.
# `readonly` ⇒ no writes/allocs/frees ⇒ memleak is impossible. Memsafe and
#   overflow still need to run (loads can deref bad ptrs; loaded values
#   feed arithmetic).
# `writeonly` ⇒ writes only, no reads ⇒ caller's invariants can still be
#   smashed; no property is droppable.
# Map clang `-W<flag>` (the bare flag, no `-W` prefix) to the feature bit
# the warning implies. A frontend warning means the source-level construct
# exists even when constant-folding deletes it from IR (e.g. `INT_MAX + 1`
# becomes a literal, with no `add` instruction left for KAMain to count).
# Without this bump, a warning-only function gets adaptive-skipped because
# `signed_arith_count == 0`.
_WARNING_FEATURE_MAP: dict[str, str] = {
    "integer-overflow": "has_arith",
    "shift-overflow": "has_shift",
    "shift-count-negative": "has_shift",
    "shift-count-overflow": "has_shift",
    "div-by-zero": "has_div",
    "division-by-zero": "has_div",
    "uninitialized": "has_deref",
    "maybe-uninitialized": "has_deref",
    "null-dereference": "has_deref",
    "array-bounds": "has_index",
    "stringop-overflow": "has_index",
    "use-after-free": "has_deref",
}


def bump_features_from_warnings(
    features: Features, scan_issues: list[dict[str, Any]],
) -> Features:
    """Re-enable feature bits implied by clang `-Wall` warnings.

    Constant-folded UB (e.g. `int x = INT_MAX + 1;`) disappears from IR but
    still produces a clang warning. Setting the corresponding bit puts the
    relevant property back into scope so the LLM is asked to assess it.
    """
    if not scan_issues:
        return features
    bumped = {
        "has_deref": features.has_deref,
        "has_alloc": features.has_alloc,
        "has_free": features.has_free,
        "has_index": features.has_index,
        "has_arith": features.has_arith,
        "has_div": features.has_div,
        "has_shift": features.has_shift,
    }
    for issue in scan_issues:
        kind = (issue.get("kind") or "").strip()
        bit = _WARNING_FEATURE_MAP.get(kind)
        if bit is not None:
            bumped[bit] = True
    return Features(**bumped)


def attrs_drops(facts: dict[str, Any] | None) -> set[str]:
    """Return the subset of PROPERTIES that LLVM attrs prove are vacuous.

    Accepts the full facts blob (`db.get_ir_facts(func.id)`); reads the
    `attrs.function` slot which KAMain emits as ``list[str]`` (we also accept
    the doc-spec value-dict shape with a ``"memory"`` key for forward-compat).
    """
    if not facts:
        return set()
    attrs = facts.get("attrs")
    if not isinstance(attrs, dict):
        return set()
    fn_attrs = attrs.get("function")
    if isinstance(fn_attrs, list):
        fn_set = {a for a in fn_attrs if isinstance(a, str)}
    elif isinstance(fn_attrs, dict):
        mem = fn_attrs.get("memory")
        fn_set = {mem} if isinstance(mem, str) else set()
    else:
        return set()
    if "readnone" in fn_set:
        return {"memsafe", "memleak"}
    if "readonly" in fn_set:
        return {"memleak"}
    return set()


def property_set(
    features: Features,
    callee_summaries: list[CodeContractSummary],
    drops: set[str] | None = None,
) -> list[str]:
    """Adaptive scoping: which subset of PROPERTIES applies here.

    A property is in scope if either (a) the function body shows behavior
    that could touch it, or (b) any callee published a non-trivial
    requires/ensures for it (so the caller may need to discharge or
    propagate).

    If `drops` is given (e.g. from `attrs_drops(facts)`), those properties
    are removed from the result regardless of features/callees — LLVM
    attrs are stronger than our heuristics.
    """
    def callee_touches(prop: str) -> bool:
        return any(
            s.has_requires(prop) or s.has_ensures(prop)
            for s in callee_summaries
        )

    props: list[str] = []
    if features.memsafe_relevant or callee_touches("memsafe"):
        props.append("memsafe")
    if features.memleak_relevant or callee_touches("memleak"):
        props.append("memleak")
    if features.overflow_relevant or callee_touches("overflow"):
        props.append("overflow")
    if drops:
        props = [p for p in props if p not in drops]
    # Defensive: PROPERTIES drives the iteration order downstream.
    return [p for p in PROPERTIES if p in props]
