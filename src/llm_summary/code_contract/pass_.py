"""CodeContractPass — SummaryPass adapter for the code-contract pipeline.

Implements the `SummaryPass` Protocol from `driver.py` so the existing
`BottomUpDriver` handles call-graph traversal, SCC iteration, and parallel
execution. One `summarize()` call per function = up to 3 LLM calls (one per
in-scope property), gated by `features.property_set`.

v0 call shape (per per-property call):
  - System message: SYSTEM_PROMPT (with `cache_system=True` for Anthropic).
  - User message: rendered PROPERTY_PROMPT[prop] with `{name}`,
    `{callee_block}`, `{source}` (and `{data_model_note}` for overflow).
  - Response: JSON validated against `PROPERTY_SCHEMA`.

Source-prep (typedef section + macro-annotated source with callee
contracts inlined) is built per-property from `prepare_source` and the
typedef section from `build_type_defs_section`. Caching the source-prep
across properties is deferred (the per-property callee inlining varies
the source slightly, so a clean cache requires either a property-
independent inlining format or a 4th-mode A/B — see plan Open Items).
"""

from __future__ import annotations

import json
import logging
from typing import Any, TextIO

from ..builder.json_utils import extract_json
from ..db import SummaryDB
from ..ir_sidecar import annotate_source_with_ir_facts
from ..llm.base import LLMBackend, make_json_response_format
from ..llm.pool import LLMPool
from ..models import Function
from .features import attrs_drops, features_for, property_set
from .inliner import build_callee_block, ordered_callee_names
from .models import (
    PROPERTIES,
    PROPERTY_SCHEMA,
    CodeContractSummary,
    is_nontrivial,
)
from .prompts import (
    PROPERTY_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
    VERIFY_SCHEMA,
    VERIFY_SYSTEM_PROMPT,
    data_model_note,
)
from .source_prep import build_type_defs_section, prepare_source
from .stdlib import STDLIB_CONTRACTS
from .svcomp_stdlib import SVCOMP_CONTRACTS, svcomp_malloc_overrides

log = logging.getLogger("code_contract.pass")


def seed_stdlib_summaries(*, svcomp: bool) -> dict[str, CodeContractSummary]:
    """Build the initial per-call-graph summaries dict.

    Always seeds libc / POSIX. When `svcomp` is True, also seeds the
    sv-comp `__VERIFIER_*` helpers and overrides `malloc` / `calloc` /
    `realloc` with the never-NULL sv-comp model.
    """
    out: dict[str, CodeContractSummary] = dict(STDLIB_CONTRACTS)
    if svcomp:
        out.update(SVCOMP_CONTRACTS)
        out.update(svcomp_malloc_overrides())
    return out




class CodeContractPass:
    """Adapter that wraps the code-contract per-function pipeline as a
    `SummaryPass` so `BottomUpDriver` can drive it."""

    name = "code_contract"

    def __init__(
        self,
        db: SummaryDB,
        model: str,
        llm: LLMBackend,
        *,
        llm_pool: LLMPool | None = None,
        svcomp: bool = False,
        data_model: str | None = None,
        cache_system: bool = True,
        log_fp: TextIO | None = None,
    ):
        self.db = db
        self.model = model
        self.llm = llm
        self.llm_pool = llm_pool
        self.svcomp = svcomp
        self.data_model_note = data_model_note(data_model)
        self.cache_system = cache_system
        # Per-task LLM log fp (mirrors `scripts/contract_pipeline.py`'s
        # `log_fp`). Caller manages open/close; we just write into it.
        self.log_fp = log_fp
        # `summarizer` slot satisfies the SummaryPass Protocol (driver
        # touches `p.summarizer._stats_lock`, `p.summarizer._progress_*`).
        self.summarizer: Any = _StatsShim()
        # Stdlib seeds — bound at construction so `summarize()` can fall
        # back when the driver doesn't pre-populate them.
        self._stdlib_seeds = seed_stdlib_summaries(svcomp=svcomp)
        # Cached call-graph edges (name → set of callee names). Lazy-built
        # on first `summarize` from `db.get_all_call_edges`.
        self._edges_cache: dict[str, set[str]] | None = None
        # Per-function verify issues, keyed by function name → property →
        # list of issue dicts. Populated by interleaved verify call inside
        # `summarize()`. Eval reads this to compute the safety verdict
        # (mirrors contract_pipeline.py: `predicted_safe = len(issues) == 0`).
        self.issues: dict[str, dict[str, list[dict[str, Any]]]] = {}

    # ── SummaryPass Protocol ────────────────────────────────────────────

    def get_cached(
        self, func_id: int, func: Function,
    ) -> CodeContractSummary | None:
        existing: CodeContractSummary | None = self.db.get_code_contract_summary(func_id)
        if existing and not self.db.needs_code_contract_update(func):
            return existing
        return None

    def summarize(
        self,
        func: Function,
        callee_summaries: dict[str, CodeContractSummary],
        **kwargs: Any,
    ) -> CodeContractSummary:
        """Produce a CodeContractSummary for `func`, then immediately verify
        the body against it (mirrors contract_pipeline.py's compositional
        loop: bottom-up topo, summarize-then-verify per function). Up to 3
        summarize + 3 verify LLM calls per function."""
        # Merge stdlib seeds with the per-call-graph callee summaries the
        # driver passes us. Stdlib seeds are static per-process, so the
        # merge is cheap.
        summaries: dict[str, CodeContractSummary] = dict(self._stdlib_seeds)
        summaries.update(callee_summaries)

        edges = self._get_edges()
        summary = self._summarize_one(func, summaries, edges)
        # Make the just-produced summary visible to the verify call (so the
        # function's own contract goes into the prompt's `own_contract`).
        summaries[func.name] = summary
        func_issues = self._verify_one(func, summary, summaries, edges)
        if func_issues:
            self.issues[func.name] = func_issues
        return summary

    def store(self, func: Function, summary: CodeContractSummary) -> None:
        self.db.store_code_contract_summary(
            func, summary, model_used=self.model,
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _get_edges(self) -> dict[str, set[str]]:
        """Build name-keyed edge map from DB once and cache it."""
        if self._edges_cache is not None:
            return self._edges_cache

        funcs = {f.id: f.name for f in self.db.get_all_functions() if f.id is not None}
        edges: dict[str, set[str]] = {n: set() for n in funcs.values()}
        for edge in self.db.get_all_call_edges():
            caller = funcs.get(edge.caller_id)
            callee = funcs.get(edge.callee_id)
            if caller and callee:
                edges[caller].add(callee)
        self._edges_cache = edges
        return edges

    def _summarize_one(
        self,
        func: Function,
        summaries: dict[str, CodeContractSummary],
        edges: dict[str, set[str]],
    ) -> CodeContractSummary:
        """Per-property summarization for one function."""
        features = features_for(func, db=self.db)
        callee_names = ordered_callee_names(func, edges, summaries)
        callee_summaries_list = [
            summaries[n] for n in callee_names if n in summaries
        ]
        ir_facts = (
            self.db.get_ir_facts(func.id) if func.id is not None else None
        ) or {}
        props = property_set(
            features, callee_summaries_list, drops=attrs_drops(ir_facts),
        )

        summary = CodeContractSummary(function=func.name, properties=props)
        # Seed noreturn from explicit attribute (extern decl with
        # `__attribute__((noreturn))` or `_Noreturn`); LLM may also emit
        # `noreturn: true` per-property below — OR-merge.
        if func.attributes and "noreturn" in func.attributes.lower():
            summary.noreturn = True

        if self.log_fp:
            self.log_fp.write(f"\n\n===== FUNCTION: {func.name} =====\n")
            self.log_fp.write(
                f"--- FEATURES ---\n{features}\n"
                f"--- PROPERTIES ---\n{props}\n"
            )

        if not props:
            if self.log_fp:
                self.log_fp.write(
                    "(no properties in scope; emitting empty summary)\n"
                )
            return summary

        response_format = make_json_response_format(
            PROPERTY_SCHEMA, name="property_summary",
        )
        # Typedef section is property-independent (depends only on identifiers
        # in func.llm_source). Build once per function.
        type_defs = build_type_defs_section(
            self.db, func.llm_source, func.file_path,
        )

        for prop in props:
            callee_block = build_callee_block(
                func, summaries, prop, callee_names,
            )
            source_inlined = prepare_source(func, summaries, edges, prop)
            # Overlay KAMain IR facts: int_ops only for overflow (no signal
            # for the others), but effects + attrs preamble apply to every
            # pass — pointer attrs (nonnull/dereferenceable) tighten the
            # memsafe/memleak reasoning too.
            if ir_facts:
                source_inlined = annotate_source_with_ir_facts(
                    source_inlined, func.line_start, ir_facts,
                    include_int_ops=(prop == "overflow"),
                    include_effects=True,
                    include_attrs_preamble=True,
                )
            # Prepend the typedef section so the LLM can resolve struct
            # field offsets and typedef'd integer widths.
            if type_defs:
                source_inlined = type_defs + "=== SOURCE ===\n" + source_inlined

            fmt_kwargs: dict[str, Any] = {
                "name": func.name,
                "callee_block": callee_block,
                "source": source_inlined,
            }
            if prop == "overflow":
                fmt_kwargs["data_model_note"] = self.data_model_note
            prompt = PROPERTY_PROMPT[prop].format(**fmt_kwargs)

            if self.log_fp:
                self.log_fp.write(f"\n--- USER ({prop}) ---\n{prompt}\n")

            # Driver-level parallelism uses LLMPool.submit at the function
            # level; per-property calls within one function are serial and
            # share `self.llm`. Backends are expected to be thread-safe.
            resp = self.llm.complete_with_metadata(
                prompt,
                system=SYSTEM_PROMPT,
                cache_system=self.cache_system,
                response_format=response_format,
            )

            if self.log_fp:
                self.log_fp.write(
                    f"--- RESPONSE ({prop}) ---\n{resp.content}\n"
                )
                self.log_fp.flush()

            try:
                data = extract_json(resp.content)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(
                    f"{func.name}/{prop}: invalid JSON: {e}\n"
                    f"{resp.content[:200]}"
                ) from e

            reqs = list(data.get("requires") or [])
            summary.requires[prop] = reqs
            summary.ensures[prop] = list(data.get("ensures") or [])
            summary.modifies[prop] = list(data.get("modifies") or [])
            summary.notes[prop] = str(data.get("notes") or "")
            # Default origin: each requires entry is "local". Future work:
            # detect verbatim callee discharge and set origin = "<callee>:<idx>".
            summary.origin[prop] = ["local"] * len(reqs)
            if bool(data.get("noreturn", False)):
                summary.noreturn = True

        return summary


    def _format_own_contract(
        self, summary: CodeContractSummary, prop: str,
    ) -> str:
        """Render `summary`'s contract for `prop` as the `{own_contract}`
        block of VERIFY_PROMPT. Mirrors contract_pipeline.py:1223."""
        reqs = [r for r in summary.requires.get(prop, []) if is_nontrivial(r)]
        ens = [e for e in summary.ensures.get(prop, []) if is_nontrivial(e)]
        mods = summary.modifies.get(prop, [])
        lines = [
            f"  requires[{prop}]: " + ("; ".join(reqs) if reqs else "true"),
            f"  ensures[{prop}]:  "
            + ("; ".join(ens) if ens else "(no observable effect)"),
        ]
        if mods:
            lines.append("  modifies: " + ", ".join(mods))
        return "\n".join(lines)

    def _verify_one(
        self,
        func: Function,
        summary: CodeContractSummary,
        summaries: dict[str, CodeContractSummary],
        edges: dict[str, set[str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Per-property verification for one function. Mirrors
        contract_pipeline.py:1236 `_verify_one`. Returns issues_by_prop;
        empty dict means body is safe under its published contract."""
        issues_by_prop: dict[str, list[dict[str, Any]]] = {}
        if not summary.properties:
            return issues_by_prop

        callee_names = ordered_callee_names(func, edges, summaries)
        ir_facts = (
            self.db.get_ir_facts(func.id) if func.id is not None else None
        ) or {}
        response_format = make_json_response_format(
            VERIFY_SCHEMA, name="verify",
        )

        if self.log_fp:
            self.log_fp.write(f"\n--- VERIFY ({func.name}) ---\n")

        from .source_prep import prepare_source

        for prop in summary.properties:
            callee_block = build_callee_block(
                func, summaries, prop, callee_names,
            )
            source_inlined = prepare_source(func, summaries, edges, prop)
            if ir_facts:
                source_inlined = annotate_source_with_ir_facts(
                    source_inlined, func.line_start, ir_facts,
                    include_int_ops=(prop == "overflow"),
                    include_effects=True,
                    include_attrs_preamble=True,
                )
            own_contract = self._format_own_contract(summary, prop)
            fmt_kwargs: dict[str, Any] = {
                "name": func.name,
                "own_contract": own_contract,
                "callee_block": callee_block,
                "source": source_inlined,
            }
            if prop == "overflow":
                fmt_kwargs["data_model_note"] = self.data_model_note
            prompt = VERIFY_PROMPT[prop].format(**fmt_kwargs)

            if self.log_fp:
                self.log_fp.write(f"\n--- VERIFY USER ({prop}) ---\n{prompt}\n")

            resp = self.llm.complete_with_metadata(
                prompt,
                system=VERIFY_SYSTEM_PROMPT,
                cache_system=self.cache_system,
                response_format=response_format,
            )

            if self.log_fp:
                self.log_fp.write(
                    f"\n--- VERIFY RESPONSE ({prop}) ---\n{resp.content}\n"
                )
                self.log_fp.flush()

            try:
                data = extract_json(resp.content)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(
                    f"{func.name}/{prop} verify: invalid JSON: {e}\n"
                    f"{resp.content[:200]}"
                ) from e

            raw_issues = data.get("issues") or []
            kept: list[dict[str, Any]] = []
            for it in raw_issues:
                if not isinstance(it, dict):
                    continue
                kind = str(it.get("kind") or "").strip()
                desc = str(it.get("description") or "").strip()
                if not kind or not desc:
                    continue
                kept.append({
                    "kind": kind,
                    "line": it.get("line"),
                    "description": desc,
                })
            if kept:
                issues_by_prop[prop] = kept

        return issues_by_prop


class _StatsShim:
    """Minimal stand-in for the `summarizer._stats_lock` / `_progress_*`
    attributes the BottomUpDriver pokes on each pass. We don't track stats
    in the new pipeline (yet); a no-op shim keeps the driver happy."""

    def __init__(self) -> None:
        import threading
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {"cache_hits": 0}
        self._progress_current = 0
        self._progress_total = 0


__all__ = [
    "CodeContractPass",
    "PROPERTIES",
    "seed_stdlib_summaries",
]
