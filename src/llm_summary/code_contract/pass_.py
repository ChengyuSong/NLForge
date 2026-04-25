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
from typing import Any

from ..builder.json_utils import extract_json
from ..db import SummaryDB
from ..ir_sidecar import annotate_source_with_ir_facts
from ..llm.base import LLMBackend, make_json_response_format
from ..llm.pool import LLMPool
from ..models import Function, SafetyIssue, VerificationSummary
from .features import (
    MAX_INLINE_BODY_LINES,
    attrs_drops,
    bump_features_from_warnings,
    features_for,
    is_inline_body,
    property_set,
    relevant_warnings_for,
)
from .inliner import build_callee_block, build_inline_body, ordered_callee_names
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
from .source_prep import (
    build_globals_section,
    build_type_defs_section,
    prepare_source,
)
from .stdlib import STDLIB_CONTRACTS
from .struggle import compute as compute_struggle
from .svcomp_stdlib import SVCOMP_CONTRACTS, svcomp_malloc_overrides

log = logging.getLogger("code_contract.pass")

_MALFORMED_JSON_MSG = (
    "Your previous response was not valid JSON. "
    "Please output only a valid JSON object matching the required schema, "
    "with no markdown fences or extra text."
)

# Composite struggle score above which we re-run the property with the
# Claude default model. Calibrated against zlib (302 pairs): the worst
# orchestrators (deflate_slow, inflate_fast, gz_*) land at ~6+ while clean
# leaves stay <2.
_RETRY_THRESHOLD: float = 5.0


def _format_scan_issues(
    scan_issues: list[dict[str, Any]], prop: str | None = None,
) -> str:
    """Render frontend warnings as a `// ... ` comment block.

    Empty input → empty string. When `prop` is given, drops warnings whose
    kind is not relevant to that property (e.g. `unused-but-set-variable`
    is hidden from both memsafe and overflow passes — only useful as
    feature-bit input via `bump_features_from_warnings`).
    """
    if not scan_issues:
        return ""
    issues = (
        relevant_warnings_for(prop, scan_issues)
        if prop is not None else scan_issues
    )
    if not issues:
        return ""
    lines = ["// FRONTEND WARNINGS (clang -Wall) — assess feasibility:"]
    for issue in issues:
        line = issue.get("line")
        kind = (issue.get("kind") or "").strip()
        msg = (issue.get("message") or "").strip()
        loc = f"line {line}" if line else "line ?"
        lines.append(f"//   {loc}: {kind}: {msg}")
    return "\n".join(lines) + "\n"


def seed_stdlib_summaries(*, svcomp: bool) -> dict[str, CodeContractSummary]:
    """Build the initial per-call-graph summaries dict.

    Loads from the global stdlib cache (~/.llm-summary/stdlib_cache.db) when
    available, then overlays hardcoded `STDLIB_CONTRACTS` (always wins) and
    optionally the sv-comp set. Cache failure is non-fatal — empty cache =
    legacy behaviour (hardcoded only).
    """
    out: dict[str, CodeContractSummary] = {}
    try:
        from ..stdlib_cache import StdlibCache
        cache = StdlibCache()
        for name in cache.list_names():
            entry = cache.get(name)
            if entry and entry.code_contract_json:
                out[name] = CodeContractSummary.from_dict(
                    json.loads(entry.code_contract_json)
                )
        cache.close()
    except Exception:
        pass
    # Hardcoded contracts always win (mirror seed_builtins force=True).
    out.update(STDLIB_CONTRACTS)
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
        log_file: str | None = None,
        verbose: bool = False,
        verify_only: bool = False,
    ):
        self.db = db
        self.model = model
        self.llm = llm
        self.llm_pool = llm_pool
        self.svcomp = svcomp
        self.data_model_note = data_model_note(data_model)
        self.cache_system = cache_system
        self.verbose = verbose
        self.verify_only = verify_only
        # Path to the LLM log file. Opened lazily in append mode on each
        # write (matches the legacy summarizer pattern; no fp lifecycle
        # for the caller to manage). None disables logging.
        self.log_file = log_file
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
        # Running totals across every `complete_with_metadata` call this
        # pass made — mirrors `contract_pipeline.py:run_one_task`'s
        # local accumulators. Drivers/eval scripts read these after
        # `driver.run` to fill `TaskResult.{llm_calls,input_tokens,output_tokens}`.
        self.calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        # Per-function struggle metadata, populated in `_summarize_one` and
        # consumed by `store()` so it can land on the DB row. Schema:
        # {func_name: {"scores": {prop: float}, "max": float,
        #              "retried": bool, "retry_model": str | None}}
        self.struggle: dict[str, dict[str, Any]] = {}
        # Lazy-initialized Claude backend used to re-run properties whose
        # primary score crosses `_RETRY_THRESHOLD`. `_disabled` short-circuits
        # repeated attempts after the first failure (no creds, same backend
        # as primary, etc.).
        self._retry_llm: LLMBackend | None = None
        self._retry_llm_disabled: bool = False
        self.struggle_retries: int = 0

    def _log(self, text: str) -> None:
        """Append `text` to the log file, no-op if logging disabled.

        Lazy open per call mirrors the legacy summarizer pattern — the
        file is created on first write and never held open across calls,
        so callers don't need to manage an fp lifecycle.
        """
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

    # ── SummaryPass Protocol ────────────────────────────────────────────

    def get_cached(
        self, func_id: int, func: Function,
    ) -> CodeContractSummary | None:
        existing: CodeContractSummary | None = self.db.get_code_contract_summary(func_id)
        if not existing:
            return None
        if self.verify_only:
            # No contract properties — nothing to verify; skip.
            if not existing.properties:
                return existing
            # Has a verification_summary already — skip.
            vs = self.db.get_verification_summary_by_function_id(func_id)
            if vs is not None:
                return existing
            # Contract exists but not yet verified — force re-run.
            return None
        if not self.db.needs_code_contract_update(func):
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
        if self.verbose:
            cur = self.summarizer._progress_current
            tot = self.summarizer._progress_total
            label = "verify-only" if self.verify_only else "code-contract"
            if tot > 0:
                print(f"  ({cur}/{tot}) {label}: {func.name}", flush=True)
            else:
                print(f"  {label}: {func.name}", flush=True)
        # Merge stdlib seeds with the per-call-graph callee summaries the
        # driver passes us. Stdlib seeds are static per-process, so the
        # merge is cheap.
        summaries: dict[str, CodeContractSummary] = dict(self._stdlib_seeds)
        summaries.update(callee_summaries)

        edges = self._get_edges()

        if self.verify_only:
            summary = self.db.get_code_contract_summary(
                func.id,  # type: ignore[arg-type]
            )
            assert summary is not None, (
                f"verify-only: no cached contract for {func.name}"
            )
        else:
            in_scc = bool(kwargs.get("in_scc"))
            summary = self._summarize_one(func, summaries, edges, in_scc=in_scc)
        # Make the just-produced summary visible to the verify call (so the
        # function's own contract goes into the prompt's `own_contract`).
        summaries[func.name] = summary
        func_issues = self._verify_one(func, summary, summaries, edges)
        if func_issues:
            self.issues[func.name] = func_issues
        return summary

    def store(self, func: Function, summary: CodeContractSummary) -> None:
        sd = self.struggle.get(func.name, {})
        self.db.store_code_contract_summary(
            func, summary, model_used=self.model,
            struggle_scores=sd.get("scores"),
            struggle_max=float(sd.get("max", 0.0)),
            retried=bool(sd.get("retried", False)),
            retry_model=sd.get("retry_model"),
        )
        func_issues = self.issues.get(func.name, {})
        safety_issues: list[SafetyIssue] = []
        for prop, issue_list in func_issues.items():
            for it in issue_list:
                safety_issues.append(SafetyIssue(
                    location=f"line {it.get('line', '?')}",
                    issue_kind=it["kind"],
                    description=f"[{prop}] {it['description']}",
                    severity="high",
                ))
        vs = VerificationSummary(
            function_name=func.name,
            issues=safety_issues,
        )
        self.db.upsert_verification_summary(func, vs, model_used=self.model)

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
        in_scc: bool = False,
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
        scan_issues = (
            self.db.get_scan_issues(func.id) if func.id is not None else []
        )
        # Re-enable any feature bits the warnings imply — constant-folded UB
        # has no IR signature but still warrants a property pass.
        features = bump_features_from_warnings(features, scan_issues)
        props = property_set(
            features, callee_summaries_list, drops=attrs_drops(ir_facts),
        )

        summary = CodeContractSummary(function=func.name, properties=props)
        # Seed noreturn from explicit attribute (extern decl with
        # `__attribute__((noreturn))` or `_Noreturn`); LLM may also emit
        # `noreturn: true` per-property below — OR-merge.
        if func.attributes and "noreturn" in func.attributes.lower():
            summary.noreturn = True

        # Inline-body shortcut: small wrappers cost more to summarize than
        # they save. Paste the (transitively-expanded) body at every
        # callsite instead. Skip when:
        #   - the function has no project-internal caller (entry points
        #     like `main` would be lost — no caller's verify pass would
        #     ever see the inlined body),
        #   - the function is inside a multi-member SCC (driver flag —
        #     would create a cycle of unresolved bodies), OR
        #   - the expansion blows past the cap.
        # In all three cases we fall through to normal per-property
        # summarization.
        has_project_caller = any(
            func.name in callees for callees in edges.values()
        )
        if has_project_caller and not in_scc and is_inline_body(func, props):
            expanded = build_inline_body(func, summaries)
            if len(expanded.splitlines()) <= MAX_INLINE_BODY_LINES:
                summary.inline_body = expanded
                summary.properties = []  # no per-property contract
                self._log(
                    f"\n\n===== FUNCTION: {func.name} (INLINE BODY) =====\n"
                    f"--- BODY ({len(expanded.splitlines())} lines) ---\n"
                    f"{expanded}\n"
                )
                return summary

        self._log(
            f"\n\n===== FUNCTION: {func.name} =====\n"
            f"--- FEATURES ---\n{features}\n"
            f"--- PROPERTIES ---\n{props}\n"
        )

        if not props:
            self._log("(no properties in scope; emitting empty summary)\n")
            return summary

        response_format = make_json_response_format(
            PROPERTY_SCHEMA, name="property_summary",
        )
        # Typedef + globals sections are property-independent. Build once.
        type_defs = build_type_defs_section(
            self.db, func.llm_source, func.file_path,
        )
        globals_section = build_globals_section(ir_facts)

        scores_by_prop: dict[str, float] = {}
        retry_model_used: str | None = None

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
            warnings_section = _format_scan_issues(scan_issues, prop)
            if warnings_section:
                source_inlined = warnings_section + source_inlined
            preamble = type_defs + globals_section
            if preamble:
                source_inlined = preamble + "=== SOURCE ===\n" + source_inlined

            fmt_kwargs: dict[str, Any] = {
                "name": func.name,
                "callee_block": callee_block,
                "source": source_inlined,
            }
            if prop == "overflow":
                fmt_kwargs["data_model_note"] = self.data_model_note
            prompt = PROPERTY_PROMPT[prop].format(**fmt_kwargs)
            # Struggle denominator: only the project-specific context the
            # model has to engage with. Excludes the static PROPERTY_PROMPT
            # template (~10K chars for memsafe/overflow) and SYSTEM_PROMPT
            # — otherwise every leaf scores `under * 4 ≈ 4` baseline before
            # any real signal. Calibrated against zlib body+callees.
            ctx_chars = len(callee_block) + len(source_inlined)

            self._log(f"\n--- USER ({prop}) ---\n{prompt}\n")

            # Driver-level parallelism uses LLMPool.submit at the function
            # level; per-property calls within one function are serial and
            # share `self.llm`. Backends are expected to be thread-safe.
            data = self._call_property_llm(
                self.llm, func.name, prop, prompt, response_format,
            )
            if data is None:
                continue

            chosen_data = data
            chosen_score = self._struggle_for(data, prop, ctx_chars)

            if chosen_score > _RETRY_THRESHOLD:
                retry_data, retry_score = self._maybe_retry(
                    func.name, prop, prompt, response_format,
                    ctx_chars, chosen_score,
                )
                if retry_data is not None and retry_score < chosen_score:
                    chosen_data = retry_data
                    chosen_score = retry_score
                    retry_llm = self._retry_llm
                    if retry_llm is not None:
                        retry_model_used = retry_llm.model
                    self.struggle_retries += 1

            scores_by_prop[prop] = chosen_score

            reqs = list(chosen_data.get("requires") or [])
            summary.analysis[prop] = str(chosen_data.get("analysis") or "")
            summary.requires[prop] = reqs
            summary.ensures[prop] = list(chosen_data.get("ensures") or [])
            summary.modifies[prop] = list(chosen_data.get("modifies") or [])
            summary.notes[prop] = str(chosen_data.get("notes") or "")
            # Default origin: each requires entry is "local". Future work:
            # detect verbatim callee discharge and set origin = "<callee>:<idx>".
            summary.origin[prop] = ["local"] * len(reqs)
            conf = str(chosen_data.get("confidence") or "").strip().lower()
            summary.confidence[prop] = (
                conf if conf in ("high", "medium", "low") else ""
            )
            if bool(chosen_data.get("noreturn", False)):
                summary.noreturn = True

        if scores_by_prop:
            self.struggle[func.name] = {
                "scores": scores_by_prop,
                "max": max(scores_by_prop.values()),
                "retried": retry_model_used is not None,
                "retry_model": retry_model_used,
            }

        return summary

    def _call_property_llm(
        self,
        llm: LLMBackend,
        func_name: str,
        prop: str,
        prompt: str,
        response_format: dict[str, Any],
        *,
        label: str = "",
    ) -> dict[str, Any] | None:
        """One per-property LLM call with one malformed-JSON retry.

        Returns the parsed JSON dict, or `None` if the response was
        malformed both times. Tokens accumulate into `self.{calls,
        input_tokens, output_tokens}`. `label` distinguishes retry
        attempts in the log.
        """
        tag = f"{prop}{label}" if label else prop
        try:
            resp = llm.complete_with_metadata(
                prompt,
                system=SYSTEM_PROMPT,
                cache_system=self.cache_system,
                response_format=response_format,
            )
        except Exception as e:
            log.warning("%s/%s: LLM call failed: %s", func_name, tag, e)
            return None
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self._log(f"--- RESPONSE ({tag}) ---\n{resp.content}\n")

        try:
            return extract_json(resp.content)
        except (json.JSONDecodeError, ValueError):
            log.warning("%s/%s: malformed JSON, retrying", func_name, tag)

        retry_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": _MALFORMED_JSON_MSG},
        ]
        try:
            resp2 = llm.complete_messages_with_metadata(
                retry_messages,
                system=SYSTEM_PROMPT,
                cache_system=self.cache_system,
                response_format=response_format,
            )
        except Exception as e:
            log.warning(
                "%s/%s: retry LLM call failed: %s", func_name, tag, e,
            )
            return None
        self.calls += 1
        self.input_tokens += resp2.input_tokens
        self.output_tokens += resp2.output_tokens
        self._log(f"--- RETRY RESPONSE ({tag}) ---\n{resp2.content}\n")
        try:
            return extract_json(resp2.content)
        except (json.JSONDecodeError, ValueError):
            log.warning(
                "%s/%s: malformed JSON after retry, skipping property",
                func_name, tag,
            )
            return None

    def _struggle_for(
        self, data: dict[str, Any], prop: str, ctx_chars: int,
    ) -> float:
        """Compute the composite struggle score for one parsed response."""
        reqs = [
            r for r in (data.get("requires") or []) if isinstance(r, str)
        ]
        return compute_struggle(
            str(data.get("analysis") or ""), reqs, prop, ctx_chars,
        ).score

    def _get_retry_llm(self) -> LLMBackend | None:
        """Lazily construct the Claude default backend used for retries.

        Returns `None` if retry is disabled (already tried, no creds, or
        primary already uses the same Claude default).
        """
        if self._retry_llm is not None:
            return self._retry_llm
        if self._retry_llm_disabled:
            return None
        try:
            from ..llm.claude import ClaudeBackend
            candidate = ClaudeBackend()
        except Exception as e:
            log.warning("retry: failed to construct Claude backend: %s", e)
            self._retry_llm_disabled = True
            return None
        if (
            isinstance(self.llm, ClaudeBackend)
            and self.llm.model == candidate.model
        ):
            log.info(
                "retry: primary already uses %s; disabling retry",
                candidate.model,
            )
            self._retry_llm_disabled = True
            return None
        self._retry_llm = candidate
        return self._retry_llm

    def _maybe_retry(
        self,
        func_name: str,
        prop: str,
        prompt: str,
        response_format: dict[str, Any],
        ctx_chars: int,
        primary_score: float,
    ) -> tuple[dict[str, Any] | None, float]:
        """Retry one property with Claude default; return (data, score) or
        (None, +inf) if retry is unavailable / failed.

        Caller compares the returned score with `primary_score` and decides
        whether to swap. We score the retry on the same prompt so the
        comparison is apples-to-apples.
        """
        retry_llm = self._get_retry_llm()
        if retry_llm is None:
            return None, float("inf")
        log.info(
            "%s/%s: struggle=%.2f, retrying with %s",
            func_name, prop, primary_score, retry_llm.model,
        )
        self._log(
            f"\n--- STRUGGLE RETRY ({prop}) ---\n"
            f"primary={primary_score:.2f} > {_RETRY_THRESHOLD}; "
            f"retrying with {retry_llm.model}\n"
        )
        retry_data = self._call_property_llm(
            retry_llm, func_name, prop, prompt, response_format,
            label=":retry",
        )
        if retry_data is None:
            return None, float("inf")
        retry_score = self._struggle_for(retry_data, prop, ctx_chars)
        if retry_score < primary_score:
            log.info(
                "%s/%s: retry improved %.2f -> %.2f",
                func_name, prop, primary_score, retry_score,
            )
            self._log(
                f"--- RETRY ACCEPTED ({prop}) ---\n"
                f"score: {primary_score:.2f} -> {retry_score:.2f}\n"
            )
        else:
            log.info(
                "%s/%s: retry didn't improve (%.2f); keeping primary",
                func_name, prop, retry_score,
            )
            self._log(
                f"--- RETRY REJECTED ({prop}) ---\n"
                f"score: {primary_score:.2f} -> {retry_score:.2f}\n"
            )
        return retry_data, retry_score

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
        if summary.inline_body:
            # Inline-body functions made no claims — verification is
            # delegated to whichever caller pastes the body.
            return issues_by_prop
        if not summary.properties:
            return issues_by_prop

        callee_names = ordered_callee_names(func, edges, summaries)
        ir_facts = (
            self.db.get_ir_facts(func.id) if func.id is not None else None
        ) or {}
        scan_issues = (
            self.db.get_scan_issues(func.id) if func.id is not None else []
        )
        response_format = make_json_response_format(
            VERIFY_SCHEMA, name="verify",
        )

        type_defs = build_type_defs_section(
            self.db, func.llm_source, func.file_path,
        )
        globals_section = build_globals_section(ir_facts)

        self._log(f"\n--- VERIFY ({func.name}) ---\n")

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
            warnings_section = _format_scan_issues(scan_issues, prop)
            if warnings_section:
                source_inlined = warnings_section + source_inlined
            preamble = type_defs + globals_section
            if preamble:
                source_inlined = preamble + "=== SOURCE ===\n" + source_inlined
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

            self._log(f"\n--- VERIFY USER ({prop}) ---\n{prompt}\n")

            resp = self.llm.complete_with_metadata(
                prompt,
                system=VERIFY_SYSTEM_PROMPT,
                cache_system=self.cache_system,
                response_format=response_format,
            )
            self.calls += 1
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens

            self._log(
                f"\n--- VERIFY RESPONSE ({prop}) ---\n{resp.content}\n"
            )

            try:
                data = extract_json(resp.content)
            except (json.JSONDecodeError, ValueError):
                log.warning("%s/%s verify: malformed JSON, retrying", func.name, prop)
                retry_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": _MALFORMED_JSON_MSG},
                ]
                resp = self.llm.complete_messages_with_metadata(
                    retry_messages,
                    system=VERIFY_SYSTEM_PROMPT,
                    cache_system=self.cache_system,
                    response_format=response_format,
                )
                self.calls += 1
                self.input_tokens += resp.input_tokens
                self.output_tokens += resp.output_tokens
                self._log(
                    f"\n--- VERIFY RETRY RESPONSE ({prop}) ---\n{resp.content}\n"
                )
                try:
                    data = extract_json(resp.content)
                except (json.JSONDecodeError, ValueError):
                    log.warning(
                        "%s/%s verify: malformed JSON after retry, skipping property",
                        func.name, prop,
                    )
                    continue

            raw_issues = data.get("issues") or []
            kept: list[dict[str, Any]] = []
            for it in raw_issues:
                if not isinstance(it, dict):
                    continue
                if not it.get("is_ub", True):
                    continue
                kind = str(it.get("kind") or "").strip()
                desc = str(
                    it.get("analysis") or it.get("description") or ""
                ).strip()
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
