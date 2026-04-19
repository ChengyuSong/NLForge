"""Base class for LLM-based summarizers with shared stats and logging."""

from __future__ import annotations

import threading
from typing import Any

from llm_summary.db import SummaryDB
from llm_summary.llm.base import LLMBackend, LLMResponse
from llm_summary.models import Function


class BaseSummarizer:
    """Common infrastructure for all summarizer classes.

    Provides:
    - Thread-safe stats accumulation (llm_calls, tokens, cache metrics)
    - ``record_response()`` to update stats from an ``LLMResponse``
    - ``_log_interaction()`` for optional prompt/response logging
    - Progress tracking (``_progress_current``, ``_progress_total``)
    """

    # Subclasses can override to add pass-specific stats keys.
    _extra_stats: dict[str, int] = {}

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        *,
        verbose: bool = False,
        log_file: str | None = None,
        pass_label: str = "",
    ) -> None:
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self._pass_label = pass_label
        self._stats: dict[str, int] = {
            "functions_processed": 0,
            "functions_skipped": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            **self._extra_stats,
        }
        self._stats_lock = threading.Lock()
        self._progress_current = 0
        self._progress_total = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return self._stats.copy()

    def record_response(self, response: LLMResponse) -> None:
        """Update stats from an LLM response (thread-safe)."""
        with self._stats_lock:
            self._stats["llm_calls"] += 1
            if response.cached:
                self._stats["cache_hits"] += 1
            self._stats["cache_read_tokens"] += response.cache_read_tokens
            self._stats["cache_creation_tokens"] += response.cache_creation_tokens
            self._stats["input_tokens"] += response.input_tokens
            self._stats["output_tokens"] += response.output_tokens

    def record_call(self) -> None:
        """Record a plain LLM call with no metadata (e.g. ``complete()``)."""
        with self._stats_lock:
            self._stats["llm_calls"] += 1

    def record_error(self) -> None:
        """Increment the error counter (thread-safe)."""
        with self._stats_lock:
            self._stats["errors"] += 1

    def should_skip(
        self,
        func: Function,
        callee_summaries: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Decide whether *func* can skip the LLM call entirely.

        Subclasses override with pass-specific predicates that combine
        IR features (``db.get_ir_facts(func.id)['features']``) and
        callee summaries. Returning ``(True, reason)`` means the
        summarizer must produce a trivial empty summary instead of
        calling the LLM. Default: never skip.
        """
        return (False, "")

    def record_skip(self) -> None:
        """Increment the skip counter (thread-safe)."""
        with self._stats_lock:
            self._stats["functions_processed"] += 1
            self._stats["functions_skipped"] += 1

    def _ir_features(self, func: Function) -> dict[str, Any]:
        """Return KAMain IR features for *func*, or ``{}`` if absent."""
        if func.id is None:
            return {}
        facts = self.db.get_ir_facts(func.id)
        if not facts:
            return {}
        return facts.get("features") or {}

    def _ir_attrs(self, func: Function) -> dict[str, Any]:
        """Return KAMain LLVM-attr block for *func*, or ``{}`` if absent.

        Block shape: ``{"function": {...}, "params": [...], "ret": {...},
        "callsites": {...}}``. Only attrs LLVM actually inferred are present.
        """
        if func.id is None:
            return {}
        facts = self.db.get_ir_facts(func.id)
        if not facts:
            return {}
        return facts.get("attrs") or {}

    # Pass names that may be hard-skipped purely from LLVM-inferred function
    # memory effects. Passes that touch reads (memsafe, overflow) are NOT in
    # this set for ``readonly`` because reads can still deref bad pointers
    # and arith on loaded values can still overflow.
    _ATTRS_PASSES_NO_WRITES = frozenset(
        {"leak", "init", "alloc", "free"},
    )

    def _attrs_skip_reason(
        self,
        func: Function,
        pass_name: str,
    ) -> str | None:
        """Return a skip reason if LLVM attrs prove this pass has no work.

        Cheaper and stronger than the bitfield rollup when present.
        Subclass ``should_skip`` should call this *before* its own
        bitfield logic and short-circuit on a non-None result.

        KAMain emits ``attrs.function`` as a flat ``list[str]`` of
        attribute names. (A future schema may use a value-dict; we
        accept both.)

        Rules:
          - ``readnone`` present → skip every pass (no memory effects).
          - ``readonly`` present → skip leak/init/alloc/free; memsafe
            and overflow still run (reads can deref bad ptrs, arith on
            loaded values can overflow).
          - ``writeonly`` → no skip (writes can smash invariants).
        """
        attrs = self._ir_attrs(func)
        if not attrs:
            return None
        fn_attrs = attrs.get("function")
        if isinstance(fn_attrs, list):
            fn_set = {a for a in fn_attrs if isinstance(a, str)}
        elif isinstance(fn_attrs, dict):
            # Forward-compat: doc-spec dict shape with a "memory" key.
            mem = fn_attrs.get("memory")
            fn_set = {mem} if isinstance(mem, str) else set()
        else:
            return None
        if "readnone" in fn_set:
            return "readnone function (no memory effects)"
        if "readonly" in fn_set and pass_name in self._ATTRS_PASSES_NO_WRITES:
            return "readonly function (no writes/allocs/frees)"
        return None

    def _log_interaction(
        self, func_name: str, prompt: str, response: str,
    ) -> None:
        """Log LLM interaction to file."""
        if not self.log_file:
            return
        import datetime

        label = f" [{self._pass_label}]" if self._pass_label else ""
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Function: {func_name}{label}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-' * 80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-' * 80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'=' * 80}\n\n")
