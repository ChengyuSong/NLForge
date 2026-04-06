"""Unified bottom-up graph traversal driver for summary passes."""

from __future__ import annotations

import json
import threading
from collections import deque
from concurrent.futures import as_completed
from typing import Any, Protocol

from .db import SummaryDB
from .llm.pool import LLMPool
from .models import (
    AllocationSummary,
    FreeSummary,
    Function,
    InitSummary,
    LeakSummary,
    MemsafeSummary,
    VerificationSummary,
)
from .ordering import ProcessingOrderer
from .verification_summarizer import IncompleteCalleeError

_MAX_SCC_ITERATIONS = 3
"""Maximum iterations for recursive SCC convergence."""

SCC_PREVIOUS_SUMMARY_SECTION = """\

## Previous Summary (from prior iteration)

The callee summaries above may have been updated since this summary was \
generated. Review whether the changes affect YOUR summary. Add a \
`"changed": true/false` field at the TOP of your JSON response.
- `true`: callee changes affect this summary — output the updated summary.
- `false`: callee changes do NOT affect this summary — output the previous \
summary unchanged.

Previous:
```json
{previous_json}
```
"""


def extract_scc_changed(parsed: dict[str, Any]) -> bool:
    """Extract the ``"changed"`` field from a parsed LLM JSON dict.

    Returns ``True`` (assume changed) if the field is missing.
    """
    return bool(parsed.get("changed", True))


class SummaryPass(Protocol):
    """Interface each pass must implement."""

    name: str
    summarizer: Any

    def get_cached(self, func_id: int, func: Function) -> Any | None:
        """Return cached summary if available, else None."""
        ...

    def summarize(self, func: Function, callee_summaries: dict[str, Any], **kwargs: Any) -> Any:
        """Generate a summary for *func* given its callee summaries."""
        ...

    def store(self, func: Function, summary: Any) -> None:
        """Persist *summary* to the database."""
        ...


PASS_TABLE_MAP: dict[str, str] = {
    "allocation": "allocation_summaries",
    "free": "free_summaries",
    "init": "init_summaries",
    "memsafe": "memsafe_summaries",
    "verify": "verification_summaries",
    "leak": "leak_summaries",
}


class AllocationPass:
    """Adapter that wraps AllocationSummarizer as a SummaryPass."""

    name = "allocation"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> AllocationSummary | None:
        existing = self.db.get_summary_by_function_id(func_id)
        if existing and not self.db.needs_update(func):
            return existing
        return None

    def summarize(
        self, func: Function,
        callee_summaries: dict[str, AllocationSummary],
        **kwargs: Any,
    ) -> AllocationSummary:
        result: AllocationSummary = self.summarizer.summarize_function(
            func, callee_summaries,
            previous_summary_json=kwargs.get("previous_summary_json"),
        )
        return result

    def store(self, func: Function, summary: AllocationSummary) -> None:
        self.db.upsert_summary(func, summary, model_used=self.model)


class FreePass:
    """Adapter that wraps FreeSummarizer as a SummaryPass."""

    name = "free"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> FreeSummary | None:
        return self.db.get_free_summary_by_function_id(func_id)

    def summarize(
        self, func: Function, callee_summaries: dict[str, FreeSummary], **kwargs: Any,
    ) -> FreeSummary:
        result: FreeSummary = self.summarizer.summarize_function(
            func, callee_summaries,
            previous_summary_json=kwargs.get("previous_summary_json"),
        )
        return result

    def store(self, func: Function, summary: FreeSummary) -> None:
        self.db.upsert_free_summary(func, summary, model_used=self.model)


class InitPass:
    """Adapter that wraps InitSummarizer as a SummaryPass."""

    name = "init"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> InitSummary | None:
        return self.db.get_init_summary_by_function_id(func_id)

    def summarize(
        self, func: Function, callee_summaries: dict[str, InitSummary], **kwargs: Any,
    ) -> InitSummary:
        result: InitSummary = self.summarizer.summarize_function(
            func, callee_summaries,
            previous_summary_json=kwargs.get("previous_summary_json"),
        )
        return result

    def store(self, func: Function, summary: InitSummary) -> None:
        self.db.upsert_init_summary(func, summary, model_used=self.model)
        # Propagate unconditional noreturn to function attributes
        if summary.noreturn and not summary.noreturn_condition:
            existing = func.attributes or ""
            if "__attribute__((noreturn))" not in existing:
                new_attrs = (
                    f"{existing} __attribute__((noreturn))".strip()
                    if existing else "__attribute__((noreturn))"
                )
                self.db.update_function_attributes(func.id, new_attrs)
                func.attributes = new_attrs


class MemsafePass:
    """Adapter that wraps MemsafeSummarizer as a SummaryPass."""

    name = "memsafe"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str, alias_builder: Any = None):
        self.summarizer = summarizer
        self.db = db
        self.model = model
        self.alias_builder = alias_builder

    def get_cached(self, func_id: int, func: Function) -> MemsafeSummary | None:
        return self.db.get_memsafe_summary_by_function_id(func_id)

    def summarize(
        self,
        func: Function,
        callee_summaries: dict[str, MemsafeSummary],
        callee_funcs: dict[str, Function] | None = None,
        **kwargs: Any,
    ) -> MemsafeSummary:
        callee_params = {name: f.params for name, f in (callee_funcs or {}).items()}
        alias_context = None
        if self.alias_builder is not None:
            callee_names = list((callee_funcs or {}).keys())
            alias_context = self.alias_builder.build_context(func, callee_names)
        result: MemsafeSummary = self.summarizer.summarize_function(
            func, callee_summaries, callee_params, alias_context=alias_context,
            previous_summary_json=kwargs.get("previous_summary_json"),
        )
        return result

    def store(self, func: Function, summary: MemsafeSummary) -> None:
        self.db.upsert_memsafe_summary(func, summary, model_used=self.model)


class VerificationPass:
    """Adapter that wraps VerificationSummarizer as a SummaryPass."""

    name = "verify"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str, alias_builder: Any = None):
        self.summarizer = summarizer
        self.db = db
        self.model = model
        self.alias_builder = alias_builder

    def get_cached(self, func_id: int, func: Function) -> VerificationSummary | None:
        return self.db.get_verification_summary_by_function_id(func_id)

    def summarize(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary],
        callee_funcs: dict[str, Function] | None = None,
        **kwargs: Any,
    ) -> VerificationSummary:
        callee_params = {name: f.params for name, f in (callee_funcs or {}).items()}
        alias_context = None
        if self.alias_builder is not None:
            callee_names = list((callee_funcs or {}).keys())
            alias_context = self.alias_builder.build_context(func, callee_names)
        result: VerificationSummary = self.summarizer.summarize_function(
            func, callee_summaries, callee_params=callee_params,
            alias_context=alias_context,
            previous_summary_json=kwargs.get("previous_summary_json"),
        )
        return result

    def store(self, func: Function, summary: VerificationSummary) -> None:
        self.db.upsert_verification_summary(func, summary, model_used=self.model)


class LeakPass:
    """Adapter that wraps LeakSummarizer as a SummaryPass."""

    name = "leak"

    def __init__(self, summarizer: Any, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> LeakSummary | None:
        return self.db.get_leak_summary_by_function_id(func_id)

    def summarize(
        self,
        func: Function,
        callee_summaries: dict[str, LeakSummary],
        **kwargs: Any,
    ) -> LeakSummary:
        result: LeakSummary = self.summarizer.summarize_function(
            func, callee_summaries,
        )
        return result

    def store(self, func: Function, summary: LeakSummary) -> None:
        self.db.upsert_leak_summary(func, summary, model_used=self.model)


class BottomUpDriver:
    """Builds the call graph once, then runs one or more summary passes
    over functions in bottom-up (callee-first) topological order."""

    def __init__(self, db: SummaryDB, verbose: bool = False, pool: LLMPool | None = None):
        self.db = db
        self.verbose = verbose
        self.pool = pool
        self._graph: dict[int, list[int]] | None = None
        self._orderer: ProcessingOrderer | None = None

    def build_graph(self) -> tuple[dict[int, list[int]], ProcessingOrderer]:
        """Build (and cache) the call graph + ProcessingOrderer."""
        if self._graph is not None and self._orderer is not None:
            return self._graph, self._orderer

        edges = self.db.get_all_call_edges()
        graph: dict[int, list[int]] = {}

        for edge in edges:
            if edge.caller_id not in graph:
                graph[edge.caller_id] = []
            graph[edge.caller_id].append(edge.callee_id)

        # Add all functions to graph (some may have no callees)
        for func in self.db.get_all_functions():
            if func.id is not None and func.id not in graph:
                graph[func.id] = []

        self._graph = graph
        self._orderer = ProcessingOrderer(graph)
        return graph, self._orderer

    def compute_affected(
        self, dirty_ids: set[int], graph: dict[int, list[int]]
    ) -> set[int]:
        """Compute the set of function IDs affected by changes to *dirty_ids*.

        This is *dirty_ids* plus all transitive callers (via reverse edges).
        """
        # Build reverse graph
        reverse: dict[int, list[int]] = {}
        for caller, callees in graph.items():
            for callee in callees:
                if callee not in reverse:
                    reverse[callee] = []
                reverse[callee].append(caller)

        affected = set(dirty_ids)
        queue: deque[int] = deque(dirty_ids)
        while queue:
            node = queue.popleft()
            for caller in reverse.get(node, []):
                if caller not in affected:
                    affected.add(caller)
                    queue.append(caller)

        return affected

    def compute_reachable(
        self, entry_ids: set[int], graph: dict[int, list[int]]
    ) -> set[int]:
        """Compute the set of function IDs reachable from *entry_ids*.

        This is *entry_ids* plus all transitive callees (forward edges).
        """
        reachable = set(entry_ids)
        queue: deque[int] = deque(entry_ids)
        while queue:
            node = queue.popleft()
            for callee in graph.get(node, []):
                if callee not in reachable:
                    reachable.add(callee)
                    queue.append(callee)

        return reachable

    def _process_func(
        self,
        func_id: int,
        func: Function,
        passes: list[SummaryPass],
        graph: dict[int, list[int]],
        results: dict[str, dict[int, Any]],
        results_lock: threading.Lock,
        force: bool,
        affected: set[int] | None,
        current: int,
        total: int,
        force_dirty: set[int] | None = None,
        scc_iter: int = 0,
    ) -> None:
        """Process a single function through all passes. Thread-safe."""
        # Skip sourceless stubs (e.g. stdlib) — nothing to summarize
        if not func.source:
            for p in passes:
                cached = p.get_cached(func_id, func)
                if cached is not None:
                    with results_lock:
                        results[p.name][func_id] = cached
            return

        # If incremental and function is not affected, load from cache
        if affected is not None and func_id not in affected:
            for p in passes:
                cached = p.get_cached(func_id, func)
                if cached is not None:
                    with results_lock:
                        results[p.name][func_id] = cached
                    with p.summarizer._stats_lock:
                        p.summarizer._stats["cache_hits"] += 1
            return

        # Determine whether to force-rerun this function:
        #  - Directly in the dirty set (callee summary changed), OR
        #  - Dynamic propagation: any direct callee was force-rerun this pass
        #    (ensures convergence in one pass rather than one hop per run).
        local_force = force or (force_dirty is not None and func_id in force_dirty)
        if not local_force and force_dirty is not None:
            callee_ids = graph.get(func_id, [])
            with results_lock:
                if any(cid in force_dirty for cid in callee_ids):
                    local_force = True

        for p in passes:
            # Check cache
            if not local_force:
                cached = p.get_cached(func_id, func)
                if cached is not None:
                    with results_lock:
                        results[p.name][func_id] = cached
                    with p.summarizer._stats_lock:
                        p.summarizer._stats["cache_hits"] += 1
                    continue

            # Gather callee summaries and Function objects from this pass's results
            callee_ids = graph.get(func_id, [])
            callee_summaries: dict[str, Any] = {}
            callee_funcs: dict[str, Function] = {}
            with results_lock:
                for callee_id in callee_ids:
                    if callee_id in results[p.name]:
                        callee_func = self.db.get_function(callee_id)
                        if callee_func:
                            callee_summaries[callee_func.name] = results[p.name][callee_id]
                            callee_funcs[callee_func.name] = callee_func

            # Set progress on the underlying summarizer
            p.summarizer._progress_current = current
            p.summarizer._progress_total = total

            # On SCC re-iterations, pass previous summary for convergence check
            prev_json: str | None = None
            if scc_iter > 0:
                with results_lock:
                    prev = results[p.name].get(func_id)
                if prev is not None:
                    prev_json = json.dumps(prev.to_dict(), indent=2)

            # Generate summary (passes that don't accept callee_funcs ignore it)
            try:
                try:
                    summary = p.summarize(
                        func, callee_summaries, callee_funcs=callee_funcs,
                        previous_summary_json=prev_json,
                    )
                except TypeError:
                    summary = p.summarize(func, callee_summaries)
            except IncompleteCalleeError as e:
                # A callee's prior pass failed (e.g. timeout). Re-run it now.
                callee_name = e.callee_name
                if self.verbose:
                    print(f"  Re-running {p.name} for incomplete callee: {callee_name}")
                # Find the callee's function ID and re-run
                retry_callee_id: int | None = next(
                    (cid for cid in callee_ids
                     if (cf := self.db.get_function(cid)) and cf.name == callee_name),
                    None,
                )
                if retry_callee_id is not None:
                    callee_func_obj = self.db.get_function(retry_callee_id)
                    if callee_func_obj:
                        # Gather the callee's own callees for the re-run
                        sub_callee_ids = graph.get(retry_callee_id, [])
                        sub_summaries: dict[str, Any] = {}
                        with results_lock:
                            for sc_id in sub_callee_ids:
                                if sc_id in results[p.name]:
                                    sc_func = self.db.get_function(sc_id)
                                    if sc_func:
                                        sub_summaries[sc_func.name] = results[p.name][sc_id]
                        try:
                            try:
                                callee_summary = p.summarize(
                                callee_func_obj, sub_summaries,
                                callee_funcs={},
                            )
                            except TypeError:
                                callee_summary = p.summarize(callee_func_obj, sub_summaries)
                            with results_lock:
                                results[p.name][retry_callee_id] = callee_summary
                                callee_summaries[callee_name] = callee_summary
                            p.store(callee_func_obj, callee_summary)
                        except Exception as retry_err:
                            if self.verbose:
                                print(f"  Re-run of {callee_name} failed: {retry_err}")
                # Retry the original function
                try:
                    try:
                        summary = p.summarize(func, callee_summaries, callee_funcs=callee_funcs)
                    except TypeError:
                        summary = p.summarize(func, callee_summaries)
                except IncompleteCalleeError as e2:
                    # Still incomplete after retry — skip this pass for this function
                    if self.verbose:
                        print(f"  Skipping {p.name} for {func.name}: "
                              f"callee '{e2.callee_name}' still incomplete after retry")
                    continue
                except Exception as e2:
                    if self.verbose:
                        print(f"  Retry of {p.name} for {func.name} failed: {e2}")
                    continue

            # Don't persist error summaries — they would poison the
            # cache and block retries on subsequent runs.
            desc = getattr(summary, "description", "") or ""
            is_error = desc.startswith("Error generating summary:") or \
                desc.startswith("Error during verification:")

            # Check if summary content actually changed (for SCC convergence).
            # The LLM sets _scc_changed=False when it determines the previous
            # summary is still valid after callee updates.
            changed = getattr(summary, "_scc_changed", True)

            with results_lock:
                results[p.name][func_id] = summary

            if not is_error:
                if changed:
                    p.store(func, summary)
                # Propagate: callers of this function should also be
                # force-rerun so that incremental mode converges in one
                # pass.  Only propagate when content actually changed.
                if force_dirty is not None and changed:
                    with results_lock:
                        force_dirty.add(func_id)

    def run(
        self,
        passes: list[SummaryPass],
        force: bool = False,
        dirty_ids: set[int] | None = None,
        target_ids: set[int] | None = None,
    ) -> dict[str, dict[int, Any]]:
        """Run all *passes* over the call graph in a single traversal.

        Args:
            passes: List of SummaryPass adapters to run.
            force: Re-summarize even if a cached summary exists.
            dirty_ids: If provided, only re-summarize these function IDs
                and their transitive callers.  Others load from cache.
            target_ids: If provided, only summarize these exact function IDs.
                Others load from cache (no transitive expansion).

        Returns:
            ``{pass.name: {func_id: summary}}``
        """
        graph, orderer = self.build_graph()

        # Compute affected set if incremental.
        # force_dirty is a mutable set that grows during traversal: when a function
        # is actually re-run, it is added so that its callers are also force-rerun
        # in the same pass (convergence in one pass instead of one hop per run).
        affected: set[int] | None = None
        force_dirty: set[int] | None = None
        if dirty_ids is not None:
            affected = self.compute_affected(dirty_ids, graph)
            force_dirty = set(dirty_ids)  # mutable copy; grows as re-runs propagate up

        # target_ids takes precedence over affected
        if target_ids is not None:
            affected = target_ids
            force_dirty = None  # targeted mode: honour global force flag only

        if self.verbose:
            stats = orderer.get_stats()
            msg = f"Processing {stats['nodes']} functions in {stats['sccs']} SCCs"
            if stats["recursive_sccs"] > 0:
                msg += f" ({stats['recursive_sccs']} recursive)"
            if target_ids is not None:
                msg += f", {len(target_ids)} targeted"
            elif affected is not None:
                msg += f", {len(affected)} affected"
                if force_dirty and dirty_ids is not None:
                    msg += f" ({len(dirty_ids)} initially dirty)"
            if self.pool is not None:
                msg += f", {self.pool.max_workers} workers"
            print(msg)

        # Prepare per-pass result dicts
        results: dict[str, dict[int, Any]] = {p.name: {} for p in passes}
        results_lock = threading.Lock()

        if self.pool is not None:
            # Parallel mode: process by depth levels
            levels = orderer.get_parallel_levels()
            total = sum(
                len(scc)
                for level in levels
                for scc in level
            )
            current = 0

            for level in levels:
                # Separate non-recursive (parallelizable) from recursive
                parallel_funcs: list[tuple[int, Function]] = []
                recursive_sccs: list[list[int]] = []
                for scc in level:
                    if len(scc) > 1:
                        recursive_sccs.append(scc)
                    else:
                        for func_id in scc:
                            current += 1
                            func = self.db.get_function(func_id)
                            if func is not None:
                                parallel_funcs.append((func_id, func))

                # Run non-recursive functions in parallel
                futures = []
                for func_id, func in parallel_funcs:
                    fut = self.pool.submit(
                        self._process_func,
                        func_id, func, passes, graph, results,
                        results_lock, force, affected, current, total,
                        force_dirty,
                    )
                    futures.append(fut)
                for fut in as_completed(futures):
                    fut.result()

                # Run recursive SCCs sequentially with convergence
                for scc in recursive_sccs:
                    for scc_iter in range(_MAX_SCC_ITERATIONS):
                        scc_changed = False
                        for func_id in scc:
                            if scc_iter == 0:
                                current += 1
                            func = self.db.get_function(func_id)
                            if func is None:
                                continue
                            dirty_before = (
                                len(force_dirty)
                                if force_dirty is not None else 0
                            )
                            self._process_func(
                                func_id, func, passes, graph, results,
                                results_lock, force, affected,
                                current, total, force_dirty,
                                scc_iter=scc_iter,
                            )
                            dirty_after = (
                                len(force_dirty)
                                if force_dirty is not None else 0
                            )
                            if dirty_after > dirty_before:
                                scc_changed = True
                        if not scc_changed:
                            break
                        if scc_iter < _MAX_SCC_ITERATIONS - 1:
                            if self.verbose:
                                print(
                                    f"  SCC iteration {scc_iter + 1}:"
                                    f" re-running {len(scc)} functions"
                                )
        else:
            # Sequential mode: original behavior
            processing_order = list(orderer.get_processing_order())
            total = sum(len(scc) for scc in processing_order)
            current = 0

            for scc in processing_order:
                is_recursive = len(scc) > 1
                max_iters = _MAX_SCC_ITERATIONS if is_recursive else 1
                for scc_iter in range(max_iters):
                    scc_changed = False
                    for func_id in scc:
                        if scc_iter == 0:
                            current += 1
                        func = self.db.get_function(func_id)
                        if func is None:
                            continue
                        dirty_before = (
                            len(force_dirty) if force_dirty is not None
                            else 0
                        )
                        self._process_func(
                            func_id, func, passes, graph, results,
                            results_lock, force, affected, current, total,
                            force_dirty, scc_iter=scc_iter,
                        )
                        dirty_after = (
                            len(force_dirty) if force_dirty is not None
                            else 0
                        )
                        if dirty_after > dirty_before:
                            scc_changed = True
                    if not scc_changed:
                        break
                    if scc_iter < max_iters - 1 and self.verbose:
                        print(f"  SCC iteration {scc_iter + 1}: "
                              f"re-running {len(scc)} functions")

        return results
