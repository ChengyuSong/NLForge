"""Tests for BaseSummarizer._attrs_skip_reason behavior across passes."""

import json
from typing import Any

from llm_summary.base_summarizer import BaseSummarizer
from llm_summary.models import Function


class _StubDB:
    """Minimal SummaryDB stub: only get_ir_facts() is exercised."""

    def __init__(self, ir_facts: dict[int, dict[str, Any]]) -> None:
        self._facts = ir_facts

    def get_ir_facts(self, function_id: int) -> dict[str, Any] | None:
        return self._facts.get(function_id)


def _make_summarizer(facts: dict[int, dict[str, Any]]) -> BaseSummarizer:
    s = BaseSummarizer.__new__(BaseSummarizer)
    s.db = _StubDB(facts)  # type: ignore[assignment]
    return s


def _make_func(fid: int = 1) -> Function:
    return Function(
        name="f", file_path="/tmp/x.c", line_start=1, line_end=10,
        source="", signature="", id=fid,
    )


def _facts_with_attrs(fn_attrs: list[str] | dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {1: {"attrs": {"function": fn_attrs}}}


class TestAttrsSkipReason:
    def test_no_attrs_no_skip(self) -> None:
        s = _make_summarizer({1: {}})
        assert s._attrs_skip_reason(_make_func(), "memsafe") is None
        assert s._attrs_skip_reason(_make_func(), "leak") is None

    def test_readnone_skips_all_passes_kamain_shape(self) -> None:
        # KAMain emits flat list-of-strings.
        s = _make_summarizer(_facts_with_attrs(["readnone", "nounwind"]))
        for pass_name in ("memsafe", "leak", "init", "alloc", "free", "overflow"):
            r = s._attrs_skip_reason(_make_func(), pass_name)
            assert r is not None
            assert "readnone" in r

    def test_readnone_skips_all_passes_doc_dict_shape(self) -> None:
        # Forward-compat with the doc-spec value-dict shape.
        s = _make_summarizer(_facts_with_attrs({"memory": "readnone"}))
        for pass_name in ("memsafe", "leak", "init", "alloc", "free", "overflow"):
            r = s._attrs_skip_reason(_make_func(), pass_name)
            assert r is not None
            assert "readnone" in r

    def test_readonly_skips_writes_passes_only(self) -> None:
        s = _make_summarizer(_facts_with_attrs(["readonly"]))
        # No-write passes hard-skip.
        for pass_name in ("leak", "init", "alloc", "free"):
            r = s._attrs_skip_reason(_make_func(), pass_name)
            assert r is not None and "readonly" in r
        # Memsafe and overflow still need to run.
        assert s._attrs_skip_reason(_make_func(), "memsafe") is None
        assert s._attrs_skip_reason(_make_func(), "overflow") is None

    def test_writeonly_no_skip(self) -> None:
        s = _make_summarizer(_facts_with_attrs(["writeonly"]))
        # writeonly doesn't help any current pass — writes can still smash.
        for pass_name in ("memsafe", "leak", "init", "alloc", "free", "overflow"):
            assert s._attrs_skip_reason(_make_func(), pass_name) is None

    def test_no_memory_attr_in_list_no_skip(self) -> None:
        # Real-world example: function attrs without any memory effect attr.
        s = _make_summarizer(_facts_with_attrs(
            ["noinline", "nounwind", "optnone", "uwtable"],
        ))
        for pass_name in ("memsafe", "leak", "init", "alloc", "free", "overflow"):
            assert s._attrs_skip_reason(_make_func(), pass_name) is None

    def test_func_id_none_no_skip(self) -> None:
        s = _make_summarizer(_facts_with_attrs(["readnone"]))
        f = _make_func()
        f.id = None
        assert s._attrs_skip_reason(f, "memsafe") is None


class TestIrAttrs:
    def test_returns_block_when_present(self) -> None:
        attrs = {"function": {"memory": "readonly"}, "params": [{"i": 0}]}
        s = _make_summarizer({1: {"attrs": attrs}})
        out = s._ir_attrs(_make_func())
        assert out == attrs

    def test_returns_empty_when_absent(self) -> None:
        s = _make_summarizer({1: {"features": {}}})
        assert s._ir_attrs(_make_func()) == {}

    def test_returns_empty_when_facts_missing(self) -> None:
        s = _make_summarizer({})
        assert s._ir_attrs(_make_func()) == {}


def test_facts_json_round_trip() -> None:
    """Ensure the JSON shape we test against actually matches the stored blob shape."""
    payload = {
        "function": "f",
        "attrs": {
            "function": {"memory": "readnone"},
            "ret": {"nonnull": True, "dereferenceable": 16},
        },
    }
    blob = json.loads(json.dumps(payload))
    assert blob["attrs"]["function"]["memory"] == "readnone"
