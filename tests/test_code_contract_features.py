"""Tests for `code_contract.features`."""

from __future__ import annotations

from typing import Any

from llm_summary.code_contract.features import (
    Features,
    attrs_drops,
    features_for,
    property_set,
)
from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.models import Function


def _func(name: str, source: str, fid: int | None = None) -> Function:
    return Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=10, source=source, signature="int(void)",
        id=fid,
    )


class _StubDB:
    def __init__(self, facts: dict[int, dict[str, Any]]) -> None:
        self._facts = facts

    def get_ir_facts(self, function_id: int) -> dict[str, Any] | None:
        return self._facts.get(function_id)


class TestFeaturesFor:
    def test_alloc_then_free_lights_up_mem_props(self) -> None:
        src = "void f(void){ char *p = malloc(10); free(p); }"
        f = features_for(_func("f", src))
        assert f.has_alloc and f.has_free
        assert f.memsafe_relevant and f.memleak_relevant

    def test_arith_lights_up_overflow(self) -> None:
        src = "int g(int a, int b){ return a + b; }"
        f = features_for(_func("g", src))
        assert f.has_arith
        assert f.overflow_relevant

    def test_pure_const_function_minimal(self) -> None:
        src = "int h(void){ return 1; }"
        f = features_for(_func("h", src))
        assert not f.memsafe_relevant
        assert not f.memleak_relevant
        # Bare `return 1;` is intentionally noisy at the regex level — the
        # mock over-approximates rather than under-approximates. We only
        # assert the memory-safety and memory-leak gates stay quiet.


class TestPropertySet:
    def _empty_features(self) -> Features:
        return Features(
            has_deref=False, has_alloc=False, has_free=False,
            has_index=False, has_arith=False, has_div=False, has_shift=False,
        )

    def test_callee_lifts_property_into_scope(self) -> None:
        feats = self._empty_features()
        callee = CodeContractSummary(
            function="callee", properties=["memsafe"],
            requires={"memsafe": ["p != NULL"]},
        )
        assert property_set(feats, [callee]) == ["memsafe"]

    def test_trivial_callee_does_not_lift_scope(self) -> None:
        feats = self._empty_features()
        callee = CodeContractSummary(
            function="callee", properties=["memsafe"],
            requires={"memsafe": ["true"]}, ensures={"memsafe": []},
        )
        assert property_set(feats, [callee]) == []

    def test_local_features_lift_scope(self) -> None:
        feats = Features(
            has_deref=True, has_alloc=False, has_free=False,
            has_index=False, has_arith=True, has_div=False, has_shift=False,
        )
        result = property_set(feats, [])
        # has_deref → memsafe; has_arith → overflow.
        assert "memsafe" in result and "overflow" in result
        assert "memleak" not in result

    def test_order_follows_properties_constant(self) -> None:
        feats = Features(
            has_deref=True, has_alloc=True, has_free=False,
            has_index=False, has_arith=True, has_div=False, has_shift=False,
        )
        # Even though we activated memsafe + memleak + overflow in any order,
        # the result follows PROPERTIES order: memsafe, memleak, overflow.
        assert property_set(feats, []) == ["memsafe", "memleak", "overflow"]

    def test_drops_remove_props(self) -> None:
        feats = Features(
            has_deref=True, has_alloc=True, has_free=False,
            has_index=False, has_arith=True, has_div=False, has_shift=False,
        )
        # readnone-equivalent: drop all memory props; overflow stays.
        assert property_set(
            feats, [], drops={"memsafe", "memleak"},
        ) == ["overflow"]


class TestFeaturesFromSidecar:
    def test_sidecar_load_count_lights_up_memsafe(self) -> None:
        # Source has no derefs; sidecar says load_count=3 → memsafe relevant.
        f = _func("g", "int g(void){ return 0; }", fid=42)
        db = _StubDB({42: {"features": {
            "load_count": 3, "store_count": 0, "ptr_params": 0,
            "alloc_count": 0, "free_count": 0,
            "signed_arith_count": 0, "div_count": 0, "shift_count": 0,
        }}})
        feats = features_for(f, db=db)  # type: ignore[arg-type]
        assert feats.has_deref
        assert feats.memsafe_relevant
        assert not feats.memleak_relevant

    def test_sidecar_alloc_lights_up_memleak(self) -> None:
        f = _func("g", "void g(void){}", fid=1)
        db = _StubDB({1: {"features": {"alloc_count": 1, "free_count": 0}}})
        feats = features_for(f, db=db)  # type: ignore[arg-type]
        assert feats.has_alloc
        assert feats.memleak_relevant

    def test_sidecar_signed_arith_lights_up_overflow(self) -> None:
        f = _func("g", "int g(void){ return 0; }", fid=2)
        db = _StubDB({2: {"features": {"signed_arith_count": 5}}})
        feats = features_for(f, db=db)  # type: ignore[arg-type]
        assert feats.has_arith
        assert feats.overflow_relevant

    def test_sidecar_empty_features_falls_through_to_regex(self) -> None:
        # Empty features dict → fall back to regex (which sees the malloc).
        f = _func("g", "void g(void){ malloc(10); }", fid=3)
        db = _StubDB({3: {"features": {}}})
        feats = features_for(f, db=db)  # type: ignore[arg-type]
        assert feats.has_alloc

    def test_no_db_uses_regex(self) -> None:
        f = _func("g", "void g(void){ malloc(10); }")
        feats = features_for(f, db=None)
        assert feats.has_alloc

    def test_func_id_none_uses_regex(self) -> None:
        f = _func("g", "void g(void){ malloc(10); }", fid=None)
        db = _StubDB({1: {"features": {"alloc_count": 0}}})
        feats = features_for(f, db=db)  # type: ignore[arg-type]
        # Regex sees malloc; sidecar would say zero — proves regex wins
        # when func.id is None.
        assert feats.has_alloc


class TestAttrsDrops:
    def test_no_facts_no_drops(self) -> None:
        assert attrs_drops(None) == set()
        assert attrs_drops({}) == set()
        assert attrs_drops({"attrs": {}}) == set()

    def test_readnone_drops_memory_props(self) -> None:
        out = attrs_drops({"attrs": {"function": ["readnone"]}})
        assert out == {"memsafe", "memleak"}

    def test_readonly_drops_only_memleak(self) -> None:
        out = attrs_drops({"attrs": {"function": ["readonly", "nounwind"]}})
        assert out == {"memleak"}

    def test_writeonly_drops_nothing(self) -> None:
        out = attrs_drops({"attrs": {"function": ["writeonly"]}})
        assert out == set()

    def test_doc_dict_shape(self) -> None:
        # Forward-compat: value-dict shape with "memory" key.
        out = attrs_drops({"attrs": {"function": {"memory": "readnone"}}})
        assert out == {"memsafe", "memleak"}

    def test_no_memory_attr_no_drops(self) -> None:
        out = attrs_drops({"attrs": {"function": ["nounwind", "noinline"]}})
        assert out == set()
