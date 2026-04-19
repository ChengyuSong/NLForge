"""Tests for ir_sidecar helpers (effect hints, attrs preamble, combined annotator)."""

from llm_summary.ir_sidecar import (
    _format_effect_hint,
    annotate_source_with_int_ops,
    annotate_source_with_ir_facts,
    format_attrs_preamble,
)

# ---------------------------------------------------------------------------
# _format_effect_hint
# ---------------------------------------------------------------------------

class TestFormatEffectHint:
    def test_plain_load_returns_none(self) -> None:
        eff = {"kind": "read", "align": 4, "loc": "f.c:10"}
        assert _format_effect_hint(eff) is None

    def test_plain_store_returns_none(self) -> None:
        eff = {"kind": "write", "align": 4, "loc": "f.c:10"}
        assert _format_effect_hint(eff) is None

    def test_volatile_load(self) -> None:
        eff = {"kind": "read", "align": 4, "volatile": True, "loc": "f.c:1"}
        assert _format_effect_hint(eff) == "volatile"

    def test_atomic_seq_cst_load(self) -> None:
        eff = {"kind": "read", "align": 4, "atomic": "seq_cst", "loc": "f.c:1"}
        assert _format_effect_hint(eff) == "atomic seq_cst"

    def test_atomic_monotonic_skipped(self) -> None:
        eff = {"kind": "read", "align": 4, "atomic": "monotonic", "loc": "f.c:1"}
        assert _format_effect_hint(eff) is None

    def test_load_md_full(self) -> None:
        eff = {
            "kind": "read", "align": 8, "loc": "f.c:1",
            "load_md": {"nonnull": True, "dereferenceable": 32,
                        "range": [[0, 256]]},
        }
        out = _format_effect_hint(eff)
        assert out is not None
        assert "nonnull" in out
        assert "dereferenceable(32)" in out
        assert "range[0,256]" in out

    def test_atomicrmw_op(self) -> None:
        eff = {
            "kind": "atomicrmw", "op": "add", "align": 4,
            "atomic": "seq_cst", "loc": "f.c:5",
        }
        out = _format_effect_hint(eff)
        assert out is not None
        assert "atomic seq_cst" in out
        assert "rmw add" in out

    def test_cmpxchg(self) -> None:
        eff = {
            "kind": "cmpxchg", "align": 8,
            "atomic_success": "seq_cst", "atomic_failure": "monotonic",
            "loc": "f.c:6",
        }
        out = _format_effect_hint(eff)
        assert out is not None
        assert "cmpxchg seq_cst" in out

    def test_call_kind_returns_none(self) -> None:
        # Only memory-touching kinds get hints.
        eff = {"kind": "call", "callee": "memcpy", "loc": "f.c:1"}
        assert _format_effect_hint(eff) is None


# ---------------------------------------------------------------------------
# format_attrs_preamble
# ---------------------------------------------------------------------------

class TestAttrsPreamble:
    def test_empty(self) -> None:
        assert format_attrs_preamble(None) is None
        assert format_attrs_preamble({}) is None

    def test_kamain_shape_flat_lists(self) -> None:
        # Actual KAMain emission: flat string lists per scope, "return" key.
        out = format_attrs_preamble({
            "function": ["readonly", "nounwind"],
            "params": [["nocapture", "noalias"], ["readonly"]],
            "return": ["nonnull"],
        })
        assert out is not None
        lines = out.split("\n")
        assert lines[0] == "// LLVM fn: readonly nounwind"
        assert lines[1] == "// LLVM arg0: nocapture noalias"
        assert lines[2] == "// LLVM arg1: readonly"
        assert lines[3] == "// LLVM ret: nonnull"

    def test_strips_low_signal_attrs(self) -> None:
        # noinline/optnone/uwtable carry no semantic signal — filter them.
        out = format_attrs_preamble({
            "function": ["noinline", "nounwind", "optnone", "uwtable"],
        })
        assert out == "// LLVM fn: nounwind"

    def test_doc_spec_dict_shape_still_accepted(self) -> None:
        # Forward-compat with the originally documented value-dict shape.
        out = format_attrs_preamble({
            "function": {"memory": "argmemonly", "nounwind": True},
            "params": [
                {"i": 0, "nocapture": True, "noalias": True,
                 "dereferenceable": 32},
                {"i": 1, "readonly": True},
            ],
            "ret": {"nonnull": True, "noalias": True},
        })
        assert out is not None
        lines = out.split("\n")
        assert lines[0] == "// LLVM fn: argmemonly nounwind"
        assert lines[1] == "// LLVM arg0: nocapture noalias dereferenceable(32)"
        assert lines[2] == "// LLVM arg1: readonly"
        assert lines[3] == "// LLVM ret: nonnull noalias"

    def test_skips_empty_param_entries(self) -> None:
        out = format_attrs_preamble({
            "function": ["readonly"],
            "params": [[]],  # no attrs on arg0
        })
        assert out == "// LLVM fn: readonly"


# ---------------------------------------------------------------------------
# annotate_source_with_ir_facts
# ---------------------------------------------------------------------------

class TestAnnotateSourceCombined:
    def test_int_ops_only_back_compat(self) -> None:
        # Original wrapper still works.
        src = "int x = a + b;\nint y = c * d;"
        ops = [
            {"id": "io1", "op": "add", "loc": "f.c:5"},
            {"id": "io2", "op": "mul", "loc": "f.c:6", "wraps_legally": True},
        ]
        out = annotate_source_with_int_ops(src, line_start=5, int_ops=ops)
        assert "// check overflow" in out
        assert "// safe" in out

    def test_effects_inline_when_enabled(self) -> None:
        src = "v = *p;\nstore_volatile(p);"
        facts = {
            "effects": [
                {"id": "e1", "kind": "read", "align": 8, "loc": "f.c:5",
                 "load_md": {"nonnull": True, "dereferenceable": 16}},
                {"id": "e2", "kind": "write", "align": 4, "volatile": True,
                 "loc": "f.c:6"},
            ],
        }
        out = annotate_source_with_ir_facts(
            src, 5, facts, include_int_ops=False, include_effects=True,
        )
        assert "nonnull" in out
        assert "dereferenceable(16)" in out
        assert "volatile" in out

    def test_attrs_preamble_prepended(self) -> None:
        src = "int f(int *p) { return *p; }"
        facts = {
            "attrs": {
                "function": ["readonly"],
                "params": [["nonnull"]],
            },
        }
        out = annotate_source_with_ir_facts(
            src, 1, facts,
            include_int_ops=False, include_attrs_preamble=True,
        )
        assert out.startswith("// LLVM fn: readonly\n")
        assert "// LLVM arg0: nonnull" in out
        assert out.endswith(src)

    def test_combined_int_ops_and_effects(self) -> None:
        src = "x = *p + 1;"
        facts = {
            "int_ops": [
                {"id": "io1", "op": "add", "loc": "f.c:10"},
            ],
            "effects": [
                {"id": "e1", "kind": "read", "align": 4,
                 "load_md": {"range": [[0, 100]]}, "loc": "f.c:10"},
            ],
        }
        out = annotate_source_with_ir_facts(
            src, 10, facts,
            include_int_ops=True, include_effects=True,
        )
        # Both hints should appear on the same line.
        assert "check overflow" in out
        assert "range[0,100]" in out

    def test_no_facts_returns_unchanged(self) -> None:
        src = "x = 1;\ny = 2;"
        out = annotate_source_with_ir_facts(src, 1, {})
        assert out == src
