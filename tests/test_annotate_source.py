"""Tests for memsafe_summarizer._annotate_source (PRE/POST condition inlining)."""

from llm_summary.memsafe_summarizer import MemsafeSummarizer
from llm_summary.models import Function, MemsafeContract, MemsafeSummary


def _make_summarizer() -> MemsafeSummarizer:
    """Create a MemsafeSummarizer without an LLM backend (for _annotate_source only)."""
    ms = MemsafeSummarizer.__new__(MemsafeSummarizer)
    ms.verbose = False
    return ms


def _make_func(
    source: str,
    callsites: list[dict],
    pp_source: str | None = None,
    line_start: int = 10,
) -> Function:
    return Function(
        name="test_func",
        file_path="/tmp/test.c",
        line_start=line_start,
        line_end=line_start + len(source.splitlines()) - 1,
        source=source,
        signature="void(int *, int)",
        callsites=callsites,
        pp_source=pp_source,
    )


class TestAnnotateSource:
    """Tests for _annotate_source."""

    def test_basic_annotation(self):
        """PRE comment is inserted before the callsite line."""
        source = "void test_func(int *buf, int n) {\n    memcpy(buf, src, n);\n}"
        callsites = [
            {"callee": "memcpy", "line": 11, "line_in_body": 1,
             "via_macro": False, "macro_name": None, "args": ["buf", "src", "n"]},
        ]
        callee_summaries = {
            "memcpy": MemsafeSummary(
                function_name="memcpy",
                contracts=[
                    MemsafeContract(target="dest", contract_kind="not_null",
                                    description="dest must not be NULL"),
                    MemsafeContract(target="dest", contract_kind="buffer_size",
                                    description="dest must have n bytes",
                                    size_expr="n", relationship="byte_count"),
                ],
            ),
        }
        callee_params = {"memcpy": ["dest", "src", "n"]}

        ms = _make_summarizer()
        func = _make_func(source, callsites)
        annotated, used = ms._annotate_source(func, callee_summaries, callee_params)

        assert used is True
        assert "/* PRE[memcpy(buf, src, n)]:" in annotated
        # formal→actual substitution: "dest" → "buf"
        assert "buf: not_null" in annotated
        assert "buf: buffer_size(n)" in annotated
        # Original callsite line preserved
        assert "memcpy(buf, src, n);" in annotated

    def test_no_callsites_returns_source(self):
        """With no callsites, returns source unchanged."""
        source = "void test_func() { return; }"
        ms = _make_summarizer()
        func = _make_func(source, callsites=[])
        annotated, used = ms._annotate_source(func, {}, {})
        assert used is False
        assert annotated == source

    def test_no_matching_callee_returns_source(self):
        """Callsites exist but none match callee_summaries."""
        source = "void test_func() {\n    foo();\n}"
        callsites = [
            {"callee": "foo", "line": 11, "line_in_body": 1,
             "via_macro": False, "macro_name": None, "args": []},
        ]
        ms = _make_summarizer()
        func = _make_func(source, callsites)
        annotated, used = ms._annotate_source(func, {"bar": MemsafeSummary("bar")}, {})
        assert used is False

    def test_via_macro_annotation(self):
        """Macro-hidden calls show [via macro NAME] header."""
        source = "void test_func(int *p) {\n    CHECK_NOT_NULL(p);\n}"
        callsites = [
            {"callee": "assert_valid", "line": 11, "line_in_body": 1,
             "via_macro": True, "macro_name": "CHECK_NOT_NULL", "args": []},
        ]
        callee_summaries = {
            "assert_valid": MemsafeSummary(
                function_name="assert_valid",
                contracts=[MemsafeContract(target="ptr", contract_kind="not_null",
                                           description="ptr must not be NULL")],
            ),
        }
        ms = _make_summarizer()
        func = _make_func(source, callsites)
        annotated, used = ms._annotate_source(func, callee_summaries, {})
        assert used is True
        assert "[via macro CHECK_NOT_NULL]" in annotated

    def test_no_callsites_with_pp_source_returns_pp_source(self):
        """With no callsites, returns pp_source (llm_source) when available."""
        source = "D(fpu, 0)"
        pp_source = "int ZSTD_cpuid_fpu(ZSTD_cpuid_t cpuid) { return (cpuid.f1d & 1) != 0; }"
        ms = _make_summarizer()
        func = _make_func(source, callsites=[], pp_source=pp_source)
        annotated, used = ms._annotate_source(func, {}, {})
        assert used is False
        assert annotated == pp_source

    def test_with_callsites_uses_original_source_not_pp(self):
        """Annotation operates on func.source (line offsets), not pp_source."""
        source = "void f(char *p) {\n    free(p);\n}"
        pp_source = "void f(char *p) { free(p); }"  # single line — different structure
        callsites = [
            {"callee": "free", "line": 11, "line_in_body": 1,
             "via_macro": False, "macro_name": None, "args": ["p"]},
        ]
        callee_summaries = {
            "free": MemsafeSummary(
                function_name="free",
                contracts=[MemsafeContract(target="ptr", contract_kind="not_null",
                                           description="ptr must not be NULL")],
            ),
        }
        callee_params = {"free": ["ptr"]}

        ms = _make_summarizer()
        func = _make_func(source, callsites, pp_source=pp_source)
        annotated, used = ms._annotate_source(func, callee_summaries, callee_params)

        assert used is True
        lines = annotated.splitlines()
        # PRE annotation should be on its own line before "free(p);"
        pre_idx = next(i for i, l in enumerate(lines) if "/* PRE[" in l)
        # The line after the annotation block (skip contract + closing */) should be the callsite
        free_idx = next(i for i, l in enumerate(lines) if "free(p);" in l)
        assert free_idx > pre_idx
        # Should use original source structure (3 lines), not pp_source (1 line)
        assert "void f(char *p) {" in lines[0]

    def test_empty_contracts_no_annotation(self):
        """Callee with empty contracts list doesn't produce PRE comment."""
        source = "void f() {\n    bar();\n}"
        callsites = [
            {"callee": "bar", "line": 11, "line_in_body": 1,
             "via_macro": False, "macro_name": None, "args": []},
        ]
        callee_summaries = {
            "bar": MemsafeSummary(function_name="bar", contracts=[]),
        }
        ms = _make_summarizer()
        func = _make_func(source, callsites)
        annotated, used = ms._annotate_source(func, callee_summaries, {})
        # used is True (we entered the annotation path) but no PRE comments emitted
        assert "/* PRE[" not in annotated
