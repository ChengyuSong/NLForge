"""Tests for the contract-check pipeline (two-stage agent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_summary.agent_tools import ToolExecutor
from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.contract_check import (
    AuditVerdict,
    ContractCheckResult,
    ContractGap,
    HazardCandidate,
    parse_public_apis,
)
from llm_summary.db import SummaryDB
from llm_summary.models import Function

# ---------------------------------------------------------------------------
# parse_public_apis
# ---------------------------------------------------------------------------


class TestParsePublicApis:
    def test_png_export_pattern(self) -> None:
        text = """
        PNG_EXPORT(1, void, png_set_sig_bytes, (png_structrp png_ptr, int num_bytes));
        PNG_EXPORTA(2, png_uint_32, png_get_io_ptr, (png_const_structrp png_ptr), PNG_DEPRECATED);
        PNG_EXPORT(3, void, png_destroy_read_struct, (png_structpp png_ptr_ptr));
        """
        names = parse_public_apis(text)
        assert names == [
            "png_set_sig_bytes",
            "png_get_io_ptr",
            "png_destroy_read_struct",
        ]

    def test_png_export_dedup_and_order(self) -> None:
        text = (
            "PNG_EXPORT(1, void, foo, (int));\n"
            "PNG_EXPORT(2, int,  bar, (void));\n"
            "PNG_EXPORT(3, void, foo, (long));\n"  # duplicate name
        )
        assert parse_public_apis(text) == ["foo", "bar"]

    def test_naive_fallback_when_no_png_export(self) -> None:
        text = """
        int do_thing(int x);
        void other_thing(struct S *s);
        // not a function declaration
        typedef int my_t;
        """
        names = parse_public_apis(text)
        assert "do_thing" in names
        assert "other_thing" in names
        assert "my_t" not in names

    def test_empty_header(self) -> None:
        assert parse_public_apis("") == []
        assert parse_public_apis("// comment only\n") == []


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_db() -> SummaryDB:
    db = SummaryDB(":memory:")
    yield db
    db.close()


def _insert_func_with_contract(db: SummaryDB, name: str) -> None:
    f = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5,
        source=f"void {name}(void) {{}}",
        signature=f"void {name}(void)",
    )
    f.id = db.insert_function(f)
    summary = CodeContractSummary(function=name, properties=["memsafe"])
    db.store_code_contract_summary(f, summary, model_used="test")


def _insert_func_no_contract(db: SummaryDB, name: str) -> None:
    f = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5,
        source=f"void {name}(void) {{}}",
        signature=f"void {name}(void)",
    )
    db.insert_function(f)


# ---------------------------------------------------------------------------
# list_apis_without_contracts handler
# ---------------------------------------------------------------------------


class TestListApisWithoutContracts:
    def test_returns_only_uncontracted(self, empty_db: SummaryDB) -> None:
        _insert_func_with_contract(empty_db, "has_contract_a")
        _insert_func_with_contract(empty_db, "has_contract_b")
        _insert_func_no_contract(empty_db, "no_contract")

        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({
            "api_names": [
                "has_contract_a",
                "no_contract",
                "not_in_db_at_all",
                "has_contract_b",
            ],
        })
        assert result["checked"] == 4
        assert set(result["missing"]) == {"no_contract", "not_in_db_at_all"}
        assert result["missing_count"] == 2

    def test_empty_input(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({"api_names": []})
        assert result == {"checked": 0, "missing_count": 0, "missing": []}

    def test_non_list_input_returns_error(
        self, empty_db: SummaryDB,
    ) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({"api_names": "foo"})
        assert "error" in result


# ---------------------------------------------------------------------------
# list_public_apis handler
# ---------------------------------------------------------------------------


class TestListPublicApisHandler:
    def test_parses_header_relative_to_project(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        header = tmp_path / "lib.h"
        header.write_text(
            "PNG_EXPORT(1, void, foo, (int));\n"
            "PNG_EXPORT(2, int,  bar, (void));\n",
        )
        ex = ToolExecutor(empty_db, project_path=tmp_path)
        result = ex._tool_list_public_apis({"header_path": "lib.h"})
        assert result["api_count"] == 2
        assert result["apis"] == ["foo", "bar"]

    def test_missing_header_returns_error(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        ex = ToolExecutor(empty_db, project_path=tmp_path)
        result = ex._tool_list_public_apis({"header_path": "nope.h"})
        assert "error" in result

    def test_no_project_path_errors(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_public_apis({"header_path": "lib.h"})
        assert "error" in result

    def test_path_traversal_blocked(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        outside = tmp_path.parent / "outside.h"
        outside.write_text("PNG_EXPORT(1, void, evil, (int));\n")
        try:
            ex = ToolExecutor(empty_db, project_path=tmp_path)
            result = ex._tool_list_public_apis({
                "header_path": "../outside.h",
            })
            assert "error" in result
        finally:
            outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# submit_hazards / submit_audit_verdict handlers
# ---------------------------------------------------------------------------


class TestSubmitHazards:
    def test_echoes_input(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        payload = {
            "summary": "scanned 30 APIs",
            "candidates": [
                {
                    "function": "png_init_io",
                    "hazard_kind": "null_input",
                    "description": (
                        "fp == NULL crashes inside fread()"
                    ),
                    "source_evidence": (
                        "pngwio.c:42 / contract.requires[memsafe]: "
                        "fp != NULL"
                    ),
                    "contract_clause": "fp != NULL",
                    "contract_property": "memsafe",
                },
            ],
        }
        result = ex._tool_submit_hazards(payload)
        assert result["accepted"] is True
        assert result["summary"] == "scanned 30 APIs"
        assert result["candidates"] == payload["candidates"]


class TestSubmitAuditVerdict:
    def test_echoes_input(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        payload = {
            "documented": False,
            "doc_searched": (
                "libpng-manual.txt §IV.3 (no warning); "
                "png.h:1023 (decl only)"
            ),
            "doc_quote": "",
            "recommendation": "Document that fp must be non-NULL.",
        }
        result = ex._tool_submit_audit_verdict(payload)
        assert result["accepted"] is True
        assert result["documented"] is False
        assert "manual" in result["doc_searched"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestHazardCandidate:
    def test_roundtrip_with_contract_clause(self) -> None:
        d = {
            "function": "png_init_io",
            "hazard_kind": "null_input",
            "description": "fp NULL crashes",
            "source_evidence": "pngwio.c:42",
            "contract_clause": "fp != NULL",
            "contract_property": "memsafe",
        }
        c = HazardCandidate.from_dict(d)
        assert c.to_dict() == d

    def test_roundtrip_without_contract_clause(self) -> None:
        d = {
            "function": "png_set_IHDR",
            "hazard_kind": "ordering",
            "description": "must follow png_create_write_struct",
            "source_evidence": "pngset.c:15",
            "contract_clause": "",
            "contract_property": "",
        }
        c = HazardCandidate.from_dict(d)
        assert c.to_dict() == d


class TestAuditVerdict:
    def test_roundtrip_undocumented(self) -> None:
        d = {
            "documented": False,
            "doc_searched": "manual.txt (no mention)",
            "doc_quote": "",
            "recommendation": "Document the NULL handling.",
        }
        v = AuditVerdict.from_dict(d)
        assert v.to_dict() == d
        assert v.documented is False

    def test_roundtrip_documented(self) -> None:
        d = {
            "documented": True,
            "doc_searched": "manual.txt:42 (clear warning)",
            "doc_quote": "fp must be non-NULL",
            "recommendation": "",
        }
        v = AuditVerdict.from_dict(d)
        assert v.to_dict() == d
        assert v.documented is True


class TestContractGap:
    def test_roundtrip_doc_and_db_gap(self) -> None:
        d = {
            "function": "f",
            "categories": ["missing_contract", "incomplete_contract"],
            "hazard_kind": "ordering",
            "description": "must call setup() before f()",
            "source_evidence": "f.c:42 / contract.requires[memsafe]: state",
            "doc_searched": "manual.txt §3 (no mention)",
            "doc_quote": "",
            "recommendation": "Document the setup() ordering rule.",
            "contract_clause": "state == INITIALIZED",
            "contract_property": "memsafe",
        }
        g = ContractGap.from_dict(d)
        assert g.to_dict() == d

    def test_roundtrip_doc_only(self) -> None:
        # missing_contract alone (no DB gap) — contract_clause/property empty.
        g = ContractGap(
            function="f",
            categories=["missing_contract"],
            hazard_kind="lifecycle",
            description="caller must free returned ptr",
            source_evidence="f.c:10 (returns malloc)",
            doc_searched="manual.txt §2 (no free mention)",
            recommendation="Document that the caller must free the result.",
        )
        d = g.to_dict()
        assert d["categories"] == ["missing_contract"]
        assert d["contract_clause"] == ""
        assert d["contract_property"] == ""


class TestContractCheckResult:
    def test_to_dict(self) -> None:
        gaps = [
            ContractGap(
                function="f",
                categories=["missing_contract"],
                hazard_kind="null_input",
                description="...",
                source_evidence="png.h:1",
                doc_searched="manual (no mention)",
                recommendation="document NULL handling",
            ),
        ]
        r = ContractCheckResult(
            library="libpng", target="png_static",
            summary="x", gaps=gaps, completed=True,
        )
        d = r.to_dict()
        assert d["library"] == "libpng"
        assert d["gap_count"] == 1
        assert d["completed"] is True
        assert len(d["gaps"]) == 1
