"""Tests for the cross-libc alias map (glibc abilist names -> musl source names)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_summary.stdlib_cache import load_libc_aliases


class TestLoadLibcAliases:
    def test_bundled_map_loads(self) -> None:
        m = load_libc_aliases()
        assert isinstance(m, dict)
        assert len(m) > 0

    def test_known_problem_cases_present(self) -> None:
        m = load_libc_aliases()
        assert m.get("fstat") == "__fstat"
        assert m.get("dlopen") == "stub_dlopen"
        assert m.get("__isoc23_strtol") == "strtol"

    def test_extra_overrides_bundled(self, tmp_path: Path) -> None:
        extra = tmp_path / "override.json"
        extra.write_text(json.dumps({
            "fstat": {"musl": "vendor_fstat", "via": "vendor"},
            "vendor_only": {"musl": "vendor_func", "via": "vendor"},
        }))
        m = load_libc_aliases(extra_paths=[extra])
        assert m["fstat"] == "vendor_fstat"
        assert m["vendor_only"] == "vendor_func"
        assert m.get("dlopen") == "stub_dlopen"

    def test_shorthand_string_value(self, tmp_path: Path) -> None:
        extra = tmp_path / "shorthand.json"
        extra.write_text(json.dumps({"foo": "bar", "baz": None}))
        m = load_libc_aliases(extra_paths=[extra])
        assert m["foo"] == "bar"
        assert m["baz"] is None

    def test_invalid_entry_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": 42}))
        with pytest.raises(ValueError, match="must be a string, null, or"):
            load_libc_aliases(extra_paths=[bad])

    def test_invalid_musl_field_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": {"musl": 42}}))
        with pytest.raises(ValueError, match="'musl' must be a"):
            load_libc_aliases(extra_paths=[bad])

    def test_top_level_must_be_object(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_libc_aliases(extra_paths=[bad])


class TestBundledMapIntegrity:
    """Sanity checks on the bundled glibc_to_musl.json contents."""

    def test_orphans_have_builtin_contracts(self) -> None:
        """Every musl=null entry must have a hand-crafted code-contract.

        This mirrors the validation init-stdlib runs at startup.
        """
        from llm_summary.code_contract.stdlib import STDLIB_CONTRACTS
        m = load_libc_aliases()
        orphans = sorted(n for n, musl in m.items() if musl is None)
        missing = [n for n in orphans if n not in STDLIB_CONTRACTS]
        assert not missing, (
            f"libc-alias orphans without builtin contracts: {missing}"
        )
