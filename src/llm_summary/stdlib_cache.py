"""Global persistent cache for external/stdlib function summaries.

Stores LLM-generated (and hand-crafted) summaries keyed by function name so
they can be reused across projects without re-querying the LLM.

Default location: ~/.llm-summary/stdlib_cache.db
"""

from __future__ import annotations

import importlib.resources
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Known-externals loader
# ---------------------------------------------------------------------------

def load_known_externals(extra_paths: list[str | Path] | None = None) -> frozenset[str]:
    """Return the set of known external function names from bundled abilist files.

    Parses lines of the form ``fun:<name>=<category>`` (DFSan abilist format).
    ``extra_paths`` may supply additional abilist files to merge in.
    """
    names: set[str] = set()

    # Bundled abilists shipped with the package
    try:
        pkg_data = importlib.resources.files("llm_summary").joinpath("data/abilists")
        for entry in pkg_data.iterdir():
            if entry.name.endswith(".txt"):
                _parse_abilist(entry.read_text(encoding="utf-8"), names)
    except (FileNotFoundError, NotADirectoryError):
        pass

    # User-supplied extra files
    for p in (extra_paths or []):
        _parse_abilist(Path(p).read_text(encoding="utf-8"), names)

    return frozenset(names)


def _parse_abilist(text: str, out: set[str]) -> None:
    pattern = re.compile(r"^fun:([^=]+)=")
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            out.add(m.group(1))


# ---------------------------------------------------------------------------
# Cross-libc alias map (glibc abilist names -> musl source names)
# ---------------------------------------------------------------------------

def load_libc_aliases(
    extra_paths: list[str | Path] | None = None,
) -> dict[str, str | None]:
    """Return ``{abilist_name: source_name_or_None}``.

    Loads the bundled JSON files under ``data/libc_aliases/`` and merges in
    any caller-supplied paths.  Later files override earlier ones so users
    can patch vendor-specific differences without editing the bundled data.

    A ``None`` source name marks a glibc-only orphan that has no musl
    equivalent — a builtin contract is required for these (validated at
    ``init-stdlib`` startup).
    """
    aliases: dict[str, str | None] = {}

    try:
        pkg_data = importlib.resources.files("llm_summary").joinpath(
            "data/libc_aliases"
        )
        for entry in pkg_data.iterdir():
            if entry.name.endswith(".json"):
                _merge_alias_file(entry.read_text(encoding="utf-8"), aliases)
    except (FileNotFoundError, NotADirectoryError):
        pass

    for p in (extra_paths or []):
        _merge_alias_file(Path(p).read_text(encoding="utf-8"), aliases)

    return aliases


def _merge_alias_file(text: str, out: dict[str, str | None]) -> None:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("libc-alias file must be a JSON object")
    for glibc_name, entry in data.items():
        if isinstance(entry, dict):
            musl = entry.get("musl")
        elif isinstance(entry, str) or entry is None:
            musl = entry  # shorthand: name -> "musl_name" or null
        else:
            raise ValueError(
                f"libc-alias entry for {glibc_name!r} must be a string, "
                "null, or an object with a 'musl' key"
            )
        if musl is not None and not isinstance(musl, str):
            raise ValueError(
                f"libc-alias entry for {glibc_name!r}: 'musl' must be a "
                "string or null"
            )
        out[glibc_name] = musl


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class StdlibCacheEntry:
    name: str
    allocation_json: str | None   # serialised AllocationSummary or None
    free_json: str | None         # serialised FreeSummary or None
    init_json: str | None         # serialised InitSummary or None
    memsafe_json: str | None      # serialised MemsafeSummary or None
    model_used: str
    code_contract_json: str | None = None   # serialised CodeContractSummary or None
    code_contract_model: str | None = None  # provenance for the code-contract blob


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_SUMMARY_TYPE_TO_COLUMN: dict[str, str] = {
    "code_contract": "code_contract_json",
    "allocation": "allocation_json",
    "free": "free_json",
    "init": "init_json",
    "memsafe": "memsafe_json",
}


class StdlibCache:
    """SQLite-backed cache of stdlib/external function summaries.

    One row per function name.  All four summary types (allocation, free,
    init, memsafe) are stored as JSON blobs; NULL means "not applicable" for
    that function.
    """

    DEFAULT_PATH: Path = Path.home() / ".llm-summary" / "stdlib_cache.db"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stdlib_cache (
                name                 TEXT PRIMARY KEY,
                allocation_json      TEXT,
                free_json            TEXT,
                init_json            TEXT,
                memsafe_json         TEXT,
                model_used           TEXT NOT NULL DEFAULT '',
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                code_contract_json   TEXT,
                code_contract_model  TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dep_headers (
                header_path     TEXT PRIMARY KEY,
                library_name    TEXT NOT NULL,
                dep_db_path     TEXT,
                resolved_by     TEXT NOT NULL DEFAULT 'heuristic',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, name: str, summary_type: str = "code_contract") -> bool:
        """Check whether *name* has a cached summary of the given type.

        ``summary_type`` must be one of ``"code_contract"``, ``"allocation"``,
        ``"free"``, ``"init"``, ``"memsafe"``, or ``"any"`` (any non-NULL blob).
        """
        col = _SUMMARY_TYPE_TO_COLUMN.get(summary_type)
        if col:
            row = self.conn.execute(
                f"SELECT 1 FROM stdlib_cache WHERE name = ? AND {col} IS NOT NULL",
                (name,),
            ).fetchone()
        elif summary_type == "any":
            row = self.conn.execute(
                "SELECT 1 FROM stdlib_cache WHERE name = ? AND ("
                "allocation_json IS NOT NULL OR free_json IS NOT NULL OR "
                "init_json IS NOT NULL OR memsafe_json IS NOT NULL OR "
                "code_contract_json IS NOT NULL)",
                (name,),
            ).fetchone()
        else:
            raise ValueError(
                f"unknown summary_type {summary_type!r}; "
                "expected one of: code_contract, allocation, free, init, memsafe, any"
            )
        return row is not None

    def get(self, name: str) -> StdlibCacheEntry | None:
        row = self.conn.execute(
            "SELECT name, allocation_json, free_json, init_json, memsafe_json, "
            "       model_used, code_contract_json, code_contract_model "
            "FROM stdlib_cache WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return StdlibCacheEntry(
            name=row[0],
            allocation_json=row[1],
            free_json=row[2],
            init_json=row[3],
            memsafe_json=row[4],
            model_used=row[5],
            code_contract_json=row[6],
            code_contract_model=row[7],
        )

    def put(
        self,
        name: str,
        allocation_json: str | None,
        free_json: str | None,
        init_json: str | None,
        memsafe_json: str | None,
        model_used: str,
        code_contract_json: str | None = None,
        code_contract_model: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO stdlib_cache
                (name, allocation_json, free_json, init_json, memsafe_json,
                 model_used, code_contract_json, code_contract_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                allocation_json = COALESCE(
                    excluded.allocation_json, stdlib_cache.allocation_json),
                free_json = COALESCE(
                    excluded.free_json, stdlib_cache.free_json),
                init_json = COALESCE(
                    excluded.init_json, stdlib_cache.init_json),
                memsafe_json = COALESCE(
                    excluded.memsafe_json, stdlib_cache.memsafe_json),
                model_used = excluded.model_used,
                code_contract_json = COALESCE(
                    excluded.code_contract_json,
                    stdlib_cache.code_contract_json),
                code_contract_model = COALESCE(
                    excluded.code_contract_model,
                    stdlib_cache.code_contract_model)
            """,
            (name, allocation_json, free_json, init_json, memsafe_json,
             model_used, code_contract_json, code_contract_model),
        )
        self.conn.commit()

    def seed_builtins(self, force: bool = False) -> int:
        """Populate cache with hand-crafted entries from stdlib.py and
        code_contract/stdlib.py.

        When force=False (default), only inserts rows that do not already
        exist.  When force=True, overwrites any existing DB-seeded entry so
        that hand-crafted builtins always take priority.  Returns the number
        of entries written.
        """
        from .code_contract.stdlib import STDLIB_CONTRACTS as CC_BUILTINS
        from .stdlib import (
            get_all_stdlib_free_summaries,
            get_all_stdlib_init_summaries,
            get_all_stdlib_memsafe_summaries,
            get_all_stdlib_summaries,
        )

        alloc = get_all_stdlib_summaries()
        free = get_all_stdlib_free_summaries()
        init = get_all_stdlib_init_summaries()
        memsafe = get_all_stdlib_memsafe_summaries()

        all_names = set(alloc) | set(free) | set(init) | set(memsafe) | set(CC_BUILTINS)
        added = 0
        for name in sorted(all_names):
            if not force and self.has(name):
                continue
            cc_json = (
                json.dumps(CC_BUILTINS[name].to_dict())
                if name in CC_BUILTINS else None
            )
            self.put(
                name=name,
                allocation_json=json.dumps(alloc[name].to_dict()) if name in alloc else None,
                free_json=json.dumps(free[name].to_dict()) if name in free else None,
                init_json=json.dumps(init[name].to_dict()) if name in init else None,
                memsafe_json=json.dumps(memsafe[name].to_dict()) if name in memsafe else None,
                model_used="builtin",
                code_contract_json=cc_json,
                code_contract_model="builtin" if cc_json else None,
            )
            added += 1
        return added

    def list_names(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM stdlib_cache ORDER BY name").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Dependency header resolution cache
    # ------------------------------------------------------------------

    def get_dep_header(self, header_path: str) -> tuple[str, str, str] | None:
        """Look up a cached header resolution.

        Returns (library_name, dep_db_path, resolved_by) or None.
        """
        row = self.conn.execute(
            "SELECT library_name, dep_db_path, resolved_by "
            "FROM dep_headers WHERE header_path = ?",
            (header_path,),
        ).fetchone()
        if row is None:
            return None
        return (row[0], row[1], row[2])

    def put_dep_header(
        self,
        header_path: str,
        library_name: str,
        dep_db_path: str | None,
        resolved_by: str = "heuristic",
    ) -> None:
        """Cache a header -> library resolution."""
        self.conn.execute(
            """
            INSERT INTO dep_headers (header_path, library_name, dep_db_path, resolved_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(header_path) DO UPDATE SET
                library_name = excluded.library_name,
                dep_db_path  = excluded.dep_db_path,
                resolved_by  = excluded.resolved_by
            """,
            (header_path, library_name, dep_db_path, resolved_by),
        )
        self.conn.commit()

    def get_all_dep_headers(self) -> list[tuple[str, str, str | None, str]]:
        """Return all cached header resolutions.

        Returns list of (header_path, library_name, dep_db_path, resolved_by).
        """
        rows = self.conn.execute(
            "SELECT header_path, library_name, dep_db_path, resolved_by "
            "FROM dep_headers ORDER BY header_path"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def close(self) -> None:
        self.conn.close()
