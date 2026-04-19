"""SQLite database for storing functions, summaries, and call graph."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import (
    AddressFlow,
    AddressFlowSummary,
    AddressTakenFunction,
    Allocation,
    AllocationSummary,
    AllocationType,
    BufferSizePair,
    CallEdge,
    ContainerSummary,
    FlowDestination,
    FreeOp,
    FreeSummary,
    Function,
    FunctionBlock,
    IndirectCallsite,
    IndirectCallTarget,
    InitOp,
    InitSummary,
    IntegerConstraint,
    IntegerOverflowSummary,
    LeakSummary,
    MemsafeContract,
    MemsafeSummary,
    OutputRange,
    ParameterInfo,
    SafetyIssue,
    VerificationSummary,
)

if TYPE_CHECKING:
    from .code_contract.models import CodeContractSummary

SCHEMA = """
-- Functions table
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,
    canonical_signature TEXT,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,
    pp_source TEXT,
    source_hash TEXT,
    params_json TEXT,
    callsites_json TEXT,
    attributes TEXT DEFAULT '',
    decl_header TEXT,
    UNIQUE(name, signature, file_path)
);

-- Allocation summaries table
CREATE TABLE IF NOT EXISTS allocation_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Free/deallocation summaries table
CREATE TABLE IF NOT EXISTS free_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Initialization summaries table
CREATE TABLE IF NOT EXISTS init_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Safety contract summaries table (pre-conditions)
CREATE TABLE IF NOT EXISTS memsafe_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Verification summaries table (Pass 5)
CREATE TABLE IF NOT EXISTS verification_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Leak summaries table (leak detection pass)
CREATE TABLE IF NOT EXISTS leak_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Integer overflow summaries table (integer overflow detection pass)
CREATE TABLE IF NOT EXISTS integer_overflow_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Code-contract summaries table (code-as-summary Hoare-style pipeline).
-- Holds requires/ensures/modifies/notes/origin per property (memsafe,
-- memleak, overflow), plus a property-independent noreturn flag. NO
-- verdict field by design — entry-point check is a separate Phase 4 pass.
CREATE TABLE IF NOT EXISTS code_contract_summaries (
    function_id INTEGER PRIMARY KEY REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    noreturn INTEGER NOT NULL DEFAULT 0,
    body_annotated TEXT,
    model TEXT NOT NULL,
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    tokens_cache_read INTEGER NOT NULL DEFAULT 0,
    tokens_cache_write INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Call graph edges (with callsite info)
CREATE TABLE IF NOT EXISTS call_edges (
    id INTEGER PRIMARY KEY,
    caller_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    callee_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    is_indirect INTEGER DEFAULT 0,
    file_path TEXT,
    line INTEGER,
    column INTEGER
);

-- Address-taken functions
CREATE TABLE IF NOT EXISTS address_taken_functions (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    signature TEXT NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'address_taken',
    UNIQUE(function_id, target_type)
);

-- Where function addresses flow to
CREATE TABLE IF NOT EXISTS address_flows (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    flow_target TEXT NOT NULL,
    file_path TEXT,
    line_number INTEGER,
    context_snippet TEXT,
    UNIQUE(function_id, flow_target, file_path, line_number)
);

-- Indirect call sites
CREATE TABLE IF NOT EXISTS indirect_callsites (
    id INTEGER PRIMARY KEY,
    caller_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    callee_expr TEXT NOT NULL,
    signature TEXT NOT NULL,
    context_snippet TEXT,
    UNIQUE(caller_function_id, file_path, line_number, callee_expr)
);

-- Resolved indirect call targets
CREATE TABLE IF NOT EXISTS indirect_call_targets (
    callsite_id INTEGER REFERENCES indirect_callsites(id) ON DELETE CASCADE,
    target_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    confidence TEXT,
    llm_reasoning TEXT,
    PRIMARY KEY(callsite_id, target_function_id)
);

-- LLM-generated flow summaries for address-taken functions (Pass 1)
CREATE TABLE IF NOT EXISTS address_flow_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    flow_destinations_json TEXT NOT NULL,
    semantic_role TEXT,
    likely_callers_json TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id)
);

-- Build configurations table
CREATE TABLE IF NOT EXISTS build_configs (
    project_path TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    build_system TEXT NOT NULL,
    configuration_json TEXT,
    script_path TEXT,
    artifacts_dir TEXT,
    compile_commands_path TEXT,
    llm_backend TEXT,
    llm_model TEXT,
    build_attempts INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_built_at TIMESTAMP
);

-- Container function summaries
CREATE TABLE IF NOT EXISTS container_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    container_arg INTEGER NOT NULL,
    store_args_json TEXT NOT NULL,
    load_return INTEGER NOT NULL DEFAULT 0,
    container_type TEXT NOT NULL,
    confidence TEXT NOT NULL,
    heuristic_score INTEGER,
    heuristic_signals_json TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id)
);

-- Type declarations extracted from source (typedefs, using aliases, struct/class/union)
CREATE TABLE IF NOT EXISTS typedefs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'typedef',
    underlying_type TEXT NOT NULL,
    canonical_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    definition TEXT,
    pp_definition TEXT,
    UNIQUE(name, kind, file_path)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_path);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee_id);
CREATE INDEX IF NOT EXISTS idx_address_flows_function ON address_flows(function_id);
CREATE INDEX IF NOT EXISTS idx_indirect_callsites_caller ON indirect_callsites(caller_function_id);
CREATE INDEX IF NOT EXISTS idx_flow_summaries_function ON address_flow_summaries(function_id);
CREATE INDEX IF NOT EXISTS idx_build_configs_name ON build_configs(project_name);
CREATE INDEX IF NOT EXISTS idx_container_summaries_function ON container_summaries(function_id);
CREATE INDEX IF NOT EXISTS idx_typedefs_name ON typedefs(name);
CREATE INDEX IF NOT EXISTS idx_typedefs_file ON typedefs(file_path);

-- Issue reviews (human annotations on verification issues)
CREATE TABLE IF NOT EXISTS issue_reviews (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    issue_index INTEGER NOT NULL,
    issue_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id, issue_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_issue_reviews_function ON issue_reviews(function_id);
CREATE INDEX IF NOT EXISTS idx_issue_reviews_status ON issue_reviews(status);

-- Function blocks (switch-case chunks for large function summarization)
CREATE TABLE IF NOT EXISTS function_blocks (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,
    suggested_name TEXT,
    suggested_signature TEXT,
    summary_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_blocks_function ON function_blocks(function_id);

-- Per-function IR facts imported from KAMain's --ir-sidecar-dir output.
-- One row per function in this DB whose name matches a function in any
-- imported sidecar. facts_json holds the raw per-function blob (effects,
-- branches, ranges, int_ops, features, ir_hash, cg_hash).
CREATE TABLE IF NOT EXISTS function_ir_facts (
    function_id INTEGER PRIMARY KEY REFERENCES functions(id) ON DELETE CASCADE,
    ir_hash TEXT,
    cg_hash TEXT,
    facts_json TEXT NOT NULL,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def compute_source_hash(source: str) -> str:
    """Compute a hash of the source code for change detection."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


class SummaryDB:
    """SQLite database interface for the summary system."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._local = threading.local()
        # Bootstrap the main-thread connection and run schema init
        self._local.conn = self._make_conn()
        self._init_schema()

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new connection with standard pragmas."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Per-thread connection (created on first access per thread)."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._make_conn()
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply schema migrations for existing databases."""
        # Add target_type column to address_taken_functions if missing
        cursor = self.conn.execute("PRAGMA table_info(address_taken_functions)")
        columns = {row[1] for row in cursor.fetchall()}
        if "target_type" not in columns:
            self.conn.execute(
                "ALTER TABLE address_taken_functions "
                "ADD COLUMN target_type TEXT NOT NULL "
                "DEFAULT 'address_taken'"
            )
            # Recreate unique index to include target_type.
            # Drop old unique constraint by recreating the table
            # is complex; instead just create a new unique index
            # (old UNIQUE(function_id) stays as-is for old DBs)
            self.conn.commit()

        # Ensure file-path index on typedefs exists (added for static_var support)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_typedefs_file ON typedefs(file_path)"
        )
        self.conn.commit()

        # Add kind column to typedefs if missing
        cursor = self.conn.execute("PRAGMA table_info(typedefs)")
        columns = {row[1] for row in cursor.fetchall()}
        if columns and "kind" not in columns:
            self.conn.execute(
                "ALTER TABLE typedefs ADD COLUMN kind TEXT NOT NULL DEFAULT 'typedef'"
            )
            self.conn.commit()
        if columns and "definition" not in columns:
            self.conn.execute("ALTER TABLE typedefs ADD COLUMN definition TEXT")
            self.conn.commit()
        if columns and "pp_definition" not in columns:
            self.conn.execute("ALTER TABLE typedefs ADD COLUMN pp_definition TEXT")
            self.conn.commit()

        # Add canonical_signature, params_json, callsites_json columns to functions if missing
        cursor = self.conn.execute("PRAGMA table_info(functions)")
        columns = {row[1] for row in cursor.fetchall()}
        if "canonical_signature" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN canonical_signature TEXT")
            self.conn.commit()
        if "params_json" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN params_json TEXT")
            self.conn.commit()
        if "callsites_json" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN callsites_json TEXT")
            self.conn.commit()
        if "pp_source" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN pp_source TEXT")
            self.conn.commit()
        if "attributes" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN attributes TEXT DEFAULT ''")
            self.conn.commit()
        if "decl_header" not in columns:
            self.conn.execute("ALTER TABLE functions ADD COLUMN decl_header TEXT")
            self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # ========== Function Operations ==========

    def insert_function(self, func: Function) -> int:
        """Insert a function and return its ID.

        Uses ON CONFLICT DO UPDATE to preserve the existing row ID on re-scan,
        so that foreign-key-referenced call_edges and summaries are not cascade-deleted.
        Only source-derived columns are updated on conflict; summaries are left intact.
        """
        import json as _json
        # Use pp_source for hash when available (macro-expanded is the LLM-visible source)
        hash_source = func.pp_source if func.pp_source else func.source
        source_hash = compute_source_hash(hash_source) if hash_source else None
        params_json = _json.dumps(func.params) if func.params else None
        callsites_json = _json.dumps(func.callsites) if func.callsites else None
        row = self.conn.execute(
            """
            INSERT INTO functions
            (name, signature, canonical_signature, file_path, line_start, line_end,
             source, pp_source, source_hash, params_json, callsites_json, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, signature, file_path) DO UPDATE SET
              canonical_signature = excluded.canonical_signature,
              line_start          = excluded.line_start,
              line_end            = excluded.line_end,
              source              = excluded.source,
              pp_source           = excluded.pp_source,
              source_hash         = excluded.source_hash,
              params_json         = excluded.params_json,
              callsites_json      = excluded.callsites_json,
              attributes          = excluded.attributes
            RETURNING id
            """,
            (
                func.name,
                func.signature,
                func.canonical_signature,
                func.file_path,
                func.line_start,
                func.line_end,
                func.source,
                func.pp_source,
                source_hash,
                params_json,
                callsites_json,
                func.attributes or "",
            ),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def insert_functions_batch(self, functions: list[Function]) -> dict[Function, int]:
        """Batch insert functions and return mapping to IDs.

        Also persists any FunctionBlock instances attached to each Function.
        Stale blocks from prior scans are always deleted.
        """
        result = {}
        for func in functions:
            func_id = self.insert_function(func)
            func.id = func_id
            result[func] = func_id
            # Always delete old blocks (handles re-scan with changed threshold)
            self.delete_function_blocks(func_id)
            if func.blocks:
                for block in func.blocks:
                    block.function_id = func_id
                self.insert_function_blocks(func.blocks)
        return result

    def get_function(self, func_id: int) -> Function | None:
        """Get a function by ID."""
        row = self.conn.execute(
            "SELECT * FROM functions WHERE id = ?", (func_id,)
        ).fetchone()
        if row:
            return self._row_to_function(row)
        return None

    def get_function_by_name(
        self, name: str, signature: str | None = None
    ) -> list[Function]:
        """Get functions by name, optionally filtering by signature."""
        if signature:
            rows = self.conn.execute(
                "SELECT * FROM functions WHERE name = ? AND signature = ?",
                (name, signature),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM functions WHERE name = ?", (name,)
            ).fetchall()
        return [self._row_to_function(row) for row in rows]

    def get_functions_by_file(self, file_path: str) -> list[Function]:
        """Get all functions in a file."""
        rows = self.conn.execute(
            "SELECT * FROM functions WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [self._row_to_function(row) for row in rows]

    def get_all_functions(self) -> list[Function]:
        """Get all functions in the database."""
        rows = self.conn.execute("SELECT * FROM functions").fetchall()
        return [self._row_to_function(row) for row in rows]

    def get_all_function_ids(self) -> list[tuple[int, str]]:
        """Get (id, name) for all functions — no source columns loaded."""
        rows = self.conn.execute(
            "SELECT id, name FROM functions",
        ).fetchall()
        return [(row["id"], row["name"]) for row in rows]

    def get_function_id_by_name(self, name: str) -> int | None:
        """Get function ID by name — no source columns loaded."""
        row = self.conn.execute(
            "SELECT id FROM functions WHERE name = ? LIMIT 1", (name,),
        ).fetchone()
        return row["id"] if row else None

    def _row_to_function(self, row: sqlite3.Row) -> Function:
        """Convert a database row to a Function object."""
        import json as _json

        def _col(name: str, default=None):
            try:
                return row[name]
            except (IndexError, KeyError):
                return default

        params_raw = _col("params_json")
        callsites_raw = _col("callsites_json")
        return Function(
            id=row["id"],
            name=row["name"],
            signature=row["signature"],
            canonical_signature=_col("canonical_signature"),
            file_path=row["file_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            source=row["source"],
            source_hash=row["source_hash"],
            params=_json.loads(params_raw) if params_raw else [],
            callsites=_json.loads(callsites_raw) if callsites_raw else [],
            pp_source=_col("pp_source"),
            attributes=_col("attributes", "") or "",
            decl_header=_col("decl_header"),
        )

    def update_callsites(
        self, function_id: int, callsites: list[dict],
    ) -> None:
        """Update the callsites_json for a function."""
        import json as _json

        self.conn.execute(
            "UPDATE functions SET callsites_json = ? WHERE id = ?",
            (_json.dumps(callsites), function_id),
        )
        self.conn.commit()

    def update_function_attributes(
        self, function_id: int | None, attributes: str,
    ) -> None:
        """Update the attributes for a function."""
        if function_id is None:
            return
        self.conn.execute(
            "UPDATE functions SET attributes = ? WHERE id = ?",
            (attributes, function_id),
        )
        self.conn.commit()

    # ========== Function Block Operations ==========

    def insert_function_blocks(self, blocks: list[FunctionBlock]) -> list[int | None]:
        """Insert function blocks and return their IDs."""
        ids: list[int | None] = []
        for block in blocks:
            cursor = self.conn.execute(
                """
                INSERT INTO function_blocks
                (function_id, kind, label, line_start, line_end, source,
                 suggested_name, suggested_signature, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.function_id,
                    block.kind,
                    block.label,
                    block.line_start,
                    block.line_end,
                    block.source,
                    block.suggested_name,
                    block.suggested_signature,
                    block.summary_json,
                ),
            )
            block.id = cursor.lastrowid
            ids.append(cursor.lastrowid)
        self.conn.commit()
        return ids

    def get_function_blocks(self, function_id: int) -> list[FunctionBlock]:
        """Get all blocks for a function, ordered by line_start."""
        rows = self.conn.execute(
            "SELECT * FROM function_blocks WHERE function_id = ? ORDER BY line_start",
            (function_id,),
        ).fetchall()
        return [
            FunctionBlock(
                id=row["id"],
                function_id=row["function_id"],
                kind=row["kind"],
                label=row["label"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                source=row["source"],
                suggested_name=row["suggested_name"],
                suggested_signature=row["suggested_signature"],
                summary_json=row["summary_json"],
            )
            for row in rows
        ]

    def delete_function_blocks(self, function_id: int) -> None:
        """Delete all blocks for a function."""
        self.conn.execute(
            "DELETE FROM function_blocks WHERE function_id = ?",
            (function_id,),
        )
        self.conn.commit()

    def update_function_block_summary(
        self,
        block_id: int,
        summary_json: str,
        suggested_name: str | None = None,
        suggested_signature: str | None = None,
    ) -> None:
        """Update a block's summary and suggested name/signature."""
        self.conn.execute(
            """
            UPDATE function_blocks
            SET summary_json = ?, suggested_name = ?, suggested_signature = ?
            WHERE id = ?
            """,
            (summary_json, suggested_name, suggested_signature, block_id),
        )
        self.conn.commit()

    # ========== Summary Operations ==========

    def get_summary(
        self, name: str, signature: str | None = None
    ) -> AllocationSummary | None:
        """Get allocation summary for a function."""
        if signature:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM allocation_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ? AND f.signature = ?
                """,
                (name, signature),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM allocation_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ?
                """,
                (name,),
            ).fetchone()

        if row:
            return self._json_to_summary(row["summary_json"])
        return None

    def get_summary_by_function_id(self, func_id: int) -> AllocationSummary | None:
        """Get allocation summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM allocation_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_summary(row["summary_json"])
        return None

    def get_summaries_by_file(self, file_path: str) -> list[AllocationSummary]:
        """Get all summaries for functions in a file."""
        rows = self.conn.execute(
            """
            SELECT s.summary_json FROM allocation_summaries s
            JOIN functions f ON s.function_id = f.id
            WHERE f.file_path = ?
            """,
            (file_path,),
        ).fetchall()
        return [self._json_to_summary(row["summary_json"]) for row in rows]

    def upsert_summary(
        self, func: Function, summary: AllocationSummary, model_used: str = ""
    ) -> None:
        """Insert or update an allocation summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO allocation_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def _json_to_summary(self, json_str: str) -> AllocationSummary:
        """Convert JSON string to AllocationSummary."""
        data = json.loads(json_str)
        allocations = [
            Allocation(
                alloc_type=AllocationType(a.get("type", "unknown")),
                source=a.get("source", ""),
                size_expr=a.get("size_expr"),
                size_params=a.get("size_params", []),
                returned=a.get("returned", False),
                stored_to=a.get("stored_to"),
                may_be_null=a.get("may_be_null", True),
            )
            for a in data.get("allocations", [])
        ]
        parameters = {
            k: ParameterInfo(
                role=v.get("role", ""),
                used_in_allocation=v.get("used_in_allocation", False),
            )
            for k, v in data.get("parameters", {}).items()
        }
        buffer_size_pairs = [
            BufferSizePair(
                buffer=p.get("buffer", ""),
                size=p.get("size", ""),
                kind=p.get("kind", "param_pair"),
                relationship=p.get("relationship", ""),
            )
            for p in data.get("buffer_size_pairs", [])
        ]
        return AllocationSummary(
            function_name=data.get("function", ""),
            allocations=allocations,
            parameters=parameters,
            buffer_size_pairs=buffer_size_pairs,
            description=data.get("description", ""),
        )

    def needs_update(self, func: Function) -> bool:
        """Check if a function's summary needs updating based on source hash."""
        hash_source = func.pp_source if func.pp_source else func.source
        current_hash = compute_source_hash(hash_source) if hash_source else None
        row = self.conn.execute(
            """
            SELECT f.source_hash, s.id FROM functions f
            LEFT JOIN allocation_summaries s ON f.id = s.function_id
            WHERE f.name = ? AND f.signature = ? AND f.file_path = ?
            """,
            (func.name, func.signature, func.file_path),
        ).fetchone()

        if not row:
            return True  # Function not in DB
        if row["id"] is None:
            return True  # No summary exists
        if row["source_hash"] != current_hash:
            return True  # Source has changed

        return False

    def needs_flow_update(self, function_id: int) -> bool:
        """Check if a function's flow summary needs updating based on source hash."""
        row = self.conn.execute(
            """
            SELECT f.source_hash, s.id, s.created_at FROM functions f
            LEFT JOIN address_flow_summaries s ON f.id = s.function_id
            WHERE f.id = ?
            """,
            (function_id,),
        ).fetchone()

        if not row:
            return True  # Function not in DB
        if row["id"] is None:
            return True  # No flow summary exists

        # Check if function source changed after flow summary was created
        # by comparing source hash (summary is stored with the function's current hash)
        return False  # Flow summary exists

    def get_function_source_hash(self, function_id: int) -> str | None:
        """Get the source hash for a function."""
        row = self.conn.execute(
            "SELECT source_hash FROM functions WHERE id = ?",
            (function_id,),
        ).fetchone()
        return row["source_hash"] if row else None

    def touch_stub_summaries(self) -> int:
        """Bump updated_at to now for all summary rows of callee stubs.

        Called after --clear-edges callgraph reimport so that the next
        --incremental run detects callers of newly-linked stubs as stale.
        Returns the number of rows touched across all summary tables.
        """
        touched = 0
        stub_ids_sql = """
            SELECT DISTINCT ce.callee_id FROM call_edges ce
            JOIN functions f ON f.id = ce.callee_id
            WHERE f.source IS NULL OR f.source = ''
        """
        for table in (
            "allocation_summaries",
            "free_summaries",
            "init_summaries",
            "memsafe_summaries",
            "verification_summaries",
        ):
            try:
                cur = self.conn.execute(
                    f"UPDATE {table} SET updated_at = datetime('now') "
                    f"WHERE function_id IN ({stub_ids_sql})"
                )
                touched += cur.rowcount
            except Exception:
                pass
        self.conn.commit()
        return touched

    def find_dirty_function_ids(self, pass_table: str) -> set[int]:
        """Find function IDs that need re-summarization for a given pass.

        A function is dirty if:
        1. No summary exists for this pass
        2. Source hash doesn't match (source code changed)
        3. Any callee's summary updated_at > this function's summary updated_at
        """
        # Whitelist valid table names to prevent SQL injection
        valid_tables = {
            "allocation_summaries", "free_summaries", "init_summaries",
            "memsafe_summaries", "verification_summaries",
            "leak_summaries", "integer_overflow_summaries",
        }
        if pass_table not in valid_tables:
            raise ValueError(f"Invalid pass table: {pass_table}")

        dirty: set[int] = set()

        # 1. No summary exists for this pass (skip sourceless stubs)
        rows = self.conn.execute(
            f"""
            SELECT f.id FROM functions f
            LEFT JOIN {pass_table} s ON s.function_id = f.id
            WHERE s.id IS NULL AND f.source IS NOT NULL AND f.source != ''
            """,
        ).fetchall()
        dirty.update(row["id"] for row in rows)

        # 2. Any callee's summary is newer than this function's summary.
        #    Only consider callers that have source — sourceless stubs cannot
        #    be re-summarized, so flagging them as dirty is pointless and
        #    would mask the real source functions that need updating.
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT ce.caller_id
            FROM call_edges ce
            JOIN functions f ON f.id = ce.caller_id
            JOIN {pass_table} callee_s ON callee_s.function_id = ce.callee_id
            JOIN {pass_table} caller_s ON caller_s.function_id = ce.caller_id
            WHERE callee_s.updated_at > caller_s.updated_at
              AND f.source IS NOT NULL AND f.source != ''
            """,
        ).fetchall()
        dirty.update(row["caller_id"] for row in rows)

        # 3. For verification pass: also dirty when a callee's earlier-pass
        #    summary (allocation, free, init, memsafe) is newer than this
        #    function's verification summary.  The verify pass reads all
        #    earlier-pass data from callees, so changes there must trigger
        #    re-verification of callers.
        if pass_table == "verification_summaries":
            upstream_tables = [
                "allocation_summaries", "free_summaries",
                "init_summaries", "memsafe_summaries",
            ]
            for upstream in upstream_tables:
                rows = self.conn.execute(
                    f"""
                    SELECT DISTINCT ce.caller_id
                    FROM call_edges ce
                    JOIN functions f ON f.id = ce.caller_id
                    JOIN {upstream} callee_s
                      ON callee_s.function_id = ce.callee_id
                    JOIN verification_summaries caller_s
                      ON caller_s.function_id = ce.caller_id
                    WHERE callee_s.updated_at > caller_s.updated_at
                      AND f.source IS NOT NULL AND f.source != ''
                    """,
                ).fetchall()
                dirty.update(row["caller_id"] for row in rows)

        return dirty

    # ========== Free Summary Operations ==========

    def upsert_free_summary(
        self, func: Function, summary: FreeSummary, model_used: str = ""
    ) -> None:
        """Insert or update a free summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO free_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_free_summary(
        self, name: str, signature: str | None = None
    ) -> FreeSummary | None:
        """Get free summary for a function by name."""
        if signature:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM free_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ? AND f.signature = ?
                """,
                (name, signature),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM free_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ?
                """,
                (name,),
            ).fetchone()

        if row:
            return self._json_to_free_summary(row["summary_json"])
        return None

    def get_free_summary_by_function_id(self, func_id: int) -> FreeSummary | None:
        """Get free summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM free_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_free_summary(row["summary_json"])
        return None

    def _json_to_free_summary(self, json_str: str) -> FreeSummary:
        """Convert JSON string to FreeSummary."""
        data = json.loads(json_str)

        def _parse_ops(items: list[dict]) -> list[FreeOp]:
            ops = []
            for f in items:
                conditional = f.get("conditional", False)
                ops.append(FreeOp(
                    target=f.get("target", ""),
                    target_kind=f.get("target_kind", "local"),
                    deallocator=f.get("deallocator", "free"),
                    conditional=conditional,
                    nulled_after=f.get("nulled_after", False),
                    condition=f.get("condition") if conditional else None,
                    description=f.get("description"),
                ))
            return ops

        return FreeSummary(
            function_name=data.get("function", ""),
            frees=_parse_ops(data.get("frees", [])),
            resource_releases=_parse_ops(data.get("resource_releases", [])),
            description=data.get("description", ""),
        )

    # ========== Init Summary Operations ==========

    def upsert_init_summary(
        self, func: Function, summary: InitSummary, model_used: str = ""
    ) -> None:
        """Insert or update an init summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO init_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_init_summary_by_function_id(self, func_id: int) -> InitSummary | None:
        """Get init summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM init_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_init_summary(row["summary_json"])
        return None

    def _json_to_init_summary(self, json_str: str) -> InitSummary:
        """Convert JSON string to InitSummary."""
        data = json.loads(json_str)
        vague_byte_counts = ("full", "N/A", "n/a", "unknown", "varies", "")
        inits = []
        for i in data.get("inits", []):
            bc = i.get("byte_count")
            if bc in vague_byte_counts:
                bc = None
            inits.append(InitOp(
                target=i.get("target", ""),
                target_kind=i.get("target_kind", "parameter"),
                initializer=i.get("initializer", "assignment"),
                byte_count=bc,
            ))
        output_ranges = [
            OutputRange(
                target=o.get("target", "return"),
                range=o.get("range", ""),
                description=o.get("description", ""),
            )
            for o in data.get("output_ranges", [])
        ]
        noreturn = bool(data.get("noreturn", False))
        noreturn_condition = data.get("noreturn_condition") or None

        return InitSummary(
            function_name=data.get("function", ""),
            inits=inits,
            output_ranges=output_ranges,
            description=data.get("description", ""),
            noreturn=noreturn,
            noreturn_condition=noreturn_condition,
        )

    # ========== Memsafe Summary Operations ==========

    def upsert_memsafe_summary(
        self, func: Function, summary: MemsafeSummary, model_used: str = ""
    ) -> None:
        """Insert or update a memsafe summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO memsafe_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_memsafe_summary_by_function_id(self, func_id: int) -> MemsafeSummary | None:
        """Get memsafe summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM memsafe_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_memsafe_summary(row["summary_json"])
        return None

    def _json_to_memsafe_summary(self, json_str: str) -> MemsafeSummary:
        """Convert JSON string to MemsafeSummary."""
        data = json.loads(json_str)
        contracts = [
            MemsafeContract(
                target=c.get("target", ""),
                contract_kind=c.get("contract_kind", "disallow_null"),
                description=c.get("description", ""),
                size_expr=c.get("size_expr"),
                relationship=c.get("relationship"),
            )
            for c in data.get("contracts", [])
        ]
        return MemsafeSummary(
            function_name=data.get("function", ""),
            contracts=contracts,
            description=data.get("description", ""),
        )

    # ========== Verification Summary Operations ==========

    def upsert_verification_summary(
        self, func: Function, summary: VerificationSummary, model_used: str = ""
    ) -> None:
        """Insert or update a verification summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO verification_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_verification_summary_by_function_id(
        self, func_id: int
    ) -> VerificationSummary | None:
        """Get verification summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM verification_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_verification_summary(row["summary_json"])
        return None

    def _json_to_verification_summary(self, json_str: str) -> VerificationSummary:
        """Convert JSON string to VerificationSummary."""
        data = json.loads(json_str)
        raw_sc = data.get("simplified_contracts")  # None or list
        if raw_sc is None:
            contracts = None
        else:
            contracts = [
                MemsafeContract(
                    target=c.get("target", ""),
                    contract_kind=c.get("contract_kind", "disallow_null"),
                    description=c.get("description", ""),
                    size_expr=c.get("size_expr"),
                    relationship=c.get("relationship"),
                )
                for c in raw_sc
            ]
        issues = [
            SafetyIssue(
                location=i.get("location", ""),
                issue_kind=i.get("issue_kind", "null_deref"),
                description=i.get("description", ""),
                severity=i.get("severity", "medium"),
                callee=i.get("callee"),
                contract_kind=i.get("contract_kind"),
            )
            for i in data.get("issues", [])
        ]
        return VerificationSummary(
            function_name=data.get("function", ""),
            simplified_contracts=contracts,
            issues=issues,
            description=data.get("description", ""),
        )

    # ========== Leak Summary Operations ==========

    def upsert_leak_summary(
        self, func: Function, summary: "LeakSummary", model_used: str = ""
    ) -> None:
        """Insert or update a leak summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO leak_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_leak_summary_by_function_id(
        self, func_id: int
    ) -> "LeakSummary | None":
        """Get leak summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM leak_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_leak_summary(row["summary_json"])
        return None

    def _json_to_leak_summary(self, json_str: str) -> "LeakSummary":
        """Convert JSON string to LeakSummary."""
        data = json.loads(json_str)
        allocations = [
            Allocation(
                alloc_type=AllocationType(a.get("type", "heap")),
                source=a.get("source", ""),
                size_expr=a.get("size_expr"),
                returned=a.get("returned", False),
                stored_to=a.get("stored_to"),
                may_be_null=a.get("may_be_null", True),
            )
            for a in data.get("simplified_allocations", [])
        ]
        frees = [
            FreeOp(
                target=f.get("target", ""),
                target_kind=f.get("target_kind", "parameter"),
                deallocator=f.get("deallocator", "free"),
                conditional=f.get("conditional", False),
                nulled_after=f.get("nulled_after", False),
                condition=f.get("condition"),
                description=f.get("description"),
            )
            for f in data.get("simplified_frees", [])
        ]
        issues = [
            SafetyIssue(
                location=i.get("location", ""),
                issue_kind=i.get("issue_kind", "memory_leak"),
                description=i.get("description", ""),
                severity=i.get("severity", "medium"),
            )
            for i in data.get("issues", [])
        ]
        return LeakSummary(
            function_name=data.get("function", ""),
            simplified_allocations=allocations,
            simplified_frees=frees,
            issues=issues,
            description=data.get("description", ""),
        )

    # ========== Integer Overflow Summary Operations ==========

    def upsert_integer_overflow_summary(
        self,
        func: "Function",
        summary: "IntegerOverflowSummary",
        model_used: str = "",
    ) -> None:
        """Insert or update an integer overflow summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO integer_overflow_summaries
                (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def get_integer_overflow_summary_by_function_id(
        self, func_id: int
    ) -> "IntegerOverflowSummary | None":
        """Get integer overflow summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM integer_overflow_summaries "
            "WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_integer_overflow_summary(row["summary_json"])
        return None

    def _json_to_integer_overflow_summary(
        self, json_str: str
    ) -> "IntegerOverflowSummary":
        """Convert JSON string to IntegerOverflowSummary."""
        data = json.loads(json_str)
        constraints = [
            IntegerConstraint(
                target=c.get("target", ""),
                range=c.get("range", ""),
                description=c.get("description", ""),
            )
            for c in data.get("constraints", [])
        ]
        output_ranges = [
            OutputRange(
                target=o.get("target", ""),
                range=o.get("range", ""),
                description=o.get("description", ""),
            )
            for o in data.get("output_ranges", [])
        ]
        issues = [
            SafetyIssue(
                location=i.get("location", ""),
                issue_kind=i.get("issue_kind", "integer_overflow"),
                description=i.get("description", ""),
                severity=i.get("severity", "medium"),
            )
            for i in data.get("issues", [])
        ]
        return IntegerOverflowSummary(
            function_name=data.get("function", ""),
            constraints=constraints,
            output_ranges=output_ranges,
            issues=issues,
            description=data.get("description", ""),
        )

    # ========== IR Sidecar Facts Operations ==========

    def upsert_ir_facts(
        self,
        function_id: int,
        ir_hash: str | None,
        cg_hash: str | None,
        facts_json: str,
    ) -> None:
        """Insert or replace per-function IR facts (KAMain sidecar)."""
        self.conn.execute(
            """
            INSERT INTO function_ir_facts
                (function_id, ir_hash, cg_hash, facts_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                ir_hash = excluded.ir_hash,
                cg_hash = excluded.cg_hash,
                facts_json = excluded.facts_json,
                imported_at = CURRENT_TIMESTAMP
            """,
            (function_id, ir_hash, cg_hash, facts_json),
        )

    def get_ir_facts(self, function_id: int) -> dict[str, Any] | None:
        """Get parsed per-function IR facts; None if not imported."""
        row = self.conn.execute(
            "SELECT facts_json FROM function_ir_facts WHERE function_id = ?",
            (function_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["facts_json"])  # type: ignore[no-any-return]

    # ========== Code-Contract Summary Operations ==========

    def store_code_contract_summary(
        self,
        func: Function,
        summary: "CodeContractSummary",  # noqa: F821 (forward ref)
        model_used: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_cache_read: int = 0,
        tokens_cache_write: int = 0,
        body_annotated: str | None = None,
    ) -> None:
        """Insert or replace a code-contract summary for `func`.

        The summary's full state lives in `summary_json` (round-trips via
        `CodeContractSummary.{to,from}_dict`). `noreturn` and
        `body_annotated` are denormalized so callers can grep without
        deserializing the JSON.
        """
        if func.id is None:
            raise ValueError("Function must have an ID")
        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO code_contract_summaries (
                function_id, summary_json, noreturn, body_annotated,
                model, tokens_input, tokens_output,
                tokens_cache_read, tokens_cache_write
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                noreturn = excluded.noreturn,
                body_annotated = excluded.body_annotated,
                model = excluded.model,
                tokens_input = excluded.tokens_input,
                tokens_output = excluded.tokens_output,
                tokens_cache_read = excluded.tokens_cache_read,
                tokens_cache_write = excluded.tokens_cache_write,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                func.id, summary_json, 1 if summary.noreturn else 0,
                body_annotated, model_used,
                tokens_input, tokens_output,
                tokens_cache_read, tokens_cache_write,
            ),
        )
        self.conn.commit()

    def get_code_contract_summary(
        self, func_id: int,
    ) -> "CodeContractSummary | None":  # noqa: F821 (forward ref)
        """Fetch the code-contract summary for `func_id`, or None."""
        from .code_contract.models import CodeContractSummary

        row = self.conn.execute(
            "SELECT summary_json FROM code_contract_summaries"
            " WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if not row:
            return None
        return CodeContractSummary.from_dict(json.loads(row["summary_json"]))

    def needs_code_contract_update(self, func: Function) -> bool:
        """Check if `func`'s code-contract summary needs regeneration.

        True iff the function isn't in DB, has no summary yet, or its
        source hash differs from what was stored.
        """
        hash_source = func.pp_source if func.pp_source else func.source
        current_hash = compute_source_hash(hash_source) if hash_source else None
        row = self.conn.execute(
            """
            SELECT f.source_hash, c.function_id
            FROM functions f
            LEFT JOIN code_contract_summaries c ON f.id = c.function_id
            WHERE f.name = ? AND f.signature = ? AND f.file_path = ?
            """,
            (func.name, func.signature, func.file_path),
        ).fetchone()
        if not row:
            return True
        if row["function_id"] is None:
            return True
        if row["source_hash"] != current_hash:
            return True
        return False

    # ========== Issue Review Operations ==========

    def upsert_issue_review(
        self,
        function_id: int,
        issue_index: int,
        fingerprint: str,
        status: str,
        reason: str | None = None,
    ) -> None:
        """Insert or update a review for a verification issue."""
        self.conn.execute(
            """
            INSERT INTO issue_reviews
                (function_id, issue_index, issue_fingerprint, status, reason)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(function_id, issue_fingerprint) DO UPDATE SET
                issue_index = excluded.issue_index,
                status      = excluded.status,
                reason      = excluded.reason,
                updated_at  = CURRENT_TIMESTAMP
            """,
            (function_id, issue_index, fingerprint, status, reason),
        )
        self.conn.commit()

    def get_issue_reviews(self, function_id: int) -> list[dict[str, Any]]:
        """Return all issue reviews for a function."""
        rows = self.conn.execute(
            """
            SELECT id, function_id, issue_index, issue_fingerprint,
                   status, reason, created_at, updated_at
            FROM issue_reviews
            WHERE function_id = ?
            ORDER BY issue_index
            """,
            (function_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_issue_reviews_by_fingerprints(
        self, function_id: int, fingerprints: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Look up reviews for a set of fingerprints.  Returns {fingerprint: row_dict}."""
        if not fingerprints:
            return {}
        placeholders = ",".join("?" for _ in fingerprints)
        rows = self.conn.execute(
            f"""
            SELECT id, function_id, issue_index, issue_fingerprint,
                   status, reason, created_at, updated_at
            FROM issue_reviews
            WHERE function_id = ? AND issue_fingerprint IN ({placeholders})
            """,
            [function_id, *fingerprints],
        ).fetchall()
        return {r["issue_fingerprint"]: dict(r) for r in rows}

    # ========== Call Graph Operations ==========

    def add_call_edge(self, edge: CallEdge) -> int | None:
        """Add a call edge between two functions. Returns the edge ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO call_edges (caller_id, callee_id, is_indirect, file_path, line, column)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                edge.caller_id,
                edge.callee_id,
                1 if edge.is_indirect else 0,
                edge.file_path,
                edge.line,
                edge.column,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_call_edges_batch(self, edges: list[CallEdge]) -> None:
        """Batch insert call edges."""
        self.conn.executemany(
            """
            INSERT INTO call_edges (caller_id, callee_id, is_indirect, file_path, line, column)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.caller_id,
                    e.callee_id,
                    1 if e.is_indirect else 0,
                    e.file_path,
                    e.line,
                    e.column,
                )
                for e in edges
            ],
        )
        self.conn.commit()

    def get_callees(self, func_id: int) -> list[int]:
        """Get IDs of functions called by the given function."""
        rows = self.conn.execute(
            "SELECT DISTINCT callee_id FROM call_edges WHERE caller_id = ?", (func_id,)
        ).fetchall()
        return [row["callee_id"] for row in rows]

    def get_callers(self, func_id: int) -> list[int]:
        """Get IDs of functions that call the given function."""
        rows = self.conn.execute(
            "SELECT DISTINCT caller_id FROM call_edges WHERE callee_id = ?", (func_id,)
        ).fetchall()
        return [row["caller_id"] for row in rows]

    def get_all_call_edges(self) -> list[CallEdge]:
        """Get all call edges."""
        rows = self.conn.execute("SELECT * FROM call_edges").fetchall()
        return [self._row_to_call_edge(row) for row in rows]

    def get_call_edges_by_caller(self, caller_id: int) -> list[CallEdge]:
        """Get all call edges from a specific caller."""
        rows = self.conn.execute(
            "SELECT * FROM call_edges WHERE caller_id = ?", (caller_id,)
        ).fetchall()
        return [self._row_to_call_edge(row) for row in rows]

    def _row_to_call_edge(self, row: sqlite3.Row) -> CallEdge:
        """Convert a database row to a CallEdge object."""
        return CallEdge(
            caller_id=row["caller_id"],
            callee_id=row["callee_id"],
            is_indirect=bool(row["is_indirect"]),
            file_path=row["file_path"],
            line=row["line"],
            column=row["column"],
        )

    def invalidate_and_cascade(self, func_id: int) -> list[int]:
        """Invalidate a function's summary and all callers. Returns invalidated IDs."""
        invalidated = []
        to_process = [func_id]

        while to_process:
            current = to_process.pop()
            if current in invalidated:
                continue

            self.conn.execute(
                "DELETE FROM allocation_summaries WHERE function_id = ?", (current,)
            )
            invalidated.append(current)

            callers = self.get_callers(current)
            to_process.extend(callers)

        self.conn.commit()
        return invalidated

    # ========== Indirect Call Operations ==========

    def add_address_taken_function(self, atf: AddressTakenFunction) -> int | None:
        """Add an address-taken function."""
        cursor = self.conn.execute(
            """
            INSERT OR REPLACE INTO address_taken_functions (function_id, signature, target_type)
            VALUES (?, ?, ?)
            """,
            (atf.function_id, atf.signature, atf.target_type),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_address_taken_functions(
        self,
        signature: str | None = None,
        target_type: str | None = None,
    ) -> list[AddressTakenFunction]:
        """Get address-taken functions, optionally filtered by signature and/or target_type."""
        query = "SELECT * FROM address_taken_functions"
        conditions = []
        params: list[str] = []

        if signature:
            conditions.append("signature = ?")
            params.append(signature)
        if target_type:
            conditions.append("target_type = ?")
            params.append(target_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        rows = self.conn.execute(query, params).fetchall()

        return [
            AddressTakenFunction(
                id=row["id"],
                function_id=row["function_id"],
                signature=row["signature"],
                target_type=row["target_type"],
            )
            for row in rows
        ]

    def add_address_flow(self, flow: AddressFlow) -> int | None:
        """Add an address flow record. Ignores duplicates."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO address_flows
            (function_id, flow_target, file_path, line_number, context_snippet)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                flow.function_id,
                flow.flow_target,
                flow.file_path,
                flow.line_number,
                flow.context_snippet,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_address_flows(self, function_id: int) -> list[AddressFlow]:
        """Get address flows for a function."""
        rows = self.conn.execute(
            "SELECT * FROM address_flows WHERE function_id = ?", (function_id,)
        ).fetchall()
        return [
            AddressFlow(
                id=row["id"],
                function_id=row["function_id"],
                flow_target=row["flow_target"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                context_snippet=row["context_snippet"],
            )
            for row in rows
        ]

    def add_indirect_callsite(self, callsite: IndirectCallsite) -> int | None:
        """Add an indirect call site. Ignores duplicates."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO indirect_callsites
            (caller_function_id, file_path, line_number, callee_expr, signature, context_snippet)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                callsite.caller_function_id,
                callsite.file_path,
                callsite.line_number,
                callsite.callee_expr,
                callsite.signature,
                callsite.context_snippet,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_indirect_callsites(
        self, caller_function_id: int | None = None
    ) -> list[IndirectCallsite]:
        """Get indirect call sites, optionally filtered by caller."""
        if caller_function_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM indirect_callsites WHERE caller_function_id = ?",
                (caller_function_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM indirect_callsites").fetchall()

        return [
            IndirectCallsite(
                id=row["id"],
                caller_function_id=row["caller_function_id"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                callee_expr=row["callee_expr"],
                signature=row["signature"],
                context_snippet=row["context_snippet"],
            )
            for row in rows
        ]

    def add_indirect_call_target(self, target: IndirectCallTarget) -> None:
        """Add a resolved indirect call target."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO indirect_call_targets
            (callsite_id, target_function_id, confidence, llm_reasoning)
            VALUES (?, ?, ?, ?)
            """,
            (
                target.callsite_id,
                target.target_function_id,
                target.confidence,
                target.llm_reasoning,
            ),
        )
        self.conn.commit()

    def get_indirect_call_targets(self, callsite_id: int) -> list[IndirectCallTarget]:
        """Get resolved targets for an indirect call site."""
        rows = self.conn.execute(
            "SELECT * FROM indirect_call_targets WHERE callsite_id = ?",
            (callsite_id,),
        ).fetchall()
        return [
            IndirectCallTarget(
                callsite_id=row["callsite_id"],
                target_function_id=row["target_function_id"],
                confidence=row["confidence"],
                llm_reasoning=row["llm_reasoning"],
            )
            for row in rows
        ]

    # ========== Address Flow Summary Operations (Pass 1 LLM) ==========

    def add_flow_summary(self, summary: AddressFlowSummary) -> int | None:
        """Add or update an LLM-generated flow summary for an address-taken function."""
        flow_destinations_json = json.dumps(
            [fd.to_dict() for fd in summary.flow_destinations]
        )
        likely_callers_json = json.dumps(summary.likely_callers)

        cursor = self.conn.execute(
            """
            INSERT INTO address_flow_summaries
            (function_id, flow_destinations_json, semantic_role, likely_callers_json, model_used)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                flow_destinations_json = excluded.flow_destinations_json,
                semantic_role = excluded.semantic_role,
                likely_callers_json = excluded.likely_callers_json,
                model_used = excluded.model_used,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                summary.function_id,
                flow_destinations_json,
                summary.semantic_role,
                likely_callers_json,
                summary.model_used,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_flow_summary(self, function_id: int) -> AddressFlowSummary | None:
        """Get the LLM-generated flow summary for a function."""
        row = self.conn.execute(
            "SELECT * FROM address_flow_summaries WHERE function_id = ?",
            (function_id,),
        ).fetchone()

        if not row:
            return None

        # Parse flow destinations
        flow_destinations = []
        try:
            destinations_data = json.loads(row["flow_destinations_json"])
            for fd in destinations_data:
                flow_destinations.append(
                    FlowDestination(
                        dest_type=fd.get("type", "unknown"),
                        name=fd.get("name", ""),
                        confidence=fd.get("confidence", "low"),
                        access_path=fd.get("access_path", ""),
                        root_type=fd.get("root_type", ""),
                        root_name=fd.get("root_name", ""),
                        file_path=fd.get("file_path", ""),
                        line_number=fd.get("line_number", 0),
                    )
                )
        except json.JSONDecodeError:
            pass

        # Parse likely callers
        likely_callers = []
        try:
            if row["likely_callers_json"]:
                likely_callers = json.loads(row["likely_callers_json"])
        except json.JSONDecodeError:
            pass

        return AddressFlowSummary(
            id=row["id"],
            function_id=row["function_id"],
            flow_destinations=flow_destinations,
            semantic_role=row["semantic_role"] or "",
            likely_callers=likely_callers,
            model_used=row["model_used"] or "",
        )

    def get_all_flow_summaries(self) -> list[AddressFlowSummary]:
        """Get all flow summaries from the database."""
        rows = self.conn.execute("SELECT function_id FROM address_flow_summaries").fetchall()
        summaries = []
        for row in rows:
            summary = self.get_flow_summary(row["function_id"])
            if summary:
                summaries.append(summary)
        return summaries

    def has_flow_summary(self, function_id: int) -> bool:
        """Check if a function already has a flow summary."""
        row = self.conn.execute(
            "SELECT 1 FROM address_flow_summaries WHERE function_id = ?",
            (function_id,),
        ).fetchone()
        return row is not None

    # ========== Container Summary Operations ==========

    def add_container_summary(self, summary: ContainerSummary) -> int | None:
        """Add or update a container function summary."""
        store_args_json = json.dumps(summary.store_args)
        heuristic_signals_json = json.dumps(summary.heuristic_signals)

        cursor = self.conn.execute(
            """
            INSERT INTO container_summaries
            (function_id, container_arg, store_args_json, load_return,
             container_type, confidence, heuristic_score, heuristic_signals_json, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                container_arg = excluded.container_arg,
                store_args_json = excluded.store_args_json,
                load_return = excluded.load_return,
                container_type = excluded.container_type,
                confidence = excluded.confidence,
                heuristic_score = excluded.heuristic_score,
                heuristic_signals_json = excluded.heuristic_signals_json,
                model_used = excluded.model_used,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                summary.function_id,
                summary.container_arg,
                store_args_json,
                1 if summary.load_return else 0,
                summary.container_type,
                summary.confidence,
                summary.heuristic_score,
                heuristic_signals_json,
                summary.model_used,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_container_summary(self, function_id: int) -> ContainerSummary | None:
        """Get the container summary for a function."""
        row = self.conn.execute(
            "SELECT * FROM container_summaries WHERE function_id = ?",
            (function_id,),
        ).fetchone()

        if not row:
            return None

        store_args = []
        try:
            store_args = json.loads(row["store_args_json"])
        except json.JSONDecodeError:
            pass

        heuristic_signals = []
        try:
            if row["heuristic_signals_json"]:
                heuristic_signals = json.loads(row["heuristic_signals_json"])
        except json.JSONDecodeError:
            pass

        return ContainerSummary(
            id=row["id"],
            function_id=row["function_id"],
            container_arg=row["container_arg"],
            store_args=store_args,
            load_return=bool(row["load_return"]),
            container_type=row["container_type"],
            confidence=row["confidence"],
            heuristic_score=row["heuristic_score"] or 0,
            heuristic_signals=heuristic_signals,
            model_used=row["model_used"] or "",
        )

    def get_all_container_summaries(self) -> list[ContainerSummary]:
        """Get all container summaries from the database."""
        rows = self.conn.execute("SELECT function_id FROM container_summaries").fetchall()
        summaries = []
        for row in rows:
            summary = self.get_container_summary(row["function_id"])
            if summary:
                summaries.append(summary)
        return summaries

    def has_container_summary(self, function_id: int) -> bool:
        """Check if a function already has a container summary."""
        row = self.conn.execute(
            "SELECT 1 FROM container_summaries WHERE function_id = ?",
            (function_id,),
        ).fetchone()
        return row is not None

    def delete_container_summaries(self, function_ids: list[int]) -> int:
        """Delete container summaries for the given function IDs. Returns count deleted."""
        if not function_ids:
            return 0
        placeholders = ",".join("?" for _ in function_ids)
        cursor = self.conn.execute(
            f"DELETE FROM container_summaries WHERE function_id IN ({placeholders})",
            function_ids,
        )
        self.conn.commit()
        return cursor.rowcount

    # ========== Typedef Operations ==========

    def insert_typedef(
        self,
        name: str,
        underlying_type: str,
        canonical_type: str,
        file_path: str,
        line_number: int | None = None,
        kind: str = "typedef",
        definition: str | None = None,
    ) -> int | None:
        """Insert a type declaration. Ignores duplicates (same name+kind+file)."""
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO typedefs
            (name, kind, underlying_type, canonical_type, file_path, line_number, definition)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, kind, underlying_type, canonical_type, file_path, line_number, definition),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_typedefs_batch(self, typedefs: list[dict]) -> None:
        """Batch insert type declarations.

        Each dict has name, kind, underlying_type, canonical_type, file_path,
        line_number, and optionally definition and pp_definition.
        """
        self.conn.executemany(
            """
            INSERT INTO typedefs
            (name, kind, underlying_type, canonical_type, file_path, line_number,
             definition, pp_definition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, kind, file_path) DO UPDATE SET
                pp_definition = COALESCE(excluded.pp_definition, typedefs.pp_definition)
            """,
            [
                (
                    td["name"],
                    td.get("kind", "typedef"),
                    td["underlying_type"],
                    td["canonical_type"],
                    td["file_path"],
                    td.get("line_number"),
                    td.get("definition"),
                    td.get("pp_definition"),
                )
                for td in typedefs
            ],
        )
        self.conn.commit()

    def get_typedef(self, name: str) -> dict | None:
        """Look up a typedef by name. Returns the first match or None."""
        row = self.conn.execute(
            "SELECT * FROM typedefs WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def get_all_typedefs(self) -> list[dict]:
        """Get all typedefs."""
        rows = self.conn.execute("SELECT * FROM typedefs").fetchall()
        return [dict(row) for row in rows]

    def get_typedefs_by_names(self, names: list[str]) -> list[dict]:
        """Look up type declarations by name. Returns all matches (may be multiple per name)."""
        if not names:
            return []
        placeholders = ",".join("?" * len(names))
        rows = self.conn.execute(
            f"SELECT * FROM typedefs WHERE name IN ({placeholders}) AND definition IS NOT NULL",
            names,
        ).fetchall()
        return [dict(row) for row in rows]

    def get_static_vars_by_file(self, file_path: str) -> list[dict]:
        """Get all file-scope variable declarations (static and global)."""
        rows = self.conn.execute(
            "SELECT * FROM typedefs WHERE file_path = ?"
            " AND kind IN ('static_var', 'global_var')"
            " AND definition IS NOT NULL",
            (file_path,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ========== Call Graph Import Helpers ==========

    def find_function_by_name(self, name: str) -> Function | None:
        """Find a single function by name. Returns None if not found or ambiguous."""
        rows = self.conn.execute(
            "SELECT * FROM functions WHERE name = ?", (name,)
        ).fetchall()
        if len(rows) == 1:
            return self._row_to_function(rows[0])
        return None

    def find_function_by_name_and_file(self, name: str, file_path: str) -> Function | None:
        """Find a function by name and file path (exact match)."""
        row = self.conn.execute(
            "SELECT * FROM functions WHERE name = ? AND file_path = ?",
            (name, file_path),
        ).fetchone()
        if row:
            return self._row_to_function(row)
        return None

    def find_function_by_name_and_file_suffix(self, name: str, file_suffix: str) -> Function | None:
        """Find a function by name and file path suffix (for cross-build-dir matching)."""
        rows = self.conn.execute(
            "SELECT * FROM functions WHERE name = ? AND file_path LIKE ?",
            (name, f"%{file_suffix}"),
        ).fetchall()
        if len(rows) == 1:
            return self._row_to_function(rows[0])
        return None

    def clear_call_edges(self) -> int:
        """Delete all call_edges. Returns count deleted."""
        cursor = self.conn.execute("DELETE FROM call_edges")
        self.conn.commit()
        return cursor.rowcount

    def clear_call_graph_stubs(self) -> int:
        """Delete stub functions (empty source) created by previous imports.

        Should be called together with clear_call_edges when re-importing
        a call graph, so that stale stubs don't pollute name-based matching.
        """
        cursor = self.conn.execute(
            "DELETE FROM functions WHERE source = '' OR source IS NULL"
        )
        self.conn.commit()
        return cursor.rowcount

    def insert_function_stub(
        self,
        name: str,
        file_path: str = "",
        line_start: int = 0,
        line_end: int = 0,
        linkage: str = "external",
        attributes: str = "",
    ) -> int:
        """Insert a minimal function entry (stub) and return its ID."""
        func = Function(
            name=name,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            source="",
            signature=f"{name}(...)",
            attributes=attributes,
        )
        return self.insert_function(func)

    def update_decl_headers(self, header_map: dict[str, str]) -> int:
        """Update decl_header for external functions (sourceless stubs).

        Args:
            header_map: mapping of function name -> header file path

        Returns:
            Number of rows updated.
        """
        updated = 0
        for name, header in header_map.items():
            cursor = self.conn.execute(
                "UPDATE functions SET decl_header = ? "
                "WHERE name = ? AND (source IS NULL OR source = '')",
                (header, name),
            )
            updated += cursor.rowcount
        self.conn.commit()
        return updated


    # ========== Utility Operations ==========

    def clear_all(self) -> None:
        """Clear all data from the database."""
        tables = [
            "indirect_call_targets",
            "indirect_callsites",
            "address_flow_summaries",
            "address_flows",
            "address_taken_functions",
            "container_summaries",
            "call_edges",
            "allocation_summaries",
            "free_summaries",
            "init_summaries",
            "memsafe_summaries",
            "verification_summaries",
            "typedefs",
            "functions",
            "build_configs",
        ]
        for table in tables:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        stats = {}
        tables = [
            "functions",
            "allocation_summaries",
            "free_summaries",
            "init_summaries",
            "memsafe_summaries",
            "verification_summaries",
            "leak_summaries",
            "call_edges",
            "address_taken_functions",
            "address_flows",
            "address_flow_summaries",
            "indirect_callsites",
            "indirect_call_targets",
            "container_summaries",
            "typedefs",
            "build_configs",
            "issue_reviews",
        ]
        for table in tables:
            row = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            stats[table] = row["cnt"]
        return stats

    # ========== Build Configuration Operations ==========

    def add_build_config(
        self,
        project_path: str,
        project_name: str,
        build_system: str,
        configuration: dict[str, Any] | None = None,
        script_path: str | None = None,
        artifacts_dir: str | None = None,
        compile_commands_path: str | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        build_attempts: int = 0,
    ) -> None:
        """Add or update a build configuration."""
        configuration_json = json.dumps(configuration) if configuration else None

        self.conn.execute(
            """
            INSERT INTO build_configs
            (project_path, project_name, build_system, configuration_json,
             script_path, artifacts_dir, compile_commands_path,
             llm_backend, llm_model, build_attempts, last_built_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project_path) DO UPDATE SET
                project_name = excluded.project_name,
                build_system = excluded.build_system,
                configuration_json = excluded.configuration_json,
                script_path = excluded.script_path,
                artifacts_dir = excluded.artifacts_dir,
                compile_commands_path = excluded.compile_commands_path,
                llm_backend = excluded.llm_backend,
                llm_model = excluded.llm_model,
                build_attempts = excluded.build_attempts,
                last_built_at = CURRENT_TIMESTAMP
            """,
            (
                project_path,
                project_name,
                build_system,
                configuration_json,
                script_path,
                artifacts_dir,
                compile_commands_path,
                llm_backend,
                llm_model,
                build_attempts,
            ),
        )
        self.conn.commit()

    def get_build_config(self, project_path: str) -> dict[str, Any] | None:
        """Get build configuration for a project."""
        row = self.conn.execute(
            "SELECT * FROM build_configs WHERE project_path = ?",
            (project_path,),
        ).fetchone()

        if not row:
            return None

        config = dict(row)
        if config["configuration_json"]:
            config["configuration"] = json.loads(config["configuration_json"])
        else:
            config["configuration"] = None
        del config["configuration_json"]

        return config

    def get_all_build_configs(self) -> list[dict[str, Any]]:
        """Get all build configurations."""
        rows = self.conn.execute("SELECT * FROM build_configs").fetchall()
        configs = []
        for row in rows:
            config = dict(row)
            if config["configuration_json"]:
                config["configuration"] = json.loads(config["configuration_json"])
            else:
                config["configuration"] = None
            del config["configuration_json"]
            configs.append(config)
        return configs
