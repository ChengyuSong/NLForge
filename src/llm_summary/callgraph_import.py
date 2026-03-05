"""Import KAMain call graph JSON into the summary database."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .db import SummaryDB
from .models import CallEdge
from .stdlib import get_stdlib_attributes


@dataclass
class ImportStats:
    """Statistics from a call graph import."""

    functions_in_json: int = 0
    functions_matched: int = 0
    functions_matched_by_name: int = 0
    functions_matched_by_file: int = 0
    functions_matched_by_suffix: int = 0
    functions_matched_by_demangle: int = 0
    stubs_created: int = 0
    edges_imported: int = 0
    direct_edges: int = 0
    indirect_edges: int = 0
    edges_skipped_self: int = 0
    unresolved_functions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Functions in JSON: {self.functions_in_json}",
            f"  Matched to DB: {self.functions_matched}",
            f"    By name: {self.functions_matched_by_name}",
            f"    By file+name: {self.functions_matched_by_file}",
            f"    By suffix+name: {self.functions_matched_by_suffix}",
            f"    By demangled name: {self.functions_matched_by_demangle}",
            f"  Stubs created: {self.stubs_created}",
            f"Edges imported: {self.edges_imported}",
            f"  Direct: {self.direct_edges}",
            f"  Indirect: {self.indirect_edges}",
        ]
        if self.edges_skipped_self:
            lines.append(f"  Self-edges skipped: {self.edges_skipped_self}")
        if self.unresolved_functions:
            lines.append(f"Unresolved (no stub): {len(self.unresolved_functions)}")
        return "\n".join(lines)


_LLVM_INTRINSIC_MAP = {
    "llvm.memcpy": "memcpy",
    "llvm.memmove": "memmove",
    "llvm.memset": "memset",
}


def _normalize_callee_name(name: str) -> str:
    """Normalize LLVM intrinsic names to their C stdlib equivalents.

    KAMain emits intrinsics like 'llvm.memcpy.p0.p0.i64' — strip the
    type suffixes and map to the canonical C name so that stdlib summaries
    (keyed on 'memcpy' etc.) are matched correctly.
    """
    for prefix, canonical in _LLVM_INTRINSIC_MAP.items():
        if name == prefix or name.startswith(prefix + "."):
            return canonical
    return name


def _demangle(name: str) -> str | None:
    """Demangle a C++ symbol name using c++filt."""
    if not name.startswith("_Z"):
        return None
    try:
        result = subprocess.run(
            ["c++filt", name],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            demangled = result.stdout.strip()
            if demangled != name:
                return demangled
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _extract_base_name(demangled: str) -> str:
    """Extract the base function name from a demangled C++ name.

    E.g., 'ns::Class::method(int, char*)' -> 'method'
    """
    # Strip everything from '(' onwards (parameters)
    paren_idx = demangled.find("(")
    if paren_idx != -1:
        demangled = demangled[:paren_idx]
    # Take the last component after '::'
    parts = demangled.split("::")
    return parts[-1]


_DOCKER_PATH_PREFIXES = ("/workspace/src/", "/workspace/build/")


def _strip_docker_prefix(path: str) -> str:
    """Strip Docker container path prefixes from a file path.

    KAMain reads debug info from .bc files compiled inside Docker, so function
    keys and file fields carry /workspace/src/ or /workspace/build/ prefixes.
    Strip them so that suffix matching against host paths in the DB works.
    """
    for prefix in _DOCKER_PATH_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _parse_ka_function_key(key: str) -> tuple[str | None, str]:
    """Parse a KAMain function key into (file_path, func_name).

    Internal: '/path/to/file.c:func_name' -> ('/path/to/file.c', 'func_name')
              'src/aio/aio.c:cleanup'     -> ('src/aio/aio.c', 'cleanup')
    External: 'func_name' -> (None, 'func_name')

    Handles edge case: C++ mangled names in internal functions like
    '/path/file.cc:_ZN3FooC2Ev' where ':' also appears in the key prefix.
    """
    if ":" in key:
        file_path, name = key.rsplit(":", 1)
        # Internal if the file part contains '/' (absolute or relative path)
        # or has a source file extension
        if "/" in file_path or "." in file_path:
            return file_path, name
    # External function: plain name
    return None, key


class CallGraphImporter:
    """Import KAMain call graph JSON into the summary database."""

    def __init__(self, db: SummaryDB, verbose: bool = False):
        self.db = db
        self.verbose = verbose
        # Cache: KAMain func key -> DB function ID
        self._id_cache: dict[str, int | None] = {}
        # Pre-build DB lookup caches
        self._db_funcs_by_name: dict[str, list[int]] = {}
        self._db_funcs_by_name_file: dict[tuple[str, str], int] = {}
        self._build_db_caches()

    def _build_db_caches(self) -> None:
        """Pre-load DB functions into lookup caches."""
        all_funcs = self.db.get_all_functions()
        for f in all_funcs:
            self._db_funcs_by_name.setdefault(f.name, []).append(f.id)
            self._db_funcs_by_name_file[(f.name, f.file_path)] = f.id

    def import_json(
        self, json_path: Path, clear_existing: bool = False
    ) -> ImportStats:
        """Import KAMain callgraph JSON into the DB."""
        with open(json_path) as f:
            data = json.load(f)

        functions = data.get("functions", {})
        metadata = data.get("metadata", {})
        stats = ImportStats(functions_in_json=len(functions))

        if self.verbose:
            print(f"KAMain metadata: {json.dumps(metadata)}")

        if clear_existing:
            deleted = self.db.clear_call_edges()
            if self.verbose:
                print(f"Cleared {deleted} existing call edges")

        # Phase 1: Resolve all function keys to DB IDs
        for ka_key, ka_info in functions.items():
            self._resolve_func_id(ka_key, ka_info, stats)

        # Phase 2: Import call edges
        edges_batch: list[CallEdge] = []

        for ka_key, ka_info in functions.items():
            caller_id = self._id_cache.get(ka_key)
            if caller_id is None:
                continue

            caller_file = ka_info.get("file", "")

            for callee_entry in ka_info.get("callees", []):
                callee_name = _normalize_callee_name(callee_entry.get("callee", ""))
                call_type = callee_entry.get("call_type", "direct")
                line = callee_entry.get("line")

                # Resolve callee - could be a full key or just a name
                callee_id = self._resolve_callee(callee_name, functions, stats)
                if callee_id is None:
                    continue

                if caller_id == callee_id:
                    stats.edges_skipped_self += 1
                    continue

                is_indirect = call_type == "indirect"
                edge = CallEdge(
                    caller_id=caller_id,
                    callee_id=callee_id,
                    is_indirect=is_indirect,
                    file_path=caller_file,
                    line=line,
                )
                edges_batch.append(edge)

                if is_indirect:
                    stats.indirect_edges += 1
                else:
                    stats.direct_edges += 1

        # Batch insert
        if edges_batch:
            self.db.add_call_edges_batch(edges_batch)
        stats.edges_imported = len(edges_batch)

        return stats

    def _resolve_callee(
        self, callee_ref: str, all_functions: dict, stats: ImportStats
    ) -> int | None:
        """Resolve a callee reference from a callees entry.

        The callee field can be:
        - A full key matching a functions dict key (e.g., '/path/file.c:func')
        - Just a function name (e.g., 'malloc')
        """
        # Check if it's already cached (as a full key)
        if callee_ref in self._id_cache:
            return self._id_cache[callee_ref]

        # Check if it matches a key in the functions dict
        if callee_ref in all_functions:
            return self._id_cache.get(callee_ref)

        # It might be a name reference - try to find the full key
        # Look for keys ending with :callee_ref
        for key in all_functions:
            _, name = _parse_ka_function_key(key)
            if name == callee_ref and key in self._id_cache:
                return self._id_cache[key]

        # Not in KAMain functions dict - try direct DB lookup
        if callee_ref not in self._id_cache:
            self._resolve_func_id(callee_ref, {}, stats)
        return self._id_cache.get(callee_ref)

    def _resolve_func_id(
        self, ka_key: str, ka_info: dict, stats: ImportStats
    ) -> int | None:
        """Resolve a KAMain function key to a DB function ID."""
        if ka_key in self._id_cache:
            return self._id_cache[ka_key]

        file_path, raw_name = _parse_ka_function_key(ka_key)
        if file_path:
            file_path = _strip_docker_prefix(file_path)
        name = _normalize_callee_name(raw_name)
        ka_file = _strip_docker_prefix(ka_info.get("file", file_path or ""))
        ka_linkage = ka_info.get("linkage", "external")
        ka_line_start = ka_info.get("line_start", 0)
        ka_line_end = ka_info.get("line_end", 0)

        func_id = None

        # Strategy 1: Match by name + file (internal functions)
        if file_path and name:
            # Exact file path match
            key = (name, file_path)
            if key in self._db_funcs_by_name_file:
                func_id = self._db_funcs_by_name_file[key]
                stats.functions_matched_by_file += 1
            else:
                # Try suffix matching (build dirs differ)
                func = self.db.find_function_by_name_and_file_suffix(
                    name, _file_suffix(file_path)
                )
                if func:
                    func_id = func.id
                    stats.functions_matched_by_suffix += 1

        # Strategy 2: Match by name only (external functions / fallback)
        if func_id is None and name:
            ids = self._db_funcs_by_name.get(name, [])
            if len(ids) == 1:
                func_id = ids[0]
                stats.functions_matched_by_name += 1
            elif len(ids) == 0 and name.startswith("_Z"):
                # Try demangling
                demangled = _demangle(name)
                if demangled:
                    base = _extract_base_name(demangled)
                    ids = self._db_funcs_by_name.get(base, [])
                    if len(ids) == 1:
                        func_id = ids[0]
                        stats.functions_matched_by_demangle += 1

        if func_id is not None:
            stats.functions_matched += 1
        else:
            # Create stub
            func_id = self.db.insert_function_stub(
                name=name,
                file_path=ka_file,
                line_start=ka_line_start,
                line_end=ka_line_end,
                linkage=ka_linkage,
                attributes=get_stdlib_attributes(name),
            )
            stats.stubs_created += 1
            # Update caches
            self._db_funcs_by_name.setdefault(name, []).append(func_id)
            if ka_file:
                self._db_funcs_by_name_file[(name, ka_file)] = func_id

            if self.verbose:
                print(f"  Stub: {ka_key} -> id={func_id}")

        self._id_cache[ka_key] = func_id
        return func_id


def _file_suffix(path: str, components: int = 3) -> str:
    """Get the last N path components as a suffix for matching.

    Strips Docker container path prefixes first so that paths like
    '/workspace/src/dh.c' match host-path DB entries ending in 'dh.c'.

    E.g., '/magma/targets/libpng/repo/png.c' -> 'repo/png.c' (components=2)
    """
    path = _strip_docker_prefix(path)
    parts = Path(path).parts
    if len(parts) <= components:
        return path
    return str(Path(*parts[-components:]))
