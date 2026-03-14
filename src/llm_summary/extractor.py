"""Function extraction from C/C++ source files using libclang."""

import os
import re
from pathlib import Path
from typing import Any

from clang.cindex import (
    Config,
    Cursor,
    CursorKind,
    Index,
    StorageClass,
    TranslationUnit,
)

from .compile_commands import CompileCommandsDB
from .models import Function, FunctionBlock, _annotate_macro_diff
from .preprocessor import PreprocessedFile, SourcePreprocessor

_SYSTEM_HEADER_PREFIXES = (
    "/usr/",
    "/lib/",
    "/include/",
    "/opt/",
    "/System/",
    "/Library/",
)


def _is_system_header(file_path: str) -> bool:
    """Return True if file_path looks like a system header (not project-local)."""
    return any(file_path.startswith(p) for p in _SYSTEM_HEADER_PREFIXES)


def _annotate_pp_definition(definition: str | None, pp_definition: str | None) -> str | None:
    """Return annotated diff of definition vs pp_definition when they differ.

    When a typedef/struct/static_var definition contains macros, the
    preprocessed form has concrete values (e.g. 200 instead of
    JMSG_LENGTH_MAX).  Annotating the diff with ``// (macro)`` comments lets
    the LLM see both the symbolic name and the expanded value.
    """
    if not definition or not pp_definition:
        return pp_definition
    if pp_definition == definition:
        return pp_definition
    return _annotate_macro_diff(definition, pp_definition)


def configure_libclang(libclang_path: str | None = None) -> None:
    """Configure libclang library path if needed."""
    # Check if already loaded - can't reconfigure
    if Config.loaded:
        return

    if libclang_path:
        Config.set_library_file(libclang_path)
    else:
        # Try common paths (prefer newer versions)
        common_paths = [
            "/usr/lib/x86_64-linux-gnu/libclang-18.so.1",
            "/usr/lib/llvm-18/lib/libclang-18.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
            "/usr/lib/llvm-14/lib/libclang-14.so.1",
            "/usr/local/lib/libclang.dylib",  # macOS
            "/opt/homebrew/opt/llvm/lib/libclang.dylib",  # macOS ARM
        ]
        for path in common_paths:
            if os.path.exists(path):
                Config.set_library_file(path)
                break


def get_type_spelling(cursor: Cursor) -> str:
    """Get a normalized type spelling for a function."""
    result_type = cursor.result_type.spelling
    params = []

    for child in cursor.get_children():
        if child.kind == CursorKind.PARM_DECL:
            param_type = child.type.spelling
            params.append(param_type)

    return f"{result_type}({', '.join(params)})"


def get_canonical_type_spelling(cursor: Cursor) -> str:
    """Get a canonical (typedef-resolved) type spelling for a function."""
    result_type = cursor.result_type.get_canonical().spelling
    params = []

    for child in cursor.get_children():
        if child.kind == CursorKind.PARM_DECL:
            param_type = child.type.get_canonical().spelling
            params.append(param_type)

    return f"{result_type}({', '.join(params)})"


def get_function_source(cursor: Cursor, file_contents: dict[str, str]) -> str:
    """Extract source code for a function from file contents."""
    file_path = str(cursor.location.file)
    if file_path not in file_contents:
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                file_contents[file_path] = f.read()
        except OSError:
            return ""

    content = file_contents[file_path]
    lines = content.splitlines()

    start_line = cursor.extent.start.line - 1
    end_line = cursor.extent.end.line

    if start_line < 0 or end_line > len(lines):
        return ""

    source_lines = lines[start_line:end_line]

    # Handle partial first/last lines
    if source_lines:
        start_col = cursor.extent.start.column - 1
        end_col = cursor.extent.end.column

        if len(source_lines) == 1:
            source_lines[0] = source_lines[0][start_col:end_col]
        else:
            source_lines[0] = source_lines[0][start_col:]
            source_lines[-1] = source_lines[-1][:end_col]

    return "\n".join(source_lines)


class FunctionExtractor:
    """Extracts function definitions from C/C++ source files."""

    def __init__(
        self,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
        compile_commands: CompileCommandsDB | None = None,
        project_root: Path | str | None = None,
        enable_preprocessing: bool = False,
    ):
        """
        Initialize the extractor.

        Args:
            compile_args: Additional compiler arguments (e.g., ["-I/path/to/include"])
            libclang_path: Path to libclang shared library
            compile_commands: Optional CompileCommandsDB for per-file compile flags
            project_root: If provided, functions defined in headers under this directory
                are also extracted (not just functions in the main source file).
                Deduplication across files is handled automatically via USR tracking.
            enable_preprocessing: If True, run clang -E on each file and store
                macro-expanded source as pp_source on each Function.
        """
        configure_libclang(libclang_path)
        self.index = Index.create()
        self.compile_args = compile_args or []
        self.compile_commands = compile_commands
        self.project_root = Path(project_root).resolve() if project_root else None
        self._file_contents: dict[str, str] = {}
        self._file_lines_cache: dict[str, list[str]] = {}
        # USR-based dedup: tracks libclang Unified Symbol Resolution strings so that
        # inline/template functions defined in headers are not extracted more than once
        # when multiple translation units include the same header.
        self._seen_usrs: set[str] = set()
        self.enable_preprocessing = enable_preprocessing
        self._preprocessor: SourcePreprocessor | None = None
        # Cache of preprocessed files: file_path -> PreprocessedFile
        self._pp_cache: dict[str, PreprocessedFile] = {}
        if enable_preprocessing:
            self._preprocessor = SourcePreprocessor(
                compile_commands=compile_commands,
                extra_args=compile_args,
            )

    def parse_file(self, file_path: str | Path) -> Any:
        """Parse a file and return the translation unit.

        Use this to parse once and pass the TU to extract_from_tu /
        extract_typedefs_from_tu for single-parse workflows.
        """
        file_path = Path(file_path).resolve()
        args = self._get_compile_args(file_path)
        return self.index.parse(
            str(file_path),
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )

    def extract_from_file(self, file_path: str | Path) -> list[Function]:
        """
        Extract all function definitions from a source file.

        Args:
            file_path: Path to the C/C++ source file

        Returns:
            List of Function objects
        """
        file_path = Path(file_path).resolve()

        try:
            tu = self.parse_file(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}") from e

        return self.extract_from_tu(tu, file_path)

    def extract_from_tu(self, tu, file_path: str | Path) -> list[Function]:
        """Extract functions from a pre-parsed translation unit."""
        file_path = Path(file_path).resolve()

        # Preprocess file once if enabled (cache across calls)
        pp_file: PreprocessedFile | None = None
        if self._preprocessor is not None:
            key = str(file_path)
            if key not in self._pp_cache:
                self._pp_cache[key] = self._preprocessor.preprocess(file_path)
            pp_file = self._pp_cache[key]

        functions: list[Function] = []
        self._extract_functions_recursive(
            tu.cursor, str(file_path), functions
        )

        # Populate pp_source from preprocessed output
        if pp_file is not None and pp_file.mappings:
            for func in functions:
                pp_src = pp_file.extract_pp_source(
                    func.file_path, func.line_start, func.line_end
                )
                if pp_src:
                    func.pp_source = pp_src

        # Update block source text from llm_source (preprocessed when available).
        # Block line ranges are absolute file lines, so re-extract from llm_source.
        for func in functions:
            if func.blocks:
                llm_lines = func.llm_source.splitlines()
                for block in func.blocks:
                    rel_start = block.line_start - func.line_start  # 0-based
                    rel_end = block.line_end - func.line_start  # 0-based inclusive
                    if 0 <= rel_start and rel_end < len(llm_lines):
                        block.source = "\n".join(llm_lines[rel_start : rel_end + 1])

        return functions

    def _get_compile_args(self, file_path: Path) -> list[str]:
        """
        Get compile arguments for a file.

        Uses compile_commands.json if available, otherwise uses default args.
        """
        args = list(self.compile_args)

        # Add per-file flags from compile_commands.json if available
        if self.compile_commands and self.compile_commands.has_file(file_path):
            args.extend(self.compile_commands.get_compile_flags(file_path))
        else:
            # Fall back to default language settings based on extension
            ext = file_path.suffix.lower()
            if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
                args.extend(["-x", "c++", "-std=c++17"])
            else:
                args.extend(["-x", "c", "-std=c11"])

        return args

    def extract_from_files(self, file_paths: list[str | Path]) -> list[Function]:
        """Extract functions from multiple files."""
        all_functions = []
        for path in file_paths:
            try:
                functions = self.extract_from_file(path)
                all_functions.extend(functions)
            except Exception as e:
                print(f"Warning: Failed to extract from {path}: {e}")
        return all_functions

    def extract_from_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"),
        recursive: bool = True,
    ) -> list[Function]:
        """Extract functions from all C/C++ files in a directory."""
        directory = Path(directory)

        files: list[str | Path]
        if recursive:
            files = [
                f for f in directory.rglob("*") if f.suffix.lower() in extensions
            ]
        else:
            files = [
                f for f in directory.glob("*") if f.suffix.lower() in extensions
            ]

        return self.extract_from_files(files)

    def _is_extractable_file(self, node_file: str, main_file: str) -> bool:
        """Return True if a node from node_file should be extracted.

        Always accepts the main translation-unit file. When project_root is set,
        also accepts header files that live inside the project source tree (i.e.
        not system headers or third-party headers outside the project).
        """
        if node_file == main_file:
            return True
        if self.project_root is None:
            return False
        try:
            Path(node_file).resolve().relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def _extract_functions_recursive(
        self,
        cursor: Cursor,
        main_file: str,
        functions: list[Function],
    ) -> None:
        """Recursively extract functions from AST."""
        for child in cursor.get_children():
            node_file = str(child.location.file) if child.location.file else None
            if node_file is None:
                continue
            if not self._is_extractable_file(node_file, main_file):
                continue

            if child.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD):
                if child.is_definition():
                    usr = child.get_usr()
                    if usr and usr in self._seen_usrs:
                        continue
                    func = self._cursor_to_function(child)
                    if func:
                        if usr:
                            self._seen_usrs.add(usr)
                        functions.append(func)

            # Recurse into namespaces and classes
            elif child.kind in (
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
                CursorKind.CLASS_TEMPLATE,
            ):
                self._extract_functions_recursive(child, main_file, functions)

    def _cursor_to_function(self, cursor: Cursor) -> Function | None:
        """Convert a clang cursor to a Function object."""
        if not cursor.location.file:
            return None

        # Get the full source
        source = self._get_full_source(cursor)
        if not source:
            return None

        # Get qualified name for C++ methods
        name = self._get_qualified_name(cursor)
        signature = get_type_spelling(cursor)
        raw_canonical = get_canonical_type_spelling(cursor)
        # Only store canonical_signature if it differs from signature
        canonical_signature = None if raw_canonical == signature else raw_canonical

        params = self._extract_params(cursor)
        callsites = self._extract_callsites(cursor)

        attributes = self._extract_attributes(cursor, source)

        func = Function(
            name=name,
            file_path=str(cursor.location.file),
            line_start=int(cursor.extent.start.line),
            line_end=int(cursor.extent.end.line),
            source=source,
            signature=signature,
            canonical_signature=canonical_signature,
            params=params,
            callsites=callsites,
            attributes=attributes,
        )

        # Extract code blocks for large functions.
        # Use a conservative threshold on raw source (10K) since preprocessed
        # source (llm_source) may be much larger after macro expansion.
        # The summarizers apply the real 40K threshold on llm_source.
        if len(source) > 10000:
            func.blocks = self._extract_blocks(
                cursor, source, cursor.extent.start.line
            )

        return func

    def _extract_attributes(self, cursor: Cursor, source: str) -> str:
        """Extract function attributes as raw text.

        Collects __attribute__((...)) from UNEXPOSED_ATTR AST children,
        plus _Noreturn / [[noreturn]] from the declaration line.
        """
        attrs: list[str] = []

        for child in cursor.get_children():
            if child.kind == CursorKind.UNEXPOSED_ATTR:
                tokens = list(child.get_tokens())
                if tokens:
                    token_text = " ".join(t.spelling for t in tokens)
                    # Reconstruct as __attribute__((...))
                    attrs.append(f"__attribute__(({token_text}))")

        # Extract __attribute__((...)) from type spelling (catches header-declared attrs)
        type_spelling = cursor.type.spelling if cursor.type else ""
        for m in re.finditer(r"__attribute__\(\(([^)]*)\)\)", type_spelling):
            attr_text = m.group(1).strip()
            if not any(attr_text in a for a in attrs):
                attrs.append(f"__attribute__(({attr_text}))")

        # Check declaration line for C11 _Noreturn / C++11 [[noreturn]]
        if source:
            first_line = source.split("\n")[0]
            if "_Noreturn" in first_line:
                if not any("noreturn" in a.lower() for a in attrs):
                    attrs.append("_Noreturn")
            if "[[noreturn]]" in first_line:
                if not any("noreturn" in a.lower() for a in attrs):
                    attrs.append("[[noreturn]]")

        return " ".join(attrs)

    def _extract_params(self, func_cursor: Cursor) -> list[str]:
        """Return formal parameter names from PARM_DECL children."""
        return [
            child.spelling
            for child in func_cursor.get_children()
            if child.kind == CursorKind.PARM_DECL and child.spelling
        ]

    def _extract_callsites(self, func_cursor: Cursor) -> list[dict]:
        """Walk CALL_EXPR nodes and collect per-callsite metadata.

        Each entry: {callee, line, line_in_body, via_macro, macro_name, args}
          - line:         1-based absolute line in source file
          - line_in_body: 0-based line offset from function start
          - via_macro:    True when the callee name is absent from the raw source line
                          (i.e. the call is inside a macro expansion)
          - macro_name:   uppercase identifier on that raw line (best-effort)
          - args:         textual representations of the actual arguments
        """
        file_path = str(func_cursor.location.file)
        func_start_line = func_cursor.extent.start.line  # 1-based

        raw_lines = self._get_file_lines(file_path)
        if not raw_lines:
            return []

        callsites: list[dict] = []
        seen: set[tuple[str, int]] = set()

        def walk(cursor: Cursor) -> None:
            for child in cursor.get_children():
                if child.kind == CursorKind.CALL_EXPR:
                    referenced = child.referenced
                    if referenced is not None:
                        callee_name = referenced.spelling
                        if callee_name:
                            line = child.location.line  # 1-based
                            key = (callee_name, line)
                            if key not in seen:
                                seen.add(key)
                                line_in_body = line - func_start_line  # 0-based
                                raw_line = raw_lines[line - 1] if 0 < line <= len(raw_lines) else ""
                                via_macro = not bool(
                                    re.search(r"\b" + re.escape(callee_name) + r"\b", raw_line)
                                )
                                macro_name = None
                                if via_macro:
                                    m = re.search(r"\b([A-Z_][A-Z0-9_]{2,})\s*\(", raw_line)
                                    macro_name = m.group(1) if m else None

                                # Argument texts: skip first child (function reference)
                                args: list[str] = []
                                children = list(child.get_children())
                                for arg_cursor in children[1:]:
                                    tokens = list(arg_cursor.get_tokens())
                                    arg_text = (
                                        " ".join(t.spelling for t in tokens)
                                        if tokens
                                        else arg_cursor.spelling or ""
                                    )
                                    # get_tokens() can return the entire macro
                                    # expansion; fall back to raw source text.
                                    if len(arg_text) > 200:
                                        raw = self._get_raw_extent_text(arg_cursor, raw_lines)
                                        if raw:
                                            arg_text = raw
                                    if arg_text:
                                        args.append(arg_text)

                                callsites.append({
                                    "callee": callee_name,
                                    "line": line,
                                    "line_in_body": line_in_body,
                                    "via_macro": via_macro,
                                    "macro_name": macro_name,
                                    "args": args,
                                })
                walk(child)

        walk(func_cursor)
        return callsites

    @staticmethod
    def _get_raw_extent_text(cursor: Cursor, raw_lines: list[str]) -> str:
        """Extract raw source text for a cursor's extent from file lines."""
        if not cursor.extent or not cursor.extent.start.file:
            return ""
        sl: int = cursor.extent.start.line - 1
        sc: int = cursor.extent.start.column - 1
        el: int = cursor.extent.end.line - 1
        ec: int = cursor.extent.end.column - 1
        if sl < 0 or el >= len(raw_lines):
            return ""
        if sl == el:
            return raw_lines[sl][sc:ec]
        parts = [raw_lines[sl][sc:]]
        for i in range(sl + 1, el):
            parts.append(raw_lines[i])
        parts.append(raw_lines[el][:ec])
        return "\n".join(parts)

    # -- Block extraction helpers ------------------------------------------------

    # Minimum lines for a child to be kept as its own block (smaller ones get
    # merged with adjacent siblings).
    _MIN_BLOCK_LINES = 10
    # Maximum chars for a single block before we recurse one level deeper.
    _MAX_BLOCK_CHARS = 40000
    # Maximum recursion depth when splitting oversized blocks.
    _MAX_SPLIT_DEPTH = 2

    def _extract_blocks(
        self,
        func_cursor: Cursor,
        source: str,
        func_line_start: int,
    ) -> list[FunctionBlock]:
        """Extract code blocks from a large function using AST subtree sizes.

        Works regardless of how clang represents switch internals (CASE_STMT
        vs COMPOUND_STMT) by measuring cursor.extent line spans of direct
        children of the relevant compound statement.

        Strategy:
          1. Find function body COMPOUND_STMT.
          2. Check for a dominant switch (>60% of function lines).
             If found, split at the switch body's children; otherwise split
             at the function body's children.
          3. Group small adjacent children; keep large ones as individual blocks.
          4. Infer kind/label from source text.
        """
        file_path = str(func_cursor.location.file)
        raw_lines = self._get_file_lines(file_path)
        if not raw_lines:
            return []

        # 1. Find function body COMPOUND_STMT
        body = self._find_compound_child(func_cursor)
        if body is None:
            return []

        # 2. Check for a dominant switch
        func_lines = func_cursor.extent.end.line - func_cursor.extent.start.line + 1
        split_target = self._find_dominant_switch_body(body, func_lines) or body

        # 3. Collect direct children and group into blocks
        children = list(split_target.get_children())
        if not children:
            return []

        return self._group_children_into_blocks(children, raw_lines, depth=0)

    @staticmethod
    def _find_compound_child(cursor: Cursor) -> Cursor | None:
        """Return the first COMPOUND_STMT child of *cursor*, or None."""
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                return child
        return None

    def _find_dominant_switch_body(
        self, body: Cursor, func_lines: int
    ) -> Cursor | None:
        """Return the COMPOUND_STMT of a switch that spans >60% of the function.

        Searches up to 3 levels deep to handle common patterns like
        ``for (...) { switch (...) { ... } }`` (e.g. sqlite3VdbeExec).
        """
        def _search(cursor: Cursor, depth: int) -> Cursor | None:
            for child in cursor.get_children():
                if child.kind == CursorKind.SWITCH_STMT:
                    switch_lines = child.extent.end.line - child.extent.start.line + 1
                    if switch_lines > func_lines * 0.6:
                        compound = self._find_compound_child(child)
                        if compound is not None:
                            return compound
                if depth < 3:
                    result = _search(child, depth + 1)
                    if result is not None:
                        return result
            return None

        return _search(body, 0)

    def _group_children_into_blocks(
        self,
        children: list[Cursor],
        raw_lines: list[str],
        depth: int,
    ) -> list[FunctionBlock]:
        """Group AST children into blocks based on their line spans.

        Small children (< _MIN_BLOCK_LINES) are merged with adjacent siblings.
        Large children are kept as individual blocks.
        If a single child exceeds _MAX_BLOCK_CHARS, recurse one level to split
        it further (up to _MAX_SPLIT_DEPTH).
        """
        # Build (start_line, end_line, cursor) tuples, skip children without extent
        spans: list[tuple[int, int, Cursor]] = []
        for child in children:
            if child.extent and child.extent.start.file:
                start = child.extent.start.line
                end = child.extent.end.line
                if end >= start:
                    spans.append((start, end, child))

        if not spans:
            return []

        # Sort by start line
        spans.sort(key=lambda s: s[0])

        # Group small spans with adjacent ones.
        # Use cur_code_lines (sum of each span's own line count) rather than
        # the file-extent distance, so that gaps from comments don't prematurely
        # stop merging in heavily-commented code like sqlite3.
        groups: list[tuple[int, int]] = []  # (start_line, end_line) of each group
        cur_start, cur_end, _ = spans[0]
        cur_code_lines = spans[0][1] - spans[0][0] + 1
        for i in range(1, len(spans)):
            s, e, _ = spans[i]
            next_span_lines = e - s + 1
            if cur_code_lines < self._MIN_BLOCK_LINES and next_span_lines < self._MIN_BLOCK_LINES:
                # Merge: extend current group
                cur_end = max(cur_end, e)
                cur_code_lines += next_span_lines
            else:
                # Flush current group
                groups.append((cur_start, cur_end))
                cur_start, cur_end = s, e
                cur_code_lines = next_span_lines
        groups.append((cur_start, cur_end))

        # Build FunctionBlock for each group
        blocks: list[FunctionBlock] = []
        for line_start, line_end in groups:
            block = self._make_block(raw_lines, line_start, line_end)
            if block is None:
                continue

            # Recurse if block is too large and we haven't hit depth limit
            if len(block.source) > self._MAX_BLOCK_CHARS and depth < self._MAX_SPLIT_DEPTH:
                # Find the cursor(s) in this range and try to split their children
                sub_children: list[Cursor] = []
                for s, e, cursor in spans:
                    if s >= line_start and e <= line_end:
                        # Try to get this cursor's compound children
                        compound = self._find_compound_child(cursor)
                        if compound is not None:
                            sub_children.extend(compound.get_children())
                        else:
                            sub_children.extend(cursor.get_children())
                if len(sub_children) > 1:
                    sub_blocks = self._group_children_into_blocks(
                        sub_children, raw_lines, depth + 1
                    )
                    if sub_blocks:
                        blocks.extend(sub_blocks)
                        continue
            blocks.append(block)

        return blocks

    @staticmethod
    def _make_block(
        raw_lines: list[str], line_start: int, line_end: int
    ) -> FunctionBlock | None:
        """Create a FunctionBlock from a line range, inferring kind/label from source."""
        if line_start < 1 or line_end > len(raw_lines) or line_end < line_start:
            return None

        block_source = "\n".join(raw_lines[line_start - 1 : line_end])

        # Infer kind and label from the first non-blank line
        kind = "block"
        label = ""
        for line in raw_lines[line_start - 1 : line_end]:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^case\s+.+:", stripped):
                kind = "switch_case"
                # Extract "case X:" as label
                m = re.match(r"^(case\s+.+?:)", stripped)
                label = m.group(1) if m else stripped[:60]
            elif stripped.startswith("default:"):
                kind = "default_case"
                label = "default:"
            else:
                label = stripped[:60]
            break

        if not label:
            label = f"lines {line_start}-{line_end}"

        return FunctionBlock(
            function_id=None,
            kind=kind,
            label=label,
            line_start=line_start,
            line_end=line_end,
            source=block_source,
        )

    def _get_qualified_name(self, cursor: Cursor) -> str:
        """Get the fully qualified name for a function/method."""
        parts = []
        c = cursor
        while c:
            if c.kind in (
                CursorKind.FUNCTION_DECL,
                CursorKind.CXX_METHOD,
                CursorKind.CONSTRUCTOR,
                CursorKind.DESTRUCTOR,
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
            ):
                if c.spelling:
                    parts.append(c.spelling)
            c = c.semantic_parent

        parts.reverse()
        return "::".join(parts) if len(parts) > 1 else (parts[0] if parts else "")

    def extract_typedefs_from_file(self, file_path: str | Path) -> list[dict]:
        """
        Extract type declarations from a source file.

        Captures:
        - C typedefs (kind='typedef')
        - C++ using aliases (kind='using')
        - struct/class/union definitions (kind='struct'/'class'/'union')

        Returns list of dicts with: name, kind, underlying_type,
        canonical_type, file_path, line_number
        """
        file_path = Path(file_path).resolve()
        try:
            tu = self.parse_file(file_path)
        except Exception:
            return []
        return self.extract_typedefs_from_tu(tu, file_path)

    def extract_typedefs_from_tu(self, tu, file_path: str | Path) -> list[dict]:
        """Extract type declarations from a pre-parsed translation unit."""
        file_path = Path(file_path).resolve()
        results: list[dict] = []
        # Reuse pp_file from cache if preprocessing was done for this file
        pp_file = self._pp_cache.get(str(file_path))
        self._extract_type_decls_recursive(tu.cursor, str(file_path), results, pp_file=pp_file)
        return results

    def _extract_type_decls_recursive(
        self, cursor: Cursor, main_file: str, results: list[dict],
        seen_locations: set[tuple[str, int]] | None = None,
        pp_file: PreprocessedFile | None = None,
    ) -> None:
        """Recursively extract type declarations from AST.

        Captures types from main_file and all included headers (including system
        headers) so that struct definitions are available for verification prompts.
        Deduplication across TUs is handled by the DB UNIQUE(name, kind, file_path)
        constraint.
        """
        if seen_locations is None:
            seen_locations = set()

        for child in cursor.get_children():
            if not child.location.file:
                continue
            child_file = str(child.location.file)

            if child.kind == CursorKind.TYPEDEF_DECL:
                loc_key = (child_file, child.location.line)
                if loc_key in seen_locations:
                    continue
                seen_locations.add(loc_key)
                underlying = child.underlying_typedef_type.spelling
                canonical = child.underlying_typedef_type.get_canonical().spelling
                definition = self._get_full_source(child) or None
                pp_definition = _annotate_pp_definition(
                    definition,
                    pp_file.extract_pp_source(
                        child_file, child.extent.start.line, child.extent.end.line
                    ) if pp_file else None,
                )
                results.append({
                    "name": child.spelling,
                    "kind": "typedef",
                    "underlying_type": underlying,
                    "canonical_type": canonical,
                    "file_path": child_file,
                    "line_number": child.location.line,
                    "definition": definition,
                    "pp_definition": pp_definition,
                })

            elif child.kind == CursorKind.TYPE_ALIAS_DECL:
                # C++ using X = Y;
                loc_key = (child_file, child.location.line)
                if loc_key in seen_locations:
                    continue
                seen_locations.add(loc_key)
                underlying = child.underlying_typedef_type.spelling
                canonical = child.underlying_typedef_type.get_canonical().spelling
                definition = self._get_full_source(child) or None
                pp_definition = _annotate_pp_definition(
                    definition,
                    pp_file.extract_pp_source(
                        child_file, child.extent.start.line, child.extent.end.line
                    ) if pp_file else None,
                )
                results.append({
                    "name": child.spelling,
                    "kind": "using",
                    "underlying_type": underlying,
                    "canonical_type": canonical,
                    "file_path": child_file,
                    "line_number": child.location.line,
                    "definition": definition,
                    "pp_definition": pp_definition,
                })

            elif child.kind in (
                CursorKind.STRUCT_DECL,
                CursorKind.CLASS_DECL,
                CursorKind.UNION_DECL,
            ):
                # Only record definitions (not forward declarations), skip anonymous
                if (child.is_definition() and child.spelling
                        and "(unnamed" not in child.spelling
                        and "(anonymous" not in child.spelling):
                    loc_key = (child_file, child.location.line)
                    if loc_key in seen_locations:
                        continue
                    seen_locations.add(loc_key)
                    kind_map = {
                        CursorKind.STRUCT_DECL: "struct",
                        CursorKind.CLASS_DECL: "class",
                        CursorKind.UNION_DECL: "union",
                    }
                    type_spelling = child.type.spelling
                    canonical = child.type.get_canonical().spelling
                    definition = self._get_full_source(child) or None
                    pp_definition = _annotate_pp_definition(
                        definition,
                        pp_file.extract_pp_source(
                            child_file, child.extent.start.line, child.extent.end.line
                        ) if pp_file else None,
                    )
                    results.append({
                        "name": child.spelling,
                        "kind": kind_map[child.kind],
                        "underlying_type": type_spelling,
                        "canonical_type": canonical,
                        "file_path": child_file,
                        "line_number": child.location.line,
                        "definition": definition,
                        "pp_definition": pp_definition,
                    })

            elif child.kind == CursorKind.VAR_DECL:
                # Capture file-scope static variables (static const arrays, etc.)
                # Include vars from main_file and from project-local headers.
                # System headers (under /usr/, /lib/, etc.) are excluded.
                # Vars from included headers are stored under main_file so that
                # get_static_vars_by_file(main_file) finds them at verify time.
                if (child.storage_class == StorageClass.STATIC
                        and not _is_system_header(child_file)):
                    loc_key = (child_file, child.location.line)
                    if loc_key not in seen_locations:
                        seen_locations.add(loc_key)
                        type_spelling = child.type.spelling
                        canonical = child.type.get_canonical().spelling
                        definition = self._get_full_source(child) or None
                        pp_definition = _annotate_pp_definition(
                            definition,
                            pp_file.extract_pp_source(
                                child_file, child.extent.start.line, child.extent.end.line
                            ) if pp_file else None,
                        )
                        results.append({
                            "name": child.spelling,
                            "kind": "static_var",
                            "underlying_type": type_spelling,
                            "canonical_type": canonical,
                            "file_path": main_file,
                            "line_number": child.location.line,
                            "definition": definition,
                            "pp_definition": pp_definition,
                        })

            # Recurse into namespaces and class/struct bodies for nested types,
            # but only if in main_file (avoid deep recursion into system headers)
            if child.kind in (
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
                CursorKind.UNION_DECL,
            ) and child_file == main_file:
                self._extract_type_decls_recursive(
                    child, main_file, results, seen_locations, pp_file=pp_file
                )

    def _get_file_lines(self, file_path: str) -> list[str]:
        """Return cached splitlines(keepends=True) for a file."""
        if file_path not in self._file_lines_cache:
            if file_path not in self._file_contents:
                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        self._file_contents[file_path] = f.read()
                except OSError:
                    return []
            self._file_lines_cache[file_path] = (
                self._file_contents[file_path].splitlines(keepends=True)
            )
        return self._file_lines_cache[file_path]

    def _get_full_source(self, cursor: Cursor) -> str:
        """Get the full source code of a function including body."""
        if not cursor.location.file:
            return ""

        file_path = str(cursor.location.file)
        lines = self._get_file_lines(file_path)
        if not lines:
            return ""

        start_line = cursor.extent.start.line - 1
        end_line = cursor.extent.end.line

        if start_line < 0 or end_line > len(lines):
            return ""

        source_lines = lines[start_line:end_line]
        return "".join(source_lines).strip()


class FunctionExtractorWithBodies(FunctionExtractor):
    """Function extractor that also parses function bodies for call extraction."""

    def __init__(
        self,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
        compile_commands: CompileCommandsDB | None = None,
    ):
        super().__init__(compile_args, libclang_path, compile_commands)

    def extract_from_file(self, file_path: str | Path) -> list[Function]:
        """Extract functions with full body parsing."""
        file_path = Path(file_path).resolve()

        # Get compile flags for this file
        args = self._get_compile_args(file_path)

        try:
            # Don't skip function bodies
            tu = self.index.parse(
                str(file_path),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}") from e

        functions: list[Function] = []
        self._extract_functions_recursive(tu.cursor, str(file_path), functions)

        return functions

    def get_translation_unit(self, file_path: str | Path) -> TranslationUnit:
        """Get the translation unit for advanced analysis."""
        file_path = Path(file_path).resolve()

        # Get compile flags for this file
        args = self._get_compile_args(file_path)

        return self.index.parse(
            str(file_path),
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
