"""Function extraction from assembly source files.

Uses `nm` on the assembled object file to get the authoritative list of
function symbols, then parses the .s/.S source to extract source text
for each function (label to next label).
"""

import os
import re
import subprocess
from pathlib import Path

from .models import Function

# Assembly file extensions
ASM_EXTENSIONS = {".s", ".S", ".asm"}

# Regex for a label at column 0 (function or otherwise)
_LABEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*):\s*$")


def get_function_symbols(object_path: Path) -> set[str]:
    """Run `nm` on an object file to get function symbol names.

    Returns the set of symbols with type T (global text) or t (local text).
    """
    try:
        result = subprocess.run(
            ["nm", str(object_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return set()

        symbols = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            # nm output: "addr T name" or "         U name"
            if len(parts) >= 3 and parts[1] in ("T", "t"):
                symbols.add(parts[2])
            elif len(parts) == 2 and parts[0] in ("T", "t"):
                symbols.add(parts[1])
        return symbols

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()


def _parse_label_positions(source_lines: list[str]) -> list[tuple[str, int]]:
    """Find all labels in source and their 1-based line numbers.

    Returns list of (label_name, line_number) in order of appearance.
    """
    labels = []
    for i, line in enumerate(source_lines):
        m = _LABEL_RE.match(line)
        if m:
            labels.append((m.group(1), i + 1))  # 1-based
    return labels


def extract_asm_functions(
    source_path: Path,
    object_path: Path | None = None,
    verbose: bool = False,
) -> list[Function]:
    """Extract functions from an assembly source file.

    Args:
        source_path: Path to the .s/.S source file
        object_path: Path to the assembled .o/.lo object file.
            If provided, nm is used to get the authoritative function list.
            If None, falls back to .type directives in the source.

    Returns:
        List of Function objects for each function found.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        return []

    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    source_lines = source_text.splitlines()

    # Get function names from nm (authoritative) or from .type directives (fallback)
    if object_path and Path(object_path).exists():
        func_names = get_function_symbols(Path(object_path))
    else:
        func_names = set()

    if not func_names:
        # Fallback: parse .type directives
        func_names = _parse_type_directives(source_lines)

    if not func_names:
        return []

    # Find all label positions in source
    all_labels = _parse_label_positions(source_lines)

    # Build function boundaries: for each function label, source extends
    # to the line before the next global/function label (or EOF)
    functions = []
    func_label_indices = []
    for i, (label, _line_num) in enumerate(all_labels):
        if label in func_names:
            func_label_indices.append(i)

    for idx_pos, label_idx in enumerate(func_label_indices):
        label_name, line_start = all_labels[label_idx]

        # Find the start of this function's block: include preceding
        # directives (.global, .type, .align, etc.) by scanning backwards
        block_start = line_start
        for back_line in range(line_start - 2, -1, -1):  # 0-based
            stripped = source_lines[back_line].strip()
            if not stripped or stripped.startswith(".") or stripped.startswith("#"):
                block_start = back_line + 1  # 1-based
            else:
                break

        # End: line before the next function's block start, or EOF
        if idx_pos + 1 < len(func_label_indices):
            next_label_idx = func_label_indices[idx_pos + 1]
            _, next_line = all_labels[next_label_idx]
            # Scan backwards from next label to find its directive block
            next_block_start = next_line
            for back_line in range(next_line - 2, -1, -1):
                stripped = source_lines[back_line].strip()
                if not stripped or stripped.startswith(".") or stripped.startswith("#"):
                    next_block_start = back_line + 1
                else:
                    break
            line_end = next_block_start - 1
        else:
            line_end = len(source_lines)

        # Trim trailing blank lines
        while line_end > block_start and not source_lines[line_end - 1].strip():
            line_end -= 1

        func_source = "\n".join(source_lines[block_start - 1:line_end])

        functions.append(Function(
            name=label_name,
            file_path=os.path.normpath(str(source_path)),
            line_start=block_start,
            line_end=line_end,
            source=func_source,
            signature=f"assembly ({source_path.suffix})",
        ))

    return functions


def _parse_type_directives(source_lines: list[str]) -> set[str]:
    """Parse .type directives to find function names.

    Looks for: .type <name>, @function  or  .type <name>, %function
    """
    type_re = re.compile(r'\.type\s+(\S+)\s*,\s*[@%]function')
    names = set()
    for line in source_lines:
        m = type_re.search(line)
        if m:
            names.add(m.group(1))
    return names
