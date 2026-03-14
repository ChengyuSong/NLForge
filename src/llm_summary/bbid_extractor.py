"""Extract BB ID → source location mapping from UCSan-instrumented LLVM IR.

Parses instructions that have both !dbg (source location) and !dfsan.bb (BB ID)
metadata, producing a mapping from BB IDs to source file:line:col.

Also extracts the comparison predicate for conditional branches (icmp before br).
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BBInfo:
    bb_id: int
    file: str
    line: int
    col: int
    is_conditional: bool = False
    # For conditional branches: the comparison
    cmp_op: str = ""      # e.g., "eq", "ult", "uge"
    cmp_operands: str = "" # e.g., "i64 %len, 1"
    is_loop_exit: bool = False
    # Branch successor BB IDs (for conditional branches)
    true_bb_id: int | None = None
    false_bb_id: int | None = None
    # Source line text (filled in later)
    source_line: str = ""


def extract_bbids(ir_path: str, source_dir: str | None = None) -> list[BBInfo]:
    """Parse instrumented IR and extract BB ID → source location mapping.

    Args:
        ir_path: Path to UCSan-instrumented .ll file (must have debug info).
        source_dir: Optional source directory for reading source lines.

    Returns:
        List of BBInfo sorted by bb_id.
    """
    ir_text = Path(ir_path).read_text()

    # Parse metadata nodes: !N = !{i32 BBID} and !N = !DILocation(...)
    md_bbid = {}   # metadata_id -> bb_id
    md_diloc = {}  # metadata_id -> (file_md_id, line, col)
    md_file = {}   # metadata_id -> filename

    # !N = !{i32 BBID}
    for m in re.finditer(r'^!(\d+) = !\{i32 (\d+)\}', ir_text, re.MULTILINE):
        md_bbid[int(m.group(1))] = int(m.group(2))

    # !N = !DILocation(line: L, column: C, scope: !S)
    for m in re.finditer(
        r'^!(\d+) = !DILocation\(line: (\d+), column: (\d+), scope: !(\d+)',
        ir_text, re.MULTILINE
    ):
        md_diloc[int(m.group(1))] = {
            "line": int(m.group(2)),
            "col": int(m.group(3)),
            "scope": int(m.group(4)),
        }

    # !N = distinct !DISubprogram(name: "...", ..., file: !F, ...)
    # !N = !DIFile(filename: "...", directory: "...")
    for m in re.finditer(
        r'^!(\d+) = !DIFile\(filename: "([^"]*)", directory: "([^"]*)"',
        ir_text, re.MULTILINE
    ):
        md_id = int(m.group(1))
        fname = m.group(2)
        directory = m.group(3)
        if fname.startswith("/"):
            md_file[md_id] = fname
        elif directory:
            md_file[md_id] = f"{directory}/{fname}"
        else:
            md_file[md_id] = fname

    # Resolve scope → file: walk DISubprogram/DILexicalBlock to find file
    scope_file = {}
    # !N = distinct !DISubprogram(... file: !F, ...)
    for m in re.finditer(
        r'^!(\d+) = distinct !DISubprogram\([^)]*file: !(\d+)',
        ir_text, re.MULTILINE
    ):
        scope_file[int(m.group(1))] = int(m.group(2))

    # !N = [distinct] !DILexicalBlock(scope: !S, file: !F, ...)
    for m in re.finditer(
        r'^!(\d+) = (?:distinct )?!DILexicalBlock\(scope: !(\d+)(?:, file: !(\d+))?',
        ir_text, re.MULTILINE
    ):
        md_id = int(m.group(1))
        parent_scope = int(m.group(2))
        file_id = int(m.group(3)) if m.group(3) else None
        if file_id is not None:
            scope_file[md_id] = file_id
        else:
            scope_file[md_id] = scope_file.get(parent_scope)

    # !N = [distinct] !DILexicalBlockFile(scope: !S, file: !F, ...)
    for m in re.finditer(
        r'^!(\d+) = (?:distinct )?!DILexicalBlockFile\(scope: !(\d+), file: !(\d+)',
        ir_text, re.MULTILINE
    ):
        scope_file[int(m.group(1))] = int(m.group(3))

    def resolve_file(scope_id):
        """Walk scope chain to find the file."""
        visited = set()
        while scope_id is not None and scope_id not in visited:
            visited.add(scope_id)
            if scope_id in scope_file:
                file_id = scope_file[scope_id]
                if isinstance(file_id, int):
                    if file_id in md_file:
                        return md_file[file_id]
                    # file_id might be a scope, keep walking
                    scope_id = file_id
                else:
                    return file_id
            else:
                break
        return "<unknown>"

    # Pass 1: Map IR BB labels → BB IDs
    # Each BB label's first instruction with !dfsan.bb gives the BB ID
    label_to_bbid: dict[str, int] = {}
    lines = ir_text.split('\n')
    current_label = ""

    for line in lines:
        # BB label: "name:" or "N:" at start of line (no leading whitespace)
        label_match = re.match(r'^(\S+):', line)
        if label_match:
            current_label = label_match.group(1)
            continue
        # First instruction with !dfsan.bb in this block
        bb_match = re.search(r'!dfsan\.bb !(\d+)', line)
        if bb_match and current_label and current_label not in label_to_bbid:
            bb_md_id = int(bb_match.group(1))
            if bb_md_id in md_bbid:
                label_to_bbid[current_label] = md_bbid[bb_md_id]

    # Pass 2: Parse instructions with both !dbg and !dfsan.bb
    results = []
    prev_icmp = None

    for i, line in enumerate(lines):
        # Track icmp instructions (they precede conditional branches)
        icmp_match = re.match(
            r'\s+%\S+ = icmp (\w+) (.+?)(?:,\s*!|$)', line
        )
        if icmp_match:
            prev_icmp = (icmp_match.group(1), icmp_match.group(2).strip())
            continue

        # Look for instructions with both !dbg !N and !dfsan.bb !M
        dbg_match = re.search(r'!dbg !(\d+)', line)
        bb_match = re.search(r'!dfsan\.bb !(\d+)', line)
        if not dbg_match or not bb_match:
            if not line.strip().startswith('%'):
                prev_icmp = None
            continue

        dbg_id = int(dbg_match.group(1))
        bb_md_id = int(bb_match.group(1))

        if bb_md_id not in md_bbid or dbg_id not in md_diloc:
            prev_icmp = None
            continue

        bb_id = md_bbid[bb_md_id]
        loc = md_diloc[dbg_id]
        file_path = resolve_file(loc["scope"])

        is_cond = 'br i1' in line
        is_loop = '!llvm.loop' in line

        info = BBInfo(
            bb_id=bb_id,
            file=file_path,
            line=loc["line"],
            col=loc["col"],
            is_conditional=is_cond,
            is_loop_exit=is_loop,
        )

        if is_cond and prev_icmp:
            info.cmp_op = prev_icmp[0]
            info.cmp_operands = prev_icmp[1]

        # Extract branch successor BB IDs for conditional branches
        if is_cond:
            br_match = re.search(
                r'br i1 \S+, label %(\S+), label %(\S+)', line
            )
            if br_match:
                true_label = br_match.group(1)
                false_label = br_match.group(2)
                info.true_bb_id = label_to_bbid.get(true_label)
                info.false_bb_id = label_to_bbid.get(false_label)

        results.append(info)
        prev_icmp = None

    # Sort by bb_id
    results.sort(key=lambda x: x.bb_id)

    # Fill in source lines if source_dir provided
    if source_dir:
        _fill_source_lines(results, source_dir)

    return results


def _fill_source_lines(infos: list[BBInfo], source_dir: str) -> None:
    """Read source files and attach the source line text to each BBInfo."""
    source_cache: dict[str, list[str]] = {}
    source_dir_path = Path(source_dir)

    for info in infos:
        fpath = info.file
        # Try resolving relative to source_dir
        if not Path(fpath).exists() and source_dir:
            # Try basename match
            candidate = source_dir_path / Path(fpath).name
            if candidate.exists():
                fpath = str(candidate)

        if fpath not in source_cache:
            try:
                source_cache[fpath] = Path(fpath).read_text().splitlines()
            except (FileNotFoundError, PermissionError):
                source_cache[fpath] = []

        lines = source_cache[fpath]
        if 0 < info.line <= len(lines):
            info.source_line = lines[info.line - 1].strip()


def format_annotated_source(infos: list[BBInfo], source_path: str) -> str:
    """Format source code with BB ID annotations in the margin.

    Returns source code with BB IDs annotated as comments on the relevant lines.
    """
    try:
        source_lines = Path(source_path).read_text().splitlines()
    except FileNotFoundError:
        return f"# Source file not found: {source_path}"

    # Group BBInfo by line number
    line_bbs: dict[int, list[BBInfo]] = {}
    for info in infos:
        if Path(info.file).name == Path(source_path).name:
            line_bbs.setdefault(info.line, []).append(info)

    out = []
    for lineno, text in enumerate(source_lines, 1):
        if lineno in line_bbs:
            bbs = line_bbs[lineno]
            annotations = []
            for bb in bbs:
                parts = [f"BB:{bb.bb_id}"]
                if bb.is_conditional:
                    parts.append("cond")
                    if bb.cmp_op:
                        parts.append(bb.cmp_op)
                    if bb.true_bb_id is not None and bb.false_bb_id is not None:
                        parts.append(f"T:{bb.true_bb_id}")
                        parts.append(f"F:{bb.false_bb_id}")
                if bb.is_loop_exit:
                    parts.append("loop")
                annotations.append(" ".join(parts))
            tag = " | ".join(annotations)
            out.append(f"{text}  /* [{tag}] */")
        else:
            out.append(text)

    return "\n".join(out)


def parse_cfg_dump(cfg_path: str) -> dict[str, list[dict]]:
    """Parse UCSanPass --ucsan-dump-cfg output.

    Format per line: FN:bbid:TYPE[:successors]
      C (conditional):   FN:bbid:C:T:true_bb:F:false_bb
      D (unconditional): FN:bbid:D:next_bb
      S (switch):        FN:bbid:S:case1_bb:case2_bb:...
      R (return):        FN:bbid:R
      U (unreachable):   FN:bbid:U

    Returns:
        Dict mapping function name to list of BB dicts.
    """
    result: dict[str, list[dict]] = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            func = parts[0]
            bb_id = int(parts[1])
            kind = parts[2]
            entry: dict = {"bb_id": bb_id, "type": kind,
                           "true_bb": None, "false_bb": None,
                           "succs": []}
            if kind == "C":
                entry["true_bb"] = int(parts[4])
                entry["false_bb"] = int(parts[6])
                entry["succs"] = [entry["true_bb"], entry["false_bb"]]
            elif kind == "D":
                entry["succs"] = [int(parts[3])]
            elif kind == "S":
                entry["succs"] = [int(p) for p in parts[3:]]
            result.setdefault(func, []).append(entry)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <instrumented.ll> [source_dir] [source_file]")
        sys.exit(1)

    ir_path = sys.argv[1]
    source_dir = sys.argv[2] if len(sys.argv) > 2 else None
    source_file = sys.argv[3] if len(sys.argv) > 3 else None

    infos = extract_bbids(ir_path, source_dir)

    if source_file:
        print(format_annotated_source(infos, source_file))
    else:
        for info in infos:
            flags = []
            if info.is_conditional:
                flags.append("cond")
                if info.true_bb_id is not None:
                    flags.append(f"T:{info.true_bb_id}")
                if info.false_bb_id is not None:
                    flags.append(f"F:{info.false_bb_id}")
            if info.is_loop_exit:
                flags.append("loop")
            if info.cmp_op:
                flags.append(f"{info.cmp_op}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            src = f"  // {info.source_line}" if info.source_line else ""
            print(f"BB {info.bb_id:6d} → {Path(info.file).name}:{info.line}:{info.col}{flag_str}{src}")
