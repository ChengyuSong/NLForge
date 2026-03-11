#!/usr/bin/env python3
"""Extract vulnerability ground truth from CGC cb-multios challenges.

Parses README.md files for CWE classifications and source files for
#ifdef PATCHED / #ifndef PATCHED blocks to build a ground truth JSON.

Usage:
    python scripts/cgc_extract_ground_truth.py [--cgc-dir /path/to/cb-multios] [-o output.json]
"""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import clang.cindex as ci

# CWE -> issue_kind mapping (only memory safety CWEs we can verify)
CWE_TO_ISSUE_KIND = {
    119: "buffer_overflow",   # Improper Restriction of Operations within the Bounds of a Memory Buffer
    120: "buffer_overflow",   # Buffer Copy without Checking Size of Input
    121: "buffer_overflow",   # Stack-based Buffer Overflow
    122: "buffer_overflow",   # Heap-based Buffer Overflow
    124: "buffer_overflow",   # Buffer Underwrite
    125: "buffer_overflow",   # Out-of-bounds Read
    126: "buffer_overflow",   # Buffer Over-read
    127: "buffer_overflow",   # Buffer Under-read
    131: "buffer_overflow",   # Incorrect Calculation of Buffer Size
    190: "buffer_overflow",   # Integer Overflow or Wraparound (often leads to buffer overflow)
    193: "buffer_overflow",   # Off-by-one Error
    680: "buffer_overflow",   # Integer Overflow to Buffer Overflow
    787: "buffer_overflow",   # Out-of-bounds Write
    788: "buffer_overflow",   # Access of Memory Location After End of Buffer
    415: "double_free",
    416: "use_after_free",
    476: "null_deref",
    457: "uninitialized_use",
    824: "use_after_free",    # Access of Uninitialized Pointer
    # CWEs that don't map to our issue_kinds get None:
    # 134 (format string), 191 (integer underflow),
    # 194 (unexpected sign extension),
    # 195 (signed to unsigned conversion), 200/201 (info exposure),
    # 285/287 (improper auth), 468 (incorrect pointer scaling), etc.
}

# Reverse mapping for prompt construction
CWE_DESCRIPTIONS = {
    119: "CWE-119: Improper Restriction of Operations within the Bounds of a Memory Buffer",
    120: "CWE-120: Buffer Copy without Checking Size of Input",
    121: "CWE-121: Stack-based Buffer Overflow",
    122: "CWE-122: Heap-based Buffer Overflow",
    124: "CWE-124: Buffer Underwrite",
    125: "CWE-125: Out-of-bounds Read",
    126: "CWE-126: Buffer Over-read",
    127: "CWE-127: Buffer Under-read",
    131: "CWE-131: Incorrect Calculation of Buffer Size",
    134: "CWE-134: Use of Externally-Controlled Format String",
    190: "CWE-190: Integer Overflow or Wraparound",
    191: "CWE-191: Integer Underflow",
    193: "CWE-193: Off-by-one Error",
    194: "CWE-194: Unexpected Sign Extension",
    195: "CWE-195: Signed to Unsigned Conversion Error",
    200: "CWE-200: Exposure of Sensitive Information",
    415: "CWE-415: Double Free",
    416: "CWE-416: Use After Free",
    457: "CWE-457: Use of Uninitialized Variable",
    468: "CWE-468: Incorrect Pointer Scaling",
    469: "CWE-469: Use of Pointer Subtraction to Determine Size",
    476: "CWE-476: NULL Pointer Dereference",
    680: "CWE-680: Integer Overflow to Buffer Overflow",
    787: "CWE-787: Out-of-bounds Write",
    788: "CWE-788: Access of Memory Location After End of Buffer",
    824: "CWE-824: Access of Uninitialized Pointer",
}


def assign_cwes_with_llm(
    llm_backend,
    challenge_name: str,
    cwes: list[int],
    vuln_desc: str,
    blocks: list[dict],
) -> None:
    """Use an LLM to assign the most appropriate CWE(s) to each PATCHED block.

    Modifies blocks in-place, setting ``cwes`` and ``issue_kind`` on each.
    Only called when there are multiple CWEs and unnumbered blocks.
    """
    cwe_list = "\n".join(
        f"  - {CWE_DESCRIPTIONS.get(c, f'CWE-{c}')}"
        for c in cwes
    )

    block_texts = []
    for i, b in enumerate(blocks):
        vuln_code = b.get("vulnerable_code", "") or "(code removed by patch)"
        patch_code = b.get("patched_code", "") or "(no replacement code)"
        block_texts.append(
            f"--- Patch {i+1} (in function `{b.get('function', '?')}`, "
            f"file `{b['file']}` line {b['line']}) ---\n"
            f"Vulnerable code:\n```c\n{vuln_code}\n```\n"
            f"Patched code:\n```c\n{patch_code}\n```"
        )

    prompt = f"""You are a vulnerability analyst. A CGC challenge "{challenge_name}" has these CWEs:
{cwe_list}

Vulnerability description from README:
{vuln_desc[:500] if vuln_desc else "(none)"}

The challenge has {len(blocks)} patches. For each patch, determine which CWE(s) from the list above it specifically fixes.

{chr(10).join(block_texts)}

Reply with a JSON array of objects, one per patch, in order:
[
  {{"patch": 1, "cwes": [<cwe_numbers>], "reasoning": "<brief explanation>"}},
  ...
]

Only assign CWEs from the provided list. Each patch should get the CWE(s) it specifically addresses.
Reply with ONLY the JSON array, no other text."""

    try:
        response = llm_backend.complete(prompt)
        # Extract JSON from response
        text = response.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        assignments = json.loads(text)

        for i, assignment in enumerate(assignments):
            if i >= len(blocks):
                break
            assigned_cwes = assignment.get("cwes", cwes)
            blocks[i]["cwes"] = assigned_cwes
            issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in assigned_cwes]
            blocks[i]["issue_kind"] = next((ik for ik in issue_kinds if ik), None)
    except Exception as e:
        print(f"  LLM CWE assignment failed for {challenge_name}: {e}",
              file=sys.stderr)
        # Fall back to assigning all CWEs
        for b in blocks:
            b["cwes"] = cwes
            issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in cwes]
            b["issue_kind"] = next((ik for ik in issue_kinds if ik), None)


def find_challenges(cgc_dir: Path) -> list[Path]:
    """Find single-binary challenges (skip multi-binary with cb_1/ subdirs)."""
    challenges_dir = cgc_dir / "challenges"
    if not challenges_dir.exists():
        print(f"Error: {challenges_dir} not found", file=sys.stderr)
        sys.exit(1)

    results = []
    for d in sorted(challenges_dir.iterdir()):
        if not d.is_dir():
            continue
        # Skip multi-binary challenges
        if (d / "cb_1").exists():
            continue
        results.append(d)
    return results


def parse_cwes(readme_path: Path) -> list[int]:
    """Extract CWE numbers from README.md."""
    if not readme_path.exists():
        return []
    text = readme_path.read_text(errors="replace")
    return [int(m) for m in re.findall(r"CWE-(\d+)", text)]


def parse_vuln_description(readme_path: Path) -> str:
    """Extract vulnerability description between ## Vulnerability and next ##."""
    if not readme_path.exists():
        return ""
    text = readme_path.read_text(errors="replace")
    m = re.search(
        r"##\s*Vulnerability\s*\n(.*?)(?=\n##\s|\Z)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return ""


class FunctionLocator:
    """Uses libclang to map source lines to enclosing function names."""

    def __init__(self, path_remap: tuple[str, str] | None = None):
        self._index = ci.Index.create()
        # file_path -> sorted list of (start_line, end_line, func_name)
        self._cache: dict[str, list[tuple[int, int, str]]] = {}
        # e.g. ("/workspace/", "/data/csong/cgc/cb-multios/")
        self._remap = path_remap

    def _remap_str(self, s: str) -> str:
        """Replace all occurrences of remap src with dst in s."""
        if self._remap and self._remap[0] in s:
            return s.replace(self._remap[0], self._remap[1])
        return s

    def _parse_file(self, file_path: str, compile_args: list[str]) -> None:
        """Parse a file with libclang and cache its function ranges."""
        if file_path in self._cache:
            return
        try:
            tu = self._index.parse(file_path, args=compile_args)
        except ci.TranslationUnitLoadError:
            self._cache[file_path] = []
            return

        funcs: list[tuple[int, int, str]] = []
        for c in tu.cursor.get_children():
            if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition():
                if c.location.file and c.location.file.name == file_path:
                    funcs.append(
                        (c.extent.start.line, c.extent.end.line, c.spelling)
                    )
        funcs.sort()
        self._cache[file_path] = funcs

    def find(self, file_path: str, target_line: int,
             compile_args: list[str]) -> str | None:
        """Find the function enclosing target_line (1-based)."""
        self._parse_file(file_path, compile_args)
        for start, end, name in self._cache[file_path]:
            if start <= target_line <= end:
                return name
        return None

    def compile_args_for_file(
        self, file_path: str, compile_commands: list[dict],
    ) -> list[str]:
        """Extract clean compile flags for file_path from compile_commands."""
        for entry in compile_commands:
            entry_file = self._remap_str(entry.get("file", ""))
            if entry_file == file_path:
                args = shlex.split(entry["command"])
                clean: list[str] = []
                skip = False
                for a in args[1:]:  # skip compiler
                    if skip:
                        skip = False
                        continue
                    if a == "-o":
                        skip = True
                        continue
                    if a in ("-c",):
                        continue
                    # Skip the source file argument
                    if self._remap_str(a) == file_path:
                        continue
                    clean.append(self._remap_str(a))
                return clean
        return ["-x", "c", "-std=c11"]


def parse_patched_blocks(
    file_path: Path, rel_path: str,
    locator: FunctionLocator | None = None,
    compile_commands: list[dict] | None = None,
) -> list[dict]:
    """Parse #ifdef PATCHED / #ifndef PATCHED blocks from a source file."""
    try:
        lines = file_path.read_text(errors="replace").splitlines()
    except Exception:
        return []

    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(
            r"#\s*(ifdef|ifndef)\s+(PATCHED(?:_(\d+))?)\s*$", stripped
        )
        if not m:
            i += 1
            continue

        directive = m.group(1)   # "ifdef" or "ifndef"
        patch_id = m.group(3)    # None or "1", "2", etc.

        # Find matching #else and #endif, handling nesting
        ifdef_line = i
        else_line = None
        endif_line = None
        depth = 1
        j = i + 1
        while j < len(lines) and depth > 0:
            s = lines[j].strip()
            if re.match(r"#\s*(ifdef|ifndef|if)\b", s):
                depth += 1
            elif re.match(r"#\s*else\b", s) and depth == 1:
                else_line = j
            elif re.match(r"#\s*endif\b", s):
                depth -= 1
                if depth == 0:
                    endif_line = j
            j += 1

        if endif_line is None:
            i += 1
            continue

        if directive == "ifdef":
            # #ifdef PATCHED: patched code in body, vulnerable in else
            patched_code = "\n".join(lines[ifdef_line + 1 : else_line or endif_line])
            if else_line is not None:
                vuln_code = "\n".join(lines[else_line + 1 : endif_line])
            else:
                vuln_code = ""  # No else → vulnerability is absence of patch
        else:
            # #ifndef PATCHED: vulnerable code in body, patched in else
            vuln_code = "\n".join(lines[ifdef_line + 1 : else_line or endif_line])
            if else_line is not None:
                patched_code = "\n".join(lines[else_line + 1 : endif_line])
            else:
                patched_code = ""  # No else → patch is to remove the vulnerable code

        if locator and compile_commands is not None:
            cc_args = locator.compile_args_for_file(
                str(file_path), compile_commands
            )
            func_name = locator.find(str(file_path), ifdef_line + 1, cc_args)
        else:
            func_name = None

        blocks.append({
            "patch_id": int(patch_id) if patch_id else None,
            "file": rel_path,
            "line": ifdef_line + 1,  # 1-indexed
            "function": func_name,
            "vulnerable_code": vuln_code.strip(),
            "patched_code": patched_code.strip(),
        })

        i = endif_line + 1

    return blocks


def extract_patch_defines(
    all_entries: list[dict], challenge_name: str
) -> list[str]:
    """Extract sorted unique PATCHED defines for a challenge from compile_commands.

    Returns e.g. ["PATCHED", "PATCHED_1", "PATCHED_2"].
    """
    defines: set[str] = set()
    challenge_pattern = f"/challenges/{challenge_name}/"
    for entry in all_entries:
        if challenge_pattern not in entry.get("file", ""):
            continue
        for m in re.findall(r"-D(PATCHED(?:_\d+)?)\b", entry.get("command", "")):
            defines.add(m)
    return sorted(defines)


def extract_challenge(
    challenge_dir: Path,
    locator: FunctionLocator | None = None,
    compile_commands: list[dict] | None = None,
    llm_backend=None,
) -> dict | None:
    """Extract ground truth for a single challenge."""
    name = challenge_dir.name
    readme = challenge_dir / "README.md"

    cwes = parse_cwes(readme)
    if not cwes:
        return None

    vuln_desc = parse_vuln_description(readme)

    # Find all source files under src/ and lib/
    source_dirs = [challenge_dir / "src", challenge_dir / "lib"]
    source_files = []
    for sd in source_dirs:
        if sd.exists():
            source_files.extend(
                sorted(sd.rglob("*.c")) + sorted(sd.rglob("*.cc"))
                + sorted(sd.rglob("*.cpp"))
            )

    # Parse PATCHED blocks
    all_blocks = []
    for sf in source_files:
        rel = str(sf.relative_to(challenge_dir))
        all_blocks.extend(parse_patched_blocks(
            sf, rel, locator, compile_commands,
        ))

    if not all_blocks:
        return None

    # Assign CWEs to vulnerabilities
    # When N PATCHED_N blocks and N CWEs, assign positionally (CGC convention)
    # Single PATCHED (patch_id=None) gets all CWEs
    numbered_blocks = [b for b in all_blocks if b["patch_id"] is not None]
    unnumbered_blocks = [b for b in all_blocks if b["patch_id"] is None]

    vulnerabilities = []

    if numbered_blocks:
        # Sort by patch_id
        numbered_blocks.sort(key=lambda b: b["patch_id"])
        for block in numbered_blocks:
            pid = block["patch_id"]
            # Assign CWE positionally: PATCHED_1 → first CWE, etc.
            if pid is not None and 1 <= pid <= len(cwes):
                block_cwes = [cwes[pid - 1]]
            else:
                block_cwes = cwes  # fallback: all CWEs
            issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in block_cwes]
            # Use first mappable issue_kind
            issue_kind = next((ik for ik in issue_kinds if ik), None)
            vulnerabilities.append({
                "patch_id": block["patch_id"],
                "file": block["file"],
                "line": block["line"],
                "function": block["function"],
                "issue_kind": issue_kind,
                "cwes": block_cwes,
                "vulnerable_code": block["vulnerable_code"],
                "patched_code": block["patched_code"],
            })

    if unnumbered_blocks and llm_backend and len(cwes) > 1:
        assign_cwes_with_llm(
            llm_backend, name, cwes, vuln_desc, unnumbered_blocks,
        )
        for block in unnumbered_blocks:
            vulnerabilities.append({
                "patch_id": block["patch_id"],
                "file": block["file"],
                "line": block["line"],
                "function": block["function"],
                "issue_kind": block.get("issue_kind"),
                "cwes": block.get("cwes", cwes),
                "vulnerable_code": block["vulnerable_code"],
                "patched_code": block["patched_code"],
            })
    else:
        for block in unnumbered_blocks:
            issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in cwes]
            issue_kind = next((ik for ik in issue_kinds if ik), None)
            vulnerabilities.append({
                "patch_id": block["patch_id"],
                "file": block["file"],
                "line": block["line"],
                "function": block["function"],
                "issue_kind": issue_kind,
                "cwes": cwes,
                "vulnerable_code": block["vulnerable_code"],
                "patched_code": block["patched_code"],
            })

    return {
        "cwes": cwes,
        "vuln_description": vuln_desc,
        "vulnerabilities": vulnerabilities,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract CGC vulnerability ground truth"
    )
    parser.add_argument(
        "--cgc-dir", type=Path,
        default=Path("/data/csong/cgc/cb-multios"),
        help="Path to cb-multios directory",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="cgc_ground_truth.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only extract challenges matching this substring",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="LLM backend for CWE-to-patch assignment (claude, gemini, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model override for LLM backend",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    challenges = find_challenges(args.cgc_dir)
    print(f"Found {len(challenges)} single-binary challenges")

    if args.filter:
        challenges = [
            c for c in challenges
            if args.filter.lower() in c.name.lower()
        ]
        print(f"Filter '{args.filter}': {len(challenges)} challenges")

    # Load compile_commands.json for patch define extraction
    cc_path = args.cgc_dir / "build" / "compile_commands.json"
    all_cc_entries = []
    if cc_path.exists():
        with open(cc_path) as f:
            all_cc_entries = json.load(f)

    # Build a per-challenge compile_commands lookup from the master file.
    # Keys are challenge names, values are the list of CC entries.
    cc_by_challenge: dict[str, list[dict]] = {}
    for entry in all_cc_entries:
        m_ch = re.search(r"/challenges/([^/]+)/", entry.get("file", ""))
        if m_ch:
            cc_by_challenge.setdefault(m_ch.group(1), []).append(entry)

    locator = FunctionLocator(
        path_remap=("/workspace/", str(args.cgc_dir) + "/"),
    )

    # Create LLM backend if requested
    llm_backend = None
    if args.backend:
        from llm_summary.llm import create_backend
        kwargs = {}
        if args.model:
            kwargs["model"] = args.model
        llm_backend = create_backend(args.backend, **kwargs)
        print(f"Using LLM backend '{args.backend}' for CWE assignment")

    result = {"challenges": {}}
    stats = {"total": 0, "extracted": 0, "no_cwes": 0, "no_patches": 0,
             "mappable": 0, "unmappable": 0, "llm_assigned": 0}

    for cdir in challenges:
        stats["total"] += 1
        challenge_cc = cc_by_challenge.get(cdir.name, [])
        # Filter to unpatched entries only (no -DPATCHED in command)
        challenge_cc = [
            e for e in challenge_cc
            if "-DPATCHED" not in e.get("command", "")
        ]
        entry = extract_challenge(cdir, locator, challenge_cc, llm_backend)
        if entry is None:
            if args.verbose:
                cwes = parse_cwes(cdir / "README.md")
                if not cwes:
                    print(f"  {cdir.name}: no CWEs found")
                    stats["no_cwes"] += 1
                else:
                    print(f"  {cdir.name}: no PATCHED blocks found")
                    stats["no_patches"] += 1
            continue

        # Add patch defines from compile_commands
        if all_cc_entries:
            entry["patch_defines"] = extract_patch_defines(
                all_cc_entries, cdir.name
            )

        stats["extracted"] += 1
        n_mappable = sum(
            1 for v in entry["vulnerabilities"] if v["issue_kind"]
        )
        n_unmappable = len(entry["vulnerabilities"]) - n_mappable
        stats["mappable"] += n_mappable
        stats["unmappable"] += n_unmappable

        if args.verbose:
            cwes_str = ",".join(f"CWE-{c}" for c in entry["cwes"])
            print(
                f"  {cdir.name}: {cwes_str}, "
                f"{len(entry['vulnerabilities'])} vuln(s), "
                f"{n_mappable} mappable"
            )

        result["challenges"][cdir.name] = entry

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary:")
    print(f"  Challenges scanned: {stats['total']}")
    print(f"  Challenges extracted: {stats['extracted']}")
    print(f"  Vulnerabilities with mappable issue_kind: {stats['mappable']}")
    print(f"  Vulnerabilities with unmappable CWEs: {stats['unmappable']}")
    print(f"\nGround truth written to: {args.output}")


if __name__ == "__main__":
    main()
