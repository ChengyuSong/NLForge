#!/usr/bin/env python3
"""Merge ASAN replay findings with patch-based ground truth.

Produces a refined ground truth where:
  - Location = ASAN manifestation site (where bug triggers), not patch site
  - bug_kind = ASAN-determined (concrete), not CWE-guessed
  - CWE = from README, cross-validated against ASAN findings
  - Disagreements between ASAN and README are flagged

Usage:
    python tools/merge_asan_ground_truth.py \
        --asan-results asan_ground_truth.json \
        --patch-gt cgc_ground_truth.json \
        -o cgc_ground_truth_refined.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ASAN bug_kind -> our issue_kind taxonomy
ASAN_TO_ISSUE_KIND = {
    "heap-buffer-overflow": "buffer_overflow",
    "stack-buffer-overflow": "buffer_overflow",
    "global-buffer-overflow": "buffer_overflow",
    "heap-use-after-free": "use_after_free",
    "stack-use-after-return": "use_after_free",
    "stack-use-after-scope": "use_after_free",
    "double-free": "double_free",
    "alloc-dealloc-mismatch": "double_free",
    "use-after-poison": "use_after_free",
    "container-overflow": "buffer_overflow",
    "dynamic-stack-buffer-overflow": "buffer_overflow",
    "stack-buffer-underflow": "buffer_overflow",
    "allocation-size-too-big": "integer_overflow",
    "calloc-overflow": "buffer_overflow",
    "stack-overflow": "stack_overflow",
    "bad-free": "double_free",
    "null_deref": "null_deref",
    "segv": "segv",
}

# CWE -> issue_kind (same as cgc_extract_ground_truth.py)
CWE_TO_ISSUE_KIND = {
    119: "buffer_overflow",
    120: "buffer_overflow",
    121: "buffer_overflow",
    122: "buffer_overflow",
    124: "buffer_overflow",
    125: "buffer_overflow",
    126: "buffer_overflow",
    127: "buffer_overflow",
    129: "buffer_overflow",
    131: "buffer_overflow",
    134: "buffer_overflow",
    190: "integer_overflow",
    193: "buffer_overflow",
    201: "buffer_overflow",
    415: "double_free",
    416: "use_after_free",
    457: "uninitialized_use",
    468: "buffer_overflow",
    476: "null_deref",
    680: "integer_overflow",
    787: "buffer_overflow",
    788: "buffer_overflow",
    824: "use_after_free",
    843: "buffer_overflow",
    763: "double_free",
}

# Files/functions to skip when finding the "real" crash site
# (these are our shims, not the actual bug location)
SKIP_FUNCTIONS = {
    "cgc_malloc", "cgc_free", "cgc_calloc", "cgc_realloc",
    "cgc_malloc_size",
    "malloc", "free", "calloc", "realloc",
    "__wrap_main", "__libc_start_main", "__libc_start_call_main",
    "_start",
}
SKIP_FILES = {"asan_malloc.c", "cgc_main_wrapper.c"}


def find_app_frame(frames: list[dict]) -> dict | None:
    """Find the first stack frame in application code (skip shims/libc)."""
    for f in frames:
        func = f.get("function")
        file_path = f.get("file") or ""
        basename = Path(file_path).name if file_path else ""
        if func and func not in SKIP_FUNCTIONS and basename not in SKIP_FILES:
            # Also skip frames without source info
            if f.get("file") and f.get("line"):
                return f
    return frames[0] if frames else None


def issue_kind_compatible(asan_kind: str, cwe_kind: str | None) -> bool:
    """Check if ASAN issue_kind is compatible with CWE-derived issue_kind.

    Some CWEs are broad (e.g., CWE-190 integer overflow can manifest as
    allocation-size-too-big or buffer_overflow). We treat these as compatible.
    """
    if cwe_kind is None:
        return True  # unmappable CWE, can't validate
    if asan_kind == cwe_kind:
        return True
    # Integer overflow can manifest as buffer overflow or allocation failure
    if cwe_kind == "integer_overflow" and asan_kind in (
        "buffer_overflow", "integer_overflow",
    ):
        return True
    # CWE says buffer_overflow but mapped from CWE-190 (integer overflow)
    # and ASAN says integer_overflow — that's fine
    if cwe_kind == "buffer_overflow" and asan_kind == "integer_overflow":
        return True
    # SEGV can be null_deref or buffer_overflow in disguise
    if asan_kind == "segv":
        return True
    return False


def merge(
    asan_results: dict,
    patch_gt: dict,
    readme_dir: Path | None = None,
) -> dict:
    """Merge ASAN findings with patch-based ground truth."""
    refined = {"challenges": {}}
    stats = {
        "total_challenges": 0,
        "asan_covered": 0,
        "no_asan": 0,
        "agree": 0,
        "disagree": 0,
        "asan_only": 0,
        "patch_only": 0,
    }

    # Index ASAN results by challenge name
    asan_by_challenge: dict[str, list[dict]] = {}
    for key, entry in asan_results.items():
        name = entry.get("challenge", key.split("/")[0])
        asan_by_challenge.setdefault(name, []).append(entry)

    all_challenges = set(patch_gt.get("challenges", {}).keys()) | set(
        asan_by_challenge.keys()
    )

    for name in sorted(all_challenges):
        stats["total_challenges"] += 1
        patch_entry = patch_gt.get("challenges", {}).get(name)
        asan_entries = asan_by_challenge.get(name, [])

        # Collect unique ASAN findings (deduplicate by bug_kind + app_frame)
        asan_findings = []
        seen = set()
        for ae in asan_entries:
            for finding in ae.get("findings", []):
                frames = finding.get("frames", [])
                app_frame = find_app_frame(frames)
                if not app_frame:
                    continue
                dedup_key = (
                    finding["bug_kind"],
                    app_frame.get("function"),
                    app_frame.get("file"),
                    app_frame.get("line"),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                asan_findings.append({
                    "raw_kind": finding["raw_kind"],
                    "bug_kind": finding["bug_kind"],
                    "issue_kind": ASAN_TO_ISSUE_KIND.get(
                        finding["bug_kind"], finding["bug_kind"]
                    ),
                    "sanitizer": finding["sanitizer"],
                    "function": app_frame.get("function"),
                    "file": app_frame.get("file"),
                    "line": app_frame.get("line"),
                    "pov": ae.get("pov"),
                    "frames": frames,
                })

        if not patch_entry and not asan_findings:
            continue

        # Get CWEs from patch GT
        cwes = patch_entry["cwes"] if patch_entry else []
        cwe_issue_kinds = [CWE_TO_ISSUE_KIND.get(c) for c in cwes]
        cwe_issue_kinds = [k for k in cwe_issue_kinds if k]

        # Parse README for vuln description if available
        vuln_desc = patch_entry.get("vuln_description", "") if patch_entry else ""

        # Build refined vulnerabilities
        vulnerabilities = []

        if asan_findings:
            stats["asan_covered"] += 1
            for af in asan_findings:
                # SEGV is a symptom, not a root cause — use CWE-derived
                # kind when available so the verifier can match it
                issue_kind = af["issue_kind"]
                if issue_kind == "segv" and cwe_issue_kinds:
                    issue_kind = cwe_issue_kinds[0]

                # Cross-validate against CWE
                compatible = any(
                    issue_kind_compatible(issue_kind, ck)
                    for ck in cwe_issue_kinds
                ) if cwe_issue_kinds else True

                if compatible:
                    stats["agree"] += 1
                    validation = "agree"
                else:
                    stats["disagree"] += 1
                    validation = "disagree"

                # Make file path relative to challenge dir
                rel_file = af["file"]
                if rel_file:
                    m = re.search(
                        rf"/challenges/{re.escape(name)}/(.+)$", rel_file,
                    )
                    if m:
                        rel_file = m.group(1)

                vulnerabilities.append({
                    "source": "asan",
                    "issue_kind": issue_kind,
                    "raw_kind": af["raw_kind"],
                    "sanitizer": af["sanitizer"],
                    "function": af["function"],
                    "file": rel_file,
                    "line": af["line"],
                    "pov": af["pov"],
                    "cwe_validation": validation,
                    "cwe_issue_kinds": cwe_issue_kinds,
                })
        else:
            stats["no_asan"] += 1

        # Keep patch-site vulns as secondary reference
        patch_vulns = []
        if patch_entry:
            for pv in patch_entry.get("vulnerabilities", []):
                patch_vulns.append({
                    "source": "patch",
                    "issue_kind": pv["issue_kind"],
                    "function": pv["function"],
                    "file": pv["file"],
                    "line": pv["line"],
                    "patch_id": pv.get("patch_id"),
                    "cwes": pv.get("cwes", cwes),
                })

        # Track coverage
        if asan_findings and not patch_entry:
            stats["asan_only"] += 1
        if patch_entry and not asan_findings:
            stats["patch_only"] += 1

        refined["challenges"][name] = {
            "cwes": cwes,
            "vuln_description": vuln_desc,
            "vulnerabilities": vulnerabilities,
            "patch_sites": patch_vulns,
        }

    refined["stats"] = stats
    return refined


def print_report(refined: dict) -> None:
    """Print a human-readable cross-validation report."""
    stats = refined["stats"]
    print("=== ASAN Cross-Validation Report ===\n")
    print(f"Total challenges:        {stats['total_challenges']}")
    print(f"ASAN-covered:            {stats['asan_covered']}")
    print(f"No ASAN findings:        {stats['no_asan']}")
    print(f"ASAN-only (no patch GT): {stats['asan_only']}")
    print(f"Patch-only (no ASAN):    {stats['patch_only']}")
    print(f"\nCross-validation:")
    print(f"  Agree (ASAN ≈ CWE):   {stats['agree']}")
    print(f"  Disagree:              {stats['disagree']}")

    # Show disagreements
    disagrees = []
    for name, entry in refined["challenges"].items():
        for v in entry["vulnerabilities"]:
            if v.get("cwe_validation") == "disagree":
                disagrees.append((name, v))

    if disagrees:
        print(f"\n--- Disagreements ({len(disagrees)}) ---")
        for name, v in disagrees:
            cwes = refined["challenges"][name]["cwes"]
            print(
                f"  {name}: ASAN={v['issue_kind']} ({v['raw_kind']}) "
                f"vs CWE={v['cwe_issue_kinds']} "
                f"(CWE-{','.join(str(c) for c in cwes)})"
            )
            print(f"    @ {v['function']} {v['file']}:{v['line']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge ASAN findings with patch-based ground truth"
    )
    parser.add_argument(
        "--asan-results", required=True,
        help="ASAN replay results JSON (from asan_replay_pov.py --batch)",
    )
    parser.add_argument(
        "--patch-gt", required=True,
        help="Patch-based ground truth JSON (from cgc_extract_ground_truth.py)",
    )
    parser.add_argument(
        "-o", "--output", default="cgc_ground_truth_refined.json",
        help="Output refined ground truth JSON",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(args.asan_results) as f:
        asan_results = json.load(f)
    with open(args.patch_gt) as f:
        patch_gt = json.load(f)

    refined = merge(asan_results, patch_gt)

    with open(args.output, "w") as f:
        json.dump(refined, f, indent=2)

    print_report(refined)
    print(f"\nRefined ground truth written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
