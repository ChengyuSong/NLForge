#!/usr/bin/env python3
"""Evaluate verifier accuracy against CGC ground truth.

Compares verification results in func-scans/cgc/<name>/functions.db against
the ground truth extracted by cgc_extract_ground_truth.py.

Usage:
    python scripts/cgc_evaluate.py [--ground-truth cgc_ground_truth.json] [-o report.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.db import SummaryDB

REPO_ROOT = Path(__file__).resolve().parent.parent
FUNC_SCANS_DIR = REPO_ROOT / "func-scans" / "cgc"


def has_verification_results(db_path: Path) -> bool:
    """Check if a functions.db has any verification summaries."""
    if not db_path.exists():
        return False
    db = SummaryDB(str(db_path))
    try:
        row = db.conn.execute(
            "SELECT COUNT(*) FROM verification_summaries"
        ).fetchone()
        return row is not None and row[0] > 0
    except Exception:
        return False
    finally:
        db.close()


def load_verifier_issues(db_path: Path) -> dict[str, list[dict]]:
    """Load all verification issues from a functions.db, keyed by function name.

    Returns: {function_name: [{"issue_kind": ..., "severity": ..., ...}, ...]}
    """
    result: dict[str, list[dict]] = {}
    db = SummaryDB(str(db_path))
    try:
        functions = db.get_all_functions()
        for func in functions:
            if func.id is None:
                continue
            vsummary = db.get_verification_summary_by_function_id(func.id)
            if vsummary and vsummary.issues:
                issues = [issue.to_dict() for issue in vsummary.issues]
                result.setdefault(func.name, []).extend(issues)
    finally:
        db.close()
    return result


def evaluate_challenge(
    name: str,
    gt_entry: dict,
    func_scans_dir: Path,
) -> dict:
    """Evaluate a single challenge. Returns per-challenge result dict."""
    db_path = func_scans_dir / name / "functions.db"
    patched_db_path = func_scans_dir / name / "functions_patched.db"
    result = {
        "challenge": name,
        "cwes": gt_entry["cwes"],
        "gt_vulnerabilities": len(gt_entry["vulnerabilities"]),
        "gt_mappable": 0,
        "true_positives": [],
        "confirmed_true_positives": [],
        "false_negatives": [],
        "false_positives": [],
        "patched_remaining_issues": [],
        "has_patched_db": patched_db_path.exists(),
        "error": None,
    }

    if not db_path.exists():
        result["error"] = "no functions.db"
        return result

    verifier_issues = load_verifier_issues(db_path)

    # Load patched verifier issues if available
    patched_issues: dict[str, list[dict]] | None = None
    if patched_db_path.exists():
        patched_issues = load_verifier_issues(patched_db_path)

    # Track which verifier issues are matched (for FP calculation)
    matched_verifier_issues: set[tuple[str, int]] = set()  # (func_name, issue_idx)

    # For each GT vulnerability with a mappable issue_kind
    for vuln in gt_entry["vulnerabilities"]:
        issue_kind = vuln.get("issue_kind")
        if not issue_kind:
            continue
        result["gt_mappable"] += 1

        func_name = vuln.get("function")
        if not func_name:
            result["false_negatives"].append({
                "vuln": _vuln_summary(vuln),
                "reason": "no function name in ground truth",
            })
            continue

        # Check if verifier found a matching issue
        func_issues = verifier_issues.get(func_name, [])
        matched = False
        for idx, issue in enumerate(func_issues):
            if issue["issue_kind"] == issue_kind:
                if (func_name, idx) not in matched_verifier_issues:
                    matched_verifier_issues.add((func_name, idx))
                    matched = True

                    tp_entry = {
                        "function": func_name,
                        "issue_kind": issue_kind,
                        "gt_file": vuln["file"],
                        "gt_line": vuln["line"],
                        "verifier_description": issue.get("description", ""),
                        "confirmed": False,
                    }

                    # Check if the issue disappeared in patched DB
                    if patched_issues is not None:
                        patched_func_issues = patched_issues.get(func_name, [])
                        still_present = any(
                            pi["issue_kind"] == issue_kind
                            for pi in patched_func_issues
                        )
                        if not still_present:
                            tp_entry["confirmed"] = True
                            result["confirmed_true_positives"].append(tp_entry)

                    result["true_positives"].append(tp_entry)
                    break

        if not matched:
            result["false_negatives"].append({
                "vuln": _vuln_summary(vuln),
                "reason": "no matching verifier issue",
            })

    # False positives: verifier issues not matched to any GT vulnerability
    for func_name, issues in verifier_issues.items():
        for idx, issue in enumerate(issues):
            if (func_name, idx) not in matched_verifier_issues:
                result["false_positives"].append({
                    "function": func_name,
                    "issue_kind": issue["issue_kind"],
                    "severity": issue.get("severity", ""),
                    "description": issue.get("description", ""),
                })

    # Issues that still appear in patched DB (should not exist if patch works)
    if patched_issues is not None:
        for func_name, issues in patched_issues.items():
            for issue in issues:
                result["patched_remaining_issues"].append({
                    "function": func_name,
                    "issue_kind": issue["issue_kind"],
                    "severity": issue.get("severity", ""),
                    "description": issue.get("description", ""),
                })

    return result


def _vuln_summary(vuln: dict) -> dict:
    """Compact summary of a GT vulnerability for reporting."""
    return {
        "function": vuln.get("function"),
        "file": vuln.get("file"),
        "line": vuln.get("line"),
        "issue_kind": vuln.get("issue_kind"),
        "cwes": vuln.get("cwes"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CGC verifier accuracy"
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=Path("cgc_ground_truth.json"),
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--func-scans-dir", type=Path, default=FUNC_SCANS_DIR,
        help="Root of func-scans/cgc directory",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="cgc_eval_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only evaluate challenges matching this substring",
    )
    parser.add_argument(
        "--verified-only", action="store_true",
        help="Only evaluate challenges that have verification results",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Error: {args.ground_truth} not found")
        print("Run cgc_extract_ground_truth.py first")
        sys.exit(1)

    with open(args.ground_truth) as f:
        gt = json.load(f)

    challenges = gt["challenges"]
    print(f"Loaded ground truth: {len(challenges)} challenges")

    if args.filter:
        filt = args.filter.lower()
        challenges = {
            k: v for k, v in challenges.items()
            if filt in k.lower()
        }
        print(f"Filter '{args.filter}': {len(challenges)} challenges")

    if args.verified_only:
        challenges = {
            k: v for k, v in challenges.items()
            if has_verification_results(args.func_scans_dir / k / "functions.db")
        }
        print(f"Verified-only: {len(challenges)} challenges with verification results")

    # Evaluate each challenge
    all_results = []
    totals = {
        "tp": 0, "fn": 0, "fp": 0, "confirmed_tp": 0,
        "gt_mappable": 0, "challenges_evaluated": 0,
        "challenges_skipped": 0, "challenges_with_patched": 0,
        "patched_remaining": 0,
    }
    per_kind = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "confirmed_tp": 0})

    for name, gt_entry in sorted(challenges.items()):
        result = evaluate_challenge(name, gt_entry, args.func_scans_dir)
        all_results.append(result)

        if result["error"]:
            totals["challenges_skipped"] += 1
            if args.verbose:
                print(f"  {name}: SKIP ({result['error']})")
            continue

        totals["challenges_evaluated"] += 1
        tp = len(result["true_positives"])
        ctp = len(result["confirmed_true_positives"])
        fn = len(result["false_negatives"])
        fp = len(result["false_positives"])
        totals["tp"] += tp
        totals["confirmed_tp"] += ctp
        totals["fn"] += fn
        totals["fp"] += fp
        totals["gt_mappable"] += result["gt_mappable"]
        if result["has_patched_db"]:
            totals["challenges_with_patched"] += 1
            totals["patched_remaining"] += len(result["patched_remaining_issues"])

        # Per-kind breakdown
        for m in result["true_positives"]:
            per_kind[m["issue_kind"]]["tp"] += 1
            if m.get("confirmed"):
                per_kind[m["issue_kind"]]["confirmed_tp"] += 1
        for m in result["false_negatives"]:
            kind = m["vuln"].get("issue_kind")
            if kind:
                per_kind[kind]["fn"] += 1
        for m in result["false_positives"]:
            per_kind[m["issue_kind"]]["fp"] += 1

        if args.verbose:
            status = "OK" if fn == 0 else "MISS"
            confirmed_str = f" confirmed={ctp}" if result["has_patched_db"] else ""
            print(
                f"  {name}: TP={tp}{confirmed_str} FN={fn} FP={fp} [{status}]"
            )

    # Compute metrics
    tp, fn, fp = totals["tp"], totals["fn"], totals["fp"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Print summary table
    print()
    print("=" * 60)
    print("CGC BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"  Challenges evaluated: {totals['challenges_evaluated']}")
    print(f"  Challenges skipped:   {totals['challenges_skipped']}")
    print(f"  GT vulnerabilities (mappable): {totals['gt_mappable']}")
    print()
    print(f"  True Positives:   {tp}")
    print(f"  False Negatives:  {fn}")
    print(f"  False Positives:  {fp}")
    print()
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # Patch confirmation stats
    if totals["challenges_with_patched"] > 0:
        ctp = totals["confirmed_tp"]
        confirm_rate = ctp / tp if tp > 0 else 0.0
        print()
        print(f"  --- Patch Confirmation ---")
        print(f"  Challenges with patched DB: {totals['challenges_with_patched']}")
        print(f"  Confirmed TPs (issue gone after patch): {ctp}/{tp} ({confirm_rate:.1%})")
        print(f"  Issues remaining in patched DBs: {totals['patched_remaining']}")

    if per_kind:
        print()
        has_confirmed = any(k["confirmed_tp"] > 0 for k in per_kind.values())
        if has_confirmed:
            print("  Per issue_kind:")
            print(f"  {'kind':<20s} {'TP':>4s} {'cTP':>4s} {'FN':>4s} {'FP':>4s} {'Prec':>6s} {'Rec':>6s}")
            print(f"  {'-'*20} {'----':>4s} {'----':>4s} {'----':>4s} {'----':>4s} {'------':>6s} {'------':>6s}")
            for kind in sorted(per_kind):
                k = per_kind[kind]
                ktp, kfn, kfp = k["tp"], k["fn"], k["fp"]
                kctp = k["confirmed_tp"]
                kprec = ktp / (ktp + kfp) if (ktp + kfp) > 0 else 0.0
                krec = ktp / (ktp + kfn) if (ktp + kfn) > 0 else 0.0
                print(
                    f"  {kind:<20s} {ktp:4d} {kctp:4d} {kfn:4d} {kfp:4d} {kprec:6.3f} {krec:6.3f}"
                )
        else:
            print("  Per issue_kind:")
            print(f"  {'kind':<20s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'Prec':>6s} {'Rec':>6s}")
            print(f"  {'-'*20} {'----':>4s} {'----':>4s} {'----':>4s} {'------':>6s} {'------':>6s}")
            for kind in sorted(per_kind):
                k = per_kind[kind]
                ktp, kfn, kfp = k["tp"], k["fn"], k["fp"]
                kprec = ktp / (ktp + kfp) if (ktp + kfp) > 0 else 0.0
                krec = ktp / (ktp + kfn) if (ktp + kfn) > 0 else 0.0
                print(
                    f"  {kind:<20s} {ktp:4d} {kfn:4d} {kfp:4d} {kprec:6.3f} {krec:6.3f}"
                )

    # Write report
    report = {
        "totals": {
            "challenges_evaluated": totals["challenges_evaluated"],
            "challenges_skipped": totals["challenges_skipped"],
            "gt_mappable": totals["gt_mappable"],
            "true_positives": tp,
            "confirmed_true_positives": totals["confirmed_tp"],
            "false_negatives": fn,
            "false_positives": fp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "challenges_with_patched": totals["challenges_with_patched"],
            "patched_remaining_issues": totals["patched_remaining"],
        },
        "per_issue_kind": {
            kind: {
                "tp": v["tp"], "confirmed_tp": v["confirmed_tp"],
                "fn": v["fn"], "fp": v["fp"],
                "precision": round(
                    v["tp"] / (v["tp"] + v["fp"])
                    if (v["tp"] + v["fp"]) > 0 else 0.0, 4
                ),
                "recall": round(
                    v["tp"] / (v["tp"] + v["fn"])
                    if (v["tp"] + v["fn"]) > 0 else 0.0, 4
                ),
            }
            for kind, v in sorted(per_kind.items())
        },
        "challenges": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
