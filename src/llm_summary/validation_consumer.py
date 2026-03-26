"""Consume thoroupy validation results and classify verdict outcomes.

Reads validation_result.json (from run_policy.py) alongside the triage
verdict and determines:

  safe + confirmed     → verdict holds, update DB contracts
  safe + rejected      → counter-example found, needs re-triage
  feasible + confirmed → bug path reachable, generate PoC
  feasible + rejected  → path infeasible, needs re-triage
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ucsan exit codes
EXIT_REASONS = {
    # internal (not real crashes)
    123: "loop_oob",
    124: "obj_oob",
    125: "stack_oob",
    # checker
    150: "ubi",
    151: "uaf",
    152: "oob",
    153: "null_deref",
    161: "oob_upcast",
    171: "panic",
}

INTERNAL_EXIT_CODES = {123, 124, 125}

# Backwards compat alias
CRASH_CODES = {k: v for k, v in EXIT_REASONS.items() if k not in INTERNAL_EXIT_CODES}


@dataclass
class ValidationOutcome:
    """Result of comparing a verdict against its validation run."""

    verdict_hypothesis: str  # "safe" or "feasible"
    confirmed: bool
    outcome: str  # "safe_confirmed", "safe_rejected", etc.

    # Evidence
    crashes: list[dict[str, Any]] = field(default_factory=list)
    assertion_failures: list[dict[str, Any]] = field(default_factory=list)
    assertion_passes: list[dict[str, Any]] = field(default_factory=list)
    traces_covered: list[dict[str, Any]] = field(default_factory=list)
    traces_infeasible: list[dict[str, Any]] = field(default_factory=list)
    traces_missed: list[dict[str, Any]] = field(default_factory=list)

    # For reflection
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.verdict_hypothesis,
            "confirmed": self.confirmed,
            "outcome": self.outcome,
            "crashes": self.crashes,
            "assertion_failures": self.assertion_failures,
            "assertion_passes": self.assertion_passes,
            "traces_covered": [t["goal"] for t in self.traces_covered],
            "traces_infeasible": [t["goal"] for t in self.traces_infeasible],
            "traces_missed": [t["goal"] for t in self.traces_missed],
            "summary": self.summary,
        }


def classify_outcome(
    verdict: dict[str, Any],
    validation: dict[str, Any],
) -> ValidationOutcome:
    """Classify the validation outcome for a single verdict.

    Args:
        verdict: Triage verdict dict (hypothesis, reasoning, etc.)
        validation: validation_result.json from run_policy.py
    """
    hypothesis = verdict.get("hypothesis", "unknown")
    raw_crashes = validation.get("crashes", [])

    plan_cov = validation.get("plan_coverage") or {}
    assertions = validation.get("assertions") or {}

    # Parse crash info — handle both old (list of ints) and new (list of dicts)
    crash_info = []
    for c in raw_crashes:
        if isinstance(c, dict):
            crash_info.append({"exit_code": c["code"], "kind": c["reason"]})
        else:
            code = c
            if code in INTERNAL_EXIT_CODES:
                continue  # not a real crash
            kind = EXIT_REASONS.get(code, f"unknown({code})")
            crash_info.append({"exit_code": code, "kind": kind})

    # Parse assertion results
    a_failures: list[dict[str, Any]] = []
    a_passes: list[dict[str, Any]] = []
    for a in assertions.get("assertions", []):
        entry = {
            "assertion_id": a["assertion_id"],
            "types": a.get("types", []),
        }
        if a.get("fail_count", 0) > 0:
            entry["fail_seeds"] = a["fail_seeds"]
            a_failures.append(entry)
        if a.get("pass_count", 0) > 0:
            entry["pass_seeds"] = a["pass_seeds"]
            a_passes.append(entry)

    # Parse trace coverage
    t_covered: list[dict[str, Any]] = []
    t_infeasible: list[dict[str, Any]] = []
    t_missed: list[dict[str, Any]] = []
    for t in plan_cov.get("traces", []):
        if t["status"] == "covered":
            t_covered.append(t)
        elif t["status"] == "infeasible":
            t_infeasible.append(t)
        else:
            t_missed.append(t)

    # Classify
    if hypothesis == "safe":
        # Safe verdict is rejected if we found crashes or assertion failures
        has_bugs = bool(crash_info) or bool(a_failures)
        confirmed = not has_bugs
        outcome = "safe_confirmed" if confirmed else "safe_rejected"

        if confirmed:
            # All counter-example traces were infeasible or missed
            parts = []
            if t_infeasible:
                parts.append(
                    f"{len(t_infeasible)} counter-example path(s) proven "
                    f"infeasible by solver"
                )
            if t_covered:
                parts.append(
                    f"{len(t_covered)} path(s) explored without violations"
                )
            summary = (
                "Safety verdict confirmed. "
                + "; ".join(parts) + "."
                if parts else "Safety verdict confirmed (no violations found)."
            )
        else:
            parts = []
            if crash_info:
                kinds = ", ".join(c["kind"] for c in crash_info)
                parts.append(f"crashes found: {kinds}")
            if a_failures:
                ids = ", ".join(str(a["assertion_id"]) for a in a_failures)
                parts.append(f"assertion failures: ids {ids}")
            summary = "Safety verdict REJECTED. " + "; ".join(parts) + "."

    elif hypothesis == "feasible":
        # Feasible verdict is confirmed if the bug path was covered
        # (crashes or assertion failures on the expected path)
        has_evidence = bool(crash_info) or bool(a_failures)
        path_covered = any(t["status"] == "covered" for t in plan_cov.get("traces", []))
        confirmed = has_evidence or path_covered
        outcome = "feasible_confirmed" if confirmed else "feasible_rejected"

        if confirmed:
            parts = []
            if crash_info:
                kinds = ", ".join(c["kind"] for c in crash_info)
                parts.append(f"crashes triggered: {kinds}")
            if t_covered:
                parts.append(
                    f"{len(t_covered)} trace(s) covered"
                )
            summary = "Feasibility confirmed. " + "; ".join(parts) + "."
        else:
            parts = []
            if t_infeasible:
                parts.append(
                    f"{len(t_infeasible)} path(s) proven infeasible"
                )
            if t_missed:
                parts.append(
                    f"{len(t_missed)} path(s) not reached (budget exhausted?)"
                )
            summary = (
                "Feasibility verdict REJECTED. " + "; ".join(parts) + "."
            )
    else:
        confirmed = False
        outcome = "unknown"
        summary = f"Unknown hypothesis: {hypothesis}"

    return ValidationOutcome(
        verdict_hypothesis=hypothesis,
        confirmed=confirmed,
        outcome=outcome,
        crashes=crash_info,
        assertion_failures=a_failures,
        assertion_passes=a_passes,
        traces_covered=t_covered,
        traces_infeasible=t_infeasible,
        traces_missed=t_missed,
        summary=summary,
    )


def consume_validation_dir(
    verdict_path: Path,
    harness_dir: Path,
) -> list[dict[str, Any]]:
    """Consume all validation results for a verdict file.

    Args:
        verdict_path: Path to verdict JSON (list of verdict dicts)
        harness_dir: Base harness directory containing per-verdict subdirs

    Returns:
        List of outcome dicts, one per verdict entry that has results.
    """
    with open(verdict_path) as f:
        verdicts = json.load(f)
    if not isinstance(verdicts, list):
        verdicts = [verdicts]

    from .models import SafetyIssue

    results: list[dict[str, Any]] = []
    for vi, v in enumerate(verdicts):
        func_name = v.get("function_name", "unknown")
        idx = v.get("issue_index", vi)

        issue_d = v.get("issue", {})
        si = SafetyIssue(
            location=issue_d.get("location", ""),
            issue_kind=issue_d.get("issue_kind", ""),
            description=issue_d.get("description", ""),
            severity=issue_d.get("severity", "medium"),
            callee=issue_d.get("callee"),
            contract_kind=issue_d.get("contract_kind"),
        )
        fp = si.fingerprint()
        vdir = harness_dir / func_name / f"v{idx}_{fp}"

        # Find all validation results in this verdict dir
        for vr_path in sorted(vdir.glob("validation_result.json")):
            with open(vr_path) as f:
                validation = json.load(f)
            outcome = classify_outcome(v, validation)
            results.append({
                "function": func_name,
                "issue_index": idx,
                "binary": validation.get("binary", ""),
                **outcome.to_dict(),
            })

    return results
