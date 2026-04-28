"""Contract-check agent: hunt for hazardous behaviors not warned about in docs.

Two-stage decoupled pipeline:

1. **Hypothesis** (one LLM call, own context, ~80 turns): the agent walks
   the public API of one library, reads source + existing contracts, and
   emits a list of *candidate hazards* via `submit_hazards`.

2. **Audit** (one LLM call PER candidate, fresh context each, ~60 turns):
   for each hazard the hypothesis flagged, a fresh agent searches the
   manual / public header / canonical example program for a warning and
   emits a verdict via `submit_audit_verdict`. Lean strict — vague or
   partial mentions count as undocumented.

3. **Aggregator** (no LLM): joins each candidate with its audit verdict.
   Documented hazards are dropped; the rest become `ContractGap`s in the
   final catalog. Each gap can carry both `missing_contract` (doc gap)
   and `incomplete_contract` (DB gap) categories — the latter when
   hypothesis flagged that our own contract DB is also missing the
   corresponding clause.

Why two stages: the original single-loop design exhausted its turn
budget in MINE/AUDIT spirals (~50 git_grep calls re-searching the same
hazard). Splitting gives audit a per-candidate budget and prevents one
bad regex spiral from poisoning the rest of the run. Hypothesis context
(which includes ~30 functions' source) is discarded after stage 1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_tools import (
    CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS,
    READ_TOOL_DEFINITIONS,
    ToolExecutor,
)
from .db import SummaryDB
from .git_tools import GIT_TOOL_DEFINITIONS as _GIT_TOOL_DEFS
from .git_tools import GitTools
from .llm.base import LLMBackend

# ---------------------------------------------------------------------------
# Header parsing — extract public API names from a C header.
# ---------------------------------------------------------------------------

# libpng / many libs that follow PNG_EXPORT(ordinal, type, name, args)
_PNG_EXPORT_RE = re.compile(
    r"^\s*PNG_EXPORTA?\s*\(\s*\d+\s*,\s*[^,]+,\s*(\w+)\s*,",
    re.MULTILINE,
)

# Naive C function declaration: `<type> name(...)`. Best-effort only —
# falls back when no PNG_EXPORT pattern is present. Allows leading
# whitespace; rejects macro/typedef noise via the `;` requirement and
# the C-keyword filter on the captured name.
_C_FUNC_DECL_RE = re.compile(
    r"^[ \t]*[\w\s\*\(\),]*?\b([A-Za-z_]\w+)\s*\([^;{]*\)\s*;",
    re.MULTILINE,
)

# Identifiers that look like function names but are actually keywords or
# common storage-class noise we want to skip in the naive fallback.
_C_KEYWORDS: set[str] = {
    "if", "for", "while", "switch", "return", "sizeof", "static",
    "extern", "inline", "const", "volatile", "struct", "union",
    "enum", "typedef", "void", "int", "char", "short", "long",
    "float", "double", "signed", "unsigned",
}


def parse_public_apis(header_text: str) -> list[str]:
    """Return ordered, de-duplicated list of public API names in *header_text*.

    Tries the libpng PNG_EXPORT macro first (catches every export the
    library considers public). If none match, falls back to a naive scan
    for `<type> name(...);` declarations.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _PNG_EXPORT_RE.findall(header_text):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if ordered:
        return ordered

    for name in _C_FUNC_DECL_RE.findall(header_text):
        if name in _C_KEYWORDS or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class HazardCandidate:
    """One candidate hazard from the hypothesis stage.

    `contract_clause` (and `contract_property`) are set iff the
    hypothesis agent observed that our code-contract DB is also missing
    the corresponding clause — this drives the `incomplete_contract`
    category in the final catalog.
    """

    function: str
    hazard_kind: str
    description: str
    source_evidence: str
    contract_clause: str = ""
    contract_property: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "hazard_kind": self.hazard_kind,
            "description": self.description,
            "source_evidence": self.source_evidence,
            "contract_clause": self.contract_clause,
            "contract_property": self.contract_property,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HazardCandidate:
        return cls(
            function=str(d.get("function", "")),
            hazard_kind=str(d.get("hazard_kind", "")),
            description=str(d.get("description", "")),
            source_evidence=str(d.get("source_evidence", "")),
            contract_clause=str(d.get("contract_clause", "")),
            contract_property=str(d.get("contract_property", "")),
        )


@dataclass
class AuditVerdict:
    """Per-candidate audit-stage verdict."""

    documented: bool
    doc_searched: str
    doc_quote: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "documented": self.documented,
            "doc_searched": self.doc_searched,
            "doc_quote": self.doc_quote,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditVerdict:
        return cls(
            documented=bool(d.get("documented", False)),
            doc_searched=str(d.get("doc_searched", "")),
            doc_quote=str(d.get("doc_quote", "")),
            recommendation=str(d.get("recommendation", "")),
        )


@dataclass
class ContractGap:
    """One undocumented hazard in the final catalog (post-aggregation).

    `categories` is a list of one or two values:
      - "missing_contract" — the documentation does not state this
        requirement, or does not state it clearly. Always present in
        the final catalog (audit verdict said undocumented).
      - "incomplete_contract" — our code-contract DB is missing the
        corresponding requires/ensures clause (FP source for verifier).
        Present iff the hypothesis carried a contract_clause through.
    """

    function: str
    categories: list[str]
    hazard_kind: str
    description: str
    source_evidence: str
    doc_searched: str
    recommendation: str
    doc_quote: str = ""
    contract_clause: str = ""
    contract_property: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "categories": list(self.categories),
            "hazard_kind": self.hazard_kind,
            "description": self.description,
            "source_evidence": self.source_evidence,
            "doc_searched": self.doc_searched,
            "doc_quote": self.doc_quote,
            "recommendation": self.recommendation,
            "contract_clause": self.contract_clause,
            "contract_property": self.contract_property,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContractGap:
        cats = d.get("categories", [])
        if not isinstance(cats, list):
            cats = []
        return cls(
            function=str(d.get("function", "")),
            categories=[str(c) for c in cats],
            hazard_kind=str(d.get("hazard_kind", "")),
            description=str(d.get("description", "")),
            source_evidence=str(d.get("source_evidence", "")),
            doc_searched=str(d.get("doc_searched", "")),
            doc_quote=str(d.get("doc_quote", "")),
            recommendation=str(d.get("recommendation", "")),
            contract_clause=str(d.get("contract_clause", "")),
            contract_property=str(d.get("contract_property", "")),
        )


@dataclass
class ContractCheckResult:
    """The output of one library audit."""

    library: str
    target: str
    summary: str = ""
    gaps: list[ContractGap] = field(default_factory=list)
    completed: bool = False  # True iff hypothesis returned candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "library": self.library,
            "target": self.target,
            "summary": self.summary,
            "gap_count": len(self.gaps),
            "gaps": [g.to_dict() for g in self.gaps],
            "completed": self.completed,
        }


# ---------------------------------------------------------------------------
# Tool definitions per stage
# ---------------------------------------------------------------------------

# Read tools the hypothesis agent uses to inspect each API.
_HYPOTHESIS_DB_TOOLS = {
    "read_function_source",
    "get_callers",
    "get_callees",
    "get_contracts",
}

# Read tools the audit agent uses (only get_contracts to re-confirm).
_AUDIT_DB_TOOLS = {"get_contracts"}

# Submit-tool names (terminal — exit each agent's loop).
_SUBMIT_HAZARDS = "submit_hazards"
_SUBMIT_AUDIT_VERDICT = "submit_audit_verdict"


def _filter_defs(
    defs: list[dict[str, Any]], names: set[str],
) -> list[dict[str, Any]]:
    return [d for d in defs if d["name"] in names]


def _hypothesis_tools() -> list[dict[str, Any]]:
    """Tools available to the hypothesis agent (full git + DB + submit)."""
    return [
        *_filter_defs(READ_TOOL_DEFINITIONS, _HYPOTHESIS_DB_TOOLS),
        *_filter_defs(
            CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS,
            {"list_public_apis", "list_apis_without_contracts",
             _SUBMIT_HAZARDS},
        ),
        *_GIT_TOOL_DEFS,
    ]


def _audit_tools() -> list[dict[str, Any]]:
    """Tools available to a per-candidate audit agent."""
    return [
        *_filter_defs(READ_TOOL_DEFINITIONS, _AUDIT_DB_TOOLS),
        *_filter_defs(
            CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS,
            {_SUBMIT_AUDIT_VERDICT},
        ),
        *_GIT_TOOL_DEFS,
    ]


HYPOTHESIS_TOOL_DEFINITIONS = _hypothesis_tools()
AUDIT_TOOL_DEFINITIONS = _audit_tools()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM_PROMPT = """\
You audit a C library for dangerous behaviors that downstream callers
must respect. Your output is a list of CANDIDATE HAZARDS that a
separate AUDIT agent will then check against the manual / header /
example program. You yourself do NOT decide whether the docs warn
about a hazard — that's the audit's job. Your job is enumeration.

The code and existing code-contracts are your source of truth.

## Inputs

You are given:
- The library name and build target.
- The contract database via tools (one row per function).
- The library's git repository via git_show / git_grep / git_ls_tree.

## What to do

1. **Find the public API surface.**
   - Use git_ls_tree at the repo root and likely subdirs (`include/`,
     `docs/`, `examples/`) to locate the public header. It is usually
     `<libname>.h` (e.g. `png.h`) or under `include/`.
   - Call list_public_apis on the header. For PNG_EXPORT-style
     libraries this returns the full export list. For other libraries
     the naive fallback may miss some — git_show the header and add
     names by hand if you spot exports the regex missed.
   - Call list_apis_without_contracts on the result to find APIs the
     contract pass never produced a row for. Each of those is an
     immediate "incomplete_contract" candidate (the audit agent will
     still check whether docs cover them).

2. **Walk each public API and identify hazardous behaviors.**
   For each interesting API:
   - get_contracts to see what we already record.
   - read_function_source to see the actual code.
   - get_callers / get_callees only when ordering / lifecycle matters.
   Then identify candidate hazards. Useful kinds:
     - ordering         — must be called before/after another API
     - lifecycle        — alloc/free/ownership rule
     - null_input       — pointer arg crashes if NULL
     - noreturn         — can longjmp / exit / abort out of the call
     - error_return     — returns NULL / negative on error and caller
                          must check
     - state_dependent  — only valid in a specific state (initialized,
                          after begin_X(), etc.)
     - callback_context — restrictions on what a user-supplied callback
                          can/cannot do
     - numeric_hazard   — overflow / div-by-zero / negative-shift risk
     - other            — anything else dangerous

3. **For each candidate, populate:**
   - `function`         — the API name
   - `hazard_kind`      — one of the above
   - `description`      — one paragraph in plain English. The audit
                          agent will use this verbatim to drive its
                          doc search, so be concrete: name the
                          parameter, the failure mode, and what a
                          warning would say.
   - `source_evidence`  — file:line and/or contract clause that
                          proves the hazard exists in the code (e.g.
                          "pngwio.c:42 / contract.requires[memsafe]:
                          fp != NULL"). NOT a doc quote — that's audit's
                          turn.
   - `contract_clause`  — set ONLY when our contract DB is also
                          missing this. A directly-addable C predicate
                          (e.g. "fp != NULL"). Empty otherwise.
   - `contract_property` — required iff contract_clause set
                           ("memsafe" / "memleak" / "overflow").

4. **Call submit_hazards EXACTLY ONCE** with all candidates. The call
   is terminal.

## Quality bar

- Skip plainly safe APIs (trivial getters, pure-arithmetic helpers).
- Aim for 20-50 high-signal candidates. More is fine if the API is
  large; do NOT pad with weak candidates.
- Every candidate MUST have concrete source_evidence.
- Don't try to judge what's documented — the audit agent does that.

## Pacing

You have a turn budget. Use it on whole-API coverage, not deep dives.
For each API: 1-3 tool calls, then move on. If the contract is rich
and the source has obvious obligations, log them and continue.
"""


AUDIT_SYSTEM_PROMPT = """\
You audit a SINGLE hazard candidate for documentation coverage in one C
library. The hypothesis agent has already established that the hazard
exists in the code; your only job is to check whether the manual /
public header / canonical example warns about it. Submit ONE
`submit_audit_verdict` call.

## Inputs

You are given:
- The library name and target.
- ONE candidate: `{function, hazard_kind, description, source_evidence}`.
- The library's git repository via git_show / git_grep / git_ls_tree.
- get_contracts — only to re-confirm the contract clause if needed.

## What to do

1. **Find the docs.** Use git_ls_tree at the repo root to locate:
   - the public header (`<libname>.h` or under `include/`)
   - a manual (`*manual*`, `docs/`, `*.md`, `README*`)
   - a canonical example program (`example*.c`, `examples/`,
     `contrib/`).

2. **Search for a warning** about THIS specific hazard. Use git_grep
   with phrases drawn from the candidate's description: function
   name, parameter name, "must", "do not", "before", "after",
   "undefined", "responsibility", "NULL". Use git_show to read the
   surrounding context of any hit.

3. **Decide. LEAN STRICT.**
   - If the docs CLEARLY warn about this exact hazard, set
     `documented=true` and quote the warning in `doc_quote`.
   - If the docs are silent, set `documented=false` and leave
     `doc_quote=""`.
   - If the docs make a vague or partial mention (e.g. "pass a valid
     FILE*" when the hazard is "fp must be non-NULL and binary mode"),
     that is NOT enough — set `documented=false`, quote the partial
     mention in `doc_quote`. The strict bar is the user's choice.

4. **Always populate `doc_searched`** with what you searched and what
   you found, even on the documented=true path: e.g.
   "libpng-manual.txt §IV.3 (clear warning at line 1234); png.h:1023
   (decl only); example.c:55 (uses without comment)". This is the
   audit trail.

5. **`recommendation`**: one-sentence change to propose to the
   upstream maintainer. Required when documented=false; empty string
   when documented=true.

6. **Call `submit_audit_verdict` EXACTLY ONCE.** Terminal.

## Pacing

You have a turn budget. Aim for 3-8 doc searches; do NOT spiral with
slight regex variations. If the first 3 reasonable searches turn up
nothing, the docs are silent — set documented=false and submit.
"""


# ---------------------------------------------------------------------------
# Hypothesis agent
# ---------------------------------------------------------------------------

DEFAULT_HYPOTHESIS_TURNS = 80
DEFAULT_AUDIT_TURNS = 60


class _ReActLoop:
    """Shared ReAct loop body used by both hypothesis and audit agents."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
        log_prefix: str = "ContractCheck",
    ) -> None:
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.project_path = project_path
        self.log_prefix = log_prefix

    def _run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        terminal_tool: str,
        max_turns: int,
    ) -> dict[str, Any] | None:
        """Run the ReAct loop. Returns the input of the terminal tool call
        if the agent submitted one; None otherwise.
        """
        if self.project_path is None:
            raise ValueError(
                f"{self.log_prefix} requires project_path (the git repo).",
            )

        git = GitTools(self.project_path)
        executor = ToolExecutor(
            self.db,
            verbose=self.verbose,
            git_tools=git,
            project_path=self.project_path,
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        terminal_input: dict[str, Any] | None = None

        for turn in range(max_turns):
            response = self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                system=system_prompt,
            )

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(
                        f"[{self.log_prefix}] LLM stopped at turn "
                        f"{turn + 1} ({stop})",
                    )
                break
            if stop != "tool_use":
                if self.verbose:
                    print(
                        f"[{self.log_prefix}] Unexpected stop_reason: "
                        f"{stop}",
                    )
                break

            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    entry: dict[str, Any] = {"type": "text", "text": block.text}
                    if getattr(block, "thought", False):
                        entry["thought"] = True
                    sig = getattr(block, "thought_signature", None)
                    if sig:
                        entry["thought_signature"] = sig
                    assistant_content.append(entry)
                    if self.verbose and not getattr(block, "thought", False):
                        print(f"  [LLM] {block.text[:300]}")

                elif block.type == "tool_use":
                    tool_entry: dict[str, Any] = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                    sig = getattr(block, "thought_signature", None)
                    if sig:
                        tool_entry["thought_signature"] = sig
                    assistant_content.append(tool_entry)

                    result = executor.execute(block.name, block.input)

                    if self.verbose:
                        err = result.get("error")
                        if err:
                            print(
                                f"  [Tool] {block.name} -> ERROR: "
                                f"{err[:150]}",
                            )
                        elif block.name == terminal_tool:
                            print(f"  [Tool] {terminal_tool} -> accepted")
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"  [Tool] {block.name}({arg})")

                    if (
                        block.name == terminal_tool
                        and result.get("accepted")
                    ):
                        terminal_input = dict(block.input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            if terminal_input is not None:
                break

        return terminal_input


class HypothesisAgent(_ReActLoop):
    """Stage 1: enumerate candidate hazards across a library's public API."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
    ) -> None:
        super().__init__(
            db, llm, verbose=verbose,
            project_path=project_path,
            log_prefix="Hypothesis",
        )

    def enumerate_hazards(
        self,
        library: str,
        target: str,
        *,
        max_turns: int = DEFAULT_HYPOTHESIS_TURNS,
    ) -> tuple[str, list[HazardCandidate]]:
        """Returns (summary, candidates)."""
        if self.verbose:
            print(
                f"\n[Hypothesis] {library}/{target} (max_turns={max_turns})",
            )
        user_prompt = self._build_prompt(library, target)
        result = self._run(
            system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=HYPOTHESIS_TOOL_DEFINITIONS,
            terminal_tool=_SUBMIT_HAZARDS,
            max_turns=max_turns,
        )
        if result is None:
            if self.verbose:
                print("[Hypothesis] No candidates submitted (turn limit)")
            return ("Hypothesis stage hit turn limit.", [])
        candidates = [
            HazardCandidate.from_dict(c)
            for c in result.get("candidates", [])
            if isinstance(c, dict)
        ]
        summary = str(result.get("summary", ""))
        if self.verbose:
            print(f"[Hypothesis] {len(candidates)} candidates submitted")
        return (summary, candidates)

    def _build_prompt(self, library: str, target: str) -> str:
        return "\n".join([
            "## Library to Audit (HYPOTHESIS stage)",
            "",
            f"- **Library**: {library}",
            f"- **Target**: {target}",
            "",
            "### Instructions",
            "Find the public API surface (use git_ls_tree to locate the "
            "header, then list_public_apis on it). Walk each interesting "
            "API: read_function_source + get_contracts. Identify candidate "
            "hazards (null_input / ordering / lifecycle / noreturn / "
            "error_return / state_dependent / callback_context / "
            "numeric_hazard). For each, populate function, hazard_kind, a "
            "concrete one-paragraph description, and source_evidence "
            "(file:line and/or a contract clause). If our contract DB is "
            "ALSO missing the corresponding clause, set contract_clause "
            "+ contract_property too.",
            "Aim for 20-50 high-signal candidates. Skip plainly safe "
            "APIs. Do NOT judge what's documented — the audit agent does "
            "that next. End with one submit_hazards call.",
        ])


class AuditAgent(_ReActLoop):
    """Stage 2: audit ONE candidate against the library docs."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
    ) -> None:
        super().__init__(
            db, llm, verbose=verbose,
            project_path=project_path,
            log_prefix="Audit",
        )

    def audit_candidate(
        self,
        candidate: HazardCandidate,
        library: str,
        target: str,
        *,
        max_turns: int = DEFAULT_AUDIT_TURNS,
    ) -> AuditVerdict:
        if self.verbose:
            print(
                f"\n[Audit] {library}/{target} :: "
                f"{candidate.function} [{candidate.hazard_kind}] "
                f"(max_turns={max_turns})",
            )
        user_prompt = self._build_prompt(candidate, library, target)
        result = self._run(
            system_prompt=AUDIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=AUDIT_TOOL_DEFINITIONS,
            terminal_tool=_SUBMIT_AUDIT_VERDICT,
            max_turns=max_turns,
        )
        if result is None:
            # Lean strict: if audit didn't finish, treat as undocumented.
            if self.verbose:
                print(
                    f"[Audit] {candidate.function}: turn limit hit, "
                    f"defaulting to undocumented",
                )
            return AuditVerdict(
                documented=False,
                doc_searched="(audit hit turn limit before submitting verdict)",
                recommendation=(
                    f"Document the {candidate.hazard_kind} hazard for "
                    f"{candidate.function}: {candidate.description}"
                ),
            )
        return AuditVerdict.from_dict(result)

    def _build_prompt(
        self, candidate: HazardCandidate, library: str, target: str,
    ) -> str:
        contract_note = ""
        if candidate.contract_clause:
            contract_note = (
                f"- DB also missing clause: "
                f"`{candidate.contract_clause}` "
                f"({candidate.contract_property or 'unspecified property'})\n"
            )
        return "\n".join([
            "## Audit ONE candidate",
            "",
            f"- **Library**: {library}",
            f"- **Target**: {target}",
            "",
            "### Candidate hazard",
            f"- **Function**: {candidate.function}",
            f"- **Kind**: {candidate.hazard_kind}",
            f"- **Description**: {candidate.description}",
            f"- **Source evidence**: {candidate.source_evidence}",
            contract_note.rstrip("\n") if contract_note else "",
            "",
            "### Instructions",
            "Find the docs (git_ls_tree to locate manual / header / "
            "example), then search for a warning about THIS hazard. "
            "Use git_grep / git_show. Lean strict: vague mentions = "
            "undocumented (quote them in doc_quote). Submit ONE "
            "submit_audit_verdict call.",
        ])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ContractCheckAgent:
    """Two-stage orchestrator: hypothesis → per-candidate audits → catalog."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.project_path = project_path

    def check_library(
        self,
        library: str,
        target: str,
        *,
        max_hypothesis_turns: int = DEFAULT_HYPOTHESIS_TURNS,
        max_audit_turns: int = DEFAULT_AUDIT_TURNS,
        audit_limit: int | None = None,
    ) -> ContractCheckResult:
        if self.project_path is None:
            raise ValueError(
                "ContractCheckAgent requires project_path (the git repo).",
            )

        # Stage 1: hypothesis
        hyp = HypothesisAgent(
            self.db, self.llm,
            verbose=self.verbose, project_path=self.project_path,
        )
        hyp_summary, candidates = hyp.enumerate_hazards(
            library, target, max_turns=max_hypothesis_turns,
        )

        if not candidates:
            return ContractCheckResult(
                library=library, target=target,
                summary=(
                    "Hypothesis stage produced no candidates. "
                    + hyp_summary
                ),
                gaps=[], completed=False,
            )

        total_candidates = len(candidates)
        if audit_limit is not None and audit_limit < total_candidates:
            if self.verbose:
                print(
                    f"\n[Orchestrator] --limit={audit_limit}: auditing first "
                    f"{audit_limit} of {total_candidates} candidates",
                )
            candidates = candidates[:audit_limit]

        # Stage 2: audit each candidate (sequential)
        audit = AuditAgent(
            self.db, self.llm,
            verbose=self.verbose, project_path=self.project_path,
        )
        verdicts: list[AuditVerdict] = []
        for i, cand in enumerate(candidates, start=1):
            if self.verbose:
                print(
                    f"\n[Orchestrator] Auditing {i}/{len(candidates)}: "
                    f"{cand.function}",
                )
            verdicts.append(
                audit.audit_candidate(
                    cand, library, target,
                    max_turns=max_audit_turns,
                ),
            )

        # Stage 3: aggregate
        gaps: list[ContractGap] = []
        for cand, verdict in zip(candidates, verdicts, strict=True):
            if verdict.documented:
                continue
            categories = ["missing_contract"]
            if cand.contract_clause:
                categories.append("incomplete_contract")
            gaps.append(
                ContractGap(
                    function=cand.function,
                    categories=categories,
                    hazard_kind=cand.hazard_kind,
                    description=cand.description,
                    source_evidence=cand.source_evidence,
                    doc_searched=verdict.doc_searched,
                    doc_quote=verdict.doc_quote,
                    recommendation=verdict.recommendation,
                    contract_clause=cand.contract_clause,
                    contract_property=cand.contract_property,
                ),
            )

        documented = len(candidates) - len(gaps)
        skipped = total_candidates - len(candidates)
        skipped_note = (
            f" ({skipped} skipped due to --limit)" if skipped else ""
        )
        summary = (
            f"{hyp_summary} "
            f"Audited {len(candidates)} of {total_candidates} candidates: "
            f"{len(gaps)} undocumented, {documented} documented"
            f"{skipped_note}."
        )
        return ContractCheckResult(
            library=library, target=target,
            summary=summary, gaps=gaps, completed=True,
        )
