"""Shared tool definitions and executor for LLM agents (triage, reflection).

Tools are defined once and exposed to different agents via allow-lists.
Read-only tools: read_function_source, get_callers, get_callees,
                 get_contracts
Write tools:     update_contracts, submit_verdict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .code_contract.models import CodeContractSummary
from .db import SummaryDB
from .git_tools import GitTools
from .llm.base import LLMBackend
from .models import Function

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool-use schema)
# ---------------------------------------------------------------------------

READ_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_function_source",
        "description": (
            "Read the full source code of a function, similar to a Read/cat "
            "tool but looked up by function name from the project database. "
            "Returns macro-annotated source (original lines shown as "
            "'// (macro)' comments above their expanded form), file path, "
            "line range, and signature."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to read.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_callers",
        "description": (
            "Search for all functions that call the given function, similar "
            "to Grep but searching the call graph instead of text. Returns "
            "caller names with signatures, file paths, and full source code "
            "so you can see how arguments are passed to the callee."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the callee function to find callers of.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_callees",
        "description": (
            "Get all functions called by the given function, including "
            "resolved indirect call targets (function pointers, vtable "
            "calls). Returns fully-qualified callee function names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function whose callees to list.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_contracts",
        "description": (
            "Get the code-contract summary for a function: Hoare-style "
            "requires (preconditions), ensures (postconditions), and "
            "modifies per safety property (memsafe, memleak, overflow). "
            "Works for the target function, its callers, or its callees."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_issues",
        "description": (
            "Get the safety issues found for a function during "
            "verification. Returns issue kind, location, severity, and "
            "description for each issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function.",
                },
            },
            "required": ["function_name"],
        },
    },
]

WRITE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "upsert_review",
        "description": (
            "Mark a verification issue as false_positive or confirmed. "
            "This updates the issue_reviews table so the issue is skipped "
            "in future triage runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function that has the issue.",
                },
                "issue_index": {
                    "type": "integer",
                    "description": "Index of the issue in the verification summary.",
                },
                "status": {
                    "type": "string",
                    "enum": ["false_positive", "confirmed"],
                    "description": "Review status.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this issue is FP or confirmed.",
                },
            },
            "required": ["function_name", "issue_index", "status", "reason"],
        },
    },
    {
        "name": "update_contracts",
        "description": (
            "Update a function's code-contract summary. Use this to correct "
            "wrong contracts that cause false positives in callers. For "
            "example, if a callee's requires says 'ptr != NULL' but the "
            "callee actually handles NULL gracefully, remove the requires "
            "clause. After updating, all callers become dirty and will be "
            "re-verified on the next incremental run.\n\n"
            "The contracts object must have the code-contract schema: "
            "properties, requires, ensures, modifies, notes per property."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function whose contracts to update.",
                },
                "contracts": {
                    "type": "object",
                    "description": (
                        "The corrected contract object. Must have "
                        "'properties' (list of property names), and per-property "
                        "'requires', 'ensures', 'modifies' (each a dict mapping "
                        "property name to list of C-expression strings)."
                    ),
                },
            },
            "required": ["function_name", "contracts"],
        },
    },
]

# Triage-specific tools (phase transitions, verdict submission)
TRIAGE_ONLY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "transition_phase",
        "description": (
            "Transition to the next workflow phase. "
            "ANALYZE->HYPOTHESIZE, HYPOTHESIZE->VERDICT. "
            "VALIDATE phase is auto-entered after submit_verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "next_phase": {
                    "type": "string",
                    "enum": ["HYPOTHESIZE", "VERDICT"],
                    "description": "The phase to transition to.",
                },
            },
            "required": ["next_phase"],
        },
    },
    {
        "name": "verify_contract",
        "description": (
            "Trial-verify a function against a proposed contract WITHOUT "
            "writing to the database. Use this in HYPOTHESIZE phase to test "
            "whether a strengthened contract resolves the issue before "
            "submitting your verdict. You must provide the FULL contract "
            "(not just changed clauses) — use get_contracts to read the "
            "current contract first, then modify and pass the complete "
            "replacement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to verify.",
                },
                "contract": {
                    "type": "object",
                    "description": (
                        "The complete trial contract. Must have "
                        "'properties' (list of property names), and "
                        "per-property 'requires', 'ensures', 'modifies' "
                        "(each a dict mapping property name to list of "
                        "C-expression strings), and 'notes' (dict mapping "
                        "property name to string)."
                    ),
                },
            },
            "required": ["function_name", "contract"],
        },
    },
    {
        "name": "submit_verdict",
        "description": (
            "Submit the final triage verdict. Only callable in VERDICT phase. "
            "You must provide either a safety proof (with updated_contracts), "
            "a contract gap (with updated_contracts verified via "
            "verify_contract), or a feasibility proof (with feasible_path)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "enum": ["safe", "contract_gap", "feasible"],
                    "description": (
                        "safe: the obligation cannot manifest given caller "
                        "constraints. contract_gap: the function's requires "
                        "is too weak for callee requires — propose "
                        "strengthened contracts. feasible: the violation "
                        "is reachable."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed natural language proof. For safety proofs, "
                        "explain which caller constraints prevent the issue. "
                        "For feasibility proofs, describe the execution path."
                    ),
                },
                "updated_contracts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {
                                "type": "string",
                                "description": "Function whose contract to update.",
                            },
                            "property": {
                                "type": "string",
                                "enum": ["memsafe", "memleak", "overflow"],
                                "description": "Which safety property.",
                            },
                            "clause_type": {
                                "type": "string",
                                "enum": ["requires", "ensures"],
                                "description": "Add/update a requires or ensures clause.",
                            },
                            "predicate": {
                                "type": "string",
                                "description": (
                                    "C-expression predicate. E.g. "
                                    "'ptr != NULL', 'size <= 1024'."
                                ),
                            },
                        },
                        "required": [
                            "function", "property", "clause_type", "predicate",
                        ],
                    },
                    "description": (
                        "For 'safe' hypothesis: updated/additional contract "
                        "clauses that prove the obligation away."
                    ),
                },
                "feasible_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For 'feasible' hypothesis: the call chain that can "
                        "trigger the issue. E.g. ['main', 'process_input', "
                        "'parse_header', 'target_func (overflow at line 42)']."
                    ),
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Constraints on inputs for this proof. For safety: "
                        "'width <= MAX_WIDTH because caller validates'. "
                        "For feasibility: 'width is user-controlled via "
                        "parse_header()'."
                    ),
                },
                "assertions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The violation condition to check. E.g. "
                        "'width * height overflows uint32_t', "
                        "'buf[offset] is out-of-bounds when offset >= len'."
                    ),
                },
                "relevant_functions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Functions relevant to this proof. Include the target "
                        "function, callers that constrain its inputs, and "
                        "callees whose behavior matters. These will be kept "
                        "as real code in symbolic validation; everything else "
                        "gets stubbed."
                    ),
                },
                "validation_plan": {
                    "type": "array",
                    "description": (
                        "How to test the relevant_functions. Each element is "
                        "a test case with an 'entries' list (function names). "
                        "If entries has one function, test it alone. If "
                        "entries has multiple functions, call them sequentially "
                        "in test() (e.g. first call sets up state, second "
                        "call is the function under test). Example for a "
                        "safety proof that depends on an invariant: "
                        "[{\"entries\": [\"setup_fn\", \"target_fn\"]}]. "
                        "Example for independent entries: "
                        "[{\"entries\": [\"entry_a\"]}, "
                        "{\"entries\": [\"entry_b\"]}]."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "entries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["entries"],
                    },
                },
            },
            "required": ["hypothesis", "reasoning", "relevant_functions"],
        },
    },
]

# Reflection-specific: submit the reflection verdict
REFLECTION_VERDICT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "submit_reflection",
        "description": (
            "Submit your final reflection verdict. Call this after you have "
            "analyzed the validation outcome, reviewed/updated any wrong "
            "contracts, and marked issues as FP or confirmed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "enum": ["safe", "feasible"],
                    "description": "Revised hypothesis after reflection.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why you believe this hypothesis, citing evidence."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["accept", "re-validate", "re-triage"],
                    "description": "What action to take next.",
                },
                "action_reason": {
                    "type": "string",
                    "description": "Why this action is needed.",
                },
                "original_correct": {"type": "boolean"},
                "practically_triggerable": {"type": "boolean"},
                "contracts_updated": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of function names whose contracts "
                        "were corrected during reflection."
                    ),
                },
            },
            "required": [
                "hypothesis", "confidence", "reasoning",
                "action", "original_correct",
            ],
        },
    },
]

# Git tool names for filtering
GIT_TOOL_NAMES = {"git_show", "git_ls_tree", "git_grep"}


# ---------------------------------------------------------------------------
# Shared tool executor
# ---------------------------------------------------------------------------


def _resolve_function(db: SummaryDB, name: str) -> Function | None:
    """Find a function by name, returning the first match or None."""
    funcs = db.get_function_by_name(name)
    return funcs[0] if funcs else None


class ToolExecutor:
    """Executes tools against the function database.

    Shared by triage and reflection agents. Each agent controls which tools
    are available via allow-lists — the executor itself has no phase gating.
    """

    def __init__(
        self,
        db: SummaryDB,
        verbose: bool = False,
        git_tools: GitTools | None = None,
        model_used: str = "",
        project_path: Path | None = None,
        llm: LLMBackend | None = None,
    ) -> None:
        self.db = db
        self.verbose = verbose
        self.git_tools = git_tools
        self.model_used = model_used
        self.llm = llm
        self._project_path = (
            project_path.resolve() if project_path
            else (git_tools.repo if git_tools else None)
        )
        self._func_cache: dict[str, Function | None] = {}

    def _get_func(self, name: str) -> Function | None:
        if name not in self._func_cache:
            self._func_cache[name] = _resolve_function(self.db, name)
        return self._func_cache[name]

    def _rel_path(self, abs_path: str) -> str:
        """Make a file path relative to the project root."""
        if self._project_path is None:
            return abs_path
        try:
            return str(Path(abs_path).relative_to(self._project_path))
        except ValueError:
            return abs_path

    def execute(
        self, tool_name: str, tool_input: dict[str, Any],
        allowed: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input dict.
            allowed: If set, restrict to these tool names.
        """
        if allowed is not None and tool_name not in allowed:
            return {
                "error": (
                    f"Tool '{tool_name}' not available. "
                    f"Available: {sorted(allowed)}"
                ),
            }
        # Git tools
        if tool_name in GIT_TOOL_NAMES:
            if self.git_tools is None:
                return {
                    "error": f"Tool '{tool_name}' unavailable: no project path",
                }
            return self.git_tools.dispatch(tool_name, tool_input)

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        result: dict[str, Any] = handler(tool_input)
        return result

    # -- read_function_source --

    def _tool_read_function_source(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        return {
            "function": func.name,
            "signature": func.signature or "",
            "file_path": self._rel_path(func.file_path),
            "line_start": func.line_start,
            "line_end": func.line_end,
            "source": func.llm_source[:20000],
        }

    # -- get_callers --

    def _tool_get_callers(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        caller_ids = self.db.get_callers(func.id)
        callers = []
        for cid in caller_ids:
            caller = self.db.get_function(cid)
            if caller is None:
                continue
            info: dict[str, Any] = {
                "name": caller.name,
                "signature": caller.signature or "",
                "file_path": self._rel_path(caller.file_path),
                "source": caller.llm_source[:8000],
            }
            callers.append(info)

        return {
            "function": name,
            "caller_count": len(callers),
            "callers": callers,
        }

    # -- get_callees --

    def _tool_get_callees(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        callee_ids = self.db.get_callees(func.id)
        edges = self.db.get_call_edges_by_caller(func.id)
        indirect_ids = {e.callee_id for e in edges if e.is_indirect}

        callees = []
        for cid in callee_ids:
            callee = self.db.get_function(cid)
            if callee is None:
                continue
            fq = f"{self._rel_path(callee.file_path)}::{callee.name}"
            if callee.signature:
                fq += f" {callee.signature}"
            if cid in indirect_ids:
                fq += " [indirect]"
            callees.append(fq)

        return {
            "function": name,
            "callee_count": len(callees),
            "callees": callees,
        }

    # -- get_contracts --

    def _tool_get_contracts(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        cc = self.db.get_code_contract_summary(func.id)
        if cc is None:
            return {"function": name, "status": "no_contract"}

        return {
            "function": name,
            "signature": func.signature or "",
            **cc.to_dict(),
        }

    # -- get_issues --

    def _tool_get_issues(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        vs = self.db.get_verification_summary_by_function_id(func.id)
        if vs is None or not vs.issues:
            return {"function": name, "issues": []}

        return {
            "function": name,
            "issues": [i.to_dict() for i in vs.issues],
        }

    # -- upsert_review --

    def _tool_upsert_review(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        issue_index = inp["issue_index"]
        status = inp["status"]
        reason = inp.get("reason", "")

        vs = self.db.get_verification_summary_by_function_id(func.id)
        if vs is None or not vs.issues:
            return {"error": f"No verification issues for '{name}'."}
        if issue_index < 0 or issue_index >= len(vs.issues):
            return {
                "error": (
                    f"Issue index {issue_index} out of range "
                    f"(0..{len(vs.issues) - 1})."
                ),
            }

        issue = vs.issues[issue_index]
        fp = issue.fingerprint()

        self.db.upsert_issue_review(
            function_id=func.id,
            issue_index=issue_index,
            fingerprint=fp,
            status=status,
            reason=reason,
        )

        return {
            "function": name,
            "issue_index": issue_index,
            "status": status,
            "fingerprint": fp,
        }

    # -- update_contracts --

    def _tool_update_contracts(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        data = inp["contracts"]
        model_used = f"triage:{self.model_used}" if self.model_used else "triage"

        try:
            data["function"] = name
            summary = CodeContractSummary.from_dict(data)
            self.db.store_code_contract_summary(
                func, summary, model_used=model_used,
            )
        except Exception as e:
            return {"error": f"Failed to update contracts: {e}"}

        return {"function": name, "updated": True}

    # -- verify_contract (triage) --

    def _tool_verify_contract(self, inp: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None:
            return {"error": "verify_contract requires an LLM backend"}

        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        contract_data = dict(inp["contract"])
        contract_data["function"] = name
        try:
            trial_summary = CodeContractSummary.from_dict(contract_data)
        except Exception as e:
            return {"error": f"Invalid contract: {e}"}

        from .code_contract.pass_ import CodeContractPass, seed_stdlib_summaries

        summaries: dict[str, CodeContractSummary] = dict(
            seed_stdlib_summaries(svcomp=False),
        )
        callee_ids = self.db.get_callees(func.id)
        for cid in callee_ids:
            callee_func = self.db.get_function(cid)
            if callee_func and callee_func.id is not None:
                cc = self.db.get_code_contract_summary(callee_func.id)
                if cc:
                    summaries[callee_func.name] = cc
        summaries[name] = trial_summary

        funcs_by_id = {
            f.id: f.name
            for f in self.db.get_all_functions() if f.id is not None
        }
        edges: dict[str, set[str]] = {}
        for edge in self.db.get_all_call_edges():
            caller = funcs_by_id.get(edge.caller_id)
            callee = funcs_by_id.get(edge.callee_id)
            if caller and callee:
                edges.setdefault(caller, set()).add(callee)

        trial_pass = CodeContractPass(
            self.db,
            model=self.model_used or "triage",
            llm=self.llm,
        )
        new_issues = trial_pass._verify_one(
            func, trial_summary, summaries, edges,
        )

        vs = self.db.get_verification_summary_by_function_id(func.id)
        orig = [i.to_dict() for i in (vs.issues if vs else [])]
        flat_new: list[dict[str, Any]] = []
        for prop, issues in new_issues.items():
            for it in issues:
                flat_new.append({"property": prop, **it})

        return {
            "function": name,
            "original_issues": orig,
            "trial_issues": flat_new,
            "resolved": len(flat_new) == 0,
        }

    # -- submit_verdict (triage) --

    def _tool_submit_verdict(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- submit_reflection --

    def _tool_submit_reflection(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- transition_phase (triage) --

    def _tool_transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"next_phase": inp["next_phase"]}
