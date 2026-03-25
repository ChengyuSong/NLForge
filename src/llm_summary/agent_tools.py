"""Shared tool definitions and executor for LLM agents (triage, reflection).

Tools are defined once and exposed to different agents via allow-lists.
Read-only tools: read_function_source, get_callers, get_callees,
                 get_summaries, get_verification_summary
Write tools:     upsert_review, update_summary, submit_verdict
"""

from __future__ import annotations

from typing import Any

from .db import SummaryDB
from .git_tools import GitTools
from .models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    BufferSizePair,
    FreeOp,
    FreeSummary,
    Function,
    InitOp,
    InitSummary,
    MemsafeContract,
    MemsafeSummary,
    ParameterInfo,
    SafetyIssue,
    VerificationSummary,
)

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
        "name": "get_summaries",
        "description": (
            "Get all analysis summaries for any function: memory safety "
            "pre-conditions, simplified contracts from verification, "
            "post-conditions (allocations, frees, initializations), and "
            "verification issues. Works for the target function, its "
            "callers, or its callees."
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
        "name": "get_verification_summary",
        "description": (
            "Get the full verification summary for a function, including "
            "all issues found and simplified contracts. Use this to see "
            "what the verifier reported for any function (callers, callees, "
            "or the target itself)."
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
            "in future validation runs."
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
        "name": "update_summary",
        "description": (
            "Update a function's summary for a specific pass. Use this to "
            "correct wrong summaries that cause false positives in callers. "
            "For example, if a callee's allocation summary says may_be_null=true "
            "but the function never returns NULL (it calls an error handler), "
            "update the allocation summary to set may_be_null=false. "
            "After updating, all callers become dirty and will be re-verified "
            "on the next incremental run.\n\n"
            "The summary_json must match the schema of the target pass."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function whose summary to update.",
                },
                "pass_name": {
                    "type": "string",
                    "enum": [
                        "allocation", "free", "init", "memsafe", "verification",
                    ],
                    "description": "Which summary pass to update.",
                },
                "summary_json": {
                    "type": "object",
                    "description": (
                        "The corrected summary object. Must match the schema "
                        "for the target pass (same format as get_summaries output)."
                    ),
                },
            },
            "required": ["function_name", "pass_name", "summary_json"],
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
        "name": "submit_verdict",
        "description": (
            "Submit the final triage verdict. Only callable in VERDICT phase. "
            "You must provide either a safety proof (with updated_contracts) "
            "or a feasibility proof (with feasible_path)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "enum": ["safe", "feasible"],
                    "description": (
                        "safe: the issue cannot manifest given caller constraints. "
                        "feasible: the violation condition is reachable."
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
                            "target": {"type": "string"},
                            "contract_kind": {
                                "type": "string",
                                "enum": [
                                    "not_null", "nullable", "not_freed",
                                    "initialized", "buffer_size",
                                ],
                            },
                            "description": {"type": "string"},
                            "size_expr": {"type": "string"},
                        },
                        "required": ["target", "contract_kind", "description"],
                    },
                    "description": (
                        "For 'safe' hypothesis: updated/additional contracts "
                        "that prove the issue away. These will be written back "
                        "to the DB so the verifier won't flag this again."
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
            "summaries, and marked issues as FP or confirmed."
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
                "summaries_updated": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of 'function_name:pass_name' entries for "
                        "summaries that were corrected during reflection."
                    ),
                },
                "issues_reviewed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of 'function_name#index:status' entries for "
                        "issues that were reviewed during reflection."
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
    ) -> None:
        self.db = db
        self.verbose = verbose
        self.git_tools = git_tools
        self.model_used = model_used
        self._func_cache: dict[str, Function | None] = {}

    def _get_func(self, name: str) -> Function | None:
        if name not in self._func_cache:
            self._func_cache[name] = _resolve_function(self.db, name)
        return self._func_cache[name]

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
            "file_path": func.file_path,
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
                "file_path": caller.file_path,
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
            fq = f"{callee.file_path}::{callee.name}"
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

    # -- get_summaries --

    def _tool_get_summaries(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        result: dict[str, Any] = {
            "function": func.name,
            "signature": func.signature,
        }

        memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if memsafe and memsafe.contracts:
            result["pre_conditions"] = [
                {
                    "target": c.target,
                    "kind": c.contract_kind,
                    "description": c.description,
                    **({"size_expr": c.size_expr} if c.size_expr else {}),
                }
                for c in memsafe.contracts
            ]
        else:
            result["pre_conditions"] = []

        vsummary = self.db.get_verification_summary_by_function_id(func.id)
        if vsummary:
            if vsummary.simplified_contracts is not None:
                result["simplified_contracts"] = [
                    {
                        "target": c.target,
                        "kind": c.contract_kind,
                        "description": c.description,
                        **({"size_expr": c.size_expr} if c.size_expr else {}),
                    }
                    for c in vsummary.simplified_contracts
                ]
            result["issues_count"] = len(vsummary.issues)

        alloc = self.db.get_summary_by_function_id(func.id)
        if alloc and alloc.allocations:
            result["allocations"] = [
                {
                    "source": a.source,
                    **({"size_expr": a.size_expr} if a.size_expr else {}),
                    "may_be_null": a.may_be_null,
                    "returned": a.returned,
                }
                for a in alloc.allocations
            ]

        free_summary = self.db.get_free_summary_by_function_id(func.id)
        if free_summary and free_summary.frees:
            result["frees"] = [
                {
                    "deallocator": f.deallocator,
                    "target": f.target,
                    "conditional": f.conditional,
                    **({"condition": f.condition} if f.condition else {}),
                    "nulled_after": f.nulled_after,
                }
                for f in free_summary.frees
            ]

        init_summary = self.db.get_init_summary_by_function_id(func.id)
        if init_summary and init_summary.inits:
            result["initializations"] = [
                {
                    "initializer": i.initializer,
                    "target": i.target,
                    **({"byte_count": i.byte_count} if i.byte_count else {}),
                }
                for i in init_summary.inits
            ]

        return result

    # -- get_verification_summary --

    def _tool_get_verification_summary(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        vsummary = self.db.get_verification_summary_by_function_id(func.id)
        if vsummary is None:
            return {"function": name, "status": "not_verified"}

        return {
            "function": name,
            "description": vsummary.description,
            "simplified_contracts": (
                [
                    {
                        "target": c.target,
                        "kind": c.contract_kind,
                        "description": c.description,
                        **({"size_expr": c.size_expr} if c.size_expr else {}),
                    }
                    for c in vsummary.simplified_contracts
                ]
                if vsummary.simplified_contracts is not None
                else None
            ),
            "issues": [i.to_dict() for i in vsummary.issues],
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

    # -- update_summary --

    def _tool_update_summary(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        pass_name = inp["pass_name"]
        data = inp["summary_json"]
        model_used = f"reflection:{self.model_used}" if self.model_used else "reflection"

        try:
            if pass_name == "allocation":
                summary = _parse_allocation_summary(data, func.name)
                self.db.upsert_summary(func, summary, model_used=model_used)
            elif pass_name == "free":
                summary_f = _parse_free_summary(data, func.name)
                self.db.upsert_free_summary(func, summary_f, model_used=model_used)
            elif pass_name == "init":
                summary_i = _parse_init_summary(data, func.name)
                self.db.upsert_init_summary(func, summary_i, model_used=model_used)
            elif pass_name == "memsafe":
                summary_m = _parse_memsafe_summary(data, func.name)
                self.db.upsert_memsafe_summary(
                    func, summary_m, model_used=model_used,
                )
            elif pass_name == "verification":
                summary_v = _parse_verification_summary(data, func.name)
                self.db.upsert_verification_summary(
                    func, summary_v, model_used=model_used,
                )
            else:
                return {"error": f"Unknown pass: {pass_name}"}
        except Exception as e:
            return {"error": f"Failed to update {pass_name}: {e}"}

        return {
            "function": name,
            "pass_name": pass_name,
            "updated": True,
        }

    # -- submit_verdict (triage) --

    def _tool_submit_verdict(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- submit_reflection --

    def _tool_submit_reflection(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- transition_phase (triage) --

    def _tool_transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        # Phase state is managed externally by the triage agent wrapper.
        # This is a passthrough that returns the requested transition.
        return {"next_phase": inp["next_phase"]}


# ---------------------------------------------------------------------------
# Summary parsers (JSON dict → model objects)
# ---------------------------------------------------------------------------


def _parse_allocation_summary(
    data: dict[str, Any], func_name: str,
) -> AllocationSummary:
    allocations = []
    for a in data.get("allocations", []):
        allocations.append(Allocation(
            alloc_type=AllocationType(a.get("type", "heap")),
            source=a.get("source", ""),
            size_expr=a.get("size_expr"),
            size_params=a.get("size_params", []),
            returned=a.get("returned", False),
            stored_to=a.get("stored_to"),
            may_be_null=a.get("may_be_null", True),
        ))
    parameters = {}
    for k, v in data.get("parameters", {}).items():
        parameters[k] = ParameterInfo(
            role=v.get("role", ""),
            used_in_allocation=v.get("used_in_allocation", False),
        )
    bsps = []
    for p in data.get("buffer_size_pairs", []):
        bsps.append(BufferSizePair(
            buffer=p.get("buffer", ""),
            size=p.get("size", ""),
            kind=p.get("kind", "param_pair"),
            relationship=p.get("relationship", "byte_count"),
        ))
    return AllocationSummary(
        function_name=data.get("function", func_name),
        allocations=allocations,
        parameters=parameters,
        buffer_size_pairs=bsps,
        description=data.get("description", ""),
    )


def _parse_free_summary(
    data: dict[str, Any], func_name: str,
) -> FreeSummary:
    frees = []
    for f in data.get("frees", []):
        frees.append(FreeOp(
            target=f.get("target", ""),
            target_kind=f.get("target_kind", "parameter"),
            deallocator=f.get("deallocator", "free"),
            conditional=f.get("conditional", False),
            nulled_after=f.get("nulled_after", False),
            condition=f.get("condition"),
        ))
    releases = []
    for r in data.get("resource_releases", []):
        releases.append(FreeOp(
            target=r.get("target", ""),
            target_kind=r.get("target_kind", "parameter"),
            deallocator=r.get("deallocator", ""),
            conditional=r.get("conditional", False),
            nulled_after=r.get("nulled_after", False),
            condition=r.get("condition"),
        ))
    return FreeSummary(
        function_name=data.get("function", func_name),
        frees=frees,
        resource_releases=releases,
        description=data.get("description", ""),
    )


def _parse_init_summary(
    data: dict[str, Any], func_name: str,
) -> InitSummary:
    inits = []
    for i in data.get("inits", []):
        inits.append(InitOp(
            target=i.get("target", ""),
            target_kind=i.get("target_kind", "parameter"),
            initializer=i.get("initializer", ""),
            byte_count=i.get("byte_count"),
            conditional=i.get("conditional", False),
            condition=i.get("condition"),
        ))
    return InitSummary(
        function_name=data.get("function", func_name),
        inits=inits,
        description=data.get("description", ""),
    )


def _parse_memsafe_summary(
    data: dict[str, Any], func_name: str,
) -> MemsafeSummary:
    contracts = []
    for c in data.get("contracts", []):
        contracts.append(MemsafeContract(
            target=c.get("target", ""),
            contract_kind=c.get("contract_kind", ""),
            description=c.get("description", ""),
            size_expr=c.get("size_expr"),
            relationship=c.get("relationship"),
            condition=c.get("condition"),
        ))
    return MemsafeSummary(
        function_name=data.get("function", func_name),
        contracts=contracts,
        description=data.get("description", ""),
    )


def _parse_verification_summary(
    data: dict[str, Any], func_name: str,
) -> VerificationSummary:
    contracts = None
    raw_contracts = data.get("simplified_contracts")
    if raw_contracts is not None:
        contracts = []
        for c in raw_contracts:
            contracts.append(MemsafeContract(
                target=c.get("target", ""),
                contract_kind=c.get("contract_kind", ""),
                description=c.get("description", ""),
                size_expr=c.get("size_expr"),
                relationship=c.get("relationship"),
                condition=c.get("condition"),
            ))
    issues = []
    for i in data.get("issues", []):
        issues.append(SafetyIssue(
            location=i.get("location", ""),
            issue_kind=i.get("issue_kind", ""),
            description=i.get("description", ""),
            severity=i.get("severity", "medium"),
            callee=i.get("callee"),
            contract_kind=i.get("contract_kind"),
        ))
    return VerificationSummary(
        function_name=data.get("function", func_name),
        simplified_contracts=contracts,
        issues=issues,
        description=data.get("description", ""),
    )
