"""Shared tool definitions and executor for LLM agents (triage, reflection).

Tools are defined once and exposed to different agents via allow-lists.
Read-only tools: read_function_source, get_callers, get_callees,
                 get_contracts
Write tools:     update_contracts, submit_verdict
"""

from __future__ import annotations

from collections import deque
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

# Contract-check-specific: enumerate public APIs, find APIs without contracts,
# submit the gap catalog. Used by `llm-summary contract-check`.
CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_public_apis",
        "description": (
            "Parse a public C header file and return the list of exported "
            "function names. Recognizes the libpng PNG_EXPORT/PNG_EXPORTA "
            "macro pattern; for headers without that pattern, falls back to "
            "naive function-declaration scanning. Use this in SEARCH phase "
            "to bound the universe of APIs to audit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "header_path": {
                    "type": "string",
                    "description": (
                        "Path to the public header, relative to the project "
                        "root (e.g. 'png.h')."
                    ),
                },
            },
            "required": ["header_path"],
        },
    },
    {
        "name": "list_apis_without_contracts",
        "description": (
            "Given a list of API names, return the subset that have no "
            "code-contract row in the database. These are 'missing_contract' "
            "gaps — the contract pipeline never produced a summary for them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "api_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Function names to check.",
                },
            },
            "required": ["api_names"],
        },
    },
    {
        "name": "submit_hazards",
        "description": (
            "Submit the candidate hazard list for this library. This is "
            "the terminal call of the HYPOTHESIS stage. A separate AUDIT "
            "agent will check each candidate against the docs. Aim for "
            "20-50 high-signal candidates — skip plainly safe APIs, "
            "focus on null inputs, ordering, lifecycle, noreturn, error "
            "returns, hidden state preconditions, and callback context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "1-2 sentence overview of what API surface was "
                        "scanned and the main themes of the hazards found."
                    ),
                },
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {
                                "type": "string",
                                "description": (
                                    "Public API function the hazard concerns."
                                ),
                            },
                            "hazard_kind": {
                                "type": "string",
                                "enum": [
                                    "ordering",
                                    "lifecycle",
                                    "null_input",
                                    "noreturn",
                                    "error_return",
                                    "state_dependent",
                                    "callback_context",
                                    "numeric_hazard",
                                    "other",
                                ],
                                "description": (
                                    "Kind of hazardous behavior."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "One-paragraph plain-English description "
                                    "of the hazard: what goes wrong, when, "
                                    "and why the docs should warn about it. "
                                    "The audit agent will use this verbatim "
                                    "to drive its doc search."
                                ),
                            },
                            "source_evidence": {
                                "type": "string",
                                "description": (
                                    "Where in the code/contracts the hazard "
                                    "is observable (e.g. "
                                    "'pngwio.c:42 / contract.requires"
                                    "[memsafe]: fp != NULL')."
                                ),
                            },
                            "contract_clause": {
                                "type": "string",
                                "description": (
                                    "Set iff our code-contract DB is ALSO "
                                    "missing this clause (FP source for the "
                                    "verifier). Directly-addable C predicate "
                                    "(e.g. 'fp != NULL'). Empty string "
                                    "otherwise."
                                ),
                            },
                            "contract_property": {
                                "type": "string",
                                "enum": [
                                    "", "memsafe", "memleak", "overflow",
                                ],
                                "description": (
                                    "Required iff contract_clause is set. "
                                    "Which safety property the clause belongs "
                                    "to. Empty string otherwise."
                                ),
                            },
                        },
                        "required": [
                            "function", "hazard_kind", "description",
                            "source_evidence",
                        ],
                    },
                    "description": "List of candidate hazards to audit.",
                },
            },
            "required": ["summary", "candidates"],
        },
    },
    {
        "name": "submit_audit_verdict",
        "description": (
            "Submit the AUDIT verdict for ONE hazard candidate. Terminal "
            "call of the audit stage. Set documented=true ONLY when the "
            "docs CLEARLY warn about this exact hazard. Vague or partial "
            "mentions count as undocumented (lean strict). Always record "
            "what you searched in doc_searched."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "documented": {
                    "type": "boolean",
                    "description": (
                        "True iff the docs clearly warn about this hazard. "
                        "Vague or partial mentions are NOT enough — set "
                        "false and quote the partial mention in doc_quote."
                    ),
                },
                "doc_searched": {
                    "type": "string",
                    "description": (
                        "What you searched in the docs and the result, "
                        "e.g. 'libpng-manual.txt §IV.3 (no warning); "
                        "png.h:1023 (decl only); example.c:55 (uses "
                        "without comment)'."
                    ),
                },
                "doc_quote": {
                    "type": "string",
                    "description": (
                        "Verbatim quote from the docs if a vague/partial "
                        "mention exists. Empty string if docs are silent "
                        "or if the hazard is fully documented (in the "
                        "documented=true case, quote the warning)."
                    ),
                },
                "recommendation": {
                    "type": "string",
                    "description": (
                        "One-sentence change to propose to the upstream "
                        "maintainer. Required when documented=false; "
                        "empty string when documented=true."
                    ),
                },
            },
            "required": ["documented", "doc_searched"],
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

# Enhanced triage tools: reachability, doc audit, entry-level verdict
ENHANCED_TRIAGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_entry_functions",
        "description": (
            "Find entry functions — functions that have no callers in the "
            "call graph and are directly callable by external code. These "
            "define the library's interface boundary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "restrict_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional: limit the search to these function names. "
                        "When omitted, searches all functions in the database."
                    ),
                },
            },
        },
    },
    {
        "name": "get_reachability_path",
        "description": (
            "Find the shortest call chain from one function to another "
            "using BFS over the call graph. Returns the path as a list "
            "of function names if reachable, or reachable=false if no "
            "path exists. Considers both direct and indirect call edges."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_function": {
                    "type": "string",
                    "description": "Source function (e.g. entry function).",
                },
                "to_function": {
                    "type": "string",
                    "description": "Target function (e.g. function with the bug).",
                },
            },
            "required": ["from_function", "to_function"],
        },
    },
    {
        "name": "get_call_chain_contracts",
        "description": (
            "Batch-read contracts for a list of functions (e.g. a call "
            "chain). Returns requires/ensures/modifies for each function "
            "in order. More efficient than calling get_contracts N times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of function names.",
                },
            },
            "required": ["function_names"],
        },
    },
    {
        "name": "submit_reachability",
        "description": (
            "Submit the reachability and path-feasibility verdict. "
            "Terminal call for the reachability stage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reachable": {
                    "type": "boolean",
                    "description": (
                        "True iff the issue is both structurally reachable "
                        "from an entry function AND the triggering condition "
                        "is satisfiable given path constraints."
                    ),
                },
                "entry_function": {
                    "type": "string",
                    "description": (
                        "The entry function through which the issue is "
                        "reachable. Empty if unreachable."
                    ),
                },
                "call_chain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Call chain from entry to bug site. E.g. "
                        "['png_read_row', 'png_read_filter_row', "
                        "'png_read_finish_row']."
                    ),
                },
                "path_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Guards and sanitizers found along the path. E.g. "
                        "'png_read_row:42 checks png_ptr != NULL before call', "
                        "'validate_header:88 clamps width to MAX_WIDTH'."
                    ),
                },
                "data_flow_trace": {
                    "type": "string",
                    "description": (
                        "How the triggering parameter flows from entry input "
                        "through the call chain to the bug site. E.g. "
                        "'width comes from PNG header (user-controlled) -> "
                        "png_read_info() stores in png_ptr->width -> "
                        "png_read_row() passes to filter_row(width)'."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed explanation of why the issue is or is "
                        "not reachable/triggerable."
                    ),
                },
                "summarizer_gap": {
                    "type": "boolean",
                    "description": (
                        "True if the verifier/summarizer missed important "
                        "details (e.g. a callee ensures already guarantees "
                        "the property). False otherwise."
                    ),
                },
            },
            "required": [
                "reachable", "entry_function", "call_chain",
                "reasoning",
            ],
        },
    },
    {
        "name": "submit_doc_audit",
        "description": (
            "Submit the documentation audit verdict for a reachable issue. "
            "Terminal call for the doc-audit stage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mitigated": {
                    "type": "boolean",
                    "description": (
                        "True iff project documentation or inline comments "
                        "CLEARLY state a constraint that prevents the bug. "
                        "Vague mentions do NOT count — lean strict."
                    ),
                },
                "doc_searched": {
                    "type": "string",
                    "description": (
                        "Audit trail: what docs/files/comments you searched "
                        "and what you found. E.g. 'README.md (no mention); "
                        "png.h:1023 (decl only); pngread.c:42 (comment says "
                        "width must be validated by caller)'."
                    ),
                },
                "doc_quote": {
                    "type": "string",
                    "description": (
                        "Verbatim quote from docs/comments if a mention "
                        "exists. Empty if docs are silent."
                    ),
                },
                "mitigating_constraint": {
                    "type": "string",
                    "description": (
                        "The specific constraint that mitigates the issue. "
                        "E.g. 'callers must validate width <= PNG_MAX_WIDTH "
                        "before calling png_read_row'. Empty if not mitigated."
                    ),
                },
            },
            "required": ["mitigated", "doc_searched"],
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

        raw_contract = inp.get("contract")
        if raw_contract is None:
            # LLM may have flattened the contract fields into inp
            raw_contract = {
                k: v for k, v in inp.items() if k != "function_name"
            }
        contract_data = dict(raw_contract)
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

    # -- list_public_apis (contract-check) --

    def _tool_list_public_apis(self, inp: dict[str, Any]) -> dict[str, Any]:
        if self._project_path is None:
            return {"error": "list_public_apis requires a project path"}
        rel = inp["header_path"]
        header = (self._project_path / rel).resolve()
        try:
            header.relative_to(self._project_path)
        except ValueError:
            return {"error": f"header path escapes project root: {rel}"}
        if not header.is_file():
            return {"error": f"header not found: {rel}"}

        try:
            text = header.read_text(errors="replace")
        except OSError as e:
            return {"error": f"failed to read header: {e}"}

        from .contract_check import parse_public_apis
        names = parse_public_apis(text)
        return {
            "header_path": rel,
            "api_count": len(names),
            "apis": names,
        }

    # -- list_apis_without_contracts (contract-check) --

    def _tool_list_apis_without_contracts(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        names = inp["api_names"]
        if not isinstance(names, list):
            return {"error": "api_names must be a list of strings"}

        missing: list[str] = []
        for name in names:
            if not isinstance(name, str):
                continue
            funcs = self.db.get_function_by_name(name)
            if not funcs:
                missing.append(name)
                continue
            has_contract = False
            for func in funcs:
                if func.id is None:
                    continue
                if self.db.get_code_contract_summary(func.id) is not None:
                    has_contract = True
                    break
            if not has_contract:
                missing.append(name)

        return {
            "checked": len(names),
            "missing_count": len(missing),
            "missing": missing,
        }

    # -- submit_hazards (contract-check, hypothesis stage) --

    def _tool_submit_hazards(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- submit_audit_verdict (contract-check, audit stage) --

    def _tool_submit_audit_verdict(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- submit_reflection --

    def _tool_submit_reflection(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- transition_phase (triage) --

    def _tool_transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"next_phase": inp["next_phase"]}

    # -- get_entry_functions (enhanced triage) --

    def _tool_get_entry_functions(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        from .code_contract.checker import find_entry_functions

        restrict_to = inp.get("restrict_to")
        entries = find_entry_functions(self.db, restrict_to=restrict_to)
        return {"entries": entries, "count": len(entries)}

    # -- get_reachability_path (enhanced triage) --

    def _build_call_graph(self) -> dict[int, list[int]]:
        """Build and cache the forward call graph."""
        if not hasattr(self, "_call_graph_cache"):
            graph: dict[int, list[int]] = {}
            for edge in self.db.get_all_call_edges():
                graph.setdefault(edge.caller_id, []).append(edge.callee_id)
            self._call_graph_cache = graph
        return self._call_graph_cache

    def _tool_get_reachability_path(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        from_name = inp["from_function"]
        to_name = inp["to_function"]

        from_func = self._get_func(from_name)
        to_func = self._get_func(to_name)
        if from_func is None:
            return {"error": f"Function '{from_name}' not found in database."}
        if to_func is None:
            return {"error": f"Function '{to_name}' not found in database."}
        if from_func.id is None or to_func.id is None:
            return {"error": "Function has no ID."}

        graph = self._build_call_graph()
        src, dst = from_func.id, to_func.id

        # BFS for shortest path
        visited: set[int] = {src}
        queue: deque[list[int]] = deque([[src]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == dst:
                names = []
                for fid in path:
                    f = self.db.get_function(fid)
                    names.append(f.name if f else f"<id:{fid}>")
                return {"reachable": True, "path": names}
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append([*path, neighbor])

        return {"reachable": False, "path": []}

    # -- get_call_chain_contracts (enhanced triage) --

    def _tool_get_call_chain_contracts(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        names = inp["function_names"]
        if not isinstance(names, list):
            return {"error": "function_names must be a list of strings"}

        contracts: list[dict[str, Any]] = []
        for name in names:
            result = self._tool_get_contracts({"function_name": name})
            contracts.append(result)
        return {"contracts": contracts}

    # -- submit_reachability (enhanced triage) --

    def _tool_submit_reachability(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        return {"accepted": True, **inp}

    # -- submit_doc_audit (enhanced triage) --

    def _tool_submit_doc_audit(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        return {"accepted": True, **inp}
