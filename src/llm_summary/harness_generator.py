"""LLM-based test harness generator for contract-guided symbolic execution.

Bitcode-based approach: generates a thin C shim (test entry + __dfsw_ callee stubs)
that links against instrumented project bitcode. The shim is compiled with ko-clang
for auto-symbolization; project bitcode goes through UCSanPass + TaintPass via opt-14.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .compile_commands import CompileCommandsDB
from .db import SummaryDB
from .git_tools import GIT_TOOL_DEFINITIONS, GitTools
from .llm.base import LLMBackend

# ---- Fill-in template prompts ----

FIX_PROMPT = """\
The following function body failed to compile. Fix the errors.

## Function:
```c
{func_signature} {{
{func_body}
}}
```

## Compiler errors:
```
{errors}
```

## Available API:
- assert_cond(result, id), assert_allocated(ptr, size, id), \
assert_init(ptr, size, id), assert_freed(ptr, id)
- assume_cond(result, id), assume_allocated(ptr, size, id) -> void *, \
assume_init(ptr, size, id), assume_freed(ptr, id) -> void *

## Rules
- Output a single ```fix fenced block with ONLY the fixed function body.
- Do NOT include the function signature or braces — just the body lines.
- Do NOT add #include, typedef, extern, or new functions.
- Use `void *` for any opaque/struct pointer types that cause errors.
- Output ONLY the fenced block, no explanation.
"""


PLAN_PROMPT = """\
You are a concolic execution planner. Given source code annotated with BB IDs \
and branch successors, produce a trace plan that tells the concolic executor \
which edges (transitions between BBs) to target.

## How concolic execution works

The executor starts with an empty input and runs symbolically. \
At each conditional branch, the SMT solver can generate a new input ("seed") \
that flips the branch. The scheduler picks which seed to run next.

Each conditional branch is annotated as:
  `/* [BB:X cond T:Y F:Z] */`
where X is the branch's BB ID, Y is the BB entered when the condition is true, \
and Z is the BB entered when false.

IMPORTANT: Source-level true/false may be INVERTED at the IR level. \
Always use the T: and F: annotations to determine which BB is reached \
for each direction — do NOT assume source-level if/else maps directly.

A branch marked `loop` is a loop back-edge.

## Context

{context}

## Annotated Source Code

{annotated_sources}

## Your Task

{goal}

## Execution Options

The concolic executor has configurable options you can tune:

- `loop_threshold`: max loop iterations before termination (default: 3). \
Increase if the function needs deeper loop exploration to reach the target.
- `solve_ub`: enable UB solving in the constraint solver (default: true). \
Disable if you only care about reachability, not UB detection.
- `checker_nullderef`: null pointer dereference checker (default: on).
- `checker_ubi`: uninitialized memory use checker (default: on).
- `trace_bounds`: bounds tracking with OOB/UAF exit on violation (default: on).
- `no_upcast`: disallow negative-offset container_of casts (default: on). \
Turn off if the code legitimately uses container_of patterns.
- `no_enlarge`: disallow object size growth beyond initial allocation (default: on). \
Turn off if the code uses realloc or flexible array members.

Output a JSON trace plan:

```json
{{
  "traces": [
    {{
      "goal": "what this path tests or what counter-example it seeks",
      "description": "which branches to take and why",
      "target_edges": [{{"from": 100000, "to": 100001}}],
      "priority": 1
    }}
  ],
  "deprioritize": [
    {{
      "bb_id": 100004,
      "reason": "why this branch is not worth exploring"
    }}
  ],
  "ucsan_config": {{
    "loop_threshold": 3,
    "solve_ub": true,
    "checker_nullderef": true,
    "checker_ubi": true,
    "trace_bounds": true,
    "no_upcast": true,
    "no_enlarge": true
  }}
}}
```

Guidelines:
- `target_edges`: Each edge is `{{"from": X, "to": Y}}` meaning we want \
execution to go from BB X to BB Y. Use the T:/F: annotations to pick the \
right successor. A trace may have multiple edges if the path requires \
multiple branches.
- `priority`: 1 = must explore, 2 = nice to have, 3 = low priority.
- `deprioritize`: branches leading to irrelevant paths (error returns with \
no memory ops, deep loop iterations, arithmetic unrelated to the goal).
- Loops: one iteration usually suffices. Mark loop back-edges as deprioritize \
unless the access pattern changes across iterations.
- `ucsan_config`: only include fields you want to change from defaults.

Output ONLY the JSON block, no other text.
"""


TRIAGE_VALIDATE_PROMPT = """\
Fill in the `/* FILL */` sections in the C template below.
The template is a shim for ucsan concolic execution that validates a triage
verdict about `{name}`.

## Triage Verdict

- Hypothesis: {hypothesis}
- Issue: [{severity}] {issue_kind} — {issue_description}
- Reasoning: {reasoning}

{assumptions_section}
{assertions_section}

## C Template

```c
{template}
```

## Instructions

Output a single ```c fenced block with the complete filled-in C code.
Copy the template exactly, replacing each `/* FILL: ... */` with C code.

Rules:
- Do NOT add new #include, typedef, or extern declarations
- Do NOT rename or change function signatures
- Do NOT add functions not in the template
- Every `/* FILL */` must be replaced with valid C code or left empty
- Keep the code minimal — only add what the contracts require
- Do NOT add comments — the code should be self-explanatory
- In stubs: use assert_* for pre-conditions, assume_* for post-conditions
- Contract-to-assertion mapping:
  - disallow_null → assert_cond(ptr != NULL, id)
  - allow_null → no assertion needed (NULL is allowed)
  - not_freed → assert_allocated(ptr, 0, id)  (still allocated)
  - buffer_size(N) → assert_allocated(ptr, N, id)
  - initialized(N) → assert_init(ptr, N, id)
  - freed → assert_freed(ptr, id)
- In test(): use assert_* to verify post-conditions after the call
- Post-condition annotations use brackets for qualifiers:
  - `[may_be_null]` means the pointer may be NULL — do NOT assert non-NULL
  - `[when COND]` means the effect is conditional — wrap the assertion in
    `if (COND) {{ ... }}` so it only checks on the relevant path
  - If a post-condition has no brackets, it is unconditional
- Only check contracts on direct parameters — skip deep field access
  through forward-declared (opaque) struct pointers (e.g. do NOT access
  s->l_desc.stat_desc->extra_bits if stat_desc's struct type is opaque)
- Use sizeof(*ptr) for allocation sizes, not hardcoded constants
"""

FILL_PROMPT = """\
Fill in the `/* FILL */` sections in the C template below.
The template is a shim for ucsan concolic execution testing `{name}`.

## Target Function

Signature: `{signature}`
Parameters: {params_json}

## Contracts

{contracts_section}

## Callee Contracts

{callee_section}

## Post-conditions

{postconds_section}

## C Template

```c
{template}
```

## Instructions

Output a single ```c fenced block with the complete filled-in C code.
Copy the template exactly, replacing each `/* FILL: ... */` with C code.

Rules:
- Do NOT add new #include, typedef, or extern declarations
- Do NOT rename or change function signatures
- Do NOT add functions not in the template
- Every `/* FILL */` must be replaced with valid C code or left empty
- Keep the code minimal — only add what the contracts require
- Do NOT add comments — the code should be self-explanatory
- In stubs: use assert_* for pre-conditions, assume_* for post-conditions
- Contract-to-assertion mapping:
  - disallow_null → assert_cond(ptr != NULL, id)
  - allow_null → no assertion needed (NULL is allowed)
  - not_freed → assert_allocated(ptr, 0, id)  (still allocated)
  - buffer_size(N) → assert_allocated(ptr, N, id)
  - initialized(N) → assert_init(ptr, N, id)
  - freed → assert_freed(ptr, id)
- In test(): use assert_* to verify post-conditions after the call
- Post-condition annotations use brackets for qualifiers:
  - `[may_be_null]` means the pointer may be NULL — do NOT assert non-NULL
  - `[when COND]` means the effect is conditional — wrap the assertion in
    `if (COND) {{ ... }}` so it only checks on the relevant path
  - If a post-condition has no brackets, it is unconditional
- Only check contracts on direct parameters — skip deep field access
  through forward-declared (opaque) struct pointers (e.g. do NOT access
  s->l_desc.stat_desc->extra_bits if stat_desc's struct type is opaque)
- Use sizeof(*ptr) for allocation sizes, not hardcoded constants
"""

SCHEDULE_PROMPT = """\
Generate a scheduling policy for the following shim.

## Shim code:
```c
{shim_code}
```

## Contracts

{contracts_section}

## Callee Contracts

{callee_section}

## Execution Options

The concolic executor has configurable options you can tune:

- `loop_threshold`: max loop iterations before termination (default: 3). \
Increase if the function needs deeper loop exploration to reach the target.
- `solve_ub`: enable UB solving in the constraint solver (default: true). \
Disable if you only care about reachability, not UB detection.
- `checker_nullderef`: null pointer dereference checker (default: on).
- `checker_ubi`: uninitialized memory use checker (default: on).
- `trace_bounds`: bounds tracking with OOB/UAF exit on violation (default: on).
- `no_upcast`: disallow negative-offset container_of casts (default: on). \
Turn off if the code legitimately uses container_of patterns.
- `no_enlarge`: disallow object size growth beyond initial allocation (default: on). \
Turn off if the code uses realloc or flexible array members.

Output a single ```json fenced block:

```json
{{{{
  "function": "{name}",
  "targets": [
    {{{{
      "type": "assume|boundary_access|callee_contract",
      "description": "what this checks",
      "contract_kind": "disallow_null|buffer_size|...",
      "target": "parameter name",
      "priority": "high|medium|low"
    }}}}
  ],
  "ucsan_config": {{{{
    "loop_threshold": 3,
    "solve_ub": true,
    "checker_nullderef": true,
    "checker_ubi": true,
    "trace_bounds": true,
    "no_upcast": true,
    "no_enlarge": true
  }}}}
}}}}
```

Only include `ucsan_config` fields you want to change from defaults.
"""


# ---------------------------------------------------------------------------
# Agent-based fix loop: tool definitions, executor, system prompt
# ---------------------------------------------------------------------------

_GIT_TOOL_NAMES = {"git_show", "git_ls_tree", "git_grep"}

FIX_SYSTEM_PROMPT = """\
You are fixing a C shim that failed to compile with ko-clang.

The shim was generated from a fill-in template for ucsan concolic execution.
Keep the template structure intact — do not reorganise the code.

## Git tools

You have read-only access to the project source via git tools. Use them to \
*understand* types, signatures, and macros — do NOT copy project code into \
the shim. The shim is a thin test stub, not a copy of the real implementation.

## Typical errors and fixes

- **Unknown type**: `git_grep` for the typedef/struct definition. Then either \
use `void *` (for opaque pointers) or add a minimal forward declaration \
(`struct foo;` or `typedef struct foo foo;`) at the top of the shim.
- **Wrong parameter name**: `git_show` the file containing the real function \
to see the actual signature and parameter names.
- **Undeclared identifier**: grep for it — it may be an enum value, macro, or \
global. Use a literal value or forward-declare as needed.
- **Incompatible types**: cast to `void *` for opaque pointer types.

## Rules

- Do NOT add `#include` directives.
- Do NOT copy function implementations from the project.
- Minimise changes — fix only what the compiler error requires.
- After patching, always call `compile_shim` to verify.
- When compilation succeeds, call `submit_harness` to finish.
"""

HARNESS_FIX_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "compile_shim",
        "description": (
            "Compile the current shim code with ko-clang. "
            "Returns {success: bool, errors: string}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "replace_function_body",
        "description": (
            "Replace the body of a named function in the shim. "
            "Provide the function name (e.g. '__shim_png_malloc') and "
            "the new body lines (without the surrounding braces)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to patch.",
                },
                "new_body": {
                    "type": "string",
                    "description": (
                        "Replacement body lines (no surrounding { })."
                    ),
                },
            },
            "required": ["function_name", "new_body"],
        },
    },
    {
        "name": "replace_full_code",
        "description": (
            "Replace the entire shim C code. Use only when patching "
            "individual functions is insufficient (e.g. adding a "
            "forward declaration at the top)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete replacement C code.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit_harness",
        "description": (
            "Accept the current shim code and finish the fix loop. "
            "Call this after compile_shim returns success."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
] + GIT_TOOL_DEFINITIONS

MAX_FIX_TURNS = 50
MAX_COMPILE_ATTEMPTS = 5


class HarnessFixExecutor:
    """Executes fix-loop tool calls against the shim code."""

    def __init__(
        self,
        c_code: str,
        ucsan_config: str,
        file_path: str | None,
        generator: "HarnessGenerator",
        git_tools: GitTools | None,
    ) -> None:
        self.c_code = c_code
        self.ucsan_config = ucsan_config
        self.file_path = file_path
        self.gen = generator
        self.git = git_tools
        self.submitted = False
        self.compile_attempts = 0

    def execute(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name in _GIT_TOOL_NAMES:
            if self.git is None:
                return {"error": f"Tool '{tool_name}' unavailable: no project path"}
            return self.git.dispatch(tool_name, tool_input)
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        result: dict[str, Any] = handler(tool_input)
        return result

    def _tool_compile_shim(self, _inp: dict[str, Any]) -> dict[str, Any]:
        self.compile_attempts += 1
        if self.compile_attempts > MAX_COMPILE_ATTEMPTS:
            return {
                "success": False,
                "errors": f"Compile attempt limit ({MAX_COMPILE_ATTEMPTS}) "
                "reached. Call submit_harness to accept current code or "
                "give up.",
            }
        ok, errors = self.gen._compile_shim(
            self.c_code, self.ucsan_config, file_path=self.file_path,
        )
        if ok:
            return {"success": True, "errors": ""}
        return {"success": False, "errors": errors}

    def _tool_replace_function_body(
        self, inp: dict[str, Any],
    ) -> dict[str, Any]:
        func_name = inp["function_name"]
        new_body = inp["new_body"]

        # Find the function in self.c_code by scanning for its name
        lines = self.c_code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match "... func_name(..." at a function definition
            if func_name not in stripped:
                continue
            if not ("(" in stripped and
                    not stripped.startswith("extern ")
                    and not stripped.startswith("/*")
                    and not stripped.startswith("*")):
                continue
            # Verify it's a definition (has or is followed by {)
            if "{" in line:
                brace_line = i
            elif i + 1 < len(lines) and lines[i + 1].strip() == "{":
                brace_line = i + 1
            else:
                continue
            # Find matching }
            depth = 0
            for j in range(brace_line, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth == 0:
                    # Replace body: 1-indexed for _apply_fix
                    self.c_code = self.gen._apply_fix(
                        self.c_code, brace_line + 1, j + 1, new_body,
                    )
                    return {"ok": True, "function": func_name}
            return {"error": f"Unbalanced braces in '{func_name}'"}
        return {"error": f"Function '{func_name}' not found in shim code"}

    def _tool_replace_full_code(self, inp: dict[str, Any]) -> dict[str, Any]:
        self.c_code = inp["code"]
        return {"ok": True}

    def _tool_submit_harness(self, _inp: dict[str, Any]) -> dict[str, Any]:
        self.submitted = True
        return {"accepted": True}


class HarnessGenerator:
    """Generates test harnesses for contract-guided symbolic execution.

    Bitcode-based: generates a thin C shim + ucsan build pipeline config.
    """

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        ko_clang_path: str | None = None,
        max_fix_attempts: int = 3,
        symsan_dir: str | None = None,
        compile_commands: CompileCommandsDB | None = None,
        project_path: str | None = None,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.ko_clang_path = ko_clang_path
        self.max_fix_attempts = max_fix_attempts
        self.compile_commands = compile_commands
        self.project_path: Path | None = Path(project_path) if project_path else None
        # symsan install dir (parent of bin/ko-clang)
        self.symsan_dir: Path | None
        if symsan_dir:
            self.symsan_dir = Path(symsan_dir)
        elif ko_clang_path:
            self.symsan_dir = Path(ko_clang_path).resolve().parent.parent
        else:
            self.symsan_dir = None
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "errors": 0,
            "fix_attempts": 0,
        }
        self._triage_context: dict[str, Any] | None = None
        self._ucsan_abilist = self._load_ucsan_abilist()
        self._check_toolchain()

    def _load_ucsan_abilist(self) -> set[str]:
        """Load function names from ucsan_abilist.txt.

        These functions have custom ucsan handlers and must NOT be shimmed.
        """
        if self.symsan_dir is None:
            return set()
        abilist = self.symsan_dir / "lib" / "symsan" / "ucsan_abilist.txt"
        if not abilist.exists():
            return set()
        names: set[str] = set()
        for line in abilist.read_text().splitlines():
            line = line.strip()
            if line.startswith("fun:") and "=" in line:
                name = line[4:line.index("=")]
                names.add(name)
        return names

    def _check_toolchain(self) -> None:
        """Validate that required toolchain binaries exist."""
        import shutil

        missing: list[str] = []

        # LLVM-14 tools (must be on PATH)
        for tool in ("clang-14", "opt-14", "llc-14"):
            if not shutil.which(tool):
                missing.append(f"{tool} (not found on PATH)")

        # ko-clang
        if self.ko_clang_path:
            if not Path(self.ko_clang_path).exists():
                missing.append(f"ko-clang: {self.ko_clang_path}")

        if missing:
            msg = "Toolchain check failed:\n" + "\n".join(f"  - {m}" for m in missing)
            raise FileNotFoundError(msg)

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def generate(
        self,
        func_name: str,
        output_dir: str | None = None,
        bc_file: str | None = None,
    ) -> tuple[str, dict] | None:
        """Generate shim + build config for a single function.

        Args:
            func_name: Target function name.
            output_dir: Directory to write output files.
            bc_file: Path to project bitcode containing the target function.

        Returns (shim_c_code, policy_dict) or None on error.
        """
        # Look up function
        funcs = self.db.get_function_by_name(func_name)
        if not funcs:
            if self.verbose:
                print(f"  Function not found: {func_name}")
            return None
        func = funcs[0]
        assert func.id is not None

        # Get memsafe contracts
        row = self.db.conn.execute(
            "SELECT summary_json FROM memsafe_summaries WHERE function_id = ?",
            (func.id,),
        ).fetchone()
        if not row:
            if self.verbose:
                print(f"  No memsafe summary for: {func_name}")
            return None

        memsafe_data = json.loads(row[0])
        contracts = memsafe_data.get("contracts", [])

        # Auto-compile bitcode if not provided
        if not bc_file and self.compile_commands and func.file_path:
            out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
            out.mkdir(parents=True, exist_ok=True)
            bc_file = self._compile_to_bc(func.file_path, out)
            if bc_file and self.verbose:
                print(f"    Compiled bitcode: {bc_file}")

        # Get callees and their contracts
        callee_contracts = self._gather_callee_contracts(func.id)

        # Gather post-conditions
        postconds = self._gather_postconditions(func.id)

        # Format contracts for prompt context
        contracts_section = self._format_contracts(contracts)
        callee_section = self._format_callee_contracts(callee_contracts)
        postconds_section = self._format_postconditions(postconds)

        # Determine which callees need shim stubs
        # Never shim functions in ucsan's abilist (they have custom handlers)
        if self._triage_context is not None:
            real_fns = set(self._triage_context.get("real_functions", []))
            shim_callees = [
                k for k in callee_contracts
                if k not in real_fns and k not in self._ucsan_abilist
            ]
        else:
            shim_callees = [
                k for k in callee_contracts
                if k not in self._ucsan_abilist
            ]

        # Build preceding function info for sequential test cases
        preceding: list[dict[str, Any]] | None = None
        if self._triage_context:
            test_seq = self._triage_context.get("test_sequence", [])
            if len(test_seq) > 1:
                preceding = []
                for pf_name in test_seq[:-1]:
                    pf_funcs = self.db.get_function_by_name(pf_name)
                    if not pf_funcs:
                        continue
                    pf = pf_funcs[0]
                    assert pf.id is not None
                    pf_postconds = self._gather_postconditions(pf.id)
                    # Merge preceding callees into callee_contracts
                    pf_callees = self._gather_callee_contracts(pf.id)
                    for ck, cv in pf_callees.items():
                        if ck not in callee_contracts:
                            callee_contracts[ck] = cv
                    preceding.append({
                        "name": pf.name,
                        "signature": pf.signature or "",
                        "params": pf.params or [],
                        "postconds": pf_postconds,
                    })

        # Build fill-in template (used for both triage and normal paths)
        template = self._build_fill_template(
            func.name, func.signature or "", func.params or [],
            callee_contracts, postconds, self._triage_context,
            contracts=contracts, file_path=func.file_path,
            preceding_functions=preceding,
        )

        # Build prompt
        if self._triage_context is not None:
            ctx = self._triage_context
            assumptions = ctx.get("assumptions", [])
            assertions = ctx.get("assertions", [])

            assumptions_text = ""
            if assumptions:
                assumptions_text = "Assumptions (contextual — for understanding, not code):\n"
                assumptions_text += "\n".join(
                    f"  {i}. {a}" for i, a in enumerate(assumptions, 1)
                )

            assertions_text = ""
            if assertions:
                assertions_text = "Expected checks (ucsan verifies automatically):\n"
                assertions_text += "\n".join(
                    f"  {i}. {a}" for i, a in enumerate(assertions, 1)
                )

            prompt = TRIAGE_VALIDATE_PROMPT.format(
                name=func.name,
                hypothesis=ctx.get("hypothesis", "unknown"),
                severity=ctx.get("severity", ""),
                issue_kind=ctx.get("issue_kind", ""),
                issue_description=ctx.get("issue_description", ""),
                reasoning=ctx.get("reasoning", ""),
                assumptions_section=assumptions_text,
                assertions_section=assertions_text,
                template=template,
            )
        else:
            prompt = FILL_PROMPT.format(
                name=func.name,
                signature=func.signature,
                params_json=json.dumps(func.params),
                contracts_section=contracts_section,
                callee_section=callee_section,
                postconds_section=postconds_section,
                template=template,
            )

        try:
            if self.verbose:
                label = (f"  Generating shim for: {func_name} "
                         f"(triage validate: {self._triage_context.get('hypothesis')})"
                         if self._triage_context else
                         f"  Generating shim for: {func_name}")
                print(label)

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(func_name, prompt, response)

            # Parse response: extract single ```c block from fill-in template
            c_code = self._extract_c_block(response)
            if not c_code:
                if self.verbose:
                    print("    Failed to extract C code from response")
                return None

            # Resolve validation scope (all relevant functions + source files)
            scope_fns: list[str] | None = None
            source_files: list[str] | None = None
            if self._triage_context is not None:
                real_fns = self._triage_context.get("real_functions", [])
                if real_fns:
                    scope_fns = list(real_fns)
                    if func_name not in scope_fns:
                        scope_fns.insert(0, func_name)
                    seen: set[str] = set()
                    source_files = []
                    for rn in real_fns:
                        rfuncs = self.db.get_function_by_name(rn)
                        if rfuncs and rfuncs[0].file_path:
                            fp = rfuncs[0].file_path
                            if fp not in seen:
                                seen.add(fp)
                                source_files.append(fp)

            # Add shim callees defined in compiled source files to scope
            # so ko-clang emits their .taint symbols (needed for function
            # pointer tables in .data.rel.ro).
            if scope_fns is not None and source_files and shim_callees:
                src_set = set(source_files)
                for sc in shim_callees:
                    sc_funcs = self.db.get_function_by_name(sc)
                    if sc_funcs and sc_funcs[0].file_path in src_set:
                        if sc not in scope_fns:
                            scope_fns.append(sc)

            # Compile-and-fix loop (ko-clang handles instrumentation)
            if self.ko_clang_path:
                ucsan_config = self._build_ucsan_config(
                    func_name, shim_callees,
                    scope_functions=scope_fns,
                )

                # First compile attempt
                compile_ok = False
                ok, errors = self._compile_shim(
                    c_code, ucsan_config, file_path=func.file_path,
                )
                if ok:
                    compile_ok = True
                    if self.verbose:
                        print("    Shim compiled successfully")
                elif self._can_use_fix_agent():
                    # Agent-based fix loop (with git tools)
                    if self.verbose:
                        print("    Compile failed, starting fix agent...")
                    fixed = self._fix_with_agent(
                        c_code, ucsan_config, func.file_path, errors,
                    )
                    if fixed:
                        c_code = fixed
                        compile_ok = True
                        if self.verbose:
                            print("    Shim compiled successfully (fix agent)")
                    elif self.verbose:
                        print("    Fix agent failed")
                else:
                    # Fallback: linear fix loop
                    for attempt in range(self.max_fix_attempts):
                        if attempt > 0:
                            ok, errors = self._compile_shim(
                                c_code, ucsan_config,
                                file_path=func.file_path,
                            )
                            if ok:
                                compile_ok = True
                                if self.verbose:
                                    print("    Shim compiled successfully"
                                          f" (after {attempt} fix(es))")
                                break

                        self._stats["fix_attempts"] += 1
                        if self.verbose:
                            print(f"    Compile failed (attempt {attempt + 1}/"
                                  f"{self.max_fix_attempts}), asking LLM to fix...")

                        fail = self._find_failing_function(c_code, errors)
                        if fail:
                            sig, body, bstart, bend = fail
                            fix_prompt = FIX_PROMPT.format(
                                func_signature=sig,
                                func_body=body,
                                errors=errors,
                            )
                        else:
                            fix_prompt = (
                                "The following C shim failed to compile. "
                                "Fix the errors.\n\n```c\n"
                                + c_code + "\n```\n\n"
                                "Compiler errors:\n```\n"
                                + errors + "\n```\n\n"
                                "Output a single ```c fenced block with "
                                "the complete fixed C code.\n"
                            )

                        fix_full = prompt + "\n\n---\n\n" + fix_prompt
                        fix_response = self.llm.complete(fix_full)
                        self._stats["llm_calls"] += 1

                        if self.log_file:
                            self._log_interaction(
                                f"{func_name}_fix{attempt + 1}",
                                f"[COMPILE ERRORS]\n{errors}",
                                fix_response,
                            )

                        if fail:
                            fix_body = self._extract_fix_block(fix_response)
                            if fix_body:
                                c_code = self._apply_fix(
                                    c_code, bstart, bend, fix_body,
                                )
                            else:
                                fixed = self._extract_c_block(fix_response)
                                if fixed:
                                    c_code = fixed
                        else:
                            fixed = self._extract_c_block(fix_response)
                            if fixed:
                                c_code = fixed
                    else:
                        if self.verbose:
                            print(f"    Failed to fix after "
                                  f"{self.max_fix_attempts} attempts")

                if not compile_ok:
                    self._stats["errors"] += 1
                    return None

            # Generate scheduling policy (separate LLM call)
            schedule_response = self.llm.complete(SCHEDULE_PROMPT.format(
                name=func_name,
                shim_code=c_code,
                contracts_section=contracts_section,
                callee_section=callee_section,
            ))
            self._stats["llm_calls"] += 1
            policy = self._extract_json_block(schedule_response)

            # Extract runtime config from LLM output
            runtime_config = self._build_runtime_config(
                policy.pop("ucsan_config", {}),
            )

            self._stats["functions_processed"] += 1

            # Write output files
            if output_dir:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)

                shim_path = out / f"shim_{func_name}.c"
                shim_path.write_text(c_code)

                policy_path = out / f"policy_{func_name}.json"
                policy_path.write_text(json.dumps(policy, indent=2))

                config_path = out / f"config_{func_name}.yaml"
                ucsan_config = self._build_ucsan_config(
                    func_name, shim_callees,
                    scope_functions=scope_fns,
                )
                config_path.write_text(ucsan_config)

                if runtime_config:
                    rt_path = out / f"runtime_{func_name}.json"
                    rt_path.write_text(json.dumps(runtime_config, indent=2))

                build_script = self._build_script(
                    func_name, out, bc_file, file_path=func.file_path,
                    source_files=source_files,
                    scope_functions=scope_fns,
                )
                script_path = out / f"build_{func_name}.sh"
                script_path.write_text(build_script)
                script_path.chmod(0o755)

                if self.verbose:
                    print(f"    Wrote: {shim_path}")
                    print(f"    Wrote: {policy_path}")
                    print(f"    Wrote: {config_path}")
                    if runtime_config:
                        print(f"    Wrote: {rt_path}")
                    print(f"    Wrote: {script_path}")

            return c_code, policy

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error generating shim for {func_name}: {e}")
                import traceback
                traceback.print_exc()
            return None

    def validate_triage(
        self,
        func_name: str,
        triage_context: dict[str, Any],
        output_dir: str | None = None,
        bc_file: str | None = None,
    ) -> tuple[str, dict] | None:
        """Generate a harness to symbolically validate a triage verdict.

        Sets triage context (hypothesis, assumptions, assertions, real_functions)
        on the generator so generate() appends TRIAGE_VALIDATE_PROMPT to the
        shim prompt.
        """
        self._triage_context = triage_context
        try:
            return self.generate(func_name, output_dir=output_dir, bc_file=bc_file)
        finally:
            self._triage_context = None

    def generate_seeds(
        self,
        func_name: str,
        triage_context: dict[str, Any],
        output_dir: str,
    ) -> list[tuple[Path, Path]]:
        """Generate seed tests for a validated triage verdict.

        Reads the existing shim and config from *output_dir*, calls the
        seed-generation agent, writes seed C files and build scripts.

        Returns list of (seed_c_path, build_script_path) tuples.
        """
        from .seed_generator import generate_seed_tests

        out = Path(output_dir)
        shim_path = out / f"shim_{func_name}.c"
        config_path = out / f"config_{func_name}.yaml"

        if not shim_path.exists() or not config_path.exists():
            if self.verbose:
                print(f"  Missing shim or config for {func_name}")
            return []

        shim_code = shim_path.read_text()
        ucsan_config = config_path.read_text()

        # Resolve source file for include flags
        file_path: str | None = None
        funcs = self.db.get_function_by_name(func_name)
        if funcs and funcs[0].file_path:
            file_path = funcs[0].file_path

        # Build include flags
        include_flags: list[str] = []
        if self.compile_commands and file_path:
            for flag in self.compile_commands.get_compile_flags(file_path):
                if flag.startswith(("-I", "-isystem", "-iquote",
                                    "-D", "-include")):
                    include_flags.append(flag)

        # Git tools for project exploration
        git = GitTools(self.project_path) if self.project_path else None

        if not self._can_use_fix_agent():
            if self.verbose:
                print("  Seed generation requires tool-use LLM backend")
            return []

        seed_paths = generate_seed_tests(
            func_name=func_name,
            shim_code=shim_code,
            triage_context=triage_context,
            output_dir=out,
            llm=self.llm,
            compile_fn=self._compile_shim,
            ucsan_config=ucsan_config,
            file_path=file_path,
            git_tools=git,
            include_flags=include_flags,
            verbose=self.verbose,
        )

        if not seed_paths:
            if self.verbose:
                print("  No seeds generated")
            return []

        # Extract callee stubs from shim into a standalone file for seed linking.
        # Seeds provide their own test() entry, so they can't link the full shim
        # (multi-def on test/main).  Instead we extract everything before the
        # "Entry point" marker into stubs_<func>.c.
        stubs_path = self._extract_stubs(func_name, out)

        # Collect project object files (everything except shim .o)
        shim_obj = out / f"shim_{func_name}.c.o"
        project_objs = [
            p for p in out.glob("*.o")
            if p != shim_obj and not p.name.startswith("seed_")
        ]

        # Generate build scripts for each seed
        results: list[tuple[Path, Path]] = []
        for seed_path in seed_paths:
            script = self._build_seed_script(
                func_name=func_name,
                seed_path=seed_path,
                config_path=config_path,
                project_objs=project_objs,
                stubs_path=stubs_path,
                output_dir=out,
                file_path=file_path,
            )
            script_path = out / f"build_{seed_path.stem}.sh"
            script_path.write_text(script)
            script_path.chmod(0o755)
            results.append((seed_path, script_path))

            if self.verbose:
                print(f"    Wrote: {script_path}")

        self._stats["llm_calls"] += 1  # approximate
        return results

    def refine_seeds(
        self,
        func_name: str,
        coverage_results: dict[str, Any],
        output_dir: str,
    ) -> list[tuple[Path, Path]]:
        """Refine seeds based on coverage gaps from a thoroupy run.

        Reads previous seeds and coverage data, generates new seeds
        targeting missed paths.

        Returns list of (seed_c_path, build_script_path) tuples.
        """
        from .seed_generator import refine_seed_tests

        out = Path(output_dir)
        shim_path = out / f"shim_{func_name}.c"
        config_path = out / f"config_{func_name}.yaml"

        if not shim_path.exists() or not config_path.exists():
            return []

        shim_code = shim_path.read_text()
        ucsan_config = config_path.read_text()

        # Read previous seeds
        previous_seeds: list[str] = []
        for sp in sorted(out.glob(f"seed_{func_name}_*.c")):
            previous_seeds.append(sp.read_text())

        if not previous_seeds:
            if self.verbose:
                print(f"  No previous seeds for {func_name}")
            return []

        # Resolve source file
        file_path: str | None = None
        funcs = self.db.get_function_by_name(func_name)
        if funcs and funcs[0].file_path:
            file_path = funcs[0].file_path

        include_flags: list[str] = []
        if self.compile_commands and file_path:
            for flag in self.compile_commands.get_compile_flags(file_path):
                if flag.startswith(("-I", "-isystem", "-iquote",
                                    "-D", "-include")):
                    include_flags.append(flag)

        git = GitTools(self.project_path) if self.project_path else None

        if not self._can_use_fix_agent():
            return []

        seed_paths = refine_seed_tests(
            func_name=func_name,
            shim_code=shim_code,
            previous_seeds=previous_seeds,
            coverage_results=coverage_results,
            output_dir=out,
            llm=self.llm,
            compile_fn=self._compile_shim,
            ucsan_config=ucsan_config,
            file_path=file_path,
            git_tools=git,
            include_flags=include_flags,
            verbose=self.verbose,
        )

        if not seed_paths:
            return []

        stubs_path = self._extract_stubs(func_name, out)

        shim_obj = out / f"shim_{func_name}.c.o"
        project_objs = [
            p for p in out.glob("*.o")
            if p != shim_obj and not p.name.startswith("seed_")
        ]

        results: list[tuple[Path, Path]] = []
        for seed_path in seed_paths:
            script = self._build_seed_script(
                func_name=func_name,
                seed_path=seed_path,
                config_path=config_path,
                project_objs=project_objs,
                stubs_path=stubs_path,
                output_dir=out,
                file_path=file_path,
            )
            script_path = out / f"build_{seed_path.stem}.sh"
            script_path.write_text(script)
            script_path.chmod(0o755)
            results.append((seed_path, script_path))

        return results

    def _build_seed_script(
        self,
        func_name: str,
        seed_path: Path,
        config_path: Path,
        project_objs: list[Path],
        output_dir: Path,
        stubs_path: Path | None = None,
        file_path: str | None = None,
    ) -> str:
        """Generate a build script for a seed test.

        Compiles the seed .c and links with existing project .o files
        plus a stubs .o (callee stubs extracted from the shim, without
        the test/main entry point that would conflict with the seed).
        """
        ko_clang = self.ko_clang_path or "$KO_CLANG"

        # Include flags
        include_flags_str = ""
        if self.compile_commands and file_path:
            iflags = []
            for flag in self.compile_commands.get_compile_flags(file_path):
                if flag.startswith(("-I", "-isystem", "-iquote",
                                    "-D", "-include")):
                    iflags.append(flag)
            if iflags:
                include_flags_str = " \\\n    ".join(
                    f'"{f}"' for f in iflags
                ) + " \\\n    "

        obj_files = " ".join(f'"{o}"' for o in project_objs)
        out_name = seed_path.stem + ".ucsan"
        out_path = output_dir / out_name

        # Build stubs compile + link steps if stubs file exists
        if stubs_path:
            stubs_compile = f"""\
# Step 2: Compile callee stubs
echo "[2/3] Compiling callee stubs..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c \\
    {include_flags_str}"{stubs_path}" -o "{stubs_path}.o"
"""
            link_step = "3"
            link_total = "3"
            link_objs = f'{obj_files} "{stubs_path}.o" "$SEED.o"'
        else:
            stubs_compile = ""
            link_step = "2"
            link_total = "2"
            link_objs = f'{obj_files} "$SEED.o"'

        return f"""\
#!/bin/bash
set -e

# Build seed for {func_name} ({seed_path.name})
KO_CLANG="{ko_clang}"
CONFIG="{config_path}"
SEED="{seed_path}"
OUT="{out_path}"

# Step 1: Compile seed
echo "[1/{link_total}] Compiling seed {seed_path.name}..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c \\
    {include_flags_str}"$SEED" -o "$SEED.o"

{stubs_compile}\
# Step {link_step}: Link with project objects
echo "[{link_step}/{link_total}] Linking..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
    "$KO_CLANG" {link_objs} -o "$OUT"

echo "Built: $OUT"
"""

    @staticmethod
    def _extract_stubs(func_name: str, output_dir: Path) -> Path | None:
        """Extract callee stubs from the shim into a standalone file.

        Takes everything before the ``/* ---- Entry point`` marker so seeds
        can link callee stubs without pulling in the shim's test()/main().
        Returns the path to the stubs file, or None if no stubs were found.
        """
        shim_path = output_dir / f"shim_{func_name}.c"
        if not shim_path.exists():
            return None

        shim_text = shim_path.read_text()
        marker = "/* ---- Entry point"
        idx = shim_text.find(marker)
        if idx < 0:
            return None

        stubs_text = shim_text[:idx]
        # Only write if there are actual stub functions
        if "/* ---- Callee stubs" not in stubs_text:
            return None

        stubs_path = output_dir / f"stubs_{func_name}.c"
        stubs_path.write_text(stubs_text)
        return stubs_path

    def generate_plan(
        self,
        func_name: str,
        output_dir: str,
        source_file: str | None = None,
        bc_file: str | None = None,
    ) -> dict | None:
        """Generate a trace plan for contract-guided exploration.

        Requires an already-built harness (shim + BC). Compiles the target BC
        with -O1 -g, runs UCSanPass to get BB IDs, annotates the source,
        and queries the LLM for a trace plan.

        Args:
            func_name: Target function name.
            output_dir: Harness output directory (contains shim, abilists, etc).
            source_file: Path to the source file containing the target function.
            bc_file: Path to project bitcode (will recompile with -g if needed).

        Returns:
            Plan dict or None on error.
        """
        from .bbid_extractor import (
            format_annotated_source,
            parse_cfg_dump,
        )

        out = Path(output_dir)

        # Look up function
        funcs = self.db.get_function_by_name(func_name)
        if not funcs:
            if self.verbose:
                print(f"  Function not found: {func_name}")
            return None
        func = funcs[0]
        assert func.id is not None

        # Resolve source file
        if not source_file and func.file_path:
            source_file = func.file_path
        if not source_file:
            if self.verbose:
                print(f"  No source file for: {func_name}")
            return None

        # CFG dump is generated by the build script (ko-clang with
        # KO_TRACE_BB=1 KO_DUMP_CFG=file).  It must exist before
        # calling generate_plan().
        cfg_path = out / f"cfg_{func_name}.txt"
        if not cfg_path.exists():
            if self.verbose:
                print(f"  No CFG dump found: {cfg_path}")
                print("  Run the build script first to generate BB IDs.")
            return None

        infos = parse_cfg_dump(str(cfg_path))
        if self.verbose:
            print(f"  Loaded CFG: {len(infos)} BBs from {cfg_path.name}")

        annotated = format_annotated_source(infos, source_file)

        if self.verbose:
            print(f"  Extracted {len(infos)} BB IDs")

        # Get contracts
        row = self.db.conn.execute(
            "SELECT summary_json FROM memsafe_summaries WHERE function_id = ?",
            (func.id,),
        ).fetchone()
        memsafe_data = json.loads(row[0]) if row else {}
        contracts = memsafe_data.get("contracts", [])

        callee_contracts = self._gather_callee_contracts(func.id)
        postconds = self._gather_postconditions(func.id)

        # Build context (contracts) and goal (exercise memory ops)
        context = (
            f"### Target Function\n\n"
            f"Name: `{func.name}`\n"
            f"Signature: `{func.signature}`\n\n"
            f"### Contracts\n\n"
            f"{self._format_contracts(contracts)}\n\n"
            f"### Callee Contracts\n\n"
            f"{self._format_callee_contracts(callee_contracts)}\n\n"
            f"### Post-conditions\n\n"
            f"{self._format_postconditions(postconds)}"
        )

        goal = (
            "Plan paths that **exercise memory operations** so the concolic "
            "executor can check the function's memory safety contracts. "
            "Only target paths where pointer dereferences, buffer accesses, "
            "or callee calls with pointer args actually happen.\n\n"
            "**Key principle**: A path that just returns an error code without "
            "performing any memory operations is trivially safe — skip it. "
            "Focus on paths that reach:\n"
            "- Array/buffer indexing: `buf[i] = ...`, `memcpy(dst, src, n)`\n"
            "- Pointer dereferences: `state->field`, `*ptr`\n"
            "- Callee calls that pass pointers: `func(state, buf, len)`"
        )

        annotated_block = f"### `{func.name}`\n\n```c\n{annotated}\n```"

        prompt = PLAN_PROMPT.format(
            context=context,
            annotated_sources=annotated_block,
            goal=goal,
        )

        try:
            if self.verbose:
                print(f"  Generating trace plan for: {func_name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(f"{func_name}_plan", prompt, response)

            # Parse JSON from response
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                plan: dict = json.loads(json_match.group(1))
            else:
                # Try parsing entire response as JSON
                plan = json.loads(response.strip())

            # Post-process: convert target_edges → target_bids (branch IDs to flip)
            # Build successor map from BBInfo: bb_id → (true_bb_id, false_bb_id)
            succ_map = {}
            for bb in infos:
                if bb.is_conditional and bb.true_bb_id is not None:
                    succ_map[bb.bb_id] = (bb.true_bb_id, bb.false_bb_id)

            for trace in plan.get("traces", []):
                edges = trace.pop("target_edges", [])
                flip_bids = set()
                for edge in edges:
                    src = edge.get("from")
                    dst = edge.get("to")
                    if src not in succ_map:
                        if self.verbose:
                            print(f"    Warning: BB {src} is not a conditional branch")
                        continue
                    true_bb, false_bb = succ_map[src]
                    if dst == true_bb:
                        # Want T path — flip needed if default takes F
                        # Scheduler can't know default direction, so always
                        # track this branch. Coverage = branch was flipped OR
                        # destination was visited.
                        flip_bids.add(src)
                    elif dst == false_bb:
                        # Want F path — same: track this branch
                        flip_bids.add(src)
                    else:
                        if self.verbose:
                            print(
                                f"    Warning: edge {src}→{dst} doesn't match "
                                f"successors T:{true_bb} F:{false_bb}"
                            )
                trace["target_bids"] = sorted(flip_bids)

            # Write plan
            plan_path = out / f"plan_{func_name}.json"
            plan_path.write_text(json.dumps(plan, indent=2))

            if self.verbose:
                n_traces = len(plan.get("traces", []))
                n_depri = len(plan.get("deprioritize", []))
                print(f"    Plan: {n_traces} traces, {n_depri} deprioritized")
                print(f"    Wrote: {plan_path}")

            return plan

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error generating plan for {func_name}: {e}")
                import traceback
                traceback.print_exc()
            return None

    def generate_validation_plan(
        self,
        verdict: dict[str, Any],
        output_dir: str,
        cfg_dump: str | None = None,
        entry_name: str | None = None,
        scope_functions: list[str] | None = None,
    ) -> dict | None:
        """Generate a cross-function trace plan to validate a triage verdict.

        Looks up all relevant_functions, annotates their source with BB IDs
        from the CFG dump, and asks the LLM to find counter-example paths.

        Args:
            verdict: Triage verdict dict (from verdict JSON).
            output_dir: Directory containing harness + CFG dump.
            cfg_dump: Path to CFG dump file. If None, looks for
                cfg_{entry_name}.txt in output_dir.
            entry_name: Entry function name (for naming output files).
                Defaults to verdict's function_name.
            scope_functions: Override relevant_functions from verdict.
                Use to limit scope per entry function.

        Returns:
            Plan dict or None on error.
        """
        from .bbid_extractor import (
            format_annotated_function,
            parse_cfg_dump,
        )

        out = Path(output_dir)
        func_name = verdict["function_name"]
        plan_name = entry_name or func_name
        relevant = scope_functions or verdict.get("relevant_functions", [func_name])
        hypothesis = verdict.get("hypothesis", "unknown")
        issue = verdict.get("issue", {})

        # Find CFG dump
        if cfg_dump:
            cfg_path = Path(cfg_dump)
        else:
            cfg_path = out / f"cfg_{plan_name}.txt"
        if not cfg_path.exists():
            if self.verbose:
                print(f"  No CFG dump found: {cfg_path}")
                print("  Run the build script first to generate BB IDs.")
            return None

        infos = parse_cfg_dump(str(cfg_path))
        if self.verbose:
            print(f"  Loaded CFG: {len(infos)} BBs from {cfg_path.name}")

        # Look up all relevant functions and annotate their source
        annotated_blocks = []
        all_funcs = []
        for rname in relevant:
            funcs = self.db.get_function_by_name(rname)
            if not funcs:
                if self.verbose:
                    print(f"  Function not found: {rname}")
                continue
            func = funcs[0]
            all_funcs.append(func)

            if not func.file_path or not func.line_start or not func.line_end:
                if self.verbose:
                    print(f"  No source location for: {rname}")
                continue

            annotated = format_annotated_function(
                infos, func.file_path, func.line_start, func.line_end,
            )
            bb_count = sum(
                1 for i in infos
                if Path(i.file).name == Path(func.file_path).name
                and func.line_start <= i.line <= func.line_end
            )
            annotated_blocks.append(
                f"### `{func.name}` ({func.file_path}:{func.line_start}-"
                f"{func.line_end}, {bb_count} BBs)\n\n```c\n{annotated}\n```"
            )

        if not annotated_blocks:
            if self.verbose:
                print("  No functions annotated — cannot generate plan")
            return None

        # Build context from verdict
        assumptions = verdict.get("assumptions", [])
        assertions = verdict.get("assertions", [])

        assumptions_text = "None."
        if assumptions:
            assumptions_text = "\n".join(
                f"  {i}. {a}" for i, a in enumerate(assumptions, 1)
            )

        assertions_text = "None."
        if assertions:
            assertions_text = "\n".join(
                f"  {i}. {a}" for i, a in enumerate(assertions, 1)
            )

        context = (
            f"### Triage Verdict\n\n"
            f"- Function: `{func_name}`\n"
            f"- Hypothesis: **{hypothesis}**\n"
            f"- Issue: [{issue.get('severity', '')}] "
            f"{issue.get('issue_kind', '')} — "
            f"{issue.get('description', '')}\n\n"
            f"### Reasoning\n\n"
            f"{verdict.get('reasoning', 'N/A')}\n\n"
            f"### Assumptions\n\n{assumptions_text}\n\n"
            f"### Assertions\n\n{assertions_text}\n\n"
            f"All functions listed below are **real code** (not stubs). "
            f"The executor explores actual code paths across function "
            f"boundaries."
        )

        # Build goal based on hypothesis
        if hypothesis == "safe":
            goal = (
                "The verdict claims this code is **safe**. Your job is to "
                "find paths that could **disprove** this — a counter-example "
                "where the safety assumption is violated.\n\n"
                "Specifically, look for paths where:\n"
                + "\n".join(
                    f"- Assumption could fail: {a}" for a in assumptions
                )
                + "\n\nIf such a path exists, ucsan will find the violation. "
                "If not, the verdict is confirmed."
            )
        else:
            feasible = verdict.get("feasible_path", [])
            path_text = " → ".join(feasible) if feasible else "see reasoning"
            goal = (
                f"The verdict claims this code is **unsafe**. Your job is to "
                f"find the specific violation path described in the verdict "
                f"and confirm the violation happens.\n\n"
                f"Target path: {path_text}\n\n"
                f"Guide the executor to reach the described violation. If "
                f"ucsan reaches it and finds the violation, verdict confirmed."
            )

        annotated_sources = "\n\n".join(annotated_blocks)

        prompt = PLAN_PROMPT.format(
            context=context,
            annotated_sources=annotated_sources,
            goal=goal,
        )

        try:
            if self.verbose:
                print(f"  Generating validation plan for: {func_name} "
                      f"({len(relevant)} functions)")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(
                    f"{func_name}_validation_plan", prompt, response,
                )

            # Parse JSON from response
            json_match = re.search(
                r"```json\s*(.*?)\s*```", response, re.DOTALL,
            )
            if json_match:
                plan: dict = json.loads(json_match.group(1))
            else:
                plan = json.loads(response.strip())

            # Post-process: convert target_edges → target_bids
            succ_map = {}
            for bb in infos:
                if bb.is_conditional and bb.true_bb_id is not None:
                    succ_map[bb.bb_id] = (bb.true_bb_id, bb.false_bb_id)

            for trace in plan.get("traces", []):
                edges = trace.pop("target_edges", [])
                flip_bids = set()
                for edge in edges:
                    src = edge.get("from")
                    dst = edge.get("to")
                    if src not in succ_map:
                        if self.verbose:
                            print(
                                f"    Warning: BB {src} is not a "
                                f"conditional branch"
                            )
                        continue
                    true_bb, false_bb = succ_map[src]
                    if dst in (true_bb, false_bb):
                        flip_bids.add(src)
                    elif self.verbose:
                        print(
                            f"    Warning: edge {src}→{dst} doesn't "
                            f"match successors T:{true_bb} F:{false_bb}"
                        )
                trace["target_bids"] = sorted(flip_bids)

            # Extract runtime config from LLM output
            runtime_config = self._build_runtime_config(
                plan.pop("ucsan_config", {}),
            )

            # Disable checkers unrelated to the issue being validated
            issue_kind = issue.get("issue_kind", "")
            if issue_kind:
                runtime_config = self._apply_issue_checker_filter(
                    runtime_config, issue_kind,
                )

            # Write plan
            plan_path = out / f"plan_{plan_name}_validation.json"
            plan_path.write_text(json.dumps(plan, indent=2))

            if runtime_config:
                rt_path = out / f"runtime_{plan_name}.json"
                rt_path.write_text(json.dumps(runtime_config, indent=2))

            if self.verbose:
                n_traces = len(plan.get("traces", []))
                n_depri = len(plan.get("deprioritize", []))
                print(f"    Plan: {n_traces} traces, {n_depri} deprioritized")
                print(f"    Wrote: {plan_path}")
                if runtime_config:
                    print(f"    Wrote: {rt_path}")

            return plan

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error generating validation plan: {e}")
                import traceback
                traceback.print_exc()
            return None

    # Primitive C types that don't need void* replacement
    _PRIMITIVE_TYPES = {
        "int", "unsigned", "unsigned int", "long", "unsigned long",
        "short", "unsigned short", "char", "unsigned char",
        "signed char", "float", "double", "long double",
        "long long", "unsigned long long", "size_t", "ssize_t",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "uintptr_t", "intptr_t", "ptrdiff_t",
        "void", "const char", "const void", "const unsigned char",
    }

    def _build_extern_decl(
        self, name: str, signature: str,
        param_names: list[str] | None = None,
    ) -> str:
        """Build extern declaration, replacing typedefs with canonical C types.

        - Pointer typedefs (e.g. gzFile -> struct gzFile_s *) become void *
        - Scalar typedefs (e.g. uLong -> unsigned long) become their canonical type
        - Pointer-to-non-primitive (e.g. deflate_state *) becomes void *
        """
        paren_idx = signature.index("(")
        ret_type = signature[:paren_idx].strip()
        params_str = signature[paren_idx + 1 : signature.rindex(")")]

        # Build typedef lookup maps
        ptr_typedefs = self._get_pointer_typedefs()
        scalar_typedefs = self._get_scalar_typedefs()

        def replace_type(t: str) -> str:
            t = t.strip()
            if not t or t == "void" or t == "...":
                return t
            is_const = t.startswith("const ")
            bare = t.removeprefix("const ").strip()
            # Pointer types: keep as-is (headers provide definitions)
            if t.endswith("*"):
                return t
            # Typedef that is actually a pointer (e.g. gzFile, z_streamp)
            if bare in ptr_typedefs:
                return t
            # Scalar typedef (e.g. uLong -> unsigned long)
            if bare in scalar_typedefs:
                canonical = scalar_typedefs[bare]
                return f"const {canonical}" if is_const else canonical
            # Unknown non-primitive, non-pointer type (e.g. enum typedef)
            # — fall back to int to avoid unknown type errors in the shim
            if bare not in self._PRIMITIVE_TYPES:
                return "int"
            return t

        if params_str.strip():
            params = [replace_type(p) for p in params_str.split(",")]
            # Attach parameter names if provided
            if param_names:
                named: list[str] = []
                for i, p in enumerate(params):
                    pname = param_names[i] if i < len(param_names) else f"p{i}"
                    if p == "...":
                        named.append(p)
                    else:
                        named.append(f"{p} {pname}")
                params_out = ", ".join(named)
            else:
                params_out = ", ".join(params)
        else:
            params_out = "void"

        ret_type = replace_type(ret_type)

        return f"extern {ret_type} {name}({params_out});"

    def _get_pointer_typedefs(self) -> set[str]:
        """Get set of typedef names that resolve to pointer types from the DB."""
        rows = self.db.conn.execute(
            "SELECT name FROM typedefs WHERE canonical_type LIKE '%*%'"
        ).fetchall()
        return {r[0] for r in rows}

    def _get_scalar_typedefs(self) -> dict[str, str]:
        """Get mapping of non-pointer typedef names to canonical C types.

        Only includes typedefs whose canonical type is a known primitive
        (e.g. uLong -> unsigned long), skipping struct/enum/opaque types.
        """
        rows = self.db.conn.execute(
            "SELECT DISTINCT name, canonical_type FROM typedefs "
            "WHERE canonical_type NOT LIKE '%*%'"
        ).fetchall()
        result = {}
        for name, canonical in rows:
            # Only map if canonical type is a primitive we recognize
            bare = canonical.removeprefix("const ").strip()
            if bare in self._PRIMITIVE_TYPES:
                result[name] = canonical
        return result

    def _compile_to_bc(self, source_file: str, output_dir: Path) -> str | None:
        """Re-compile a source file to LLVM bitcode using compile_commands flags.

        Uses clang-14 with the same flags from compile_commands.json but
        outputs bitcode via -emit-llvm -c.

        Returns path to .bc file or None on failure.
        """
        if not self.compile_commands:
            return None

        source_path = Path(source_file)
        if not source_path.is_absolute() and self.project_path:
            source_path = self.project_path / source_path
        if not source_path.exists():
            if self.verbose:
                print(f"    Source file not found: {source_file}")
            return None

        flags = self.compile_commands.get_compile_flags(source_file)
        if not flags:
            if self.verbose:
                print(f"    No compile flags found for: {source_file}")
            return None

        # Filter out flags incompatible with clang-14 or bitcode generation
        filtered = []
        for f in flags:
            # Skip LTO flags
            if f.startswith("-flto"):
                continue
            # Skip save-temps
            if f.startswith("-save-temps"):
                continue
            # Skip optimization flags that might cause issues
            if f.startswith("-g"):
                continue
            filtered.append(f)

        bc_name = source_path.stem + ".bc"
        bc_path = output_dir / bc_name

        cmd = ["clang-14"] + filtered + [
            "-emit-llvm", "-c", str(source_path), "-o", str(bc_path),
        ]

        if self.verbose:
            print(f"    Compiling to bitcode: {source_path.name} -> {bc_name}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            if self.verbose:
                print(f"    Bitcode compilation failed:\n{result.stderr[:500]}")
            return None

        return str(bc_path)

    def _build_ucsan_config(
        self, func_name: str,
        shim_callees: list[str] | None = None,
        scope_functions: list[str] | None = None,
    ) -> str:
        """Build ucsan config YAML (entry + scope + shims).

        Args:
            func_name: Primary target function (used as default scope).
            shim_callees: Functions that get __shim_ stubs.
            scope_functions: All functions to instrument. Defaults to
                [func_name] if not provided.
        """
        scoped = scope_functions or [func_name]
        lines = ["entry: test", "scope:"]
        for s in scoped:
            lines.append(f"  - {s}")
        if shim_callees:
            lines.append("shims:")
            for cname in shim_callees:
                lines.append(f"  - {cname}")
        return "\n".join(lines) + "\n"


    def _build_script(
        self, func_name: str, out_dir: Path, bc_file: str | None,
        file_path: str | None = None,
        source_files: list[str] | None = None,
        scope_functions: list[str] | None = None,
    ) -> str:
        """Generate a shell build script using ko-clang.

        ko-clang handles UCSanPass + TaintPass instrumentation internally
        using built-in ucsan_abilist.txt and dfsan_abilist.txt.

        When source_files are provided, compiles from source (enables
        KO_TRACE_BB / KO_DUMP_CFG for BB ID extraction).  Falls back
        to compiling from bc_file otherwise.
        """
        ko_clang = self.ko_clang_path or "$KO_CLANG"

        config_file = out_dir / f"config_{func_name}.yaml"
        shim_file = out_dir / f"shim_{func_name}.c"
        cfg_dump = out_dir / f"cfg_{func_name}.txt"

        # Resolve file_path against project root
        if file_path and not Path(file_path).is_absolute() and self.project_path:
            file_path = str(self.project_path / file_path)

        # Build include flags from compile_commands
        include_flags_str = ""
        if self.compile_commands and file_path:
            iflags = []
            for flag in self.compile_commands.get_compile_flags(file_path):
                if flag.startswith(("-I", "-isystem", "-iquote",
                                    "-D", "-include")):
                    iflags.append(flag)
            if iflags:
                include_flags_str = " \\\n    ".join(
                    f'"{f}"' for f in iflags
                ) + " \\\n    "

        # Build compile flags for source files (full flags, not just includes)
        compile_flags_str = ""
        if self.compile_commands and file_path:
            cflags = []
            for flag in self.compile_commands.get_compile_flags(file_path):
                if flag.startswith(("-flto", "-save-temps")):
                    continue
                cflags.append(flag)
            if cflags:
                compile_flags_str = " \\\n    ".join(
                    f'"{f}"' for f in cflags
                ) + " \\\n    "

        # Determine source files to compile (resolve relative paths)
        raw_srcs = source_files or ([file_path] if file_path else [])
        srcs = []
        for s in raw_srcs:
            p = Path(s)
            if not p.is_absolute() and self.project_path:
                p = self.project_path / p
            srcs.append(str(p))

        if srcs:
            # Compile from source — enables BB tracing
            src_compile_steps = []
            obj_files = []
            for src in srcs:
                src_name = Path(src).name
                obj_name = Path(src).stem + ".o"
                obj_path = out_dir / obj_name
                obj_files.append(str(obj_path))
                src_compile_steps.append(
                    f'echo "  Compiling {src_name}..."\n'
                    f'METADATA="$CONFIG" KO_CC=clang-14 '
                    f'KO_DONT_OPTIMIZE=1 KO_TRACE_BB=1 KO_DUMP_CFG="$CFG" \\\n'
                    f'    "$KO_CLANG" -c -g -fno-inline-functions \\\n'
                    f'    {compile_flags_str}"{src}" -o "{obj_path}"'
                )

            src_steps = "\n".join(src_compile_steps)
            link_objs = " ".join(f'"{o}"' for o in obj_files)

            # Build globalize step: for each source .o, check if
            # scope .taint symbols are local (t) and promote to global (T).
            # This is needed when target functions are static.
            scoped = scope_functions or [func_name]
            taint_syms = [f"{s}.taint" for s in scoped]
            globalize_lines: list[str] = []
            for obj in obj_files:
                # For each .o, check and globalize any local .taint symbols
                sym_checks = []
                for sym in taint_syms:
                    sym_checks.append(
                        f'  if nm "{obj}" 2>/dev/null '
                        f"| grep -q ' t {sym}$'; then\n"
                        f'    objcopy --globalize-symbol=\'{sym}\' "{obj}"\n'
                        f'    echo "    Globalized {sym}"\n'
                        f'  fi'
                    )
                globalize_lines.extend(sym_checks)
            globalize_block = "\n".join(globalize_lines)

            script = f"""\
#!/bin/bash
set -e

# Build harness for {func_name}
KO_CLANG="{ko_clang}"
CONFIG="{config_file}"
SHIM="{shim_file}"
CFG="{cfg_dump}"
OUT="{out_dir}/{func_name}.ucsan"

# Step 1: Compile shim
echo "[1/4] Compiling shim..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c \\
    {include_flags_str}"$SHIM" -o "$SHIM.o"

# Step 2: Compile project source (with BB tracing)
echo "[2/4] Compiling project source..."
rm -f "$CFG"
{src_steps}

# Step 3: Globalize static .taint symbols (if any)
echo "[3/4] Checking for static scope symbols..."
{globalize_block}

# Step 4: Link
echo "[4/4] Linking..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
    "$KO_CLANG" {link_objs} "$SHIM.o" -o "$OUT"

echo "Built: $OUT"
echo "CFG dump: $CFG"
"""
        else:
            # Fallback: compile from BC (no BB tracing)
            bc = bc_file or "$1"
            script = f"""\
#!/bin/bash
set -e

# Build harness for {func_name}
KO_CLANG="{ko_clang}"
CONFIG="{config_file}"
SHIM="{shim_file}"
BC="${{1:-{bc}}}"
OUT="{out_dir}/{func_name}.ucsan"

# Step 1: Compile shim
echo "[1/3] Compiling shim..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c \\
    {include_flags_str}"$SHIM" -o "$SHIM.o"

# Step 2: Compile target bitcode
echo "[2/3] Compiling project bitcode..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c "$BC" -o "$BC.o"

# Step 3: Link
echo "[3/3] Linking..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
    "$KO_CLANG" "$BC.o" "$SHIM.o" -o "$OUT"

echo "Built: $OUT"
"""
        return script

    def _compile_shim(
        self, c_code: str, ucsan_config: str,
        file_path: str | None = None,
    ) -> tuple[bool, str]:
        """Compile shim with ko-clang (handles instrumentation internally).

        ko-clang applies UCSanPass + TaintPass using built-in abilists
        (ucsan_abilist.txt, dfsan_abilist.txt).  Shim code is instrumented
        so assert_*/assume_* calls get proper labels.

        Returns (success, error_output).
        """
        if not self.ko_clang_path:
            return False, "ko_clang_path not set"

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = f"{tmpdir}/shim.c"
            obj_path = f"{tmpdir}/shim.o"
            cfg_path = f"{tmpdir}/config.yaml"

            with open(src_path, "w") as f:
                f.write(c_code)
            with open(cfg_path, "w") as f:
                f.write(ucsan_config)

            env = {
                "METADATA": cfg_path,
                "KO_CC": "clang-14",
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
            }

            # Build include flags from compile_commands
            include_flags: list[str] = []
            if self.compile_commands and file_path:
                for flag in self.compile_commands.get_compile_flags(file_path):
                    if flag.startswith(("-I", "-isystem", "-iquote",
                                        "-D", "-include")):
                        include_flags.append(flag)

            cmd = [
                self.ko_clang_path, "-c",
                *include_flags,
                src_path, "-o", obj_path,
            ]
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, env=env,
            )
            if r.returncode != 0:
                return False, (r.stderr + r.stdout).strip()

            return True, ""

    def _can_use_fix_agent(self) -> bool:
        """Check if the agent-based fix loop can be used.

        Requires tool-use support from the LLM backend (complete_with_tools
        must be overridden from the base class).
        """
        from .llm.base import LLMBackend
        return type(self.llm).complete_with_tools is not LLMBackend.complete_with_tools

    def _fix_with_agent(
        self,
        c_code: str,
        ucsan_config: str,
        file_path: str | None,
        initial_errors: str,
    ) -> str | None:
        """Use a ReAct agent loop to fix compilation errors.

        The agent can inspect project source via git tools, patch functions,
        recompile, and iterate until the shim compiles or turns are exhausted.

        Returns the fixed C code, or None if it could not be fixed.
        """
        git = GitTools(self.project_path) if self.project_path else None
        executor = HarnessFixExecutor(
            c_code, ucsan_config, file_path, self, git,
        )

        user_prompt = (
            "The following shim failed to compile. Fix the errors.\n\n"
            "## Current shim code\n\n```c\n" + c_code + "\n```\n\n"
            "## Compiler errors\n\n```\n" + initial_errors + "\n```\n\n"
            "Use the tools to investigate and fix the errors. "
            "When compilation succeeds, call submit_harness."
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        # Filter out git tools if no project path
        tools = list(HARNESS_FIX_TOOL_DEFINITIONS)
        if git is None:
            tools = [t for t in tools if t["name"] not in _GIT_TOOL_NAMES]

        for turn in range(MAX_FIX_TURNS):
            response = self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                system=FIX_SYSTEM_PROMPT,
            )
            self._stats["llm_calls"] += 1

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(f"    [fix-agent] LLM stopped at turn {turn + 1}")
                break

            if stop != "tool_use":
                if self.verbose:
                    print(f"    [fix-agent] Unexpected stop: {stop}")
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
                        print(f"    [fix-agent] {block.text[:200]}")

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
                    self._stats["fix_attempts"] += 1

                    if self.verbose:
                        err = result.get("error")
                        if err:
                            print(f"    [fix-agent] {block.name} -> "
                                  f"ERROR: {err[:150]}")
                        elif block.name == "compile_shim":
                            ok = result.get("success")
                            print(f"    [fix-agent] compile_shim -> "
                                  f"{'OK' if ok else 'FAIL'}")
                        elif block.name == "submit_harness":
                            print("    [fix-agent] submit_harness -> accepted")
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"    [fix-agent] {block.name}({arg})")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            if executor.submitted:
                return executor.c_code

        # Agent didn't submit — return None
        if self.verbose:
            print(f"    [fix-agent] Failed after {turn + 1} turns "
                  f"({executor.compile_attempts} compiles)")
        return None

    def _gather_callee_contracts(self, func_id: int) -> dict[str, dict]:
        """Get memsafe contracts for all direct callees of func_id."""
        edges = self.db.get_all_call_edges()
        callee_ids = {e.callee_id for e in edges if e.caller_id == func_id}

        result = {}
        for cid in callee_ids:
            rows = self.db.conn.execute(
                "SELECT f.name, f.signature, f.params_json, m.summary_json "
                "FROM functions f JOIN memsafe_summaries m ON m.function_id = f.id "
                "WHERE f.id = ?",
                (cid,),
            ).fetchall()
            for name, sig, params_json, summary_json in rows:
                data = json.loads(summary_json)
                if data.get("contracts"):
                    params = json.loads(params_json) if params_json else []
                    result[name] = {
                        "signature": sig,
                        "params": params,
                        "contracts": data["contracts"],
                    }
        return result

    def _gather_postconditions(self, func_id: int) -> dict:
        """Gather post-conditions from allocation, init, and free summaries."""
        result: dict = {"allocations": [], "inits": [], "frees": []}

        for table, key in [
            ("allocation_summaries", "allocations"),
            ("init_summaries", "inits"),
            ("free_summaries", "frees"),
        ]:
            row = self.db.conn.execute(
                f"SELECT summary_json FROM {table} WHERE function_id = ?",
                (func_id,),
            ).fetchone()
            if row:
                data = json.loads(row[0])
                result[key] = data.get(key, [])

        # Replace local size_expr with persistent equivalents from
        # buffer_size_pairs when available.
        if result["allocations"]:
            self._resolve_local_size_exprs(func_id, result["allocations"])

        return result

    def _resolve_local_size_exprs(
        self, func_id: int, allocations: list[dict[str, Any]],
    ) -> None:
        """Replace local-variable size_expr with persistent struct fields.

        Uses buffer_size_pairs from the allocation summary to find persistent
        size expressions (struct fields, params, globals) for allocations whose
        size_expr references a local variable.
        """
        row = self.db.conn.execute(
            "SELECT summary_json FROM allocation_summaries "
            "WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if not row:
            return
        data = json.loads(row[0])
        pairs = data.get("buffer_size_pairs", [])
        if not pairs:
            return

        # Build mapping: buffer field -> persistent size expression
        # Normalize keys by stripping array indexing (e.g. "[i]", "[]")
        buf_to_size: dict[str, str] = {}
        for bp in pairs:
            buf_key = re.sub(r"\[.*?\]", "", bp.get("buffer", ""))
            size_field = bp.get("size", "")
            if buf_key and size_field:
                buf_to_size[buf_key] = size_field

        for alloc in allocations:
            size_expr = alloc.get("size_expr")
            if not size_expr:
                continue
            # Skip if already persistent (contains -> or is a known macro)
            if "->" in size_expr or "." in size_expr or size_expr.isupper():
                continue
            # Strip local: prefix if present
            if size_expr.startswith("local:"):
                size_expr = size_expr[len("local:"):].strip()
                alloc["size_expr"] = size_expr
            # Look up stored_to in buffer_size_pairs
            stored = alloc.get("stored_to", "")
            stored_norm = re.sub(r"\[.*?\]", "", stored)
            if stored_norm in buf_to_size:
                alloc["size_expr"] = buf_to_size[stored_norm]

    def _format_postconditions(self, postconds: dict) -> str:
        """Format post-conditions for the prompt."""
        lines = []

        for alloc in postconds.get("allocations", []):
            target = "return value" if alloc.get("returned") else alloc.get("stored_to", "?")
            size = alloc.get("size_expr", "?")
            cond = alloc.get("condition", "")
            may_null = alloc.get("may_be_null", True)
            line = f"- ALLOCATES: `{target}` (size: {size}, may_be_null: {may_null})"
            if cond:
                line += f" [when {cond}]"
            lines.append(line)

        for init in postconds.get("inits", []):
            target = init.get("target", "?")
            kind = init.get("target_kind", "?")
            byte_count = init.get("byte_count", "?")
            cond = init.get("condition", "")
            line = f"- INITIALIZES: `{target}` ({kind}, {byte_count} bytes)"
            if cond:
                line += f" [when {cond}]"
            lines.append(line)

        for free in postconds.get("frees", []):
            target = free.get("target", "?")
            kind = free.get("target_kind", "?")
            cond = free.get("condition", "")
            line = f"- FREES: `{target}` ({kind})"
            if cond:
                line += f" [when {cond}]"
            lines.append(line)

        if not lines:
            return "No post-conditions."
        return "\n".join(lines)

    def _format_contracts(self, contracts: list[dict]) -> str:
        if not contracts:
            return "No contracts (no pre-conditions required)."
        lines = []
        for c in contracts:
            kind = c["contract_kind"]
            target = c["target"]
            desc = c.get("description", "")
            if kind == "buffer_size":
                size = c.get("size_expr", "?")
                rel = c.get("relationship", "byte_count")
                lines.append(f"- `{target}`: buffer_size({size}, {rel}) -- {desc}")
            else:
                lines.append(f"- `{target}`: {kind} -- {desc}")
            if c.get("condition"):
                lines[-1] += f" [when {c['condition']}]"
        return "\n".join(lines)

    def _format_callee_contracts(self, callee_contracts: dict[str, dict]) -> str:
        if not callee_contracts:
            return "No callees with contracts."
        lines = []
        for name, info in callee_contracts.items():
            sig = info["signature"]
            params = info["params"]
            lines.append(f"### `{name}` -- `{sig}`")
            lines.append(f"Parameters: {params}")
            for c in info["contracts"]:
                kind = c["contract_kind"]
                target = c["target"]
                if kind == "buffer_size":
                    size = c.get("size_expr", "?")
                    rel = c.get("relationship", "byte_count")
                    lines.append(f"  - `{target}`: buffer_size({size}, {rel})")
                else:
                    lines.append(f"  - `{target}`: {kind}")
            lines.append("")
        return "\n".join(lines)

    def _build_fill_template(
        self,
        func_name: str,
        func_signature: str,
        func_params: list[str],
        callee_contracts: dict[str, dict],
        postconds: dict,
        triage_context: dict[str, Any] | None = None,
        contracts: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
        preceding_functions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a fill-in-the-blank C template for shim generation.

        Generates the complete shim structure with `/* FILL: ... */` markers
        where the LLM needs to add code. Everything else is fixed.

        Args:
            preceding_functions: For sequential test cases, info dicts for
                functions to call before the target. Each dict has keys:
                name, signature, params, postconds.
        """
        real_fns = set((triage_context or {}).get("real_functions", []))
        # Only generate stubs for callees NOT in real_functions or ucsan abilist
        stub_callees = {
            k: v for k, v in callee_contracts.items()
            if k not in real_fns and k not in self._ucsan_abilist
        }

        lines: list[str] = []

        # --- Header ---
        lines.append(
            "/* Auto-generated shim for contract-guided concolic execution */")
        lines.append("#include <stdlib.h>")
        lines.append("#include <stdint.h>")
        lines.append("#include <stddef.h>")
        lines.append("#include <string.h>")
        lines.append("")

        # --- API reference as comments ---
        lines.append("/*")
        lines.append(" * Summary function API (use in stubs and test):")
        lines.append(" *")
        lines.append(" * Assertions (verify a condition holds):")
        lines.append(" *   assert_cond(result, id)"
                      "              -- boolean check")
        lines.append(" *   assert_allocated(ptr, size, id)"
                      "      -- ptr is allocated >= size bytes")
        lines.append(" *   assert_init(ptr, size, id)"
                      "           -- ptr allocated + all bytes initialized")
        lines.append(" *   assert_freed(ptr, id)"
                      "                -- ptr has been freed")
        lines.append(" *")
        lines.append(" * Assumptions (establish a condition in callee stubs):")
        lines.append(" *   assume_cond(result, id)"
                      "              -- constrain solver, exit if false")
        lines.append(" *   ptr = assume_allocated(ptr, size, id)"
                      " -- ensure ptr is allocated with size bytes.")
        lines.append(
            " *     ptr must be an existing variable."
            " To allocate new memory, use malloc(size).")
        lines.append(" *   assume_init(ptr, size, id)"
                      "           -- mark size bytes as initialized")
        lines.append(" *   ptr = assume_freed(ptr, id)"
                      "           -- mark as freed (returns new ptr!)")
        lines.append(" *")
        lines.append(
            " * Use the assigned id from the contract comment above.")
        lines.append(" *")
        lines.append(" * Example stub for: void read_data(void *buf, "
                      "size_t len)")
        lines.append(
            " *   assert_allocated(buf, len, 1); "
            " // id=1 buf: buffer_size(len)")
        lines.append(
            " *   assume_init(buf, len, 2); "
            "     // id=2 post: buf initialized")
        lines.append(" */")
        lines.append("")

        # --- Extern declarations for summary functions ---
        lines.append("/* Summary functions (instrumentation pass rewrites "
                      "these) */")
        lines.append("extern void assert_cond(uint8_t result, uint64_t id);")
        lines.append("extern void assert_allocated(void *ptr, size_t size, "
                      "uint64_t id);")
        lines.append("extern void assert_init(void *ptr, size_t size, "
                      "uint64_t id);")
        lines.append("extern void assert_freed(void *ptr, uint64_t id);")
        lines.append("extern void assume_cond(uint8_t result, uint64_t id);")
        lines.append("extern void *assume_allocated(void *ptr, size_t size, "
                      "uint64_t id);")
        lines.append("extern void assume_init(void *ptr, size_t size, "
                      "uint64_t id);")
        lines.append("extern void *assume_freed(void *ptr, uint64_t id);")
        lines.append("")

        # --- Include project headers from the source file ---
        if file_path:
            src_headers = self._extract_source_includes(file_path)
            for h in src_headers:
                lines.append(f'#include "{h}"')
            if src_headers:
                lines.append("")

        # --- Emit definitions for types only defined in .c files ---
        source_typedefs = self._get_source_only_typedefs(stub_callees)
        if source_typedefs:
            lines.append("/* Types defined in source files (not headers) */")
            for defn in source_typedefs:
                lines.append(defn)
            lines.append("")

        # --- Preceding function externs (sequential test case) ---
        if preceding_functions:
            lines.append("/* Preceding functions (in bitcode, called before "
                         "target to establish state) */")
            for pf in preceding_functions:
                pf_extern = self._build_extern_decl(
                    pf["name"], pf["signature"], pf["params"],
                )
                lines.append(pf_extern)
            lines.append("")

        # --- Target function extern ---
        target_extern = self._build_extern_decl(
            func_name, func_signature, func_params,
        )
        lines.append("/* Target function (in bitcode) */")
        lines.append(target_extern)
        lines.append("")

        # --- ID map + callee stubs ---
        # Assign a unique ID per contract for diagnostics feedback
        id_counter = 1
        id_map: list[str] = []  # "id: func:target:kind"

        if stub_callees:
            lines.append("/* ---- Callee stubs ---- */")
            lines.append("")
            for cname, cinfo in stub_callees.items():
                sig = cinfo["signature"]
                params = cinfo["params"]
                ccontracts = cinfo["contracts"]

                shim_sig = self._build_shim_signature(
                    cname, sig, params,
                )

                # Format contracts as comments with assigned IDs
                lines.append(f"/* Stub for {cname}: {sig}")
                lines.append(" * Contracts (use the given ID as last arg):")
                for c in ccontracts:
                    kind = c["contract_kind"]
                    target = c["target"]
                    cid = id_counter
                    id_counter += 1
                    id_map.append(f"{cid}: {cname}:{target}:{kind}")
                    if kind == "buffer_size":
                        size_expr = c.get("size_expr", "?")
                        lines.append(
                            f" *   id={cid} {target}: "
                            f"{kind}({size_expr})")
                    else:
                        lines.append(
                            f" *   id={cid} {target}: {kind}")
                lines.append(" */")
                lines.append(shim_sig + " {")
                lines.append("    /* FILL: check pre-conditions (assert_*) "
                             "and establish post-conditions (assume_*) */")
                lines.append("}")
                lines.append("")
        else:
            lines.append("/* No callee stubs needed — all callees are in "
                         "bitcode */")
            lines.append("")

        # --- test() entry ---
        lines.append("/* ---- Entry point ---- */")
        lines.append("/* FILL: Write the complete test() function.")
        lines.append(" *")
        lines.append(" * test() is the entry point for ucsan concolic "
                     "execution.")
        lines.append(" * Its parameters become symbolic inputs.")
        lines.append(" *")

        # List all functions to call in order
        all_call_funcs: list[dict[str, Any]] = []
        if preceding_functions:
            all_call_funcs.extend(preceding_functions)
        all_call_funcs.append({
            "name": func_name,
            "signature": func_signature,
            "params": func_params,
            "postconds": postconds,
            "is_target": True,
        })

        lines.append(" * Call these functions in order:")
        for si, cf in enumerate(all_call_funcs, 1):
            cf_sig = cf["signature"]
            cf_paren = cf_sig.index("(")
            cf_ret = cf_sig[:cf_paren].strip()
            cf_params_str = cf_sig[cf_paren + 1:cf_sig.rindex(")")]
            role = ("function under test" if cf.get("is_target")
                    else "establishes state")
            lines.append(
                f" *   {si}. {cf_ret} {cf['name']}"
                f"({cf_params_str})"
                f"  — {role}")
            cf_postconds = self._format_postcond_comments(
                cf.get("postconds", {}))
            for pc in cf_postconds:
                lines.append(f" *      Post: {pc}")

        # Post-condition IDs for the target function
        postcond_comments = self._format_postcond_comments(postconds)
        if postcond_comments:
            lines.append(" *")
            lines.append(
                " * Assert these post-conditions on the target function:")
            for pc in postcond_comments:
                cid = id_counter
                id_counter += 1
                id_map.append(f"{cid}: {func_name}:post:{pc}")
                lines.append(f" *   id={cid} {pc}")

        lines.append(" *")
        lines.append(" * Rules:")
        lines.append(
            " * - Pointer params: use assume_allocated(input_X, "
            "sizeof(*X), ID) to make them symbolic")
        lines.append(
            " * - Shared params: if multiple functions take the same"
            " struct pointer, use one variable")
        lines.append(
            " * - Return values: wire them if the next function needs"
            " the result")
        lines.append(
            " * - Post-conditions: assert on the target function only,"
            " using the IDs above")
        lines.append(" */")
        lines.append("")

        # Insert ID map comment at the top (after headers, before stubs)
        if id_map:
            map_lines = ["/* Assertion ID map (use these IDs as the last "
                         "arg to assert/assume functions):"]
            for entry in id_map:
                map_lines.append(f" *   {entry}")
            map_lines.append(" */")
            map_lines.append("")
            # Find insert point: after extern declarations, before stubs
            insert_idx = next(
                (idx for idx, ln in enumerate(lines)
                 if "Callee stubs" in ln or "No callee stubs" in ln),
                len(lines),
            )
            for j, ml in enumerate(map_lines):
                lines.insert(insert_idx + j, ml)

        return "\n".join(lines)

    def _build_shim_signature(
        self, name: str, signature: str, params: list[str],
    ) -> str:
        """Build a __shim_ stub function signature from callee info.

        Plain C signature (no dfsan_label params) — the instrumentation
        pass provides labels. Named __shim_<name> so ucsan can redirect
        calls from the original function.
        """
        paren = signature.index("(")
        ret_type_raw = signature[:paren].strip()
        params_str = signature[paren + 1:signature.rindex(")")]

        # Resolve types
        param_types = [
            self._resolve_type(t.strip()) for t in params_str.split(",")
        ] if params_str.strip() else []

        parts: list[str] = []
        for ptype, pname in zip(param_types, params, strict=False):
            parts.append(f"{ptype} {pname}")

        params_out = ", ".join(parts)
        ret_type = self._resolve_type(ret_type_raw)

        return f"__attribute__((used)) {ret_type} __shim_{name}({params_out})"

    def _build_test_params(
        self, func_name: str, signature: str, params: list[str],
    ) -> tuple[str, str]:
        """Build test() parameter list and call arguments.

        Returns (test_param_str, call_args_str).
        Pointer params become void *input_X, scalars keep their type.
        """
        paren = signature.index("(")
        params_str = signature[paren + 1:signature.rindex(")")]
        param_types = [t.strip() for t in params_str.split(",")
                       ] if params_str.strip() else []

        ptr_typedefs = self._get_pointer_typedefs()
        test_params: list[str] = []
        call_args: list[str] = []

        for ptype, pname in zip(param_types, params, strict=False):
            if ptype.endswith("*") or ptype in ptr_typedefs:
                test_params.append(f"{ptype} input_{pname}")
                call_args.append(pname)  # local var from malloc
            else:
                c_type = self._resolve_type(ptype)
                test_params.append(f"{c_type} {pname}")
                call_args.append(pname)

        return ", ".join(test_params), ", ".join(call_args)

    def _resolve_type(self, t: str) -> str:
        """Resolve a single type to a valid C type for the shim."""
        t = t.strip()
        if not t or t == "void" or t == "...":
            return t
        is_const = t.startswith("const ")
        bare = t.removeprefix("const ").strip()
        # Pointer types: keep as-is (headers or emitted typedefs provide defs)
        if t.endswith("*"):
            return t
        if bare in self._get_pointer_typedefs():
            return t
        scalar_typedefs = self._get_scalar_typedefs()
        if bare in scalar_typedefs:
            canonical = scalar_typedefs[bare]
            return f"const {canonical}" if is_const else canonical
        if bare not in self._PRIMITIVE_TYPES:
            return "int"
        return t

    def _get_source_only_typedefs(
        self, stub_callees: dict[str, dict[str, Any]] | None,
    ) -> list[str]:
        """Return C definitions for types only defined in .c files.

        Scans all parameter types in stub_callees, finds those whose base
        type exists in the typedefs table only in .c files (not headers),
        and returns their definitions to emit in the shim.
        """
        if not stub_callees:
            return []

        # Collect base type names from all stub signatures
        base_names: set[str] = set()
        ptr_typedefs = self._get_pointer_typedefs()
        for cinfo in stub_callees.values():
            sig = cinfo["signature"]
            paren = sig.index("(")
            ret_raw = sig[:paren].strip()
            params_str = sig[paren + 1:sig.rindex(")")]
            all_types = [ret_raw]
            if params_str.strip():
                all_types.extend(t.strip() for t in params_str.split(","))
            for t in all_types:
                bare = (t.removeprefix("const ").strip()
                        .rstrip("*").rstrip()
                        .removeprefix("struct ").strip())
                if (bare and bare not in self._PRIMITIVE_TYPES
                        and bare not in ptr_typedefs
                        and bare not in self._get_scalar_typedefs()):
                    base_names.add(bare)

        if not base_names:
            return []

        # Check which are source-only and have definitions
        definitions: list[str] = []
        for name in sorted(base_names):
            rows = self.db.conn.execute(
                "SELECT file_path, definition, pp_definition, line_number "
                "FROM typedefs WHERE name = ?", (name,)
            ).fetchall()
            if not rows:
                continue
            if not all(r[0].endswith(".c") for r in rows):
                continue  # available from a header
            # Prefer pp_definition (macro-expanded), then definition
            defn = rows[0][2] or rows[0][1]
            if defn:
                definitions.append(defn.rstrip())
                continue
            # No stored definition — extract from source file
            src_path, line_no = rows[0][0], rows[0][3]
            extracted = self._extract_typedef_from_source(src_path, line_no)
            if extracted:
                definitions.append(extracted)

        return definitions

    @staticmethod
    def _extract_typedef_from_source(
        file_path: str, start_line: int,
    ) -> str | None:
        """Extract a typedef/struct definition from a source file.

        Reads from start_line and collects lines until a closing brace +
        semicolon is found (handles ``typedef struct { ... } name;``).
        """
        from pathlib import Path
        src = Path(file_path)
        if not src.exists() or not start_line:
            return None
        all_lines = src.read_text(errors="replace").splitlines()
        if start_line < 1 or start_line > len(all_lines):
            return None
        collected: list[str] = []
        brace_depth = 0
        for line in all_lines[start_line - 1:]:
            collected.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 and ";" in line:
                break
            if len(collected) > 100:
                return None  # safety limit
        return "\n".join(collected) if collected else None

    @staticmethod
    def _format_postcond_comments(postconds: dict) -> list[str]:
        """Format post-conditions as comment lines for the template."""
        comments: list[str] = []
        for alloc in postconds.get("allocations", []):
            target = ("return value" if alloc.get("returned")
                      else alloc.get("stored_to", "?"))
            size = alloc.get("size_expr", "?")
            may_null = alloc.get("may_be_null", True)
            line = f"ALLOCATES {target} (size: {size})"
            if may_null:
                line += " [may_be_null]"
            cond = alloc.get("condition", "")
            if cond:
                line += f" [when {cond}]"
            comments.append(line)
        for init in postconds.get("inits", []):
            target = init.get("target", "?")
            byte_count = init.get("byte_count", "?")
            line = f"INITIALIZES {target} ({byte_count} bytes)"
            cond = init.get("condition", "")
            if cond:
                line += f" [when {cond}]"
            comments.append(line)
        for free in postconds.get("frees", []):
            target = free.get("target", "?")
            line = f"FREES {target}"
            cond = free.get("condition", "")
            if cond:
                line += f" [when {cond}]"
            comments.append(line)
        return comments

    @staticmethod
    def _extract_c_block(response: str) -> str | None:
        """Extract the first ```c fenced block from a response."""
        match = re.search(r"```c\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    @staticmethod
    def _build_runtime_config(ucsan_config: dict[str, Any]) -> dict[str, Any]:
        """Convert LLM ucsan_config to run_policy.py --config format.

        The LLM outputs flat booleans/ints; this converts to the nested
        structure expected by run_policy.py DEFAULT_CONFIG.
        """
        runtime: dict[str, Any] = {}

        # ucsan_config.termination.loop.threshold
        if "loop_threshold" in ucsan_config:
            runtime["ucsan_config"] = {
                "termination": {
                    "loop": {"threshold": ucsan_config["loop_threshold"]}
                }
            }

        # UCSAN_OPTIONS flags
        opts: dict[str, str] = {}
        for flag in ("checker_nullderef", "checker_ubi", "trace_bounds",
                      "no_upcast", "no_enlarge"):
            if flag in ucsan_config:
                opts[flag] = "1" if ucsan_config[flag] else "0"
        if opts:
            runtime["ucsan_options"] = opts

        # TAINT_OPTIONS flags
        taint: dict[str, str] = {}
        if "solve_ub" in ucsan_config:
            taint["solve_ub"] = "1" if ucsan_config["solve_ub"] else "0"
        if "trace_bounds" in ucsan_config:
            taint["trace_bounds"] = "1" if ucsan_config["trace_bounds"] else "0"
        if taint:
            runtime["taint_options"] = taint

        return runtime

    @staticmethod
    def _apply_issue_checker_filter(
        runtime: dict[str, Any], issue_kind: str,
    ) -> dict[str, Any]:
        """Disable checkers unrelated to the issue being validated.

        Prevents false positives from unrelated checker exits (e.g.
        null_deref exit when validating a buffer_overflow issue).
        """
        # Map issue_kind → which checkers are relevant
        # If not in the map, leave all checkers enabled
        relevant_checkers: dict[str, set[str]] = {
            "null_deref": {"checker_nullderef"},
            "buffer_overflow": {"trace_bounds"},
            "use_after_free": {"trace_bounds"},
            "out_of_bounds": {"trace_bounds"},
            "uninitialized_use": {"checker_ubi"},
        }
        relevant = relevant_checkers.get(issue_kind)
        if relevant is None:
            return runtime

        all_checkers = {"checker_nullderef", "checker_ubi", "trace_bounds"}
        disable = all_checkers - relevant

        opts = runtime.get("ucsan_options", {})
        for checker in disable:
            opts[checker] = "0"
        runtime["ucsan_options"] = opts

        return runtime

    @staticmethod
    def _extract_json_block(response: str) -> dict[str, Any]:
        """Extract the first ```json fenced block as a dict."""
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                result: dict[str, Any] = json.loads(match.group(1))
                return result
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _extract_fix_block(response: str) -> str | None:
        """Extract the first ```fix fenced block from a response."""
        match = re.search(r"```fix\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def _find_failing_function(
        c_code: str, errors: str,
    ) -> tuple[str, str, int, int] | None:
        """Find the function that caused a compile error.

        Parses error line numbers, finds which function body contains them.
        Returns (signature_line, body, body_start_line, body_end_line)
        or None if not found.
        """
        lines = c_code.split("\n")

        # Extract error line numbers from compiler output
        error_lines: set[int] = set()
        for m in re.finditer(r"shim\.c:(\d+):", errors):
            error_lines.add(int(m.group(1)))
        if not error_lines:
            return None

        # Find function boundaries: signature line, { line, matching } line
        func_ranges: list[tuple[str, int, int]] = []  # (sig, body_start, body_end)
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            # Match function definitions (not externs, not comments)
            if (("{" in line or (i + 1 < len(lines) and
                 lines[i + 1].strip() == "{"))
                and not line.startswith("extern ")
                and not line.startswith("/*")
                and not line.startswith(" *")
                and ("void " in line or "int " in line or
                     "unsigned " in line or "char " in line or
                     "size_t " in line or "long " in line)
                and "(" in line):
                # Find the opening brace
                if "{" in line:
                    brace_line = i
                    sig = line[:line.index("{")].rstrip()
                else:
                    sig = line
                    brace_line = i + 1
                # Find matching closing brace
                depth = 0
                for j in range(brace_line, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if depth == 0:
                        # 1-indexed line numbers for error matching
                        func_ranges.append((sig, brace_line + 1, j + 1))
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1

        # Find which function contains an error line
        for sig, start, end in func_ranges:
            for eline in error_lines:
                if start <= eline <= end:
                    # Extract body (lines between { and })
                    body_lines = lines[start:end - 1]  # 0-indexed, skip { and }
                    body = "\n".join(body_lines)
                    return sig, body, start, end
        return None

    @staticmethod
    def _apply_fix(c_code: str, body_start: int, body_end: int,
                   new_body: str) -> str:
        """Replace a function body in c_code.

        body_start/body_end are 1-indexed line numbers (from _find_failing_function).
        body_start points to the { line, body_end points to the } line.
        """
        lines = c_code.split("\n")
        # Keep the { line and } line, replace everything between
        new_lines = (
            lines[:body_start]  # up to and including { line (0-indexed: body_start-1 is {)
            + [new_body]
            + lines[body_end - 1:]  # } line and after
        )
        return "\n".join(new_lines)

    def _extract_source_includes(self, file_path: str) -> list[str]:
        """Extract project-local #include "..." directives from a source file.

        Returns absolute paths for quoted includes resolved relative to the
        source file's directory.
        """
        import re as _re
        src = Path(file_path)
        if not src.exists():
            return []
        includes: list[str] = []
        src_dir = src.parent
        for line in src.read_text(errors="replace").splitlines():
            m = _re.match(r'\s*#\s*include\s+"([^"]+)"', line)
            if m:
                hdr = m.group(1)
                resolved = src_dir / hdr
                if resolved.exists():
                    includes.append(str(resolved.resolve()))
                else:
                    # Try include paths from compile_commands
                    if self.compile_commands:
                        for flag in self.compile_commands.get_compile_flags(
                            file_path,
                        ):
                            if flag.startswith("-I"):
                                inc_dir = Path(flag[2:])
                                candidate = inc_dir / hdr
                                if candidate.exists():
                                    includes.append(
                                        str(candidate.resolve()))
                                    break
        return includes

    def _collect_referenced_types(
        self,
        signature: str,
        contracts: list[dict[str, Any]],
        callee_contracts: dict[str, dict[str, Any]],
    ) -> set[str]:
        """Collect type names referenced by signatures and contracts.

        Parses pointer types from function signatures and maps contract
        targets with '->' to their parameter's declared type.
        """
        types: set[str] = set()
        # Primitives and stdlib types we don't need to extract
        skip = {
            "void", "char", "int", "unsigned", "long", "short", "float",
            "double", "size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t",
            "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "uintptr_t",
            "bool", "FILE",
        }

        def _types_from_sig(sig: str) -> None:
            """Extract non-primitive pointer types from a signature."""
            # sig is like "int(deflate_state *, int, const char *)"
            paren = sig.find("(")
            if paren < 0:
                return
            # Return type
            ret = sig[:paren].strip().rstrip("*").strip()
            if ret and ret not in skip:
                types.add(ret)
            # Param types
            params_str = sig[paren + 1:sig.rfind(")")]
            for part in params_str.split(","):
                part = part.strip().rstrip("*").strip()
                # Remove const/volatile/struct qualifiers
                for qual in ("const ", "volatile ", "struct ", "enum "):
                    part = part.replace(qual, "")
                part = part.strip()
                if part and part not in skip:
                    types.add(part)

        _types_from_sig(signature)
        for info in callee_contracts.values():
            _types_from_sig(info.get("signature", ""))

        return types

    def _find_type_headers(
        self, file_path: str, type_names: set[str],
    ) -> set[str]:
        """Find header files that define the given struct/typedef types.

        Runs clang -E and uses line markers to map struct definitions
        back to their originating header file.
        """
        if not type_names or not self.compile_commands:
            return set()

        from .preprocessor import SourcePreprocessor
        pp = SourcePreprocessor(
            compile_commands=self.compile_commands,
            verbose=self.verbose,
        )
        result = pp.preprocess(file_path)
        if result.error or not result.mappings:
            return set()

        # For each type, find the line that defines it (} type_name ;)
        # and look up which file that line came from via mappings
        headers: set[str] = set()
        source_path = str(Path(file_path).resolve())

        for type_name in type_names:
            # Search for "} type_name ;" pattern in preprocessed lines
            for m in result.mappings:
                line = m.pp_line.strip()
                if (line.startswith("}") and type_name in line
                        and line.endswith(";")):
                    # Check it's a typedef close: "} type_name;"
                    after_brace = line[1:].strip().rstrip(";").strip()
                    # Could be "} type_name" or "} *type_name, type_name"
                    if type_name in after_brace.replace(",", " ").split():
                        orig = str(Path(m.orig_file).resolve())
                        if orig != source_path:
                            headers.add(m.orig_file)
                        if self.verbose:
                            print(f"    Type {type_name} defined in "
                                  f"{m.orig_file}")
                        break

        return headers

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        if not self.log_file:
            return
        import datetime
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [shim generation]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")


