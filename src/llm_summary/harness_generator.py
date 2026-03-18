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
  ]
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

Output a single ```json fenced block:

```json
{{{{
  "function": "{name}",
  "targets": [
    {{{{
      "type": "assume|boundary_access|callee_contract",
      "description": "what this checks",
      "contract_kind": "not_null|buffer_size|...",
      "target": "parameter name",
      "priority": "high|medium|low"
    }}}}
  ],
  "loop_bound": 3,
  "timeout_ms": 10000
}}}}
```
"""


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
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.ko_clang_path = ko_clang_path
        self.max_fix_attempts = max_fix_attempts
        self.compile_commands = compile_commands
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
        self._check_toolchain()

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
        if self._triage_context is not None:
            real_fns = set(self._triage_context.get("real_functions", []))
            shim_callees = [k for k in callee_contracts if k not in real_fns]
        else:
            shim_callees = list(callee_contracts.keys())

        # Build fill-in template (used for both triage and normal paths)
        template = self._build_fill_template(
            func.name, func.signature or "", func.params or [],
            callee_contracts, postconds, self._triage_context,
            contracts=contracts, file_path=func.file_path,
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

            # Compile-and-fix loop (ko-clang handles instrumentation)
            if self.ko_clang_path:
                ucsan_config = self._build_ucsan_config(
                    func_name, shim_callees,
                    scope_functions=scope_fns,
                )

                for attempt in range(self.max_fix_attempts):
                    ok, errors = self._compile_shim(
                        c_code, ucsan_config, file_path=func.file_path,
                    )
                    if ok:
                        if self.verbose:
                            print("    Shim compiled successfully"
                                  + (f" (after {attempt} fix(es))" if attempt else ""))
                        break

                    self._stats["fix_attempts"] += 1
                    if self.verbose:
                        print(f"    Compile failed (attempt {attempt + 1}/"
                              f"{self.max_fix_attempts}), asking LLM to fix...")

                    # Find which function failed and ask for a patch
                    fail = self._find_failing_function(c_code, errors)
                    if fail:
                        sig, body, bstart, bend = fail
                        fix_prompt = FIX_PROMPT.format(
                            func_signature=sig,
                            func_body=body,
                            errors=errors,
                        )
                    else:
                        # Can't locate failing function — send full
                        # code in a fallback prompt
                        fix_prompt = (
                            "The following C shim failed to compile. "
                            "Fix the errors.\n\n```c\n"
                            + c_code + "\n```\n\n"
                            "Compiler errors:\n```\n"
                            + errors + "\n```\n\n"
                            "Output a single ```c fenced block with "
                            "the complete fixed C code.\n"
                        )

                    # Prepend original prompt for KV cache reuse
                    fix_full = prompt + "\n\n---\n\n" + fix_prompt
                    fix_response = self.llm.complete(fix_full)
                    self._stats["llm_calls"] += 1

                    if self.log_file:
                        self._log_interaction(
                            f"{func_name}_fix{attempt + 1}",
                            f"[COMPILE ERRORS]\n{errors}", fix_response,
                        )

                    # Apply fix: patch or full replacement
                    if fail:
                        fix_body = self._extract_fix_block(fix_response)
                        if fix_body:
                            c_code = self._apply_fix(
                                c_code, bstart, bend, fix_body,
                            )
                        else:
                            # LLM may have returned ```c block instead
                            fixed = self._extract_c_block(fix_response)
                            if fixed:
                                c_code = fixed
                    else:
                        fixed = self._extract_c_block(fix_response)
                        if fixed:
                            c_code = fixed
                else:
                    if self.verbose:
                        print(f"    Failed to fix after {self.max_fix_attempts} attempts")

            # Generate scheduling policy (separate LLM call)
            schedule_response = self.llm.complete(SCHEDULE_PROMPT.format(
                name=func_name,
                shim_code=c_code,
                contracts_section=contracts_section,
                callee_section=callee_section,
            ))
            self._stats["llm_calls"] += 1
            policy = self._extract_json_block(schedule_response)

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

                build_script = self._build_script(
                    func_name, out, bc_file, file_path=func.file_path,
                    source_files=source_files,
                )
                script_path = out / f"build_{func_name}.sh"
                script_path.write_text(build_script)
                script_path.chmod(0o755)

                if self.verbose:
                    print(f"    Wrote: {shim_path}")
                    print(f"    Wrote: {policy_path}")
                    print(f"    Wrote: {config_path}")
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

            # Write plan
            plan_path = out / f"plan_{plan_name}_validation.json"
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

        # Determine source files to compile
        srcs = source_files or ([file_path] if file_path else [])

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
                    f'KO_TRACE_BB=1 KO_DUMP_CFG="$CFG" \\\n'
                    f'    "$KO_CLANG" -c -g -fno-inline-functions \\\n'
                    f'    {compile_flags_str}"{src}" -o "{obj_path}"'
                )

            src_steps = "\n".join(src_compile_steps)
            link_objs = " ".join(f'"{o}"' for o in obj_files)

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
echo "[1/3] Compiling shim..."
METADATA="$CONFIG" KO_CC=clang-14 \\
    "$KO_CLANG" -c \\
    {include_flags_str}"$SHIM" -o "$SHIM.o"

# Step 2: Compile project source (with BB tracing)
echo "[2/3] Compiling project source..."
rm -f "$CFG"
{src_steps}

# Step 3: Link
echo "[3/3] Linking..."
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

        return result

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
    ) -> str:
        """Build a fill-in-the-blank C template for shim generation.

        Generates the complete shim structure with `/* FILL: ... */` markers
        where the LLM needs to add code. Everything else is fixed.
        """
        real_fns = set((triage_context or {}).get("real_functions", []))
        # Only generate stubs for callees NOT in real_functions
        stub_callees = {
            k: v for k, v in callee_contracts.items()
            if k not in real_fns
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
                      " -- ensure allocation (returns new ptr!)")
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

        test_params, call_args = self._build_test_params(
            func_name, func_signature, func_params,
        )

        lines.append(f"void test({test_params}) {{")

        # assume_allocated for pointer params — tells the engine
        # the buffer is allocated with the correct size
        buf_size_map: dict[str, str] = {}
        for c in (contracts or []):
            if c.get("contract_kind") == "buffer_size" and c.get("size_expr"):
                buf_size_map[c["target"]] = c["size_expr"]

        paren = func_signature.index("(")
        param_types = func_signature[paren + 1:func_signature.rindex(")")
                                     ].split(",")
        for ptype, pname in zip(param_types, func_params, strict=False):
            ptype = ptype.strip()
            if ptype.endswith("*") or ptype in self._get_pointer_typedefs():
                size_expr = buf_size_map.get(pname, f"sizeof(*{pname})")
                aid = id_counter
                id_counter += 1
                id_map.append(
                    f"{aid}: test:{pname}:assume_allocated")
                lines.append(
                    f"    {ptype} {pname} = assume_allocated("
                    f"input_{pname}, {size_expr}, {aid});")

        lines.append("")
        ret_type = func_signature[:paren].strip()
        if ret_type != "void":
            c_ret = self._resolve_type(ret_type)
            lines.append(f"    {c_ret} result = {func_name}({call_args});")
        else:
            lines.append(f"    {func_name}({call_args});")

        # Post-conditions with IDs
        lines.append("")
        postcond_comments = self._format_postcond_comments(postconds)
        if postcond_comments:
            lines.append(
                "    /* FILL: verify post-conditions with assert_init / "
                "assert_allocated / assert_cond")
            for pc in postcond_comments:
                cid = id_counter
                id_counter += 1
                id_map.append(f"{cid}: {func_name}:post:{pc}")
                lines.append(f"     *   id={cid} {pc}")
            lines.append("     */")
        else:
            lines.append(
                "    /* FILL: post-condition assertions (if any) */")

        lines.append("}")
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
        # Pointer types: keep as-is (headers provide definitions)
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


