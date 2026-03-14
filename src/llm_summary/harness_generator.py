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

from .compile_commands import CompileCommandsDB
from .db import SummaryDB
from .llm.base import LLMBackend


# ---- Shim template: thin C file linked against project bitcode ----

SHIM_TEMPLATE = """\
/* Auto-generated shim for contract-guided concolic execution (SymSan/ucsan) */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef uint32_t dfsan_label;

/* Comparison predicates for __taint_assert */
enum taint_predicate {{
    bveq  = 32,  /* == (unsigned) */
    bvneq = 33,  /* != */
    bvugt = 34,  /* >  (unsigned) */
    bvuge = 35,  /* >= (unsigned) */
    bvult = 36,  /* <  (unsigned) */
    bvule = 37,  /* <= (unsigned) */
    bvsgt = 38,  /* >  (signed) */
    bvsge = 39,  /* >= (signed) */
    bvslt = 40,  /* <  (signed) */
    bvsle = 41,  /* <= (signed) */
}};

extern void __taint_assert(uint64_t v1, dfsan_label l1,
                           uint64_t v2, dfsan_label l2,
                           int predicate);
extern void __taint_check_bounds(dfsan_label addr_label, uintptr_t addr,
                                 dfsan_label size_label, uint64_t size);

/* Post-condition assertion primitives */
extern void __assert_cond(int cond);
extern void __assert_init(const void *ptr, size_t size);
extern void __assert_allocated(const void *ptr);
extern void __assert_freed(const void *ptr);

/* ---- Target function (extern — defined in project bitcode) ---- */
{target_extern}

/* ---- Callee stubs (__dfsw_ wrappers) ---- */
{stubs}

/* ---- Entry (args auto-symbolized by SymSan) ---- */
{test_func}
"""

# ---- LLM prompt for bitcode-based shim ----

SHIM_PROMPT = """\
You are generating a C shim for SymSan/ucsan concolic execution.
The target function is in pre-compiled project bitcode — you do NOT redefine it.
You generate __dfsw_ callee stubs and a test() entry point.

## Target Function

Name: `{name}`
Signature: `{signature}`
Parameters: {params_json}

## Target Function's Contracts (pre-conditions)

{contracts_section}

## Target Function's Post-conditions

{postconds_section}

## Callee Contracts

{callee_section}

## Important: Callee stub convention

Callee stubs receive extra `dfsan_label` arguments — one per original parameter, \
plus a `dfsan_label *ret_label` pointer at the end (ONLY if the function returns non-void).

**CRITICAL**: If the callee returns `void`, do NOT add `ret_label` — dfsan does \
not pass one for void functions, so reading/writing it causes a segfault.

Example for non-void: `int foo(char *buf, size_t len)`:
```c
int foo(char *buf, size_t len,
        dfsan_label buf_label, dfsan_label len_label,
        dfsan_label *ret_label) {{
    __taint_assert((uint64_t)buf, buf_label, 0, 0, bvneq);  // not_null
    __taint_check_bounds(buf_label, (uintptr_t)buf, len_label, len);  // buffer_size
    *ret_label = 0;
    return 0;
}}
```

Example for void: `void bar(void *state, int err, const char *msg)`:
```c
void bar(void *state, int err, const char *msg,
         dfsan_label state_label, dfsan_label err_label, dfsan_label msg_label) {{
    __taint_assert((uint64_t)state, state_label, 0, 0, bvneq);
    // NO ret_label parameter, NO *ret_label = 0
}}
```

Available check primitives:
- `__taint_assert(v1, l1, v2, l2, predicate)` — symbolic constraint. \
Predicates: `bvneq` (!=), `bveq` (==), `bvugt`/`bvuge`/`bvult`/`bvule` (unsigned), \
`bvsgt`/`bvsge`/`bvslt`/`bvsle` (signed). Cast pointers to `(uint64_t)`.
- `__taint_check_bounds(addr_label, (uintptr_t)ptr, size_label, size)` — \
for buffer_size contracts (triggers GEP bounds check)

## Generate the following sections, each in its own fenced block:

### Section 1: Callee stubs (```stubs ... ```)

For each callee that has contracts above, generate a stub with the **original callee name**:
- Each stub receives the original parameters PLUS a `dfsan_label` for each \
parameter AND a `dfsan_label *ret_label` at the end (ONLY if non-void return)
- **If the callee returns void**: NO `ret_label` parameter, NO `*ret_label = 0`
- For `not_null` contracts: \
`__taint_assert((uint64_t)ptr, ptr_label, 0, 0, bvneq);`
- For `buffer_size` contracts: \
`__taint_check_bounds(ptr_label, (uintptr_t)ptr, size_label, size);`
- Set `*ret_label = 0` and return a reasonable default (0, NULL, etc.)
- Use `void *` for ALL struct/opaque/typedef pointer types in the signature
- Do NOT reconstruct struct layouts, do NOT dereference struct fields
- Do NOT define any typedefs or struct definitions
- Keep stubs minimal: just check contracts on direct parameters, then return
- If no callees have contracts, output an empty block

### Section 2: test() function (```test ... ```)

A plain `void test(...)` C function. \
Its arguments are auto-symbolized by SymSan:
- **Pointer parameters** of the target become `void *` arguments in `test()`. \
Inside test(), allocate a fresh buffer and **memcpy** from the symbolic input \
to give it a valid concrete address while preserving symbolic taint on fields:
  ```c
  void test(void *input_ptr, int x) {{
      void *ptr = malloc(256);
      memcpy(ptr, input_ptr, 256);
      target_func(ptr, x);
  }}
  ```
  - For `buffer_size` contracts: use the contract size instead of 256
  - The `malloc + memcpy` pattern is required — without memcpy the struct fields \
are concrete zeros and branches on them won't be symbolic
- **Scalar parameters** become plain C type arguments (`int`, `size_t`, etc.)
- SymSan auto-tracks malloc buffer sizes, no guards or assumes needed
- Call the target function, casting `void *` to the expected type if needed
- **After the call**, assert post-conditions using these primitives:
  - `__assert_cond(expr)` — assert a boolean condition (e.g., return value check)
  - `__assert_init(ptr, size)` — assert ptr is initialized for size bytes
  - `__assert_allocated(ptr)` — assert ptr points to allocated memory
  - `__assert_freed(ptr)` — assert ptr has been freed
- Post-condition assertions may be conditional: \
`if (result == 0) {{ __assert_init(buf, len); }}`
- **ONLY assert post-conditions on values directly accessible in test()**: \
return value, direct pointer args passed to the target. \
**SKIP** any post-condition on struct sub-fields (e.g., `s->a`, `list->next`) \
— these are not accessible from the shim.
- Do NOT call `free()`. Do NOT define `main()`
- Do NOT define any typedefs, structs, or type definitions
- Do NOT try to initialize struct fields — treat all structs as opaque
- Use `void *` for ALL non-primitive pointer types

### Section 3: Scheduling policy (```json ... ```)

```json
{{
  "function": "{name}",
  "targets": [
    {{
      "type": "assume|boundary_access|callee_contract",
      "description": "what this checks",
      "contract_kind": "not_null|buffer_size|...",
      "target": "parameter name",
      "priority": "high|medium|low"
    }}
  ],
  "loop_bound": 3,
  "timeout_ms": 10000
}}
```

Output ONLY the three fenced blocks, no other text.
"""

FIX_PROMPT = """\
The following C shim failed to compile. Fix the errors.

## Current shim code:
```c
{harness_code}
```

## Compiler errors:
```
{errors}
```

## Rules
- Output TWO fixed sections (stubs and test), each in its own fenced block.
- The fixed template (includes, extern target, dfsan_label typedef, \
__symsan_assume, __taint_check_bounds) are NOT your responsibility.
- Use `void *` for any opaque/struct pointer types.
- Do NOT redefine the target function. Do NOT add `main()` or `free()`.
- Output ONLY the two fenced blocks, no explanation.

```stubs
... fixed __dfsw_ stubs ...
```

```test
... fixed test() ...
```
"""


PLAN_PROMPT = """\
You are a concolic execution planner. Given a function's source code annotated \
with BB IDs and branch successors, its memory safety contracts, and callee \
contracts, produce a trace plan that tells the concolic executor which \
edges (transitions between BBs) to target.

## How concolic execution works

The executor starts with an empty input and runs the function symbolically. \
At each conditional branch, the SMT solver can generate a new input ("seed") \
that flips the branch. The scheduler picks which seed to run next.

Each conditional branch is annotated as:
  `/* [BB:X cond <pred> T:Y F:Z] */`
where X is the branch's BB ID, Y is the BB entered when the condition is true, \
and Z is the BB entered when false.

IMPORTANT: Source-level true/false may be INVERTED at the IR level. \
Always use the T: and F: annotations to determine which BB is reached \
for each direction — do NOT assume source-level if/else maps directly.

A branch marked `loop` is a loop back-edge.

## Target Function

Name: `{name}`
Signature: `{signature}`

## Contracts to Assess

{contracts_section}

## Callee Contracts

{callee_section}

## Post-conditions

{postconds_section}

## Annotated Source Code (with BB IDs and branch successors)

```c
{annotated_source}
```

## Your Task

Analyze the code and contracts. For each contract/post-condition, determine \
which execution path exercises it by identifying the **edge** (BB transition) \
that must be taken.

Output a JSON trace plan with edges:

```json
{{
  "function": "{name}",
  "traces": [
    {{
      "goal": "what contract/post-condition this trace assesses",
      "description": "brief explanation of the path",
      "target_edges": [{{"from": 100000, "to": 100001}}],
      "priority": 1
    }}
  ],
  "deprioritize": [
    {{
      "bb_id": 100004,
      "reason": "why this branch is not worth exploring for contract assessment"
    }}
  ]
}}
```

Guidelines:
- `target_edges`: Each edge is `{{"from": X, "to": Y}}` meaning we want execution \
to transition from BB X to BB Y. Use the T:/F: annotations to pick the right \
successor. A trace may have multiple edges if the path requires multiple branches.
- `priority`: 1 = must explore, 2 = nice to have, 3 = low priority.
- `deprioritize`: branches that are redundant for contract assessment \
(e.g., deeper loop iterations, arithmetic branches unrelated to memory safety).
- Focus on paths that exercise **different** contract-relevant behaviors \
(e.g., different buffer access patterns, null vs non-null paths, error vs success).
- Don't plan traces for unreachable paths or paths that test the exact same \
contract clause as another trace.
- Loops: typically one iteration is enough to test bounds. Mark loop back-edges \
as deprioritize unless the access pattern changes across iterations.

Output ONLY the JSON block, no other text.
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

        # Build target extern declaration
        # func.signature is like "int(gzFile, int)" — insert function name
        # and replace opaque/typedef pointer types with void *
        target_extern = self._build_extern_decl(func.name, func.signature)

        # Gather post-conditions
        postconds = self._gather_postconditions(func.id)

        # Build prompt
        contracts_section = self._format_contracts(contracts)
        callee_section = self._format_callee_contracts(callee_contracts)
        postconds_section = self._format_postconditions(postconds)

        prompt = SHIM_PROMPT.format(
            name=func.name,
            signature=func.signature,
            params_json=json.dumps(func.params),
            contracts_section=contracts_section,
            callee_section=callee_section,
            postconds_section=postconds_section,
        )

        try:
            if self.verbose:
                print(f"  Generating shim for: {func_name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(func_name, prompt, response)

            # Parse LLM response
            stubs, test_func, policy = self._parse_response(response, func_name)

            # Post-process: add __dfsw_ prefix to callee stub function names
            stubs = self._add_dfsw_prefix(stubs, callee_contracts)
            # Post-process: fix void stubs that incorrectly have ret_label
            stubs = self._fix_void_ret_labels(stubs, callee_contracts)

            # Assemble shim
            c_code = SHIM_TEMPLATE.format(
                target_extern=target_extern,
                stubs=stubs,
                test_func=test_func,
            )

            # Compile-and-fix loop (shim via clang-14 -> opt-14 -> llc-14)
            if self.symsan_dir:
                ucsan_config = self._build_ucsan_config(func_name)

                for attempt in range(self.max_fix_attempts):
                    ok, errors = self._compile_shim(c_code, ucsan_config)
                    if ok:
                        if self.verbose:
                            print(f"    Shim compiled successfully"
                                  + (f" (after {attempt} fix(es))" if attempt else ""))
                        break

                    self._stats["fix_attempts"] += 1
                    if self.verbose:
                        print(f"    Compile failed (attempt {attempt + 1}/"
                              f"{self.max_fix_attempts}), asking LLM to fix...")

                    fix_response = self.llm.complete(FIX_PROMPT.format(
                        harness_code=c_code,
                        errors=errors,
                    ))
                    self._stats["llm_calls"] += 1

                    if self.log_file:
                        self._log_interaction(
                            f"{func_name}_fix{attempt + 1}",
                            f"[COMPILE ERRORS]\n{errors}", fix_response,
                        )

                    stubs = self._extract_block(fix_response, "stubs") or stubs
                    test_func = self._extract_block(fix_response, "test") or test_func

                    c_code = SHIM_TEMPLATE.format(
                        target_extern=target_extern,
                        stubs=stubs,
                        test_func=test_func,
                    )
                else:
                    if self.verbose:
                        print(f"    Failed to fix after {self.max_fix_attempts} attempts")

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
                ucsan_config = self._build_ucsan_config(func_name)
                config_path.write_text(ucsan_config)

                # Write abilist files (always — even leaf functions need them)
                if self.symsan_dir:
                    # Shim abilist (same file for both ucsan and taint passes)
                    shim_abl = out / f"shim_abilist_{func_name}.txt"
                    shim_abl.write_text(self._build_shim_abilist())

                    # Target ucsan abilist (standard + callees as taint)
                    target_ucsan_abl = out / f"target_ucsan_abilist_{func_name}.txt"
                    target_ucsan_abl.write_text(
                        self._build_target_ucsan_abilist(callee_contracts))

                    # Target taint abilist (callees as uninstrumented+custom)
                    target_taint_abl = out / f"target_taint_abilist_{func_name}.txt"
                    taint_abl_content = self._build_target_taint_abilist(callee_contracts)
                    target_taint_abl.write_text(taint_abl_content or "# No callee contracts\n")

                # Write build script
                build_script = self._build_script(func_name, out, bc_file)
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
            extract_bbids, format_annotated_source, parse_cfg_dump,
        )

        out = Path(output_dir)

        # Look up function
        funcs = self.db.get_function_by_name(func_name)
        if not funcs:
            if self.verbose:
                print(f"  Function not found: {func_name}")
            return None
        func = funcs[0]

        # Resolve source file
        if not source_file and func.file_path:
            source_file = func.file_path
        if not source_file:
            if self.verbose:
                print(f"  No source file for: {func_name}")
            return None

        # Compile to BC with -O1 -g for debug info + loop detection
        if not bc_file and self.compile_commands and source_file:
            bc_file = self._compile_to_bc_debug(source_file, out)
        if not bc_file:
            if self.verbose:
                print(f"  No bitcode available for: {func_name}")
            return None

        # Run UCSanPass to get BB ID annotations and CFG dump
        result_path = self._instrument_for_bbids(func_name, bc_file, out)
        if not result_path:
            if self.verbose:
                print(f"  Failed to instrument for BB IDs: {func_name}")
            return None

        # Extract BB IDs from IR (for source line mapping)
        ir_path = out / f"bbids_{func_name}.ll"
        source_dir = str(Path(source_file).parent)
        infos = extract_bbids(str(ir_path), source_dir)

        # Merge successor info from CFG dump (authoritative T/F mapping)
        cfg_path = out / f"cfg_{func_name}.txt"
        if cfg_path.exists():
            cfg = parse_cfg_dump(str(cfg_path))
            cfg_bbs = {e["bb_id"]: e for e in cfg.get(func_name, [])}
            for info in infos:
                if info.bb_id in cfg_bbs:
                    entry = cfg_bbs[info.bb_id]
                    if entry["type"] == "C":
                        info.is_conditional = True
                        info.true_bb_id = entry["true_bb"]
                        info.false_bb_id = entry["false_bb"]
            if self.verbose:
                print(f"  Loaded CFG: {len(cfg_bbs)} BBs from {cfg_path.name}")

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

        # Build prompt
        prompt = PLAN_PROMPT.format(
            name=func.name,
            signature=func.signature,
            contracts_section=self._format_contracts(contracts),
            callee_section=self._format_callee_contracts(callee_contracts),
            postconds_section=self._format_postconditions(postconds),
            annotated_source=annotated,
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
                plan = json.loads(json_match.group(1))
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

    def _compile_to_bc_debug(self, source_file: str, output_dir: Path) -> str | None:
        """Compile source to BC with -O1 -g (for loop detection + debug info)."""
        if not self.compile_commands:
            return None

        source_path = Path(source_file)
        if not source_path.exists():
            return None

        flags = self.compile_commands.get_compile_flags(source_file)
        if not flags:
            return None

        # Filter and force -O1 -g
        filtered = []
        for f in flags:
            if f.startswith(("-flto", "-save-temps", "-g", "-O")):
                continue
            filtered.append(f)

        bc_name = source_path.stem + "_dbg.bc"
        bc_path = output_dir / bc_name

        cmd = ["clang-14"] + filtered + [
            "-O1", "-g",
            "-emit-llvm", "-c", str(source_path), "-o", str(bc_path),
        ]

        if self.verbose:
            print(f"    Compiling with -O1 -g: {source_path.name} -> {bc_name}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            if self.verbose:
                print(f"    Debug BC compilation failed:\n{result.stderr[:500]}")
            return None

        return str(bc_path)

    def _instrument_for_bbids(
        self, func_name: str, bc_file: str, output_dir: Path,
    ) -> str | None:
        """Run UCSanPass only on BC to get BB ID annotations and CFG dump.

        Returns path to the IR file, or None on failure.
        CFG dump is written as a side-effect to cfg_{func_name}.txt.
        """
        if not self.symsan_dir:
            return None

        ucsan_pass = self.symsan_dir / "lib" / "symsan" / "UCSanPass.so"
        # Use target ucsan abilist if it exists, otherwise standard
        target_abilist = output_dir / f"target_ucsan_abilist_{func_name}.txt"
        if not target_abilist.exists():
            target_abilist = self.symsan_dir / "lib" / "symsan" / "ucsan_abilist.txt"

        config_path = output_dir / f"config_{func_name}.yaml"
        ir_path = output_dir / f"bbids_{func_name}.ll"
        cfg_path = output_dir / f"cfg_{func_name}.txt"

        env = {
            "METADATA": str(config_path),
            "KO_USE_THOROUPY": "1",
            "KO_CC": "clang-14",
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
        }

        cmd = [
            "opt-14",
            "-load", str(ucsan_pass),
            f"-load-pass-plugin={ucsan_pass}",
            f"-ucsan-abilist={target_abilist}",
            "-ucsan-with-taint=true",
            "-ucsan-trace-bb",
            f"-ucsan-dump-cfg={cfg_path}",
            "-passes=ucsan",
            "-S", "-disable-verify",
            bc_file, "-o", str(ir_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env,
        )
        if result.returncode != 0:
            if self.verbose:
                print(f"    UCSan instrumentation failed:\n{result.stderr[:500]}")
            return None

        return str(ir_path)

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

    def _build_extern_decl(self, name: str, signature: str) -> str:
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
            # Explicit pointer to non-primitive type
            if t.endswith("*"):
                base = t[:-1].strip().removeprefix("const ").strip()
                if base not in self._PRIMITIVE_TYPES and base != "void":
                    if is_const:
                        return "const void *"
                    return "void *"
                return t
            # Typedef that is actually a pointer (e.g. gzFile, z_streamp)
            if bare in ptr_typedefs:
                return "void *"
            # Scalar typedef (e.g. uLong -> unsigned long)
            if bare in scalar_typedefs:
                canonical = scalar_typedefs[bare]
                return f"const {canonical}" if is_const else canonical
            return t

        if params_str.strip():
            params = [replace_type(p) for p in params_str.split(",")]
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

    def _build_ucsan_config(self, func_name: str) -> str:
        """Build ucsan config YAML (entry + scope only, no files key)."""
        lines = ["entry: test", "scope:", f"  - {func_name}"]
        return "\n".join(lines) + "\n"

    def _build_shim_abilist(self) -> str:
        """Build abilist for shim compilation (used by both UCSanPass and TaintPass).

        Marks __dfsw_* stubs and __taint_* runtime functions as uninstrumented
        so both passes leave their bodies intact.
        """
        return (
            "# Shim abilist: preserve __dfsw_ stub and __taint_* bodies\n"
            "fun:__dfsw_*=uninstrumented\n"
            "fun:__taint_*=uninstrumented\n"
            "fun:malloc=custom\n"
            "fun:calloc=custom\n"
            "fun:realloc=custom\n"
        )

    def _build_target_ucsan_abilist(self, callee_contracts: dict[str, dict]) -> str:
        """Build ucsan abilist for target BC compilation.

        Merges the standard ucsan_abilist.txt with callee entries marked as
        taint so UCSanPass treats them as WK_TaintCustom (preserves bodies
        for TaintPass to handle __dfsw_ wrapping).
        """
        if not self.symsan_dir:
            return ""
        ucsan_path = self.symsan_dir / "lib" / "symsan" / "ucsan_abilist.txt"
        if not ucsan_path.exists():
            return ""

        lines = ucsan_path.read_text().splitlines()
        if callee_contracts:
            lines.append("")
            lines.append("# Contract-guided callee stubs")
            for name in callee_contracts:
                lines.append(f"fun:{name}=taint")

        return "\n".join(lines) + "\n"

    def _build_target_taint_abilist(self, callee_contracts: dict[str, dict]) -> str:
        """Build taint abilist for target BC compilation.

        Callee entries are uninstrumented+custom so TaintPass rewrites calls
        to __dfsw_ variants (resolved by shim stubs at link time).
        """
        if not callee_contracts:
            return ""
        lines = ["# Contract-guided callee stubs (uninstrumented+custom → __dfsw_)"]
        for name in callee_contracts:
            lines.append(f"fun:{name}=uninstrumented")
            lines.append(f"fun:{name}=custom")
        return "\n".join(lines) + "\n"

    def _build_script(
        self, func_name: str, out_dir: Path, bc_file: str | None,
    ) -> str:
        """Generate a shell build script for the full pipeline.

        Two compilation contexts with different abilists:
        - Shim: __dfsw_*=uninstrumented, __taint_*=uninstrumented (preserve stub bodies)
        - Target BC: callees=taint (ucsan), callees=uninstrumented+custom (taint)
        """
        symsan_dir = self.symsan_dir or Path("$SYMSAN_DIR")
        ko_clang = self.ko_clang_path or f"{symsan_dir}/bin/ko-clang"
        ucsan_pass = f"{symsan_dir}/lib/symsan/UCSanPass.so"
        taint_pass = f"{symsan_dir}/lib/symsan/TaintPass.so"
        dfsan_abilist = f"{symsan_dir}/lib/symsan/dfsan_abilist.txt"

        config_file = out_dir / f"config_{func_name}.yaml"
        shim_file = out_dir / f"shim_{func_name}.c"
        shim_abilist = out_dir / f"shim_abilist_{func_name}.txt"
        target_ucsan_abilist = out_dir / f"target_ucsan_abilist_{func_name}.txt"
        target_taint_abilist = out_dir / f"target_taint_abilist_{func_name}.txt"

        bc = bc_file or "$1"

        script = f"""\
#!/bin/bash
set -e

# Build harness for {func_name}
# Usage: {out_dir}/build_{func_name}.sh [path/to/project.bc]

SYMSAN_DIR="{symsan_dir}"
KO_CLANG="{ko_clang}"
UCSAN_PASS="{ucsan_pass}"
TAINT_PASS="{taint_pass}"
DFSAN_ABILIST="{dfsan_abilist}"
SHIM_ABILIST="{shim_abilist}"
TARGET_UCSAN_ABILIST="{target_ucsan_abilist}"
TARGET_TAINT_ABILIST="{target_taint_abilist}"
CONFIG="{config_file}"
SHIM="{shim_file}"
BC="${{1:-{bc}}}"
OUT="{out_dir}/{func_name}.ucsan"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Step 1: Compile shim via clang-14 -> opt-14 -> llc-14
# Shim abilist: __dfsw_*=uninstrumented, __taint_*=uninstrumented
echo "[1/5] Compiling shim to bitcode..."
clang-14 -emit-llvm -c "$SHIM" -o "$TMPDIR/shim.bc"

echo "[2/5] Instrumenting shim (UCSanPass + TaintPass)..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
opt-14 \\
    -load "$UCSAN_PASS" -load-pass-plugin="$UCSAN_PASS" \\
    -ucsan-abilist="$SHIM_ABILIST" \\
    -ucsan-with-taint=true \\
    -ucsan-trace-bb \\
    -load "$TAINT_PASS" -load-pass-plugin="$TAINT_PASS" \\
    -taint-abilist="$SHIM_ABILIST" \\
    -taint-with-ucsan=true \\
    -passes=ucsan,taint \\
    -S -disable-verify \\
    "$TMPDIR/shim.bc" -o "$TMPDIR/shim.s"

echo "[3/5] Compiling shim to object..."
llc-14 -filetype=obj --relocation-model=pic -o "$TMPDIR/shim.o" "$TMPDIR/shim.s"

# Step 2: Instrument target BC
# ucsan: callees=taint (preserve for TaintPass); taint: callees=uninstrumented+custom (__dfsw_)
echo "[4/5] Instrumenting project bitcode..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
opt-14 \\
    -load "$UCSAN_PASS" -load-pass-plugin="$UCSAN_PASS" \\
    -ucsan-abilist="$TARGET_UCSAN_ABILIST" \\
    -ucsan-with-taint=true \\
    -ucsan-trace-bb \\
    -load "$TAINT_PASS" -load-pass-plugin="$TAINT_PASS" \\
    -taint-abilist="$DFSAN_ABILIST" \\
    -taint-abilist="$TARGET_TAINT_ABILIST" \\
    -taint-with-ucsan=true \\
    -passes=ucsan,taint \\
    -S -disable-verify \\
    "$BC" -o "$TMPDIR/target.s"

llc-14 -filetype=obj --relocation-model=pic -o "$TMPDIR/target.o" "$TMPDIR/target.s"

# Step 3: Link with ko-clang
echo "[5/5] Linking..."
METADATA="$CONFIG" KO_USE_THOROUPY=1 KO_CC=clang-14 \\
    "$KO_CLANG" "$TMPDIR/target.o" "$TMPDIR/shim.o" -o "$OUT"

echo "Built: $OUT"
"""
        return script

    def _compile_shim(
        self, c_code: str, ucsan_config: str,
    ) -> tuple[bool, str]:
        """Compile shim via clang-14 -> opt-14 -> llc-14 pipeline.

        Both UCSanPass and TaintPass use a minimal abilist that marks
        __dfsw_* and __taint_* as uninstrumented, preserving stub bodies.

        Returns (success, error_output).
        """
        if not self.symsan_dir:
            return False, "symsan_dir not set"

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = f"{tmpdir}/shim.c"
            bc_path = f"{tmpdir}/shim.bc"
            ir_path = f"{tmpdir}/shim.s"
            obj_path = f"{tmpdir}/shim.o"
            cfg_path = f"{tmpdir}/config.yaml"
            abilist_path = f"{tmpdir}/shim_abilist.txt"

            with open(src_path, "w") as f:
                f.write(c_code)
            with open(cfg_path, "w") as f:
                f.write(ucsan_config)
            with open(abilist_path, "w") as f:
                f.write(self._build_shim_abilist())

            ucsan_pass = self.symsan_dir / "lib" / "symsan" / "UCSanPass.so"
            taint_pass = self.symsan_dir / "lib" / "symsan" / "TaintPass.so"

            env = {
                "METADATA": cfg_path,
                "KO_USE_THOROUPY": "1",
                "KO_CC": "clang-14",
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
            }

            # Step 1: clang-14 -> bitcode
            r = subprocess.run(
                ["clang-14", "-emit-llvm", "-c", src_path, "-o", bc_path],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return False, (r.stderr + r.stdout).strip()

            # Step 2: opt-14 with UCSanPass + TaintPass
            # Same abilist for both passes: __dfsw_*=uninstrumented, __taint_*=uninstrumented
            opt_cmd = [
                "opt-14",
                "-load", str(ucsan_pass),
                f"-load-pass-plugin={ucsan_pass}",
                f"-ucsan-abilist={abilist_path}",
                "-ucsan-with-taint=true",
                "-ucsan-trace-bb",
                "-load", str(taint_pass),
                f"-load-pass-plugin={taint_pass}",
                f"-taint-abilist={abilist_path}",
                "-taint-with-ucsan=true",
                "-passes=ucsan,taint",
                "-S", "-disable-verify",
                bc_path, "-o", ir_path,
            ]
            r = subprocess.run(
                opt_cmd, capture_output=True, text=True, timeout=30, env=env,
            )
            if r.returncode != 0:
                return False, (r.stderr + r.stdout).strip()

            # Step 3: llc-14 -> object
            r = subprocess.run(
                ["llc-14", "-filetype=obj", "--relocation-model=pic",
                 "-o", obj_path, ir_path],
                capture_output=True, text=True, timeout=30,
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

    def _fix_void_ret_labels(self, stubs: str, callee_contracts: dict[str, dict]) -> str:
        """Remove ret_label from __dfsw_ stubs for void-returning callees.

        dfsan does not pass ret_label for void functions — having it causes
        the stub to read garbage from the stack and segfault on *ret_label = 0.
        """
        if not stubs or not callee_contracts:
            return stubs
        for name, info in callee_contracts.items():
            sig = info.get("signature", "")
            # Check if return type is void
            paren_idx = sig.find("(")
            if paren_idx < 0:
                continue
            ret_type = sig[:paren_idx].strip()
            if ret_type != "void":
                continue
            # Remove ", dfsan_label *ret_label" or ",\n  dfsan_label *ret_label"
            # from the __dfsw_ stub signature
            stubs = re.sub(
                rf'((?:__dfsw_)?{re.escape(name)}\s*\([^)]*?)'
                r',\s*\n?\s*dfsan_label\s*\*\s*ret_label\)',
                r'\1)',
                stubs,
            )
            # Remove "*ret_label = 0;" lines
            stubs = re.sub(
                r'\s*\*ret_label\s*=\s*0;\s*\n',
                '\n',
                stubs,
            )
        return stubs

    def _add_dfsw_prefix(self, stubs: str, callee_contracts: dict[str, dict]) -> str:
        """Post-process stubs: add __dfsw_ prefix and __attribute__((used))."""
        if not stubs or not callee_contracts:
            return stubs
        for name in callee_contracts:
            # Add __dfsw_ prefix and __attribute__((used)) to prevent DCE
            # Match return type + function name at definition
            stubs = re.sub(
                rf'^(\w[\w\s\*]*?)\s+(?<!__dfsw_){re.escape(name)}\s*\(',
                rf'__attribute__((used)) \1 __dfsw_{name}(',
                stubs,
                flags=re.MULTILINE,
            )
        return stubs

    def _gather_postconditions(self, func_id: int) -> dict:
        """Gather post-conditions from allocation, init, and free summaries."""
        result = {"allocations": [], "inits": [], "frees": []}

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
            line = f"- ALLOCATES: `{target}` (size: {size}, may_be_null: {alloc.get('may_be_null', True)})"
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

    def _parse_response(
        self, response: str, func_name: str,
    ) -> tuple[str, str, dict]:
        """Parse LLM response into (stubs, test_func, policy)."""
        stubs = self._extract_block(response, "stubs") or ""
        test_func = self._extract_block(response, "test") or ""

        # Fallback: try generic ```c blocks
        if not stubs and not test_func:
            c_blocks = re.findall(r"```c\s*(.*?)\s*```", response, re.DOTALL)
            if len(c_blocks) >= 2:
                stubs = c_blocks[0]
                test_func = c_blocks[1]
            elif len(c_blocks) == 1:
                test_func = c_blocks[0]

        # Extract JSON policy
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                policy = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                policy = {"function": func_name, "targets": [], "error": "parse_failed"}
        else:
            policy = {"function": func_name, "targets": [], "error": "no_json_block"}

        return stubs, test_func, policy

    def _extract_block(self, response: str, block_name: str) -> str | None:
        """Extract content from a ```<block_name> ... ``` fenced block."""
        pattern = rf"```{re.escape(block_name)}\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1) if match else None

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
