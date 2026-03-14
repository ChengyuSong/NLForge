# Contract-Guided Concolic Execution (ucsan / SymSan)

This document describes the harness generation workflow that bridges LLM-derived
memory safety summaries with concrete dynamic verification via SymSan/ucsan.

## Overview

After the five-pass summarization pipeline produces verification issues, those
issues are LLM-inferred and may be false positives. The `gen-harness` command
generates a thin C shim that exercises the target function under SymSan's
taint-tracking fuzzer (ucsan), allowing concrete path exploration to confirm or
refute each issue.

```
verification_summaries ──▶ gen-harness ──▶ harness (.c, .bc, .sh)
                                │                    │
                           issue_reviews        ucsan / SymSan
                           (confirmed /              │
                           false_positive)      Thoroupy scheduler
                                                     │
                                              assess-issue
                                              (targeted assertion)
```

## Prerequisites

- **SymSan / ucsan**: `ko-clang` binary for instrumented compilation
- **Thoroupy**: policy scheduler at `~/fuzzing/ucsan/thoroupy/`
- **Project bitcode**: pre-compiled `.bc` from Phase 0 (`build-learn`), or
  `compile_commands.json` so `gen-harness` can recompile

## Generating Harnesses

```bash
source ~/project/llm-summary/venv/bin/activate

llm-summary gen-harness \
  --db func-scans/zlib/zlibstatic/functions.db \
  -f gzputc \
  --ko-clang-path ~/fuzzing/symsan/b3/bin/ko-clang \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \
  --project-path /data/csong/opensource/zlib \
  -v
```

Omit `-f` to generate harnesses for **all** functions with memsafe contracts.

### Key options

| Option | Description |
|--------|-------------|
| `--db` | Path to `functions.db` (required) |
| `-f / --function` | Target function(s); omit to process all |
| `--ko-clang-path` | Path to `ko-clang`; enables compilation and error-fix loop |
| `--compile-commands` | `compile_commands.json` for recompiling source to bitcode |
| `--project-path` | Host-side source root (for Docker path remapping) |
| `--build-dir` | Host-side build dir (for Docker build path remapping) |
| `--bc-file` | Pre-compiled project bitcode (skips recompilation) |
| `--plan` | Also generate a Thoroupy trace plan after harness generation |
| `--plan-only` | Only (re)generate the plan; requires existing harness files |
| `--assess-issue N` | Inject assertion for issue index N and rebuild |
| `-o / --output-dir` | Output directory (default: `harnesses/<project>/`) |

### Output files

For a function `<func>`, the following files are written to the output directory:

| File | Purpose |
|------|---------|
| `<func>.c` | C shim with `test()` entry point and `__dfsw_` callee stubs |
| `<func>.bc` | Compiled shim bitcode (when `--ko-clang-path` is set) |
| `<func>.sh` | Build/run script |
| `<func>.ucsan.cfg` | ucsan configuration |
| `<func>.abilist` | DFSan ABI list for shim functions |
| `<func>_target.abilist` | DFSan ABI list for target function |
| `<func>_taint.abilist` | Taint propagation ABI list |
| `<func>.ucsan` | Final instrumented binary |

## How the Shim Works

The LLM generates two components from the function's memsafe contracts and
post-conditions:

1. **`test()` entry point** — allocates inputs, calls the target function, and
   asserts any post-conditions. SymSan taints the inputs and explores paths.

2. **`__dfsw_<callee>` stubs** — thin wrappers for each callee. Stubs return 0
   (success) by default so ucsan explores the continuation paths where memory
   operations happen, rather than early-return error paths.

DFSan label propagation is handled automatically through the ABI lists.

## Plan Generation (Thoroupy)

The `--plan` flag instruments the target binary with basic-block IDs, then asks
the LLM to produce an exploration strategy: an ordered list of BB targets that
represent a path likely to trigger the contract violation.

```bash
# Generate plan only (existing harness already compiled)
llm-summary gen-harness \
  --db func-scans/zlib/zlibstatic/functions.db \
  -f gzputc --plan-only \
  --symsan-dir ~/fuzzing/symsan/b3 \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \
  --project-path /data/csong/opensource/zlib -v
```

The plan is saved as `<func>.plan.json` and consumed by Thoroupy's policy
scheduler. Use `scripts/run_thoroupy.sh` to launch:

```bash
scripts/run_thoroupy.sh harnesses/zlib/gzputc.ucsan harnesses/zlib/gzputc.plan.json
```

Thoroupy logs go to `thoroupy_gzputc.log` next to the plan file.

## Issue Assessment (`--assess-issue`)

To confirm a specific verification issue, inject a targeted assertion into an
existing shim and rebuild:

```bash
# First, view issues for the function
llm-summary show-issues --db func-scans/openssh/scp/functions.db \
  --name sftp_path_append --severity high

# Then assess issue index 0
llm-summary gen-harness \
  --db func-scans/openssh/scp/functions.db \
  -f sftp_path_append \
  --assess-issue 0 \
  --ko-clang-path ~/fuzzing/symsan/b3/bin/ko-clang \
  --compile-commands /data/csong/build-artifacts/openssh/compile_commands.json \
  --project-path /data/csong/opensource/openssh-portable -v
```

The LLM reads the existing shim and the issue description, adds an assertion
that should fire if the issue is real, and recompiles. Run ucsan on the result
to see if it triggers.

## Issue Triage

After running ucsan (manually or via Thoroupy), record the verdict:

```bash
# Mark a false positive
llm-summary review-issue sftp_path_append 0 \
  --status false_positive \
  --reason "short-circuit eval guards this path" \
  --db func-scans/openssh/scp/functions.db

# Mark a confirmed bug
llm-summary review-issue png_set_IHDR 2 \
  --status confirmed \
  --reason "ucsan triggered buffer overflow at line 312" \
  --db func-scans/libpng/libpng16/functions.db
```

### Listing Issues

```bash
# All pending high-severity issues
llm-summary show-issues --db func-scans/openssh/scp/functions.db \
  --severity high --status pending

# All confirmed issues as JSON
llm-summary show-issues --db func-scans/libpng/libpng16/functions.db \
  --status confirmed --format json
```

Review statuses:

| Status | Meaning |
|--------|---------|
| `pending` | Not yet triaged (default) |
| `confirmed` | Verified as a real bug |
| `false_positive` | LLM hallucination or infeasible path |
| `wontfix` | Real but intentionally not addressed |

## Workflow Summary

```
1. summarize --type verify      # Pass 5: find issues
2. show-issues                  # Review what was flagged
3. gen-harness -f <func>        # Generate shim + compile
4. gen-harness --plan-only      # Generate Thoroupy plan
5. run_thoroupy.sh <bin> <plan> # Explore paths
6.   -- or --
   gen-harness --assess-issue N # Inject assertion for issue N
7. review-issue <func> N        # Record verdict
```
