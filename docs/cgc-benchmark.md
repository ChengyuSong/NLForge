# CGC Benchmark Pipeline

This document describes the Cyber Grand Challenge (CGC) benchmark used to
evaluate the verifier's precision and recall against known vulnerabilities.

## Overview

The [DARPA Cyber Grand Challenge](https://github.com/trailofbits/cb-multios)
`cb-multios` dataset contains hundreds of small C programs, each with a
known vulnerability protected by an `#ifdef PATCHED` guard. This makes it
possible to extract ground-truth bugs automatically and measure how many the
verifier finds (recall) versus how many flagged issues are real (precision).

```
cb-multios challenges
  │  ├─ #ifdef PATCHED  ──▶ ground truth extraction
  │  └─ CWE metadata    ──▶ bug classification
  │
  ▼
cgc_run.sh pipeline
  Phase 0: extract ground truth   (cgc_extract_ground_truth.py)
  Phase 1: scan + call graph      (cgc_prepare.py)
  Phase 3: summarize              (llm-summary summarize)
  Phase 4: verify                 (llm-summary summarize --type verify)
  Phase 5: patch re-scan          (cgc_prepare.py --patch)
  Phase 6: patched summarize+verify
  Phase 7: evaluate               (cgc_evaluate.py)
```

## Prerequisites

- `cb-multios` checked out (default: `/data/csong/cgc/cb-multios`)
- KAMain / kanalyzer binary on `$PATH` or passed via `--kamain-bin`
- LLM backend configured (same as main pipeline)
- venv activated: `source ~/project/llm-summary/venv/bin/activate`

## Running the Benchmark

```bash
source ~/project/llm-summary/venv/bin/activate

scripts/cgc_run.sh \
  --backend claude \
  --model claude-sonnet-4-6 \
  --cgc-dir /data/csong/cgc/cb-multios \
  --verbose
```

### Common Options

| Option | Description |
|--------|-------------|
| `--backend NAME` | LLM backend: claude, gemini, ollama, llamacpp (required) |
| `--model NAME` | Model override |
| `--from-phase N` | Start from phase N (0–7); skip earlier phases |
| `--filter NAME` | Only process challenges matching this substring |
| `--exclude NAME` | Skip challenges matching this substring (repeatable) |
| `--max-functions N` | Skip challenges with more than N functions |
| `--limit N` | Process at most N challenges |
| `--force` | Re-summarize/verify even if cached |
| `--incremental` | Only re-summarize functions with stale callee summaries |
| `--cgc-dir PATH` | Path to cb-multios (default: `/data/csong/cgc/cb-multios`) |
| `--kamain-bin PATH` | Path to KAMain binary |

### Example: run a single challenge

```bash
scripts/cgc_run.sh --backend claude --filter CADET_00003 --verbose
```

### Resume from a later phase

```bash
# Skip extraction and scan; start from summarize
scripts/cgc_run.sh --backend claude --from-phase 3
```

## Pipeline Phases

### Phase 0: Extract Ground Truth (`cgc_extract_ground_truth.py`)

Parses each challenge's C source for `#ifdef PATCHED` / `#ifndef PATCHED`
blocks to identify what code is removed by the patch (i.e., the bug). Also
reads `cb-multios` metadata for CWE classifications.

Output: `cgc_ground_truth.json` — list of challenges with affected functions,
CWE IDs, and patch context.

### Phase 1: Prepare + Scan + Call Graph (`cgc_prepare.py`)

For each challenge:
1. Compiles the challenge with LLVM (`-flto=full -save-temps=obj`) inside a
   Docker container to produce `.bc` bitcode
2. Runs `llm-summary scan` to extract functions and build the AST call graph
3. Runs KAMain (CFL-reachability) to produce a precise call graph and V-snapshot
4. Imports the call graph into `func-scans/cgc/<challenge>/functions.db`

Output: `func-scans/cgc/<challenge>/functions.db` per challenge

### Phase 3: Summarize

Runs all four post/pre-condition passes (allocation, free, init, memsafe) on
each challenge DB using `llm-summary summarize`.

### Phase 4: Verify

Runs verification pass (Pass 5) on each challenge DB. Produces `SafetyIssue`
records in `verification_summaries`.

### Phase 5: Patch Re-scan (`cgc_prepare.py --patch`)

Re-scans the challenge with `-DPATCHED` defined so the buggy code is compiled
out. This produces `functions_patched.db` containing the patched version.

### Phase 6: Patched Summarize + Verify (incremental)

Runs summarization and verification on `functions_patched.db`. Uses
`--incremental` to only re-summarize functions whose source changed due to the
patch. The expectation: issues present in the unpatched DB should disappear
(or have lower severity) in the patched DB — confirming the verifier correctly
attributed the issue to the patched code.

### Phase 7: Evaluate (`cgc_evaluate.py`)

Computes precision, recall, and F1 by comparing:
- **Ground truth**: functions identified by Phase 0 as containing the bug
- **Detected**: functions with `high`-severity issues in `verification_summaries`
- **Patch-confirmed**: issues that disappear after patching (Phase 6)

Outputs `cgc_eval_report.json` and prints a running summary after each challenge:

```
True Positives:  12
False Negatives: 8
False Positives: 3
Precision: 0.80
Recall:    0.60
F1:        0.686
```

## Output Files

| File | Description |
|------|-------------|
| `cgc_ground_truth.json` | Ground-truth bugs extracted from cb-multios |
| `func-scans/cgc/<challenge>/functions.db` | Per-challenge unpatched analysis DB |
| `func-scans/cgc/<challenge>/functions_patched.db` | Per-challenge patched DB |
| `cgc_eval_report.json` | Precision/recall/F1 evaluation report |

## CWE Coverage

The ground truth extractor maps cb-multios challenge metadata to CWE IDs.
The verifier pipeline covers the following CWEs:

| CWE | Class | Verifier pass |
|-----|-------|---------------|
| CWE-119 / CWE-120 / CWE-787 | Buffer overflow | Pass 5 (`buffer_overflow`) |
| CWE-476 | Null dereference | Pass 5 (`null_deref`) |
| CWE-416 | Use-after-free | Pass 5 (`use_after_free`) |
| CWE-415 | Double free | Pass 5 (`double_free`) |
| CWE-457 | Uninitialized use | Pass 5 (`uninitialized_use`) |
| CWE-843 | Type confusion | Pass 5 |
| CWE-129 | Improper array index validation | Pass 5 |
| CWE-134 | Uncontrolled format string | Pass 5 |

## Interpreting Results

- **High recall, low precision**: verifier over-reports; consider tightening
  prompts or adding more post-condition summaries so more issues are ruled out.
- **Low recall**: verifier misses bugs; check if the buggy function was
  summarized (it may have been skipped as a stub) or if the issue type is
  outside the current CWE set.
- **Patch-confirmed rate**: fraction of detected issues that disappear after
  patching — a proxy for true positive rate without manual review.
