#!/bin/bash
# Single-project pipeline: run all phases for one project from gpr_projects.json.
#
# Make sure you activate the virtual environment first!
# source ~/project/llm-summary/venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND=""
MODEL=""
LLM_HOST=""
LLM_PORT=""
FROM_PHASE=0
SKIP_VERIFY=0
WITH_CONTAINERS=0
PREPROCESS=0
FORCE=""
INCREMENTAL=""
VERBOSE=""
PASSTHROUGH=()

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <project-name> --backend <backend> [options] [-- extra-args...]

Run the full analysis pipeline for a single project.

Positional:
  PROJECT              Project name from gpr_projects.json

Required:
  --backend NAME       LLM backend (claude, gemini, ollama, llamacpp)

Optional:
  --model NAME         Model override
  --llm-host HOST      Host for local backends (ollama, llamacpp)
  --llm-port PORT      Port for local backends
  --from-phase N       Start from phase N (0-6), skip earlier phases (default: 0)
  --skip-build         Shorthand for --from-phase 2 (skip build-learn and discover-link-units)
  --skip-verify        Skip verification phase (phase 5)
  --with-containers    Run container detection phase (phase 6)
  --preprocess         Run clang -E to expand macros (pp_source) during scan
  --force              Force re-summarize even if cached
  --incremental        Only re-summarize functions with stale callee summaries
  -v, --verbose        Verbose output
  --                   Pass remaining args through to batch scripts

Phases:
  0  build-learn          (batch_build_learn.py)
  1  discover-link-units  (batch_discover_link_units.py)
  2  scan targets         (batch_scan_targets.py, incl. extern header extraction)
  3  call graph           (batch_call_graph_gen.py)
  4  summarize            (batch_summarize.py, incl. cross-project import-dep)
  5  verify               (batch_verify.py)
  6  container detect     (batch_container_detect.py, requires --with-containers)
EOF
    exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    usage
fi

PROJECT="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --llm-host)
            LLM_HOST="$2"; shift 2 ;;
        --llm-port)
            LLM_PORT="$2"; shift 2 ;;
        --from-phase)
            FROM_PHASE="$2"; shift 2 ;;
        --skip-build)
            FROM_PHASE=2; shift ;;
        --skip-verify)
            SKIP_VERIFY=1; shift ;;
        --with-containers)
            WITH_CONTAINERS=1; shift ;;
        --preprocess)
            PREPROCESS=1; shift ;;
        --force|-f)
            FORCE="--force"; shift ;;
        --incremental)
            INCREMENTAL="--incremental"; shift ;;
        --verbose|-v)
            VERBOSE="--verbose"; shift ;;
        --)
            shift; PASSTHROUGH=("$@"); break ;;
        *)
            echo "Unknown option: $1"
            usage ;;
    esac
done

if [[ -z "$BACKEND" ]]; then
    echo "ERROR: --backend is required"
    usage
fi

# ── Build arg strings ────────────────────────────────────────────────────────
FILTER_ARGS="--filter $PROJECT --limit 1"

LLM_ARGS="--backend $BACKEND"
[[ -n "$MODEL" ]]    && LLM_ARGS="$LLM_ARGS --model $MODEL"
[[ -n "$LLM_HOST" ]] && LLM_ARGS="$LLM_ARGS --llm-host $LLM_HOST"
[[ -n "$LLM_PORT" ]] && LLM_ARGS="$LLM_ARGS --llm-port $LLM_PORT"

# ── Helpers ───────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)

run_phase() {
    local phase_num="$1"
    local label="$2"
    shift 2

    if [[ $phase_num -lt $FROM_PHASE ]]; then
        echo "--- Phase $phase_num: $label [SKIPPED] ---"
        return 0
    fi

    echo ""
    echo "=== Phase $phase_num: $label ==="
    local start=$(date +%s)

    "$@"
    local rc=$?

    local elapsed=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then
        echo "FAILED phase $phase_num ($label) after ${elapsed}s (exit $rc)"
        exit $rc
    fi
    echo "--- Phase $phase_num done in ${elapsed}s ---"
}

# ── Print config ──────────────────────────────────────────────────────────────
echo "Pipeline: single-project"
echo "Project:  $PROJECT"
echo "Backend:  $BACKEND"
[[ -n "$MODEL" ]]    && echo "Model:    $MODEL"
[[ -n "$LLM_HOST" ]] && echo "LLM host: $LLM_HOST"
[[ -n "$LLM_PORT" ]] && echo "LLM port: $LLM_PORT"
echo "From:     phase $FROM_PHASE"
[[ $SKIP_VERIFY -eq 1 ]] && echo "Verify:   skipped"
[[ $PREPROCESS -eq 1 ]]  && echo "Preproc:  yes"
[[ -n "$FORCE" ]]        && echo "Force:       yes"
[[ -n "$INCREMENTAL" ]] && echo "Incremental: yes"
[[ -n "$VERBOSE" ]]      && echo "Verbose:  yes"
echo ""

# ── Phase 0: build-learn ─────────────────────────────────────────────────────
run_phase 0 "build-learn" \
    python3 scripts/batch_build_learn.py \
        $FILTER_ARGS $LLM_ARGS $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 1: discover-link-units ─────────────────────────────────────────────
run_phase 1 "discover-link-units" \
    python3 scripts/batch_discover_link_units.py \
        $FILTER_ARGS $LLM_ARGS $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 2: scan targets ────────────────────────────────────────────────────
PREPROCESS_ARG=""
[[ $PREPROCESS -eq 1 ]] && PREPROCESS_ARG="--preprocess"

run_phase 2 "scan targets" \
    python3 scripts/batch_scan_targets.py \
        $FILTER_ARGS $PREPROCESS_ARG $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 3: call graph generation ───────────────────────────────────────────
run_phase 3 "call graph" \
    python3 scripts/batch_call_graph_gen.py \
        $FILTER_ARGS --compositional $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 4: summarize ───────────────────────────────────────────────────────
run_phase 4 "summarize" \
    python3 scripts/batch_summarize.py \
        $FILTER_ARGS $LLM_ARGS --init-stdlib $FORCE $INCREMENTAL $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 5: verify ──────────────────────────────────────────────────────────
if [[ $SKIP_VERIFY -eq 0 ]]; then
    run_phase 5 "verify" \
        python3 scripts/batch_verify.py \
            $FILTER_ARGS $LLM_ARGS $FORCE $INCREMENTAL $VERBOSE \
            "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
else
    echo ""
    echo "--- Phase 5: verify [SKIPPED] ---"
fi

# ── Phase 6: container detection (optional) ──────────────────────────────────
if [[ $WITH_CONTAINERS -eq 1 ]]; then
    run_phase 6 "container detection" \
        python3 scripts/batch_container_detect.py \
            $FILTER_ARGS $LLM_ARGS $FORCE $VERBOSE \
            "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
else
    echo ""
    echo "--- Phase 6: container detection [SKIPPED] (use --with-containers) ---"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "=== Pipeline complete for '$PROJECT' in ${TOTAL_ELAPSED}s ==="
