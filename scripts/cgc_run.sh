#!/bin/bash
# CGC benchmark pipeline: extract ground truth, scan, summarize, verify, evaluate.
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
FILTER=""
LIMIT=""
FORCE=""
INCREMENTAL=""
VERBOSE=""
CGC_DIR="/data/csong/cgc/cb-multios"
KAMAIN_BIN="kanalyzer"
FUNC_SCANS_DIR="func-scans/cgc"
GT_FILE="cgc_ground_truth.json"

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --backend <backend> [options]

Run the CGC benchmark pipeline: extract ground truth, scan, summarize, verify, evaluate.

Required:
  --backend NAME       LLM backend (claude, gemini, ollama, llamacpp)

Optional:
  --model NAME         Model override
  --llm-host HOST      Host for local backends
  --llm-port PORT      Port for local backends
  --from-phase N       Start from phase N (0-6), skip earlier phases (default: 0)
  --filter NAME        Only process challenges matching this substring
  --limit N            Limit to at most N challenges
  --cgc-dir PATH       Path to cb-multios directory (default: $CGC_DIR)
  --kamain-bin PATH    Path to KAMain/kanalyzer binary (default: $KAMAIN_BIN)
  --force              Force re-summarize/verify even if cached
  --incremental        Only re-summarize functions with stale callee summaries
  -v, --verbose        Verbose output

Phases:
  0  extract ground truth       (cgc_extract_ground_truth.py)
  1  prepare + scan + callgraph (cgc_prepare.py)
  3  summarize                  (batch_summarize.py)
  4  verify                     (batch_verify.py)
  5  patch re-scan              (cgc_prepare.py --patch)
  6  patched summarize+verify   (llm-summary summarize --incremental)
  7  evaluate                   (cgc_evaluate.py)
EOF
    exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────────
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
        --filter)
            FILTER="$2"; shift 2 ;;
        --limit)
            LIMIT="$2"; shift 2 ;;
        --cgc-dir)
            CGC_DIR="$2"; shift 2 ;;
        --kamain-bin)
            KAMAIN_BIN="$2"; shift 2 ;;
        --force|-f)
            FORCE="--force"; shift ;;
        --incremental)
            INCREMENTAL="--incremental"; shift ;;
        --verbose|-v)
            VERBOSE="--verbose"; shift ;;
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
FILTER_ARGS=""
[[ -n "$FILTER" ]] && FILTER_ARGS="--filter $FILTER"
[[ -n "$LIMIT" ]]  && FILTER_ARGS="$FILTER_ARGS --limit $LIMIT"

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
    echo "  > $*"
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
echo "Pipeline: CGC benchmark"
echo "Backend:  $BACKEND"
[[ -n "$MODEL" ]]    && echo "Model:    $MODEL"
[[ -n "$FILTER" ]]   && echo "Filter:   $FILTER"
[[ -n "$LIMIT" ]]    && echo "Limit:    $LIMIT"
echo "CGC dir:  $CGC_DIR"
echo "From:     phase $FROM_PHASE"
[[ -n "$FORCE" ]]       && echo "Force:       yes"
[[ -n "$INCREMENTAL" ]] && echo "Incremental: yes"
[[ -n "$VERBOSE" ]]     && echo "Verbose:  yes"
echo ""

# ── Phase 0: extract ground truth ────────────────────────────────────────────
run_phase 0 "extract ground truth" \
    python3 scripts/cgc_extract_ground_truth.py \
        --cgc-dir "$CGC_DIR" -o "$GT_FILE" $FILTER_ARGS $VERBOSE

# ── Phase 1: prepare + scan + call graph ─────────────────────────────────────
run_phase 1 "prepare + scan + callgraph" \
    python3 scripts/cgc_prepare.py \
        --cgc-dir "$CGC_DIR" --func-scans-dir "$FUNC_SCANS_DIR" \
        --kamain-bin "$KAMAIN_BIN" \
        $FILTER_ARGS $VERBOSE

# ── Phases 3-7: per-challenge pipeline ───────────────────────────────────────
if [[ 3 -le $FROM_PHASE ]] && [[ $FROM_PHASE -le 7 ]] || [[ $FROM_PHASE -le 3 ]]; then
    echo ""
    echo "=== Phases 3-7: per-challenge summarize + verify + patch + evaluate ==="

    # Collect challenge dirs matching filter
    CHALLENGE_DIRS=()
    for db in "$FUNC_SCANS_DIR"/*/functions.db; do
        [[ -f "$db" ]] || continue
        challenge_name="$(basename "$(dirname "$db")")"

        if [[ -n "$FILTER" ]]; then
            if ! echo "$challenge_name" | grep -qi "$FILTER"; then
                continue
            fi
        fi

        CHALLENGE_DIRS+=("$challenge_name")
    done

    if [[ -n "$LIMIT" ]]; then
        CHALLENGE_DIRS=("${CHALLENGE_DIRS[@]:0:$LIMIT}")
    fi

    TOTAL=${#CHALLENGE_DIRS[@]}
    echo "  Challenges to process: $TOTAL"
    echo ""

    N_OK=0
    N_FAIL=0

    for i in "${!CHALLENGE_DIRS[@]}"; do
        challenge_name="${CHALLENGE_DIRS[$i]}"
        idx=$((i + 1))
        db_path="$FUNC_SCANS_DIR/$challenge_name/functions.db"

        echo "[$idx/$TOTAL] $challenge_name"
        challenge_start=$(date +%s)

        # Phase 3: summarize
        if [[ $FROM_PHASE -le 3 ]]; then
            echo "  summarize..."
            if ! llm-summary summarize \
                --db "$db_path" \
                --type allocation --type free --type init --type memsafe \
                $LLM_ARGS --init-stdlib $FORCE $INCREMENTAL $VERBOSE; then
                echo "  FAILED summarize"
                N_FAIL=$((N_FAIL + 1))
                continue
            fi
        fi

        # Phase 4: verify
        if [[ $FROM_PHASE -le 4 ]]; then
            echo "  verify..."
            if ! llm-summary summarize \
                --db "$db_path" \
                --type verify \
                $LLM_ARGS $FORCE $INCREMENTAL $VERBOSE; then
                echo "  FAILED verify"
                N_FAIL=$((N_FAIL + 1))
                continue
            fi
        fi

        # Phase 5: patch re-scan
        if [[ $FROM_PHASE -le 5 ]]; then
            echo "  patch re-scan..."
            python3 scripts/cgc_prepare.py \
                --cgc-dir "$CGC_DIR" --func-scans-dir "$FUNC_SCANS_DIR" \
                --patch --filter "$challenge_name" $VERBOSE
        fi

        # Phase 6: patched incremental summarize + verify
        patched_db="$FUNC_SCANS_DIR/$challenge_name/functions_patched.db"
        if [[ $FROM_PHASE -le 6 ]] && [[ -f "$patched_db" ]]; then
            echo "  patched summarize (incremental)..."
            llm-summary summarize \
                --db "$patched_db" \
                --type allocation --type free --type init --type memsafe \
                $LLM_ARGS --incremental --init-stdlib $VERBOSE

            echo "  patched verify (incremental)..."
            llm-summary summarize \
                --db "$patched_db" \
                --type verify \
                $LLM_ARGS --incremental $VERBOSE
        fi

        challenge_elapsed=$(( $(date +%s) - challenge_start ))
        echo "  done (${challenge_elapsed}s)"
        N_OK=$((N_OK + 1))

        # Phase 7: evaluate (running total)
        if [[ $FROM_PHASE -le 7 ]]; then
            python3 scripts/cgc_evaluate.py \
                --ground-truth "$GT_FILE" \
                --func-scans-dir "$FUNC_SCANS_DIR" \
                $FILTER_ARGS -o cgc_eval_report.json 2>/dev/null | grep -E "True Pos|False Neg|False Pos|Precision|Recall|F1|Confirmed"
        fi

        echo ""
    done

    echo "=== Per-challenge pipeline done: $N_OK ok, $N_FAIL failed ==="
fi

# ── Final evaluation ─────────────────────────────────────────────────────────
echo ""
echo "=== Final evaluation ==="
python3 scripts/cgc_evaluate.py \
    --ground-truth "$GT_FILE" \
    --func-scans-dir "$FUNC_SCANS_DIR" \
    $FILTER_ARGS $VERBOSE

# ── Done ──────────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "=== CGC benchmark pipeline complete in ${TOTAL_ELAPSED}s ==="
