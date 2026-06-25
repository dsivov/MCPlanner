#!/usr/bin/env bash
# A/B/C policy sweep across 3 SOPs.
#
# Runs sequentially to avoid SQLite write contention. Each invocation captures
# per-session JSONL for downstream analysis.

set -e
cd "$(dirname "$0")/.."

OUTDIR="/tmp/abc_sweep"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"

# Append (resumable) — don't truncate, so a re-run preserves prior cell logs.
SOPS=(
    "seed:credit_card_activation.json"
    "seed:car_insurance_renewal.json"
    "seed:medical_appointment_booking.json"
)
POLICIES=("llm_top1" "llm_topk" "bandit")
SESSIONS=10
MAX_TURNS=6
CONCURRENCY=1  # uvicorn --reload + concurrency>1 caused mid-request worker recycles in prior run

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG" ; }

log "=== A/B/C policy sweep ==="
log "  SOPs:             ${SOPS[*]}"
log "  Policies:         ${POLICIES[*]}"
log "  Sessions/cell:    $SESSIONS"
log "  Max turns:        $MAX_TURNS"
log "  Per-script concurrency: $CONCURRENCY"

for sop in "${SOPS[@]}"; do
    sop_slug=$(echo "$sop" | sed -E 's|seed:||; s|\.json||')
    for pol in "${POLICIES[@]}"; do
        cell="${sop_slug}__${pol}"
        out_file="$OUTDIR/${cell}.jsonl"
        if [ -s "$out_file" ] && [ "$(wc -l < "$out_file")" -ge "$SESSIONS" ]; then
            log "--- $cell --- (already has $(wc -l < "$out_file") lines, skipping)"
            continue
        fi
        log "--- $cell ---"
        .venv/bin/python scripts/run_benchmark.py \
            --sop "$sop" \
            --modes simulate \
            --preset balanced \
            --sessions-per-mode "$SESSIONS" \
            --max-turns "$MAX_TURNS" \
            --concurrency "$CONCURRENCY" \
            --rollout-policy "$pol" \
            --no-router \
            --no-data-prefetch \
            --out "$out_file" \
            >> "$LOG" 2>&1
        log "    -> $out_file"
    done
done

log "=== sweep complete ==="
