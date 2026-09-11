#!/usr/bin/env bash
# VPS-side pull loop. Polls the box for per-job completion markers and rsyncs each finished
# job the moment it appears - one background transfer and one log per job.
#
# NO GLOBAL STOP SIGNAL. The previous version exited as soon as it saw a marker file named
# ALL_DONE, which each stage driver touched on exit. That killed the loop twice in one day:
# once at the end of stage C (stage E's results then sat unpulled on the box) and once
# again after stage E, and each time the loop had to be noticed as dead and restarted by
# hand. A stage finishing is not a reason to stop pulling - another stage may follow.
#
# The loop now ends ONLY on an explicit command:
#   touch <STOPFILE>           (default var/scratch/b200_run_20260908/PULL_STOP)
#   or SIGTERM/SIGINT
# In both cases it waits for in-flight transfers before returning, so a stop never
# truncates a job.
#
# Markers are per job: the box-side driver writes /workspace/markers/<job_id>.done LAST,
# after the job directory is complete. Anything not ending in .done is ignored, so a
# stage-level or bookkeeping marker can never be mistaken for a job.
set -uo pipefail
: "${GPU_HOST:?set GPU_HOST=root@<ip>}"
: "${GPU_PORT:?set GPU_PORT=<ssh port>}"
BASE=${BASE:-var/scratch/b200_run_20260908}
DEST=${DEST:-$BASE/pulled}
LOGS=${LOGS:-$BASE/logs}
KEY=${KEY:-$HOME/.ssh/id_ed25519_vast}
STOPFILE=${STOPFILE:-$BASE/PULL_STOP}
INTERVAL=${INTERVAL:-20}
SSH="ssh -p $GPU_PORT -i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=20"
mkdir -p "$DEST" "$LOGS"
rm -f "$STOPFILE"

stopping=0
trap 'stopping=1; echo "[$(date -Is)] signal received, finishing in-flight transfers"' TERM INT

declare -A pulled
echo "[$(date -Is)] pull loop started; dest=$DEST; stop with: touch $STOPFILE"
while true; do
  markers=$($SSH "$GPU_HOST" 'ls /workspace/markers/*.done 2>/dev/null' 2>/dev/null || true)
  for m in $markers; do
    job=$(basename "$m" .done)
    [[ -n "${pulled[$job]:-}" ]] && continue
    echo "[$(date -Is)] pulling $job"
    rsync -a --partial --info=stats2 -e "$SSH" \
      "$GPU_HOST:/workspace/out/$job/" "$DEST/$job/" \
      > "$LOGS/rsync_$job.log" 2>&1 &
    pulled[$job]=1
  done
  if [[ -f "$STOPFILE" || $stopping -eq 1 ]]; then
    echo "[$(date -Is)] stop requested, waiting for ${#pulled[@]} transfer slots to drain"
    wait
    echo "[$(date -Is)] pull loop finished cleanly, ${#pulled[@]} jobs pulled"
    exit 0
  fi
  sleep "$INTERVAL"
done
