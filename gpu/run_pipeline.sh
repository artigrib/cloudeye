#!/bin/bash
# The single entrypoint for one pipeline job: video -> keyframes -> vocabulary ->
# MapAnything inference -> floor alignment -> SAM3 objects -> occupancy grid -> GLB.
#
# Usage: run_pipeline.sh <job_dir>
#
# Never held open over a live SSH session - the backend starts this fire-and-forget
# (`nohup setsid bash run_pipeline.sh <job_dir> > <job_dir>/logs/run_pipeline.log 2>&1 &`)
# and polls status.json separately (see app/services/gpu_client.py). set -u (not -e) so
# every stage's exit code is handled explicitly rather than the script dying silently
# partway through a stage's own error handling.
set -u
exec < /dev/null

JOB_DIR="${1:?usage: run_pipeline.sh <job_dir>}"
cd "$JOB_DIR" || exit 2

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$JOB_DIR/logs"
echo $$ > "$JOB_DIR/pipeline.pid"

STAGE_TIMEOUT="${STAGE_TIMEOUT:-900}"
N_STAGES=7

# WORKSPACE_DIR mirrors app.config.settings.gpu_workspace_dir - GPU_SSH_HOST and
# GPU_WORKSPACE_DIR together are the entire "switch GPU provider" surface. Default to
# deriving it from this script's own location ($PIPELINE_DIR is $WORKSPACE_DIR/pipeline)
# rather than hardcoding /workspace, since a non-Vast host (e.g. a non-root GCE user)
# won't have /workspace at all. GPU_WORKSPACE_DIR is accepted as an override for the
# rare case this script is invoked with the pipeline checked out somewhere unexpected,
# but note start_pipeline() in gpu_client.py never sets it - it just cds into job_dir
# and runs this script from its rsync'd location, so the derived default is what
# actually executes in production.
WORKSPACE_DIR="${GPU_WORKSPACE_DIR:-$(cd "$PIPELINE_DIR/.." && pwd)}"
VIDMAP_PY="$WORKSPACE_DIR/envs/vidmap/bin/python"
MAPANYTHING_PY="$WORKSPACE_DIR/envs/mapanything/bin/python"
SAM3_PY="$WORKSPACE_DIR/envs/sam3/bin/python"

# Per-job OpenRouter key, pushed by the backend via SSH stdin into a mode-600 file
# scoped to this job only (never in argv, never logged). Missing is fine - stage_vocab.py
# falls back to the default vocabulary and never fails the job over it.
if [ -f "$JOB_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  . "$JOB_DIR/.env"
  set +a
fi

run_stage() {
  local index="$1" name="$2" venv="$3" script="$4"
  shift 4
  python3 "$PIPELINE_DIR/job_io.py" status "$JOB_DIR" running \
    --stage "$name" --stage-index "$index" --n-stages "$N_STAGES" --message "starting $name"

  local log="$JOB_DIR/logs/$name.log"
  timeout "$STAGE_TIMEOUT" "$venv" "$script" "$@" > "$log" 2>&1 < /dev/null
  local rc=$?
  echo "EXIT:$rc" >> "$log"

  if [ "$rc" -ne 0 ]; then
    local tail_err
    tail_err="$(tail -n 40 "$log" | tr -d '\000')"
    python3 "$PIPELINE_DIR/job_io.py" status "$JOB_DIR" failed \
      --stage "$name" --stage-index "$index" --n-stages "$N_STAGES" \
      --error "stage '$name' exited $rc: $tail_err"
    echo "STAGE $name FAILED (rc=$rc)"
    exit 1
  fi
  echo "stage $name OK"
}

run_stage 1 keyframes  "$VIDMAP_PY"      "$PIPELINE_DIR/stage_keyframes.py"  "$JOB_DIR"
run_stage 2 vocab      "$VIDMAP_PY"      "$PIPELINE_DIR/stage_vocab.py"      "$JOB_DIR"
run_stage 3 infer      "$MAPANYTHING_PY" "$PIPELINE_DIR/stage_infer.py"      "$JOB_DIR"
run_stage 4 align      "$MAPANYTHING_PY" "$PIPELINE_DIR/stage_align.py"      "$JOB_DIR"
run_stage 5 objects    "$SAM3_PY"        "$PIPELINE_DIR/stage_objects.py"    "$JOB_DIR"
run_stage 6 occupancy  "$MAPANYTHING_PY" "$PIPELINE_DIR/stage_occupancy.py"  "$JOB_DIR"
run_stage 7 export_glb "$MAPANYTHING_PY" "$PIPELINE_DIR/stage_export_glb.py" "$JOB_DIR"

python3 "$PIPELINE_DIR/job_io.py" finalize "$JOB_DIR"
python3 "$PIPELINE_DIR/job_io.py" status "$JOB_DIR" done \
  --stage finalize --stage-index "$N_STAGES" --n-stages "$N_STAGES" --message "pipeline complete"

# Best-effort cleanup of the per-job secret - never fail the pipeline over this.
rm -f "$JOB_DIR/.env" 2>/dev/null || true

echo "PIPELINE DONE"
