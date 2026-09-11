#!/bin/bash
# --onstart-cmd for a fresh RTX 4090: make the box usable by this pipeline, then say so.
#
# Two jobs, in this order, because the second is useless if the first fails:
#
#   1. Keep sshd able to accept our key. vastai/pytorch images write
#      $HOME/.ssh/authorized_keys with ownership or modes sshd rejects, and the image's own
#      entrypoint can rewrite the file after us - instance 50254121 was lost to exactly that.
#      The fix runs in a loop for the first 5 minutes. `start instance` cannot carry this
#      (the flag is create-only), which is precisely why it belongs here.
#   2. Rebuild venv `nb` from scratch: torch 2.9.1+cu128 and the nvblox_torch 0.0.10 wheel,
#      plus what the run steps import (numpy, open3d), plus rsync for the push/pull.
#
# Ends by writing /workspace/PROVISIONED containing the output of
#   python -c "import nvblox_torch, torch; print(torch.cuda.is_available())"
# so a waiter can poll one file instead of guessing. Absent file = not ready; present file
# whose last line is not "True" = ready but no CUDA, which is a different problem.
#
# IDEMPOTENT: re-running is a no-op once the import check passes. Safe as an onstart (which
# fires on every boot, not only the first) and safe to run again by hand over ssh.
#
# Pins, and why each one (from nvblox_v2/results/versions.json, the validated install):
#   torch 2.9.1+cu128  - the wheel's METADATA pins torch<=2.9.1; on the image's own torch
#                        2.11 the import dies at dlopen with
#                        `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`
#                        because torch changed that symbol's 4th parameter int -> unsigned.
#   numpy 1.26.4       - the wheel wants numpy<1.27.
#   open3d 0.19.0      - fit_floor and small_components import it; 0.19 is what the floor
#                        rule's behaviour was measured against.
# Never install nvblox into the image's /venv/main - hence the dedicated venv.
set -uo pipefail

VENV=${VENV:-/workspace/venvs/nb}
MARKER=${MARKER:-/workspace/PROVISIONED}
LOG=${LOG:-/workspace/provision.log}
WHEEL_BASE=https://github.com/nvidia-isaac/nvblox/releases/download
NVBLOX_VER=${NVBLOX_VER:-0.0.10}
TORCH_VER=${TORCH_VER:-2.9.1}
NUMPY_VER=${NUMPY_VER:-1.26.4}
OPEN3D_VER=${OPEN3D_VER:-0.19.0}

mkdir -p /workspace "$(dirname "$LOG")" 2>/dev/null
exec >>"$LOG" 2>&1
echo "=== onstart_nvblox.sh $(date -Is) ==="

# --- 1. ssh key permissions, in the background, for the first 5 minutes ----------------
(
  for _ in $(seq 1 30); do
    chown -R root:root $HOME/.ssh 2>/dev/null
    chmod 700 $HOME/.ssh 2>/dev/null
    chmod 600 $HOME/.ssh/authorized_keys 2>/dev/null
    sleep 10
  done
) &
echo "ssh-perm-fixer started (pid $!)"

check() { "$VENV/bin/python" -c "import nvblox_torch, torch; print(torch.cuda.is_available())" 2>&1; }

# --- idempotence: already good? then just refresh the marker and stop ------------------
if [ -x "$VENV/bin/python" ]; then
  OUT="$(check)"
  if [ "${OUT##*$'\n'}" = "True" ] || [ "$OUT" = "True" ]; then
    echo "already provisioned: $OUT"
    printf '%s\n' "$OUT" > "$MARKER"
    echo "marker refreshed at $MARKER"
    exit 0
  fi
  echo "venv exists but the import check said: $OUT - rebuilding"
fi

rm -f "$MARKER"

echo "--- 0. environment ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
nvcc --version | tail -2 || true
. /etc/os-release; echo "ubuntu $VERSION_ID"
UBU=${VERSION_ID%%.*}
CUDA_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)
CUDA_MAJOR=${CUDA_MAJOR:-12}
WHEEL="$WHEEL_BASE/v$NVBLOX_VER/nvblox_torch-$NVBLOX_VER%2Bcu${CUDA_MAJOR}ubuntu${UBU}-py3-none-linux_x86_64.whl"
echo "wheel: $WHEEL"

echo "--- 1. system deps ---"
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip libglib2.0-0t64 libgl1 rsync \
  || apt-get install -y -qq python3-venv python3-pip libglib2.0-0 libgl1 rsync

echo "--- 2. venv + torch $TORCH_VER ---"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cu128 \
  || { echo "FATAL: torch install failed"; exit 1; }

echo "--- 3. nvblox_torch $NVBLOX_VER (official prebuilt wheel) ---"
"$VENV/bin/pip" install "$WHEEL" || { echo "FATAL: nvblox wheel install failed"; exit 1; }

echo "--- 4. what the run steps import ---"
"$VENV/bin/pip" install -q "numpy==$NUMPY_VER" "open3d==$OPEN3D_VER" \
  || { echo "FATAL: numpy/open3d install failed"; exit 1; }

echo "--- 5. verify and mark ---"
OUT="$(check)"
echo "check: $OUT"
printf '%s\n' "$OUT" > "$MARKER"
"$VENV/bin/python" - <<'PY' >> "$MARKER" 2>&1 || true
import json, torch, nvblox_torch, numpy, open3d
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(),
                  "nvblox_torch": getattr(nvblox_torch, "__version__", "unknown"),
                  "numpy": numpy.__version__, "open3d": open3d.__version__}))
PY
echo "=== done $(date -Is); marker: $MARKER ==="
cat "$MARKER"
