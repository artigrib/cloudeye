#!/bin/bash
# Provision NVIDIA Isaac Sim 6.0.1 (standalone binary, not pip) on the rented GPU box,
# for validating app/services/usd_export.py's output with tools/isaac_validate.py. See
# docs/ISAAC_VALIDATION.md for the full runbook this fits into.
#
#   !!! UNVERIFIED !!!
#   This script has never been run. The download URL was checked reachable (HTTP 200,
#   ~13GB, application/zip) from https://docs.isaacsim.omniverse.nvidia.com/6.0.1/
#   installation/download.html and quick-install.html's extract/post_install steps at
#   authoring time, but nothing past that (post_install.sh's own behavior, the code_editor
#   extension's flags, the exact isaac-sim.sh CLI surface) has been exercised. Record
#   what actually happened in docs/ISAAC_VALIDATION.md's "Verified runs" table once
#   you've run this.
#
# Usage: run directly ON the GPU box (not via this repo's gpu/ rsync path - see
# docs/ISAAC_VALIDATION.md's Step 2 for why tools/ and this script are copied
# separately from gpu/run_pipeline.sh's territory):
#   ssh gpu 'mkdir -p /workspace/deploy'
#   rsync -avz deploy/install_isaacsim_gpu.sh gpu:/workspace/deploy/
#   ssh gpu 'bash /workspace/deploy/install_isaacsim_gpu.sh'
#
# Safety contract:
#   - Installs ONLY under $VOLUME (default /workspace), which must already be a real
#     mounted volume, separate from the container's root disk - resolved via
#     `findmnt`, never assumed from a path string. Aborts if it isn't mounted, or if
#     it resolves to the same device as /, UNLESS ALLOW_ROOT_VOLUME=1 is explicitly
#     set (see resolve_volume below) - opt-in only, for a single-disk throwaway
#     instance with no separate persistent volume attached. Defaults to refusing.
#   - The only path this script ever deletes is $VOLUME/isaacsim/, and only when
#     --reinstall is passed. Everything else (existing cache directories) is migrated
#     (moved), never discarded, before being replaced with a symlink. The downloaded
#     zip is deleted after a successful extraction unless KEEP_ISAACSIM_ZIP=1 is set.
#   - Nothing is written to the container's root disk - not the 13GB download, not the
#     extracted install, not the shader/extension caches.
set -euo pipefail

VOLUME="${VOLUME:-/workspace}"
ALLOW_ROOT_VOLUME="${ALLOW_ROOT_VOLUME:-0}"
KEEP_ISAACSIM_ZIP="${KEEP_ISAACSIM_ZIP:-0}"
# Omniverse/Isaac Sim refuses to run as root by default (most rented GPU instances log
# in as root, with no other user configured) - exported here, and into
# start_code_editor_server.sh below, so this is a one-time script concern, not
# something every future invocation has to remember to set by hand.
export OMNI_KIT_ALLOW_ROOT=1
ISAACSIM_DIR="$VOLUME/isaacsim"
CACHE_DIR="$VOLUME/isaac-cache"
ENV_FILE="$VOLUME/env.sh"
ISAAC_SIM_VERSION="6.0.1"
ZIP_NAME="isaac-sim-standalone-${ISAAC_SIM_VERSION}-linux-x86_64.zip"
ZIP_URL="${ISAAC_SIM_ZIP_URL:-https://downloads.isaacsim.nvidia.com/${ZIP_NAME}}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"  # 13GB download + extracted install + post_install cache
CODE_EDITOR_PORT="${CODE_EDITOR_PORT:-8226}"
REINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --reinstall) REINSTALL=1 ;;
    *) echo "unknown argument: $arg (only --reinstall is accepted)" >&2; exit 2 ;;
  esac
done

log() { echo "[install_isaacsim_gpu] $*"; }
die() { echo "[install_isaacsim_gpu] ERROR: $*" >&2; exit 1; }

command -v findmnt >/dev/null 2>&1 || die "findmnt not found - needed to verify \$VOLUME is a real mounted volume, not a guess"
command -v curl >/dev/null 2>&1 || die "curl not found"
command -v unzip >/dev/null 2>&1 || die "unzip not found"

# --- Resolve $VOLUME from the mount table, never from the path string alone. Refuses
# to proceed onto the container's root disk - a 13GB+ download/extract there is exactly
# the kind of accident this check exists to prevent.
resolve_volume() {
  [ -d "$VOLUME" ] || die "$VOLUME does not exist - mount the persistent volume there before running this script"
  local vol_src root_src
  vol_src=$(findmnt -no SOURCE -T "$VOLUME" 2>/dev/null) || die "$VOLUME is not on any mounted filesystem (findmnt found nothing covering it)"
  root_src=$(findmnt -no SOURCE /)
  [ -n "$vol_src" ] || die "findmnt returned an empty source for $VOLUME - refusing to proceed"
  if [ "$vol_src" = "$root_src" ]; then
    if [ "$ALLOW_ROOT_VOLUME" -eq 1 ]; then
      log "$VOLUME resolves to the same device as / ($root_src) - normally refused, but ALLOW_ROOT_VOLUME=1 was set (single-disk throwaway instance, no separate persistent volume attached). Proceeding onto the root disk."
    else
      die "$VOLUME resolves to the same device as / ($root_src) - it is NOT a separate mounted volume. Refusing to install Isaac Sim onto the container's root disk. Mount the persistent volume at $VOLUME first, or set VOLUME=<mount point>; or, ONLY for a single-disk throwaway instance with no separate volume, set ALLOW_ROOT_VOLUME=1 to proceed anyway."
    fi
  fi
  log "resolved \$VOLUME=$VOLUME -> device $vol_src (separate from root device $root_src)"
}
resolve_volume

# --- Disk check on the resolved volume specifically (not on / , which may be tiny in
# a container image regardless of how much space the volume has).
avail_kb=$(df --output=avail -k "$VOLUME" | tail -n1 | tr -d ' ')
avail_gb=$((avail_kb / 1024 / 1024))
log "free space on $VOLUME: ${avail_gb}GB (require >= ${MIN_FREE_GB}GB)"
if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
  die "only ${avail_gb}GB free on $VOLUME, need at least ${MIN_FREE_GB}GB (13GB download + extracted install + post_install caches). Free up space (or override with MIN_FREE_GB=N) before retrying."
fi

# --- Install directory. The only rm -rf in this entire script, and only for this one
# path, and only when --reinstall was explicitly passed.
if [ -x "$ISAACSIM_DIR/isaac-sim.sh" ]; then
  if [ "$REINSTALL" -eq 1 ]; then
    log "--reinstall passed: removing existing install at $ISAACSIM_DIR"
    rm -rf "$ISAACSIM_DIR"
  else
    log "$ISAACSIM_DIR already has an install (isaac-sim.sh present) - skipping download/extract/post_install. Pass --reinstall to force a fresh install."
  fi
fi

if [ ! -x "$ISAACSIM_DIR/isaac-sim.sh" ]; then
  mkdir -p "$ISAACSIM_DIR"
  zip_path="$ISAACSIM_DIR/$ZIP_NAME"

  log "downloading $ZIP_URL (~13GB - this takes a while)"
  # Downloaded straight into $VOLUME, never into /tmp or any root-disk path - the
  # container's root disk may have nowhere near enough room for a 13GB file.
  curl -fL --retry 3 --retry-delay 5 -o "$zip_path" "$ZIP_URL" \
    || die "download failed. If ISAAC_SIM_ZIP_URL has moved, check https://docs.isaacsim.omniverse.nvidia.com/${ISAAC_SIM_VERSION}/installation/download.html for the current Linux x86_64 standalone link and set ISAAC_SIM_ZIP_URL to override."

  log "verifying zip integrity"
  unzip -tq "$zip_path" >/dev/null || die "$zip_path failed unzip's integrity test - the download is corrupt, delete it and retry"

  log "extracting into $ISAACSIM_DIR"
  unzip -q "$zip_path" -d "$ISAACSIM_DIR"
  if [ "$KEEP_ISAACSIM_ZIP" -eq 1 ]; then
    log "KEEP_ISAACSIM_ZIP=1 set - keeping $zip_path (not deleting)"
  else
    rm -f "$zip_path"  # the extracted install is what matters; the archive itself just burns space
  fi

  [ -x "$ISAACSIM_DIR/isaac-sim.sh" ] || die "extraction finished but $ISAACSIM_DIR/isaac-sim.sh is missing - the zip's internal layout may not match what this script expects (see quick-install.html)"

  log "running post_install.sh"
  (cd "$ISAACSIM_DIR" && ./post_install.sh) \
    || die "post_install.sh failed - see its output above. UNVERIFIED step, see this script's header."
fi

ISAACSIM_PYTHON_EXE="$ISAACSIM_DIR/python.sh"
[ -x "$ISAACSIM_PYTHON_EXE" ] || die "$ISAACSIM_PYTHON_EXE not found or not executable after install - the standalone bundle should ship its own python.sh"

# --- Cache symlinks: shader/extension caches survive an instance recycle only if they
# live on the persistent volume. Existing content is MOVED (never discarded) into
# $CACHE_DIR before the real path is replaced with a symlink into it.
link_cache_dir() {
  local real_path="$1" volume_subdir="$2"
  mkdir -p "$(dirname "$real_path")" "$volume_subdir"

  if [ -L "$real_path" ]; then
    local current_target
    current_target=$(readlink -f "$real_path" 2>/dev/null || true)
    if [ "$current_target" = "$(readlink -f "$volume_subdir")" ]; then
      log "$real_path already symlinked to $volume_subdir - OK"
      return
    fi
    log "$real_path is a symlink to something else ($current_target) - relinking to $volume_subdir"
    rm -f "$real_path"
  elif [ -d "$real_path" ]; then
    log "$real_path exists as a real directory - migrating its contents into $volume_subdir (not discarding)"
    # Merge rather than clobber: an empty $volume_subdir from mkdir -p above is fine
    # to merge into; a non-empty one (e.g. a prior run) just gets additional files.
    cp -a "$real_path"/. "$volume_subdir"/ 2>/dev/null || true
    rm -rf "$real_path"
  fi

  ln -s "$volume_subdir" "$real_path"
  log "linked $real_path -> $volume_subdir"
}

link_cache_dir "$HOME/.cache/ov" "$CACHE_DIR/cache-ov"
link_cache_dir "$HOME/.local/share/ov" "$CACHE_DIR/local-share-ov"
link_cache_dir "$HOME/.nvidia-omniverse" "$CACHE_DIR/nvidia-omniverse"

# --- Environment file other tooling (tools/isaac_validate.py's own launch, a future
# Claude Code Isaac skill) can source rather than re-deriving these paths.
cat > "$ENV_FILE" <<EOF
# Written by deploy/install_isaacsim_gpu.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Source this
# to get ISAACSIM_PATH/ISAACSIM_PYTHON_EXE without re-deriving them:  source $ENV_FILE
export ISAACSIM_PATH="$ISAACSIM_DIR"
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PYTHON_EXE"
EOF
log "wrote $ENV_FILE"

# --- System dependency: libMaterialXRenderHw.so (pulled in by Isaac Sim's bundled
# renderer extensions) dlopen's libXt at runtime - not installed by default on the
# minimal container images these GPU instances tend to use, and not something
# post_install.sh checks for. Missing it doesn't fail the install above; it surfaces
# later as a load error from the code_editor probe right below. Installed here,
# before that probe, so a fresh box needs no manual intervention.
if command -v apt-get >/dev/null 2>&1; then
  log "installing libxt6 (runtime dep of Isaac Sim's bundled libMaterialXRenderHw.so)"
  apt-get update -qq && apt-get install -y -qq libxt6 \
    || die "apt-get install libxt6 failed - see output above"
else
  log "WARNING: apt-get not found - skipping libxt6 install; if the code_editor probe below fails to load libMaterialXRenderHw.so, install libxt6 (or your distro's equivalent) manually"
fi

# --- code_editor extension probe: confirms the extension loads and can bind
# $CODE_EDITOR_PORT in THIS install, using the exact headless invocation this scene
# validator/skill workflow is expected to use (no venv - the bundled isaac-sim.sh).
# UNVERIFIED: the --enable / --/exts/.../port flag syntax below is unconfirmed for
# 6.0.1 specifically.
log "probing isaacsim.code_editor.python_server (one-shot headless launch, port $CODE_EDITOR_PORT)"
probe_log="$VOLUME/.isaac_extension_probe.log"
if timeout 300 "$ISAACSIM_DIR/isaac-sim.sh" --no-window \
    --enable isaacsim.code_editor.python_server \
    --/exts/isaacsim.code_editor.python_server/port="$CODE_EDITOR_PORT" \
    --/app/quitAfter=5 \
    > "$probe_log" 2>&1; then
  if grep -qi "code_editor" "$probe_log"; then
    log "isaacsim.code_editor.python_server appears in the launch log - see $probe_log for details"
  else
    log "WARNING: launch succeeded but no code_editor mention found in $probe_log - verify the extension actually enabled (its extension.toml, or 'isaac-sim.sh --help')"
  fi
else
  log "WARNING: probe launch exited non-zero or timed out - see $probe_log. --/app/quitAfter may not be a real setting for this build; if so, this probe just needs a different way to exit rather than indicating a real failure."
fi

# --- Reusable helper to start the code-editor server later, kept separate from
# "install" - this script provisions, it does not leave a long-running process behind.
cat > "$ISAACSIM_DIR/start_code_editor_server.sh" <<EOF
#!/bin/bash
# Starts Isaac Sim headless with isaacsim.code_editor.python_server enabled on port
# $CODE_EDITOR_PORT and leaves it running in the foreground (Ctrl-C to stop).
set -eu
# Omniverse/Isaac Sim refuses to run as root by default - most rented GPU instances
# have no other user configured, so this is baked in rather than left to whoever
# invokes this script later to remember.
export OMNI_KIT_ALLOW_ROOT=1
exec "$ISAACSIM_DIR/isaac-sim.sh" --no-window \\
  --enable isaacsim.code_editor.python_server \\
  --/exts/isaacsim.code_editor.python_server/port=$CODE_EDITOR_PORT
EOF
chmod +x "$ISAACSIM_DIR/start_code_editor_server.sh"
log "wrote $ISAACSIM_DIR/start_code_editor_server.sh - run it manually when you want the server up"

log "done. Isaac Sim $ISAAC_SIM_VERSION installed at $ISAACSIM_DIR"
log "next: source $ENV_FILE, then run tools/isaac_validate.py against an exported scene - see docs/ISAAC_VALIDATION.md Step 4"
log "record what actually happened (including any deviations from this script) in docs/ISAAC_VALIDATION.md's Verified runs table"
