# nvblox / stdgpu patches for CUDA 12.8

Patches needed to get `nvblox` + `nvblox_torch` compiling on CUDA 12.8 (nvcc),
GCC 13, Ubuntu 24.04 (RTX 4090, PyTorch 2.11.0+cu128). Discovered building
CloudEye's GPU pipeline; not currently used by that pipeline in production
(see [Why we're not using nvblox](#why-were-not-using-nvblox-right-now) below).

## Repos and exact commits

We built against **`valtsblukis/nvblox`**, a personal fork of the real
upstream (`nvidia-isaac/nvblox`, formerly `NVIDIA-ISAAC-ROS/nvblox`) that
nvblox_torch's own README points to, because it adds
`PRE_CXX11_ABI_LINKABLE` support needed to link against a PyTorch build.
nvblox in turn vendors **`stotko/stdgpu`** via CMake `FetchContent`, pinned
to a specific commit in `nvblox/thirdparty/stdgpu/stdgpu.cmake`.

| Repo | Commit | Date | Note |
|---|---|---|---|
| `valtsblukis/nvblox` | `7bda93e993594ca69c479dde64213ac35835eb8` | 2024-01-30 | Fork's tip — no commits since. This is what patches 02 and 04 apply to. |
| `stotko/stdgpu` | `e10f6f3ccc9902d693af4380c3bcd188ec34a2e6` | 2021-07-26 | Pinned by the fork's `stdgpu.cmake`. This is what patches 01 and 03 apply to. |

**A note on these hashes**, for anyone diffing against our original install
log: our first pass at this (`~/gpu-backup/setup-log.md` on the build
machine) recorded the nvblox commit as `e10f6f3ccc9902d693af4380c3bcd188ec34a2e`
(missing its final `6`) and the stdgpu commit as
`7bda93e993594ca69c479dde64213ac35835eb8` — i.e. the two hashes were
swapped, and the stdgpu one was truncated by one character. Both are
corrected in the table above and verified directly: cloned each repo fresh,
confirmed the commit exists and resolves, and confirmed it's the one
actually referenced by the other repo's build config (`stdgpu.cmake`'s
`GIT_TAG` for stdgpu; the fork's own `git log -1` for nvblox).

## Applying

Each numbered patch targets one repo and one concern. Apply with `git
apply` (or `patch -p1`) from the repo root at the commit above:

```bash
# nvblox (valtsblukis/nvblox @ 7bda93e993594ca69c479dde64213ac35835eb8)
git apply 02-nvtx3-include.patch
git apply 04-rates-array-include.patch

# stdgpu (stotko/stdgpu @ e10f6f3ccc9902d693af4380c3bcd188ec34a2e6)
# — in practice this is nvblox's build/_deps/ext_stdgpu-src/ after
#   FetchContent has pulled it, before you run `make`
git apply 01-findthrust-cmake-comment-parse.patch
git apply 03-stdgpu-forward-adl-qualification.patch
```

Verified (2026-08-29) applying cleanly with `git apply --check` against
fresh worktrees checked out at the exact commits above, and that the
resulting diff is byte-identical to what was originally captured off the
build machine. **Not re-verified by an actual `make` in this session** — no
GPU instance was available; the original successful build (confirmed by
`libpy_nvblox.so` linking and nvblox_torch's own bundled TSDF/depth-image
demo running on `cuda:0`) is recorded in `~/gpu-backup/setup-log.md`. If you
apply these fresh, a real compile is the only way to be sure — treat "patch
applies" as necessary, not sufficient.

## Upstream status (checked 2026-08-29)

Two of the four are already fixed in **real** upstream (`nvidia-isaac/nvblox`,
not the fork) — the fork is just stale:

| Patch | Fixes | Upstream status |
|---|---|---|
| `01-findthrust-cmake-comment-parse` | CMake configure fails: `Could NOT find thrust: Found unsuitable version "ERROR.ERROR.ERROR"` | **Still broken** in `nvidia-isaac/nvblox` (its stdgpu pin predates stdgpu's own fix). Open issue: [nvidia-isaac/nvblox#54](https://github.com/nvidia-isaac/nvblox/issues/54), no PR linked. Draft PR: `upstream/01-nvblox-pr-bump-stdgpu-pin.md`. |
| `02-nvtx3-include` | ~35 "already defined" NVTX redefinition errors compiling `gpu_layer_view.cu` | **Already fixed** in `nvidia-isaac/nvblox`'s `public` branch. No contribution needed there. |
| `03-stdgpu-forward-adl-qualification` | ~12 ambiguous-overload errors on `forward`/`construct_at`/`destroy_at` | **Still present** in `stotko/stdgpu` master (verified directly against current `memory_detail.h`). No existing issue found. Draft issue: `upstream/02-stdgpu-issue-forward-adl-ambiguity.md`. |
| `04-rates-array-include` | `CircularBuffer::buffer_` has incomplete type | **Already fixed** in `nvidia-isaac/nvblox`'s `public` branch. No contribution needed there. |

### Why we ended up on a stale fork at all

`valtsblukis/nvblox` exists only for `PRE_CXX11_ABI_LINKABLE`. As of
2026-08-29, real upstream `nvidia-isaac/nvblox` has that option natively
(`cmake/setup_pytorch_cpp11_abi.cmake`). If that's confirmed to work for
nvblox_torch's actual use case, the fork may no longer be necessary at
all — see `upstream/03-nvblox_torch-issue-stale-fork-readme.md` for a draft
suggesting nvblox_torch's README point at real upstream instead. That would
make patches 02 and 04 moot for future users (they'd never hit them), and
narrow this whole patch set down to whatever nvidia-isaac/nvblox itself
still needs (patch 01, and possibly patch 03's stdgpu fix depending on
which commit it pins).

## Two more things this build hit, not build errors

Not part of this patch set because they're functional, not compile,
issues — see `upstream/04-nvblox_torch-issue-query-sdf-sentinel.md`:

- `mapper.query_sdf(...)` against a successfully-fused ESDF layer returns
  the "unobserved" sentinel for ~90% of queries, including at points
  sampled directly from the reconstructed surface.
- `get_occupied_voxels_on_grid` is unimplemented (`pass`) in the installed
  `nvblox_torch==0.0.post1.dev15`.

## Why we're not using nvblox right now

`update_mesh`/`save_mesh` produced a correct, room-scale mesh (confirming
depth fusion itself works), but `query_sdf` was not usable for occupancy
queries for the reason above, and there was no working fallback path to
pull a raw occupancy grid out of nvblox directly. CloudEye's occupancy grid
is currently built independently, from point density in an XZ grid over the
fused point cloud (`gpu/stage_occupancy.py`) — see the main
[README's Caveats section](../../README.md#caveats). These patches get
nvblox *compiling*; they don't (and can't) fix the separate query-side bug
above.
