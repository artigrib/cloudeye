DRAFT — NOT SUBMITTED. For review before posting.

Target: NVlabs/nvblox_torch
Type: Issue
Template: same two-prompt template as the other nvblox_torch draft.

Note: searched existing open (3) and closed (8) issues on this repo
(2026-08-29) for anything matching — closest was a closed issue about
`query_sdf` segfaulting (different symptom: a crash, not a silent
wrong-value return), closed 2023-12-15. Nothing found matching this
symptom or mentioning `get_occupied_voxels_on_grid`. This search used
WebSearch/WebFetch rather than an authenticated GitHub search, so it's
worth a manual double-check before submitting.

---

## Issue title

`query_sdf` returns the "unobserved" sentinel for ~90% of queries at known-surface points; `get_occupied_voxels_on_grid` unimplemented

## Issue body

**How are you using nvblox_torch (docker/native/conda/?):** native (uv venv), `nvblox-torch==0.0.post1.dev15`.

**Issue Details:**

### Summary

After successfully fusing ~26 depth views into a TSDF map and confirming
the fusion itself is correct (via `update_mesh` + `save_mesh`, which
produces a well-formed, room-scale, correctly-metric mesh), `query_sdf`
against the same map's ESDF layer returns the -100.0 "never touched"
sentinel for the large majority of queries — including queries placed
directly at points sampled from the reconstructed surface, which by
definition should be integrated.

### Repro

```python
from nvblox_torch.mapper import Mapper

mapper = Mapper(voxel_sizes=[0.02], integrator_types=["tsdf"])
for depth, pose, intrinsics in views:  # 26 views, real room walkthrough
    mapper.add_depth_frame(depth, pose, intrinsics, 0)
mapper.update_esdf(0)

# spheres_xyzr sampled directly from the reconstructed point cloud —
# i.e. points that demonstrably ARE on integrated surfaces
mapper.query_sdf(spheres_xyzr_tensor, out, True, 0)
# -> -100.0 sentinel ("never touched") for ~90% of the 500 test points
```

`mapper.update_mesh(0)` + `save_mesh(...)` on the same map produces
21,708 vertices with a plausible bbox
(X=[-2.65,0.59] Y=[-1.25,1.07] Z=[-0.47,2.75]) — confirming depth fusion
itself works. This is specifically a `query_sdf` / ESDF-query problem.

### What we ruled out

- **Truncation band too narrow** for cross-view depth noise: tested
  `voxel_size` in `{0.02, 0.05, 0.10}` (truncation = 4× voxel_size =
  8cm/20cm/40cm). Valid-query count stayed flat: 45/43/53 out of 500 — not
  a truncation-band issue.
- **Pose convention** (camera-to-world vs. world-to-camera): tried
  `np.linalg.inv(camera_pose)` on the pose tensor passed to
  `add_depth_frame`. No meaningful change (55/500 valid).

Both attempts left coverage in the same 40-55/500 range, which suggests
the bug isn't in how we're calling `add_depth_frame`/`update_esdf` — the
integration itself is verifiably correct via the mesh — but somewhere in
how `query_sdf` samples the ESDF layer or interprets the query points.

### Secondary: no fallback extraction path

`get_occupied_voxels_on_grid` is unimplemented (`pass`, no docstring
explaining why) in `mapper.py` in this version, so there was no way to
pull a direct occupied/free/unknown grid out of nvblox as an alternative
to trusting `query_sdf`.

### Environment

- CUDA 12.8, PyTorch 2.11.0+cu128, RTX 4090, Ubuntu 24.04
- `nvblox-torch==0.0.post1.dev15`, built from source per the README against
  `valtsblukis/nvblox` (patches for CUDA 12.8 compile issues applied
  separately, unrelated to this — fusion and meshing both work correctly
  post-patch, this is a distinct runtime/query bug)

Happy to share the actual view data / a minimal script if that'd help
reproduce — didn't attach it to this draft since we don't have a public
place to host it yet.
