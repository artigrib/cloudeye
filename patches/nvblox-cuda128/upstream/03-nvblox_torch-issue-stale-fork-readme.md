DRAFT — NOT SUBMITTED. For review before posting.

Target: NVlabs/nvblox_torch
Type: Issue
Template: repo uses `.github/ISSUE_TEMPLATE/issue-template.md`, which is
just two prompts: "How are you using nvblox_torch (docker/native/conda/?)"
and "Issue Details:". It also auto-assigns to balakumar-s — that's
automatic on the real GitHub form, not something to add manually. Filled
in below in that shape, with a longer body under "Issue Details".

---

## Issue title

README points to a fork of nvblox that's 2+ years stale and missing fixes real upstream already has

## Issue body

**How are you using nvblox_torch (docker/native/conda/?):** native (uv venv), building nvblox from source per the README's instructions.

**Issue Details:**

The README's build instructions clone `valtsblukis/nvblox` (for
`PRE_CXX11_ABI_LINKABLE` support) rather than real upstream
(`nvidia-isaac/nvblox`, formerly `NVIDIA-ISAAC-ROS/nvblox`). That fork's
last commit is from 2024-01-30 — it's fallen behind real upstream, and
building it under CUDA 12.8 hits three compile failures that are already
fixed on `nvidia-isaac/nvblox`'s `public` branch:

- Legacy `<nvToolsExt.h>` include causing ~35 NVTX symbol redefinition
  errors against CUDA 12.8's Thrust/CUB (fixed upstream: switched to
  `<nvtx3/nvToolsExt.h>`)
- Missing `#include <array>` in `rates.h` causing an incomplete-type error
  under GCC 13's libstdc++ (fixed upstream)
- A vendored-stdgpu CMake configure failure parsing `THRUST_VERSION`
  (still open upstream too, as nvidia-isaac/nvblox#54 — not fork-specific)

Checking real upstream (2026-08-29), it now has
`cmake/setup_pytorch_cpp11_abi.cmake` implementing
`PRE_CXX11_ABI_LINKABLE` natively — which was the fork's whole reason for
existing. If that's confirmed to satisfy nvblox_torch's actual linking
requirements, would it make sense for the README to point at
`nvidia-isaac/nvblox` directly instead of the fork? That would sidestep
the first two issues above entirely for anyone building fresh, and leave
only the (already-tracked) stdgpu pin issue.

I don't have visibility into why the fork was chosen originally (maybe
`PRE_CXX11_ABI_LINKABLE` landed upstream after nvblox_torch's README was
written, or maybe there's a subtlety the fork handles that upstream
doesn't) — raising this as a question rather than a PR, since I can't
verify from the nvblox_torch side whether upstream's version is actually
equivalent for your linking needs.

Full patch set + notes for the three build issues above, for CUDA 12.8 /
GCC 13 / Ubuntu 24.04, in case useful regardless of which nvblox source
the README ends up recommending: [link to our patches/nvblox-cuda128/ once
that's public].
