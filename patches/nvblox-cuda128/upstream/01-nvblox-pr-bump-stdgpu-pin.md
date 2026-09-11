DRAFT — NOT SUBMITTED. For review before posting.

Target: nvidia-isaac/nvblox (real upstream, not the valtsblukis fork)
Type: Pull Request
References: nvidia-isaac/nvblox#54 (open, no PR linked, filed 2024-07-25)

No CONTRIBUTING.md or PR template found in the repo (checked 2026-08-29;
`.github/` has only CI workflows, `.gitlab/merge_request_templates` exists
but content wasn't accessible via unauthenticated fetch — worth checking
manually before submitting in case it has required fields we should
match).

---

## PR title

Bump vendored stdgpu pin to pick up Findthrust.cmake CUDA 12.x fix (fixes #54)

## PR body

### Problem

`cmake ..` configure fails on CUDA 12.x with:

```
CMake Error at build/_deps/ext_stdgpu-src/cmake/Findthrust.cmake:22 (math):
  math cannot parse the expression: "ERROR / 100000": syntax error,
  unhandled operator...
-- Could NOT find thrust: Found unsuitable version "ERROR.ERROR.ERROR",
  but required is at least "..." (found .../include)
```

Reproduced on CUDA 12.8 / nvcc, CMake 3.28, Ubuntu 24.04, and reported
independently in #54 on CUDA 12.5.82.

### Root cause

`thirdparty/stdgpu/stdgpu.cmake` pins stdgpu via `FetchContent` to
`71a5aef26626eda47d15e5f577ca3b1538ff996a` (2024-02-11). stdgpu's vendored
`cmake/Findthrust.cmake` extracts `THRUST_VERSION` from CUDA's
`thrust/version.h` via regex and passes it straight to `math(EXPR ...)`.
CUDA 12.x's `thrust/version.h` has a trailing comment on that `#define`
line (e.g. `#define THRUST_VERSION 200700 // macro expansion...` or, per
#54, one with a `##` token-paste comment), which the old regex doesn't
strip, so `math()` receives a non-numeric string and fails.

stdgpu itself already fixed this in commit
`1f0b2d51718692ec9046fe1b36173a591c611bdb` (2024-03-13, "Fix bug #407:
THRUST_VERSION may be followed by comments") — after the commit this repo
currently pins.

### Fix

Bump the `GIT_TAG` in `thirdparty/stdgpu/stdgpu.cmake` to a commit at or
after `1f0b2d5171...`. I haven't picked a specific target commit for this
draft — whoever finalizes this PR should pick current stdgpu `master` (or
their preferred pinned point) and confirm nothing else changed between the
old and new pin that affects nvblox's build (I did not diff the full
range).

```diff
- GIT_TAG        71a5aef26626eda47d15e5f577ca3b1538ff996a
+ GIT_TAG        <commit at or after 1f0b2d5171>
```

### Alternative considered

A local patch to the vendored `Findthrust.cmake` (regex fix, not a pin
bump) — see `patches/01-findthrust-cmake-comment-parse.patch` in
[our patch set]. Left out of this PR since bumping the pin is the more
maintainable fix (picks up any other stdgpu fixes in the same range, and
doesn't leave a divergent vendored file to reconcile on the next real pin
bump).

### Testing

Confirmed the *symptom* is fixed by direct string manipulation of the pin
target's `Findthrust.cmake` (i.e. verified stdgpu's own fixed file no
longer produces the `ERROR.ERROR.ERROR` string for CUDA 12.8's
`thrust/version.h`). **Have not done a full nvblox build against the
bumped pin** — no GPU instance available when preparing this PR. Whoever
submits this should do a clean `cmake .. && make` against the new pin on
an actual CUDA 12.x machine before merging, not just trust this
description.

### Environment this was found on

- CUDA 12.8 (nvcc), CMake 3.28, Ubuntu 24.04, GCC 13
- GPU: RTX 4090
