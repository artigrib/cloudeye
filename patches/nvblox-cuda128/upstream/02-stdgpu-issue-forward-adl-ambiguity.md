DRAFT — NOT SUBMITTED. For review before posting.

Target: stotko/stdgpu
Type: Issue (not a PR — see rationale below)
Per stdgpu's contributing guide (stotko.github.io/stdgpu/development/contributing.html):
include a clear problem summary, expected vs. actual behavior, and a
minimal reproducible example.

Why an issue and not a PR: the mechanical fix (qualify every call site as
`stdgpu::forward<...>` etc.) is straightforward and we have it working,
but it's a tree-wide, somewhat blunt change, and the maintainer may prefer
a different approach (e.g. a `using` alias at namespace scope, or scoping
the fix to only the call sites that are actually reachable with
thrust-associated types) — this is the kind of judgment call better left
to whoever owns the codebase's style conventions. Happy to convert our
patch into a PR if the maintainer confirms the qualify-everywhere approach
is the one they want.

---

## Issue title

Ambiguous overload: unqualified `forward`/`construct_at`/`destroy_at` calls conflict with `cuda::std` under CUDA 12.x libcu++

## Issue body

### Summary

Compiling stdgpu-dependent code (via nvblox, in our case) against CUDA
12.8 fails with ambiguous-overload errors on stdgpu's own internal helper
functions when they're instantiated with a `thrust::pair<...>` argument.

### Environment

- CUDA 12.8 (nvcc), GCC 13, Ubuntu 24.04
- stdgpu commit `e10f6f3ccc9902d693af4380c3bcd188ec34a2e6` (2021-07-26,
  reached via nvblox's `FetchContent` pin) — but current stdgpu `master`
  still has the same unqualified call sites in `memory_detail.h` as of
  2026-08-29, so this isn't specific to that old commit.

### Error (representative, ~12 occurrences)

```
error: more than one instance of overloaded function "stdgpu::forward"
matches the argument list:
            function "stdgpu::forward<T>(...)"
            function "cuda::std::__4::forward<T>(...)"
            argument types are: (...)
```

Same pattern for `construct_at` / `destroy_at`.

### Root cause (as best we can tell)

CUDA 12.x's libcu++ (CCCL) aliases `thrust::pair` into the `cuda::std`
namespace. stdgpu's internal helpers — `forward<Args>(...)`,
`construct_at(...)`, `destroy_at(...)` — are called unqualified inside
`stdgpu::`'s own headers (e.g. `src/stdgpu/impl/memory_detail.h`,
`unordered_base_detail.cuh`, `unordered_map_detail.cuh`,
`unordered_set_detail.cuh`, `vector_detail.cuh`, `deque_detail.cuh`,
`functional_detail.h`, `functional.h`). When one of these is instantiated
with a `thrust::pair<...>` argument, argument-dependent lookup (ADL) now
also associates `cuda::std` (via the `pair` alias) and finds
`cuda::std::forward` / `cuda::std::destroy_at` alongside stdgpu's own —
two equally-good candidates, ambiguous overload.

This is a change in behavior driven by CUDA 12.x's libcu++, not something
that changed in stdgpu — so it likely affects any consumer that
instantiates these templates with a `thrust::` type on a modern CUDA
toolkit, not just nvblox.

### Reproduction

Minimal repro is awkward to isolate outside of nvblox's actual usage (it
requires instantiating stdgpu's unordered containers/allocator machinery
with a `thrust::pair` value type under CUDA 12.x). We hit it building
nvblox's `src/gpu_hash/gpu_layer_view.cu`, which does exactly that.
Happy to try to reduce this to a standalone `.cu` file if useful — didn't
attempt it for this report since the nvblox reproduction was already in
hand.

### Suggested fix

Qualify the ambiguous call sites as `stdgpu::forward<...>`,
`stdgpu::construct_at(...)`, `stdgpu::destroy_at(...)` — a qualified call
skips ADL entirely, removing the ambiguity regardless of what `cuda::std`
aliases. We have a patch doing this tree-wide across the 8 affected files
(functional.h, deque_detail.cuh, functional_detail.h, memory_detail.h,
unordered_base_detail.cuh, unordered_map_detail.cuh,
unordered_set_detail.cuh, vector_detail.cuh) and can open a PR if that's
the direction you'd want — flagging as an issue first since it's a
tree-wide change and you may prefer a different fix shape (e.g. a
namespace-scope `using` declaration instead of qualifying every call
site).
