#!/usr/bin/env python3
"""Where the CPU stages' interpreter lives - resolved in ONE place.

PACK, LAYERS, REACH and EXPORT do not run under this repo's own `.venv`: they need
open3d, and EXPORT needs `pxr` (usd-core), neither of which the API/worker venv carries.
They run under a separate CPU-only venv instead.

That venv used to be hardcoded as `/tmp/o3d-venv/bin/python` in six places. `/tmp` is not
a home for a build dependency - `systemd-tmpfiles-clean` reclaims it on a timer, and on
2026-09-09 it was removed by hand after `lsof` showed no open files (the wrong test for
"unused": nothing holds a venv open between runs). Every CPU stage then died with a bare
`FileNotFoundError` traceback naming a path, with nothing saying what the path was for.

So: one default, one env var to override it, and a guard that says what is missing and how
to put it back.

    PIPELINE_CPU_PYTHON=/some/other/venv/bin/python python3 pipeline/run_pipeline.py ...

Rebuilding it (the pins are the validated stack - see PIPELINE.md 2):

    uv venv ~/venvs/o3d-cpu --python 3.12
    uv pip install --python ~/venvs/o3d-cpu/bin/python \\
        numpy==2.5.2 scipy==1.18.1 open3d==0.19.0 matplotlib usd-core trimesh
"""

from __future__ import annotations

import os
from pathlib import Path

#: Override with an absolute path to a python that has the stack below.
ENV_VAR = "PIPELINE_CPU_PYTHON"

#: Deliberately NOT under /tmp. See the module docstring.
DEFAULT = str(Path.home() / "venvs" / "o3d-cpu" / "bin" / "python")

#: What that interpreter must be able to import, and why - so the guard can say more than
#: "missing". numpy/scipy: every stage. open3d: mesh IO in PACK/REACH/EXPORT. matplotlib:
#: the LAYERS and REACH renders. pxr: isaac_export.py's USD writer. trimesh: mesh repair.
REQUIRES = ("numpy==2.5.2", "scipy==1.18.1", "open3d==0.19.0", "matplotlib", "usd-core",
            "trimesh")

REBUILD_CMD = (f"uv venv {Path(DEFAULT).parents[1]} --python 3.12 && "
               f"uv pip install --python {DEFAULT} " + " ".join(REQUIRES))


def cpu_python() -> str:
    """The configured path. Existence is NOT checked here - see `require_cpu_python`."""
    return os.environ.get(ENV_VAR) or DEFAULT


def missing_reason(path: str | None = None) -> str | None:
    """None when the interpreter is usable, else a ONE-LINE reason naming the path, the
    env var and the rebuild command. One line on purpose: this is printed instead of a
    traceback, and a traceback is what it exists to replace."""
    p = Path(path or cpu_python())
    if p.is_file() and os.access(p, os.X_OK):
        return None
    return (f"CPU interpreter not found at {p} (override with {ENV_VAR}); "
            f"rebuild it with: {REBUILD_CMD}")


def require_cpu_python(path: str | None = None) -> str:
    """The path, or RuntimeError with the one-line reason. Call before spawning it."""
    p = path or cpu_python()
    reason = missing_reason(p)
    if reason:
        raise RuntimeError(reason)
    return p
