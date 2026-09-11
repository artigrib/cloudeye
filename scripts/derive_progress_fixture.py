#!/usr/bin/env python3
"""Turn one recorded pipeline run into the replay fixture the progress screen is tested on.

The source is READ-ONLY and is never modified: `var/scratch/pipeline_live/scene161054/
status.json`, the own_0901_161054 run of 2026-09-09. Only the final snapshot survives on
disk - pipeline-v1 rewrites status.json in place after every change (docs/JOB_SPEC.md
section 2), so the intermediate states are gone. They are RECONSTRUCTED here, and the
reconstruction is arithmetic, not invention:

  * every step name, state, `started`, `finished`, `duration_s` and `error` is copied
    verbatim from the recorded file;
  * a snapshot is emitted at each step boundary, holding exactly the steps that had
    started by then, with the step in flight marked `state: "running"` and its
    `finished`/`duration_s` null - which is what `Status.enter` writes and what
    `finish_step` later overwrites;
  * the top-level `state` of each snapshot is the step in flight, and the last snapshot
    is the recorded file's own terminal state.

Nothing else is added. In particular the fixture carries NO `substage`/`detail`/`since`/
`attempt`/`next_retry`: this run was written by a build that predates those fields, they
are absent from the recorded file, and the whole point of replaying it is to prove the
screen omits every row it has no value for rather than printing a placeholder. The
substage cases are covered by the synthetic fixture beside it, which is labelled as
synthetic precisely because no recorded run on this box exercises them.

    uv run python scripts/derive_progress_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("var/scratch/pipeline_live/scene161054/status.json")
OUT = Path(__file__).resolve().parents[1] / "frontend/src/fixtures/progressReplay.json"

#: Copied per step. `artifacts`, `cost_estimate_usd` and the `boxes` blob are dropped -
#: the progress screen reads none of them, and carrying 16 KB of nvblox artifact paths
#: into a unit test would obscure what the test is about.
STEP_FIELDS = ("state", "started", "finished", "duration_s", "error")


def main() -> int:
    doc = json.loads(SOURCE.read_text())
    steps = doc["steps"]
    names = list(steps)  # insertion order == the order the states were entered

    snapshots = []
    for i, name in enumerate(names):
        live = {}
        for done_name in names[:i]:
            live[done_name] = {f: steps[done_name].get(f) for f in STEP_FIELDS}
        # The step in flight, as `Status.enter` writes it before `finish_step` runs.
        live[name] = {
            "state": "running",
            "started": steps[name].get("started"),
            "finished": None,
            "duration_s": None,
            "error": None,
        }
        snapshots.append({
            "state": name,
            "started": doc.get("started"),
            "finished": None,
            "error": None,
            "steps": live,
        })

    # The recorded terminal snapshot, verbatim in every field the screen reads.
    snapshots.append({
        "state": doc.get("state"),
        "started": doc.get("started"),
        "finished": doc.get("finished"),
        "error": doc.get("error"),
        "steps": {n: {f: steps[n].get(f) for f in STEP_FIELDS} for n in names},
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_source": str(SOURCE),
        "_scene": doc.get("scene"),
        "_derivation": "scripts/derive_progress_fixture.py - snapshots reconstructed at "
                       "step boundaries from the recorded final status.json; every step "
                       "name, state and timestamp is verbatim from that file",
        "snapshots": snapshots,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(snapshots)} snapshots -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
