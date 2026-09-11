#!/usr/bin/env python3
"""Re-run ONLY stage_align.py's alignment logic against an already-completed job's
`per_view/*.npz` (produced by stage_infer.py), without touching that job's existing
`aligned_room.ply` / `alignment_transform.npz` / `alignment.json` / `cameras_aligned.json`
- writes a fresh set of those four files into a sibling directory instead (default
`aligned_v2/`).

Built for manually verifying a floor_ceiling.py/stage_align.py algorithm change against
a real, already-captured scene (e.g. the hotel-room floor-selection fix - see
docs/DECISIONS.md's 2026-09-03 "Floor selection fixed..." entry) without re-running the
whole 7-stage pipeline: keyframes/vocab/inference are unaffected by an alignment-only
change, so there's no reason to redo them.

Runs under the mapanything venv (same as stage_align.py itself - imports open3d,
sklearn). Usage, on the GPU box:

    <workspace>/envs/mapanything/bin/python <workspace>/pipeline/rerun_align.py <job_dir> [--out-name aligned_v2]

Prints the OLD run's floor_y/floor_support/points_below_floor_frac/floor_likely_wrong
(read from the job's existing alignment.json, if any - "floor_likely_wrong" reads back
as "N/A (schema < 4)" for a run from before this field existed), then the NEW run's
full per-candidate diagnostics (height/support/n_views/hull_area for every horizontal
plane found, flagging which one was picked as floor/ceiling) and the same summary
numbers - so a before/after comparison is one command instead of two directory diffs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import job_paths  # noqa: E402
from stage_align import run_align  # noqa: E402


def _print_old_run(job_dir: Path) -> None:
    alignment_json = job_paths(job_dir)["alignment_json"]
    print("=== OLD RUN ===")
    if not alignment_json.exists():
        print(f"  no existing {alignment_json} - nothing to compare against")
        return
    old = json.loads(alignment_json.read_text())
    print(f"  path:                       {alignment_json}")
    print(f"  schema_version:             {old.get('schema_version')}")
    print(f"  chosen_band:                {old.get('chosen_band')}")
    print(f"  floor_y:                    {old.get('floor_y')}")
    print(f"  floor_support:              {old.get('floor_support')}")
    print(f"  points_below_floor_frac:    {old.get('points_below_floor_frac')}")
    print(f"  reflective_floor_suspected: {old.get('reflective_floor_suspected')}")
    print(f"  floor_likely_wrong:         {old.get('floor_likely_wrong', 'N/A (schema < 4)')}")
    print(f"  conf_available:             {old.get('conf_available', 'N/A (schema < 5)')}")
    print(f"  conf_cutoff:                {old.get('conf_cutoff', 'N/A (schema < 5)')}")


def _print_new_run(summary: dict) -> None:
    print(f"\n=== NEW RUN (written to {summary['out_dir']}) ===")
    print("  candidates (all near-horizontal planes found, either side of the room):")
    for c in summary["diagnostics"]["candidates"]:
        flag = ""
        if c["is_chosen_floor"]:
            flag = "  <- FLOOR"
        elif c["is_chosen_ceiling"]:
            flag = "  <- CEILING"
        print(
            f"    side={c['side']:<4} y={c['plane_y']:+8.4f}  support={c['support']:<7} "
            f"n_views={c['n_views']:<3} hull_area={c['hull_area']:7.2f}m^2 "
            f"tilt={c['tilt_deg']:5.2f}deg{flag}"
        )
    print(f"  room_bbox_xy_area:          {summary['diagnostics'].get('room_bbox_xy_area')}")
    print(f"  chosen_band:                {summary['chosen_band']}")
    print(f"  floor_y:                    {summary['floor_y']:.4f}")
    print(f"  floor_support:              {summary['floor_support']}")
    print(f"  points_below_floor_frac:    {summary['points_below_floor_frac']:.4%}")
    print(f"  reflective_floor_suspected: {summary['reflective_floor_suspected']}")
    print(f"  floor_likely_wrong:         {summary['floor_likely_wrong']}")
    print(f"  residual_tilt_deg:          {summary['residual_tilt_deg']:.3f}")
    print(f"  conf_available:             {summary['conf_available']}")
    if summary["conf_available"]:
        print(
            f"  conf_cutoff:                {summary['conf_cutoff']:.4f} "
            f"(percentile={summary['conf_threshold_percentile']:.1f})"
        )
    if summary["warnings"]:
        print("  warnings:")
        for w in summary["warnings"]:
            print(f"    - {w}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", help="existing job directory (must already contain per_view/*.npz)")
    ap.add_argument(
        "--out-name",
        default="aligned_v2",
        help="subdirectory under job_dir to write the new run's 4 output files into "
        "(default: aligned_v2) - never overwrites the job's existing artifacts",
    )
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    if not job_dir.is_dir():
        ap.error(f"job_dir does not exist: {job_dir}")
    per_view_dir = job_paths(job_dir)["per_view_dir"]
    if not per_view_dir.is_dir() or not any(per_view_dir.glob("view_*.npz")):
        ap.error(f"no per_view/*.npz found under {per_view_dir} - run stage_infer.py first")

    _print_old_run(job_dir)
    summary = run_align(job_dir, out_dir=job_dir / args.out_name)
    _print_new_run(summary)


if __name__ == "__main__":
    main()
