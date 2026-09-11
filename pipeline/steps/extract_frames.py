"""Extract an explicit list of raw video frame indices to JPEGs in a single ffmpeg
decode pass (one `select` filter expression, not one ffmpeg invocation per frame).

Output filenames are `frame_{raw_frame_index:06d}.jpg` - stable, sortable, and directly
traceable back to motion_sampled_frames.json / passA_frames.json / passB_windows.json,
all of which key on raw_frame_index.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def extract(video_path: Path, raw_indices: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_indices = sorted(set(raw_indices))
    select_expr = "+".join(f"eq(n\\,{i})" for i in raw_indices)
    tmp_pattern = str(out_dir / "_seq_%06d.jpg")
    # `-autorotate 1` is ffmpeg's default; stated explicitly so the frames' orientation is
    # a decision this file makes rather than a default it inherits. frames_by_fps.py records
    # the container rotation and the resulting orientation alongside the frame list.
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-autorotate", "1", "-i", str(video_path),
        "-vf", f"select='{select_expr}'", "-vsync", "0", "-qscale:v", "2",
        tmp_pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    produced = sorted(out_dir.glob("_seq_*.jpg"))
    if len(produced) != len(raw_indices):
        raise RuntimeError(
            f"ffmpeg produced {len(produced)} frames but {len(raw_indices)} were "
            f"requested - select expression or frame indices are wrong"
        )
    for seq_path, raw_idx in zip(produced, raw_indices):
        seq_path.rename(out_dir / f"frame_{raw_idx:06d}.jpg")
    print(f"extracted {len(raw_indices)} frames -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames-json", required=True, help="JSON with a 'frames' list of {raw_frame_index: int, ...}")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.frames_json).read_text())
    raw_indices = [f["raw_frame_index"] for f in data["frames"]]
    extract(Path(args.video), raw_indices, Path(args.out_dir))


if __name__ == "__main__":
    main()
