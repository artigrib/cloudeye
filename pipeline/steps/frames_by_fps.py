#!/usr/bin/env python3
"""Uniform frame list at a fixed sampling rate, expressed as an INTEGER decimation
step over the video's native fps. Emits the shape extract_frames.py / run_pass_a.py
already consume: {"frames": [{"raw_frame_index", "t_sec", "file"}, ...]}.

No blur / histogram-similarity filtering: the point of this ladder is that the number
of frames reaching the network is a deterministic function of the knob and the video
length, which the production extractor's content-dependent filters (they drop 6%-77%)
would destroy.
"""
import argparse, json, subprocess, math
from pathlib import Path


def probe(video: str) -> tuple[int, float]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,duration",
        "-of", "json", video]).decode()
    s = json.loads(out)["streams"][0]
    nb = int(s["nb_read_frames"])
    num, den = s["avg_frame_rate"].split("/")
    return nb, float(num) / float(den)


def probe_rotation(video: str) -> dict:
    """Read the container's rotation metadata and state what the extracted frames will be.

    Phone captures carry a rotation the pixels themselves do not have - the hero video is
    tagged -90. ffmpeg APPLIES it by default when decoding, so frames come out upright and
    `frames_orientation_deg` is 0; with `-noautorotate` they would come out turned by
    `rotation_deg`. This is recorded rather than assumed because an unapplied rotation moves
    the cameras' "up" by a quarter turn, which downstream shows up as a floor fit that
    cannot find a floor - see nvblox_scenes.py's `camera_up_variants`.

    Both spellings are read: modern ffprobe reports `side_data_list[].rotation`, older files
    carry a `rotate` tag.
    """
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", video]).decode()
    st = (json.loads(out).get("streams") or [{}])[0]
    rot = None
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
            break
    if rot is None and (st.get("tags") or {}).get("rotate") is not None:
        rot = float(st["tags"]["rotate"])
    rot = 0.0 if rot is None else rot
    return {"container_rotation_deg": rot,
            "autorotate": True,
            "frames_orientation_deg": 0.0,
            "note": ("ffmpeg autorotates by default, so the extracted frames are upright "
                     "regardless of container_rotation_deg; frames_orientation_deg is what "
                     "the frames actually are, not what the container says")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--step", type=int, help="keep every Nth raw frame")
    g.add_argument("--fps", type=float,
                   help="target sampling rate; the step is derived from the video's own "
                        "native fps and rounded to an integer, because the decimation this "
                        "ladder is built on is integer by definition. The achieved rate is "
                        "reported as sample_fps and will differ from the request whenever "
                        "native/fps is not a whole number - that difference is recorded, "
                        "not hidden.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    nb, native = probe(a.video)
    step = a.step if a.step is not None else max(1, round(native / a.fps))
    idx = list(range(0, nb, step))
    doc = {
        "video": a.video,
        "video_meta": {"nb_frames": nb, "native_fps": native,
                       **probe_rotation(a.video)},
        "sampling": {"method": "uniform_integer_decimation", "step": step,
                     "requested_fps": a.fps,
                     "sample_fps": native / step, "filters": "none"},
        "n_frames": len(idx),
        "frames": [{"raw_frame_index": i, "t_sec": i / native,
                    "file": f"frame_{i:06d}.jpg"} for i in idx],
    }
    Path(a.out).write_text(json.dumps(doc, indent=2))
    print(f"{a.video}: nb={nb} native={native:.3f}fps step={step} "
          f"-> {len(idx)} frames @ {native/step:.4f} fps -> {a.out}")


if __name__ == "__main__":
    main()
