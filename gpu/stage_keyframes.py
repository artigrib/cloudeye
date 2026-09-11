#!/usr/bin/env python3
"""Stage 1: extract keyframes from the job's input video, then enforce the VRAM budget
by uniformly subsampling if vidmap.py kept more than max_keyframes.

Runs under the vidmap venv. Shells out to vidmap.py (already fully parameterized, no
changes needed there) rather than importing it, keeping this stage a thin, testable
wrapper around a script that's already proven correct on its own.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import job_paths, load_params  # noqa: E402


DEFAULT_FPS = 2.0
# 100.0 rejected every frame of two different real indoor walkthroughs (a corridor, a
# bedroom) whose frames were visibly sharp on inspection - Laplacian variance on a
# large smooth painted wall reads as "blurry" regardless of actual focus, since the
# metric measures edge/texture content, not focus. 15.0 is the value that let both
# through; both of this project's known-good reference videos (bathroom, tiled/fixture-
# heavy) cleared 100.0 without needing an override, so there was real margin to lower
# this without needing every upload to know about blur_threshold. See setup-log.md.
DEFAULT_BLUR_THRESHOLD = 15.0
DEFAULT_SIMILARITY_THRESHOLD = 0.98
DEFAULT_MAX_KEYFRAMES = 200


def main() -> None:
    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)

    fps = params.get("keyframe_fps", DEFAULT_FPS)
    blur_threshold = params.get("blur_threshold", DEFAULT_BLUR_THRESHOLD)
    similarity_threshold = params.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    max_keyframes = params.get("max_keyframes", DEFAULT_MAX_KEYFRAMES)

    vidmap_script = Path(__file__).resolve().parent / "vidmap.py"
    cmd = [
        sys.executable,
        str(vidmap_script),
        str(paths["input"]),
        str(paths["job_dir"]),
        "--fps", str(fps),
        "--blur-threshold", str(blur_threshold),
        "--similarity-threshold", str(similarity_threshold),
    ]
    print(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)

    keyframes_dir = paths["keyframes_dir"]
    kept = sorted(keyframes_dir.glob("kf_*.jpg"))
    n_extracted = len(kept)
    print(f"vidmap kept {n_extracted} keyframes")

    subsampled = False
    if n_extracted > max_keyframes:
        # Uniform subsample rather than "first N" - a walkthrough's later frames are just
        # as informative as its earlier ones, and MapAnything's GPU memory scales
        # ~78MB/frame + 4.7GB fixed (measured), so this is a real VRAM budget guard, not
        # a formality.
        step = n_extracted / max_keyframes
        indices = {int(i * step) for i in range(max_keyframes)}
        for i, path in enumerate(kept):
            if i not in indices:
                path.unlink()
        kept = sorted(keyframes_dir.glob("kf_*.jpg"))
        subsampled = True
        print(
            f"exceeded max_keyframes={max_keyframes}: uniformly subsampled "
            f"{n_extracted} -> {len(kept)}"
        )

    paths["keyframes_json"].write_text(
        json.dumps(
            {
                "extracted_count": n_extracted,
                "kept_count": len(kept),
                "subsampled": subsampled,
                "max_keyframes": max_keyframes,
                "fps": fps,
                "blur_threshold": blur_threshold,
                "similarity_threshold": similarity_threshold,
            },
            indent=2,
        )
    )

    if len(kept) < 3:
        # A room walkthrough that yields fewer than 3 usable keyframes almost certainly
        # means the source video is broken (too short, too blurry throughout, wrong
        # file) - fail loudly here rather than letting MapAnything choke on it later
        # with a much less legible error.
        raise RuntimeError(
            f"only {len(kept)} usable keyframes extracted from the input video - "
            "too few for a 3D reconstruction. Check the source video isn't corrupt, "
            "too short, or uniformly blurry."
        )

    print(f"DONE: {len(kept)} keyframes ready in {keyframes_dir}")


if __name__ == "__main__":
    main()
