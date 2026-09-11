#!/usr/bin/env python3
"""Extract sharp, non-redundant keyframes from a continuous walkthrough video."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def extract_frames(video_path: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(video_path), "-vf", f"fps={fps}", "-qscale:v", "2", pattern]
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    return sorted(out_dir.glob("frame_*.jpg"))


def laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale histogram correlation. 1.0 = identical, lower = more different."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ha = cv2.calcHist([ga], [0], None, [256], [0, 256])
    hb = cv2.calcHist([gb], [0], None, [256], [0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--blur-threshold", type=float, default=100.0,
                     help="drop frames with Laplacian variance below this")
    ap.add_argument("--similarity-threshold", type=float, default=None,
                     help="drop frames with histogram correlation >= this vs last kept frame (0-1); omit to disable dedup")
    ap.add_argument("--keep-raw", action="store_true", help="keep the raw extracted frames dir")
    args = ap.parse_args()

    raw_dir = args.out_dir / "_raw"
    kept_dir = args.out_dir / "keyframes"
    kept_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = extract_frames(args.video, raw_dir, args.fps)
    manifest = []
    last_kept_img = None
    n_kept = 0

    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sharpness = float(laplacian_variance(img))
        blurry = bool(sharpness < args.blur_threshold)

        similar = False
        sim_score = None
        if not blurry and args.similarity_threshold is not None and last_kept_img is not None:
            sim_score = float(frame_similarity(img, last_kept_img))
            similar = bool(sim_score >= args.similarity_threshold)

        keep = not blurry and not similar
        if keep:
            n_kept += 1
            dest = kept_dir / f"kf_{n_kept:05d}.jpg"
            cv2.imwrite(str(dest), img)
            last_kept_img = img

        manifest.append({
            "source_frame": p.name,
            "sharpness_laplacian_var": round(sharpness, 2),
            "blurry": blurry,
            "similarity_to_last_kept": round(sim_score, 4) if sim_score is not None else None,
            "dropped_as_similar": similar,
            "kept": keep,
        })

    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if not args.keep_raw:
        for p in frame_paths:
            p.unlink()
        raw_dir.rmdir()

    print(f"extracted {len(frame_paths)} frames @ {args.fps}fps -> kept {n_kept} keyframes in {kept_dir}")


if __name__ == "__main__":
    main()
