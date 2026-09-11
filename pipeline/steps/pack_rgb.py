#!/usr/bin/env python3
"""PACK_RGB: the colour half of the packed stack, so nvblox can integrate it.

PACK produces `depth_u16.npy` and nothing else. MapAnything's `per_view/*.npz` carry
`pts3d`, `mask` and `conf` - no RGB at all - and FRAMES writes a frame INDEX, not JPEGs, so
there is no colour anywhere on the VPS side of the pipeline. That is the real reason every
live mesh is flat grey: not a lost setting, an input that was never assembled.

This re-derives colour from the source video in ONE ffmpeg pass and writes `rgb_u8.npy`
beside `depth_u16.npy`, shaped (N, h, w, 3) uint8 in `meta.json`'s own view order, so
`nvblox_scenes.py --rgb-file rgb_u8.npy` can hand frame i's colour to the same pose and
sensor that took frame i's depth.

The preprocessing must match MapAnything's own 518x294 input pixel-for-pixel, or the colour
lands offset from the geometry. The transform is NOT guessed: nvblox_v2/extract_hero_rgb.sh
calibrated "scale the short side to 294, centre-crop the long side to 518" against the real
`img_no_norm` images of the 74-view hero run at NCC 0.9637, against 0.9579 for a plain
resize and 0.9389 for pad-to-width, winning in the top, middle and bottom thirds separately.
1080 -> 294 puts the long side at 522.67 px, so ffmpeg is given 523 and a 2 px top crop.
`scripts/ingest/colorize_cloud.py` on the ingest-15fps branch re-derived the same recipe
independently and its output is md5-identical on 5 frames, which is the closest thing to a
second opinion this recipe has.

Why an .npy stack and not the JPEGs: the box's venv has numpy but NOT Pillow
(pipeline/box/onstart_nvblox.sh installs torch, the nvblox wheel, numpy and open3d only),
and the driver's existing push/pull/transfer-gate machinery moves named files out of the
packed dir. One .npy needs neither a new dependency on the box nor a new transfer path.
It is bigger on the wire than a JPEG tarball (525 MB vs 23 MB for the hero) - if the link
becomes the constraint, the tarball is the change to make, not a different recipe.

    pack_rgb.py --video <mp4> --packed <scene-dir> [--out rgb_u8.npy]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

#: The short side every MapAnything input has, whichever way up the phone was held.
SHORT_SIDE = 294


def display_size(video: Path) -> tuple[int, int]:
    """The frame size AFTER rotation metadata, which is what ffmpeg decodes and what
    MapAnything saw.

    This is the whole reason `own_0828_152850` came out transposed. Both it and the hero are
    stored 1920x1080; the hero carries `rotation: -90` and so decodes as 1080x1920 portrait,
    while 152850 carries none and decodes landscape. Reading the stored size alone would
    build a portrait recipe for a landscape frame and paint the colour on sideways - and
    nothing downstream would notice, because a sideways-coloured mesh is still a mesh.
    """
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "csv=p=0:nk=1", str(video)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ffprobe failed on {video}: {p.stderr.strip()[:300]}")
    nums = [int(float(x)) for x in p.stdout.replace("\n", ",").split(",") if x.strip()]
    w, h = nums[0], nums[1]
    rot = nums[2] if len(nums) > 2 else 0
    return (h, w) if abs(rot) % 180 == 90 else (w, h)


def build_recipe(src_w: int, src_h: int, tgt_w: int, tgt_h: int, step: int) -> str:
    """`extract_hero_rgb.sh`'s rule, stated for either orientation.

    That script hardcodes `scale=294:523,crop=294:518:0:2` for the portrait hero, and the
    rule behind those numbers is written above them: "scale the short side to 294,
    centre-crop the long side to 518". Applied to the hero's own 1080x1920 this function
    returns exactly that string, which is what makes it a generalisation of the calibrated
    recipe rather than a second, uncalibrated one.

    An orientation MISMATCH between video and pack is refused rather than rotated into
    place: it means the pack was not built from this video, and no amount of scaling fixes
    that.
    """
    if (src_w >= src_h) != (tgt_w >= tgt_h):
        raise SystemExit(
            f"video decodes {src_w}x{src_h} but the pack wants {tgt_w}x{tgt_h} - opposite "
            f"orientations. That is a different video, or rotation metadata this did not "
            f"see; it is not something to scale away.")
    if tgt_h >= tgt_w:                                  # portrait: short side is the width
        sw, sh = tgt_w, round(src_h * tgt_w / src_w)
        cx, cy = 0, (sh - tgt_h) // 2
    else:                                               # landscape: short side is the height
        sh, sw = tgt_h, round(src_w * tgt_h / src_h)
        cx, cy = (sw - tgt_w) // 2, 0
    return (f"select='not(mod(n\\,{step}))',"
            f"scale={sw}:{sh}:flags=bicubic,crop={tgt_w}:{tgt_h}:{cx}:{cy}")


def derive_step(raw: list[int]) -> int:
    """The integer decimation step behind these raw frame indices.

    `frames_by_fps.py` samples by an integer step over native fps, so the indices are an
    arithmetic run from 0 - which is what lets one `select='not(mod(n,step))'` reproduce
    them. Anything else (a gap, a non-zero start, a variable step) would silently pair
    frame i's colour with frame j's depth, so it is refused rather than approximated.
    """
    if len(raw) < 2:
        raise SystemExit(f"need at least 2 views to derive a step, got {len(raw)}")
    step = raw[1] - raw[0]
    if raw[0] != 0 or step <= 0 or raw != list(range(0, raw[-1] + 1, step)):
        raise SystemExit(
            f"raw_frame_index is not an arithmetic run from 0 (start={raw[0]}, "
            f"step={step}, n={len(raw)}, last={raw[-1]}). One ffmpeg `select` by modulo "
            f"cannot reproduce it, and guessing the mapping would pair one frame's colour "
            f"with another frame's depth. Extract by explicit index instead.")
    return step


def extract(video: Path, vf: str, dest: Path) -> list[Path]:
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(video),
           "-vf", vf, "-vsync", "0", "-qscale:v", "2", str(dest / "_seq_%06d.jpg")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed (rc={p.returncode}): {p.stderr.strip()[:400]}")
    return sorted(dest.glob("_seq_*.jpg"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--packed", required=True,
                    help="the packed scene dir - the one holding meta.json and depth_u16.npy")
    ap.add_argument("--out", default="rgb_u8.npy", help="written inside --packed")
    ap.add_argument("--keep-frames", default=None,
                    help="keep the extracted JPEGs here instead of a temp dir (debugging)")
    a = ap.parse_args()

    from PIL import Image                                    # only this script needs it

    packed = Path(a.packed)
    meta = json.loads((packed / "meta.json").read_text())
    views = meta["views"]
    h, w = meta["shape_hw"]
    if min(w, h) != SHORT_SIDE:
        raise SystemExit(f"{packed}/meta.json says shape_hw {[h, w]}; its short side is "
                         f"{min(w, h)}, not {SHORT_SIDE}. The transform is calibrated for a "
                         f"{SHORT_SIDE}-px short side and a different one needs re-calibrating "
                         f"against real img_no_norm images, not rescaling.")

    raw = [int(v["raw_frame_index"]) for v in views]
    step = derive_step(raw)
    src_w, src_h = display_size(Path(a.video))
    vf = build_recipe(src_w, src_h, w, h, step)

    t0 = time.time()
    tmp = Path(a.keep_frames) if a.keep_frames else Path(tempfile.mkdtemp(prefix="pack_rgb_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        jpgs = extract(Path(a.video), vf, tmp)
        need = raw[-1] // step + 1
        if len(jpgs) < need:
            raise SystemExit(f"ffmpeg produced {len(jpgs)} frames, need at least {need} "
                             f"to cover raw_frame_index up to {raw[-1]} at step {step}")

        out_path = packed / a.out
        arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                        shape=(len(views), h, w, 3))
        for i, r in enumerate(raw):
            im = Image.open(jpgs[r // step]).convert("RGB")
            if im.size != (w, h):
                raise SystemExit(f"{jpgs[r // step]}: {im.size} != {(w, h)}")
            arr[i] = np.asarray(im, np.uint8)
        arr.flush()
        del arr
    finally:
        if not a.keep_frames:
            shutil.rmtree(tmp, ignore_errors=True)

    doc = {"source_video": str(Path(a.video).resolve()),
           "n_views": len(views), "shape": [len(views), h, w, 3], "dtype": "uint8",
           "frame_step": step, "raw_frame_index_first_last": [raw[0], raw[-1]],
           "recipe": vf, "video_display_size": [src_w, src_h],
           "recipe_calibrated_by": "nvblox_v2/extract_hero_rgb.sh (NCC 0.9637 vs img_no_norm)",
           # There is no per-view ground truth to check against here: the pulled bundles
           # carry pts3d/mask/conf and no img_no_norm, so the NCC gate that calibrated the
           # recipe cannot be re-run per scene. Said plainly rather than implied by silence.
           "ncc_gate": "not run - the pulled per_view npz carry no img_no_norm to compare to",
           "bytes": out_path.stat().st_size, "wall_s": round(time.time() - t0, 2)}
    (packed / "rgb_meta.json").write_text(json.dumps(doc, indent=2))
    print(f"wrote {out_path} {doc['shape']} uint8, {doc['bytes'] / 1e6:.1f} MB "
          f"in {doc['wall_s']}s (step {step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
