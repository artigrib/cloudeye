"""Gate for step 2: does our preprocessed frame match MapAnything's own input?

Ground truth = img_no_norm from the hero-74 pipeline run (74 views, 518x294 float32 0..1).
Our candidate = the calibrated recipe scale=294:523:bicubic,crop=294:518:0:2 applied to the
same source video. The hero-74 view -> raw_frame_index mapping is NOT recorded anywhere
(manifest.json keys on keyframe sequence numbers, and grid_meta.json says
frame_index_mapping_confirmed: false), so it is SEARCHED here: for each probe view we scan a
window of raw indices and keep the argmax. A mapping that is real shows up as a consistent
offset across views, not as five unrelated best matches.
"""
import json, subprocess, sys, random
from pathlib import Path
import numpy as np
from PIL import Image

SCENE = Path("var/scratch/hero-frozen/own_0901_173903/scene")
VIDEO = "var/uploads/c6bdbc7e-3e33-49d6-92dd-5316a412ede1.mp4"
TMP = Path(sys.argv[1]) / "ncc"; TMP.mkdir(parents=True, exist_ok=True)
WINDOW = 8            # +/- raw frames searched around the predicted index
NATIVE_FPS = 29.98897137610395
KEYFRAME_FPS = 2.0

def preprocess(raw_idx: int) -> np.ndarray:
    """Our candidate preprocessing, byte-identical to nvblox_v2/extract_hero_rgb.sh
    (verified: md5 match on 5 frames)."""
    p = TMP / f"f_{raw_idx:06d}.jpg"
    if not p.exists():
        subprocess.run(["ffmpeg","-y","-nostdin","-loglevel","error","-i",VIDEO,
            "-vf", rf"select='eq(n\,{raw_idx})',scale=294:523:flags=bicubic,crop=294:518:0:2",
            "-vsync","0","-qscale:v","2","-frames:v","1",str(p)], check=True)
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64); y = b.reshape(-1).astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d else 0.0

kept = [i for i, e in enumerate(json.loads((SCENE/"manifest.json").read_text())) if e["kept"]]
print(f"kept keyframes: {len(kept)} of 153")

random.seed(0)
probes = sorted(random.sample(range(74), 5))
rows = []
for v in probes:
    seq = kept[v]                                   # 0-based keyframe sequence number
    predicted = int(round(seq / KEYFRAME_FPS * NATIVE_FPS))
    gt = np.load(SCENE / f"per_view/view_{v:03d}.npz")["img_no_norm"]
    best = max(((ncc(gt, preprocess(r)), r)
                for r in range(max(0, predicted-WINDOW), min(2295, predicted+WINDOW+1))))
    rows.append((v, seq, predicted, best[1], best[1]-predicted, best[0]))
    print(f"  view_{v:03d}  seq={seq:3d}  predicted_raw={predicted:5d}  "
          f"best_raw={best[1]:5d}  offset={best[1]-predicted:+3d}  NCC={best[0]:.4f}")

nccs = np.array([r[5] for r in rows]); offs = [r[4] for r in rows]
print(f"\nNCC over 5 views: min {nccs.min():.4f}  mean {nccs.mean():.4f}  max {nccs.max():.4f}")
print(f"offsets: {offs}  -> {'CONSISTENT' if len(set(offs))==1 else 'NOT consistent'}")
print(f"GATE (>=0.95 on all 5): {'PASS' if (nccs>=0.95).all() else 'FAIL'}")
json.dump({"probes": rows, "ncc_min": float(nccs.min()), "ncc_mean": float(nccs.mean()),
           "offsets": offs}, open(TMP/"gate.json","w"), indent=1)
