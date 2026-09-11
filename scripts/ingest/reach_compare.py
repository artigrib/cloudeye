"""Reachability on the nvblox layers, and the reconciliation against nvblox_v2's own table.

Free space here = EDT over (esdf_slice <= 0 OR unobserved) >= r, which is what
`pathfinding.inflate`'s disk dilation by ceil(r/res) cells computes on this grid - so the
stock endpoint needs no new code path. Both are computed below and compared, because
"equivalent" is a claim that should carry a number.

The nvblox table used a different obstacle set: mesh vertices in the 0.1-1.5 m BAND above the
floor, not one 0.3 m slice. The band-vs-slice cell counts are measured here from our own
coloured cloud, on the same 0.05 m grid.
"""
import json, sys
import numpy as np
from scipy import ndimage

S = sys.argv[1]
SD = sys.argv[2] if len(sys.argv) > 2 else "var/uploads/scenes/62a4e533-00a5-4b67-a5ce-a2fb02597889/"
LAY = SD + "nvblox/"
REACH = json.load(open("var/nvblox_v2/results/reach/own_0901_173903/reachability.json"))
CELL = 0.05

esdf = np.load(LAY + "esdf_slice_0.3m.npy")
unobs = np.load(LAY + "unobserved_mask.npy")
obst = unobs | (np.nan_to_num(esdf, nan=-1.0) <= 0.0)
observed = ~unobs
edt = ndimage.distance_transform_edt(~obst) * CELL

print(f"grid {obst.shape}  obstacle cells {obst.sum()}  observed {observed.sum()} "
      f"({observed.mean():.4f})  EDT max {edt.max():.4f} m")
print(f"nvblox: observed_frac {REACH['observed_frac']:.4f}  nav2d_clearance_max_m "
      f"{REACH['nav2d_clearance_max_m']:.4f}  esdf3d_slice_max_m {REACH['esdf3d_slice_max_m']:.4f}\n")


def inflate_like_pathfinding(cells_obst, r):
    k = int(np.ceil(r / CELL))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return ndimage.binary_dilation(cells_obst, structure=(xx ** 2 + yy ** 2) <= k * k)


print("r        free_frac(EDT)  free_frac(inflate)  cells differing")
for r in (0.10, 0.2496, 0.30, 0.5528, 0.60):
    f_edt = edt >= r
    f_inf = ~inflate_like_pathfinding(obst, r)
    print(f"{r:.4f}   {f_edt[observed].mean():.4f}          {f_inf[observed].mean():.4f}"
          f"            {int((f_edt ^ f_inf).sum())}")


def quadrant_anchors(free, dist):
    H, W = free.shape
    out = {}
    for name, rs, cs in (("NW", slice(0, H // 2), slice(0, W // 2)),
                         ("NE", slice(0, H // 2), slice(W // 2, W)),
                         ("SW", slice(H // 2, H), slice(0, W // 2)),
                         ("SE", slice(H // 2, H), slice(W // 2, W))):
        sub = np.where(free[rs, cs], dist[rs, cs], -1)
        if sub.max() <= 0:
            out[name] = None
            continue
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        out[name] = (i + (rs.start or 0), j + (cs.start or 0))
    return out


print("\nr        free_frac  n_components  reachable_pairs")
res = {}
for r in (0.10, 0.2496, 0.30, 0.5528, 0.60):
    free = edt >= r
    lab, n = ndimage.label(free, structure=np.ones((3, 3)))
    anch = quadrant_anchors(free, edt)
    names = "NW NE SW SE".split()
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ok = sum(1 for a, b in pairs
             if anch[a] and anch[b] and lab[anch[a]] == lab[anch[b]] != 0)
    res[r] = (float(free[observed].mean()), int(n), ok)
    print(f"{r:.4f}   {free[observed].mean():.4f}     {n:3d}           {ok}/6")

r03 = [x for x in REACH["runs"] if x["radius_m"] == 0.3]
print(f"\nnvblox r=0.3: free_frac_nav2d {r03[0]['free_frac_nav2d']:.4f}, "
      f"reachable {sum(1 for x in r03 if x['reachable'])}/{len(r03)}")
for kk in ("free_frac_esdf3d", "free_frac_nav2d_no_component_filter",
           "free_frac_if_unknown_traversable"):
    v = REACH.get(kk, r03[0].get(kk) if r03 else None)
    print(f"nvblox {kk}: {v}")

d = np.load(S + "/out/cloud_002.npz")
P = d["xyz"]
gm = json.load(open(LAY + "grid_meta.json"))["our_grid"]
u0, v0 = gm["u_range"][0], gm["v_range"][0]
nu, nv = obst.shape
band = P[(P[:, 2] >= 0.1) & (P[:, 2] <= 1.5)]
iu = np.floor((band[:, 0] - u0) / CELL).astype(int)
iv = np.floor((band[:, 1] - v0) / CELL).astype(int)
k = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
band_obst = np.zeros_like(obst)
band_obst[iu[k], iv[k]] = True
slice_obst = np.nan_to_num(esdf, nan=-1.0) <= 0.0
print("\nobstacle cells on the same 75x140 grid, observed area only:")
print(f"  0.1-1.5 m band (from our cloud): {int((band_obst & observed).sum())}")
print(f"  0.3 m slice only (esdf<=0):      {int((slice_obst & observed).sum())}")
print(f"  band adds, slice misses:         {int((band_obst & ~slice_obst & observed).sum())}")
print(f"  slice has, band misses:          {int((slice_obst & ~band_obst & observed).sum())}")
print(f"  both:                            {int((band_obst & slice_obst & observed).sum())}")
json.dump({str(kk): vv for kk, vv in res.items()}, open(S + "/out/reach.json", "w"), indent=1)
