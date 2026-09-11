"""Attribute the global-cloud render error. Splatting 3.26M points from 1148 views into one
camera has no depth test beyond the painter's order, so points seen THROUGH gaps (behind
walls, other side of the room) land on pixels they do not belong to. Gate the same render on
agreement with that view's OWN depth and see what the surviving pixels score."""
import json, sys
import numpy as np
from PIL import Image
S=sys.argv[1]
LAY="var/nvblox_v2/results/frontend_layers/own_0901_173903/"
RUN="var/scratch/b200_run_20260908/pulled/own_0901_173903__step2/"
gm=json.load(open(LAY+"grid_meta.json"))["our_grid"]; fp=json.load(open(LAY+"floor_plane.json"))
O=np.array(gm["origin_world"]);AU=np.array(gm["axis_u"]);AV=np.array(gm["axis_v"])
UP=np.array(fp["normal_up"]);FPT=np.array(fp["point_on_plane"])
R=np.stack([AU,AV,UP]); t=np.array([-O@AU,-O@AV,-FPT@UP])
d=np.load(S+"/out/cloud_002.npz"); Pw=(d["xyz"].astype(np.float64)-t)@R; C=d["rgb"]
poses={v["raw_frame_index"]:v for v in json.load(open(RUN+"poses.json"))["views"]}
def ncc(a,b):
    x=a.ravel().astype(float);y=b.ravel().astype(float);x-=x.mean();y-=y.mean()
    n=np.linalg.norm(x)*np.linalg.norm(y); return float(x@y/n) if n else 0.
print("raw   all_px_NCC   depth_agreeing_px   agree_NCC   see_through_frac")
tot=[]
for raw in (200,800,1400,2000,2280):
    v=poses[raw]; K=np.array(v["intrinsics"]); T=np.array(v["camera_pose"])
    own=np.load(RUN+f"per_view/view_{raw:06d}.npz"); m=own["mask"]
    own_d=np.full((518,294),np.nan)
    oc=(own["pts3d"][m].astype(np.float64)-T[:3,3])@T[:3,:3]
    rows,cols=np.nonzero(m); own_d[rows,cols]=oc[:,2]
    Pc=(Pw-T[:3,3])@T[:3,:3]; z=Pc[:,2]; k=z>0.05
    uv=Pc[k]@K.T; uv=uv[:,:2]/uv[:,2:3]
    u=np.round(uv[:,0]).astype(int); vv=np.round(uv[:,1]).astype(int)
    ok=(u>=0)&(u<294)&(vv>=0)&(vv<518)
    u,vv,zz,cc=u[ok],vv[ok],z[k][ok],C[k][ok]
    o=np.argsort(-zz)
    img=np.zeros((518,294,3),np.uint8); dep=np.full((518,294),np.nan); hit=np.zeros((518,294),bool)
    img[vv[o],u[o]]=cc[o]; dep[vv[o],u[o]]=zz[o]; hit[vv,u]=True
    frame=np.asarray(Image.open(f"var/nvblox_v2/hero_rgb/frame_{raw:06d}.jpg").convert("RGB"))
    agree=hit&np.isfinite(own_d)&(np.abs(dep-own_d)<0.15)
    n_all=ncc(img[hit],frame[hit]); n_ag=ncc(img[agree],frame[agree])
    see=1-agree.sum()/max(hit.sum(),1); tot.append((n_all,n_ag,see))
    print(f"{raw:5d}   {n_all:.4f}      {100*agree.mean():6.2f}%           {n_ag:.4f}      {100*see:5.1f}%")
a=np.array(tot); print(f"\nmean: all {a[:,0].mean():.4f}   depth-agreeing {a[:,1].mean():.4f}   see-through {100*a[:,2].mean():.1f}%")
