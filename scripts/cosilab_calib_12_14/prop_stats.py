import numpy as np, sys
sys.path.insert(0,'.')
from lib.utils import cosilab
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_vit003_extra'
for c in [12,13,14,15]:
    d=np.load(f'{E}/cam{c}_batch01/proposals.npz',allow_pickle=True)
    b=d['box']; h=b[:,3]-b[:,1]; f=d['frame']
    per=np.bincount(f-f.min())
    # pairwise overlap: fraction of boxes with IoU>0.3 with another box same frame
    ov=0
    for fr in np.unique(f):
        bb=b[f==fr]
        x1=np.maximum(bb[:,None,0],bb[None,:,0]);y1=np.maximum(bb[:,None,1],bb[None,:,1]);x2=np.minimum(bb[:,None,2],bb[None,:,2]);y2=np.minimum(bb[:,None,3],bb[None,:,3])
        inter=(x2-x1).clip(0)*(y2-y1).clip(0); a=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1]); iou=inter/(a[:,None]+a[None]-inter); np.fill_diagonal(iou,0)
        ov+=(iou.max(1)>0.3).sum() if len(bb)>1 else 0
    t=cosilab.raw_frame_table(cosilab.DATA_ROOT,c)
    dt=t['dt'][6000:6600]
    print(f'cam{c}: {len(np.unique(d["pid"]))} people, {per.mean():.1f} per frame, box height px median {np.median(h):.0f} (p10 {np.percentile(h,10):.0f}, p90 {np.percentile(h,90):.0f}), boxes with IoU>0.3 to another: {ov/len(b):.0%}; raw file {set(t["files"][i] for i in t["file"][6000:6600])}, |dt| max {np.abs(dt).max()*1000:.1f} ms')
