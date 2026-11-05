"""Timing check: projected 4-cam joints (visibility.npz joint_uv, batch frame t) vs detector VIT003 keypoints at t+k."""
import sys, numpy as np
from scipy.optimize import linear_sum_assignment
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_vit003_extra'
for c in sys.argv[1:]:
    v=np.load(f'{E}/cam{c}_batch01/visibility.npz'); d=np.load(f'{E}/cam{c}_batch01/detections.npz')
    det=d['kind']=='detector'; df=d['frame'][det]; dkp=d['kp'][det]
    byf={f:dkp[df==f] for f in np.unique(df)}
    vf=v['frame']; res=[]
    for k in range(-5,6):
        errs=[]
        for f in range(6005,6595,3):
            m=(vf==f)&(v['vis_frac']>=0.5)
            if f+k not in byf or not m.any(): continue
            uv=v['joint_uv'][m]; jv=v['joint_vis'][m]; kp=byf[f+k]
            ok=jv[:,None]&(kp[None,:,:,2]>0.3)
            e=np.linalg.norm(uv[:,None]-kp[None,:,:,:2],axis=-1)
            with np.errstate(all='ignore'):
                M=np.nanmedian(np.where(ok,e,np.nan),-1)
            M=np.where(ok.sum(-1)>=5,M,1e3)
            r,cc=linear_sum_assignment(M)
            errs+= [x for x in M[r,cc] if x<100]
        res.append((k,np.median(errs),len(errs)))
    print(f'cam{c}: '+'  '.join(f'{k:+d}:{m:.1f}' for k,m,n in res), '| n', res[5][2])
