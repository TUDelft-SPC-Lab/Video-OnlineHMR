"""Bodies at frame t vs detected keypoints at t+k, stored extrinsics, clearly-visible people; share within 40 px per k.
Usage: [BATCH=2] python scripts/cosilab_calib_12_14/time_scan.py 12 13"""
import os
import sys, types, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from scipy.optimize import linear_sum_assignment
from lib.models.smpl import SMPL
from lib.utils import cosilab
from cosilab_project_boxes import collect_bodies
from cosilab_refit_extrinsic import project, clearly_visible
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
a=types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam',single_root=f'{E}/cosilab_smpl',kp_conf=0.3)
B=int(os.environ.get('BATCH',1))
best=collect_bodies(a,(1,2,3,4),B,6000,6600)
keys=sorted(best); rows=[best[k] for k in keys]
with torch.no_grad():
    J=SMPL()(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for *_,d,i in rows]))[:,None],body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for *_,d,i in rows]))[:,1:],
        betas=torch.tensor(np.stack([d['pred_shape'][i] for *_,d,i in rows])),transl=torch.tensor(np.stack([d['world_trans'][i] for *_,d,i in rows])),pose2rot=False).joints[:,cosilab.CONFLAB_TO_SPIN].numpy().astype(float)
fr=np.array([k[0] for k in keys])
for cam in map(int,sys.argv[1:]):
    k=cosilab.load_camera(cosilab.DATA_ROOT,cam,use_refit=False)
    uv,front,dist=project(J.reshape(-1,3),k['rvec'],k['t_w2c'],k['K'],k['D']); uv=uv.reshape(-1,17,2); front=front.reshape(-1,17); dist=dist.reshape(-1,17)
    det=np.load(f'{E}/cosilab_vit003_extra/cam{cam}_batch{B:02d}/detections.npz'); m=det['kind']=='detector'
    byf={f:det['kp'][m&(det['frame']==f)] for f in np.unique(det['frame'][m])}
    out=[]
    for kk in range(-300,301,15):
        n=hit=0
        for t in range(6000,6600,10):
            if t+kk not in byf: continue
            s=fr==t; u,fv=uv[s],front[s]; vis=clearly_visible(u,fv,dist[s],0.3,0.2)
            if not vis.any(): continue
            u,fv=u[vis],fv[vis]; kp=byf[t+kk]
            ok=fv[:,None]&(kp[None,:,:,2]>0.3); e=np.linalg.norm(u[:,None]-kp[None,:,:,:2],axis=-1)
            with np.errstate(all='ignore'): M=np.nanmedian(np.where(ok,e,np.nan),-1)
            M=np.where(ok.sum(-1)>=6,M,1e4); r,c=linear_sum_assignment(M); n+=len(r); hit+=(M[r,c]<40).sum()
        out.append((kk,hit/max(n,1),n))
    b=max(out,key=lambda x:x[1])
    print(f'cam{cam} batch{B:02d}: best k {b[0]:+d} frames ({b[0]/59.94:+.2f} s) u40 {b[1]:.2f}; k=0 u40 {[x[1] for x in out if x[0]==0][0]:.2f}; profile '+' '.join(f'{x[0]:+d}:{x[1]:.2f}' for x in out[::2]),flush=True)
