"""Which world->camera extrinsic of cam12/14 best explains the detected VIT003 keypoints (projected 4-cam joints)."""
import sys, json, types, numpy as np, torch, cv2
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from scipy.optimize import linear_sum_assignment
from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye
from cosilab_project_boxes import collect_bodies
torch.set_grad_enabled(False)
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
C='/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/idiap_params'
def rt(path):
    d=json.load(open(path)); return cv2.Rodrigues(np.array(d['rvec'],float).reshape(3))[0], np.array(d['tvec'],float).reshape(3)
c04=cosilab.load_camera(cosilab.DATA_ROOT,4); R4,t4=c04['R_w2c'],c04['t_w2c']
args=types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam', single_root=f'{E}/cosilab_smpl', kp_conf=0.3)
best=collect_bodies(args,(1,2,3,4),1,6000,6600)
keys=[k for k in sorted(best) if k[0]%10==0]
smpl=SMPL(); rows=[best[k] for k in keys]
J=smpl(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for *_,d,i in rows]))[:,None],
       body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for *_,d,i in rows]))[:,1:],
       betas=torch.tensor(np.stack([d['pred_shape'][i] for *_,d,i in rows])),
       transl=torch.tensor(np.stack([d['world_trans'][i] for *_,d,i in rows])),pose2rot=False).joints[:,cosilab.CONFLAB_TO_SPIN].numpy()
fr=np.array([k[0] for k in keys])
for cam in map(int,sys.argv[1:]):
    cm=cosilab.load_camera(cosilab.DATA_ROOT,cam); K,D=cm['K'],cm['D']
    Rr,tr=rt(f'{C}/relative_mingling_2_side/camera-{cam}/extrinsic.json'); Rs,ts=cm['R_w2c'],cm['t_w2c']
    Ri,ti=R4.T,-R4.T@t4  # cam04 c2w
    Cs=-Rs.T@ts; cands={}
    for dy in np.arange(-0.5,4.01,0.5):
        for dx in (-0.5,0,0.5):
            for dz in (0,0.5):
                Cn=Cs+np.array([dx,dy,dz]); cands[f'shift {dx:+.1f},{dy:+.1f},{dz:+.1f}']=(Rs,-Rs@Cn)
    d=np.load(f'{E}/cosilab_vit003_extra/cam{cam}_batch01/detections.npz'); det=d['kind']=='detector'
    res=[]
    for name,(R,t) in cands.items():
        Xc=J@R.T+t; front=Xc[...,2]>0
        uv=fisheye.project_torch(torch.tensor(Xc,dtype=torch.float32),torch.tensor(K,dtype=torch.float32),torch.tensor(D,dtype=torch.float32)).numpy()
        errs=[]; inside=[]
        for f in np.unique(fr):
            m=fr==f; kp=d['kp'][det&(d['frame']==f)]
            u=uv[m]; ok=front[m][:,None]&(kp[None,:,:,2]>0.3)
            inside.append(((u[...,0]>=0)&(u[...,0]<1920)&(u[...,1]>=0)&(u[...,1]<1080)&front[m]).all(1).mean())
            e=np.linalg.norm(u[:,None]-kp[None,:,:,:2],axis=-1)
            with np.errstate(all='ignore'): M=np.nanmedian(np.where(ok,e,np.nan),-1)
            M=np.where(ok.sum(-1)>=5,M,1e4); r,c=linear_sum_assignment(M); errs+=[x for x in M[r,c] if x<1e4]
        errs=np.array(errs)
        res.append((np.median(errs),f'cam{cam} {name:<22} centre {(-R.T@t).round(2)}  bodies fully inside {np.mean(inside):.2f}  matched median {np.median(errs):6.1f} px, <40 px {np.mean(errs<40):.2f}'))
    for r in sorted(res)[:4]: print(r[1])
