"""Fit cam04's pose at side-calibration time (anchor), shared by cam12 and cam14 whose poses relative to it come from the boards."""
import json, sys, types, numpy as np, cv2
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from scipy.optimize import minimize
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
C='/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/idiap_params/relative_mingling_2_side'
def rel(c):
    d=json.load(open(f'{C}/camera-{c}/extrinsic.json')); return cv2.Rodrigues(np.array(d['rvec'],float).reshape(3))[0], np.array(d['tvec'],float).reshape(3)
a=types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam',single_root=f'{E}/cosilab_smpl',kp_conf=0.3,det_dir=f'{E}/cosilab_vit003_extra',
    neighbours=None,start=6000,num_frames=600,every=10,all_people=False,occ_depth=0.3,max_occ=0.2)
cams={}
for c in (12,14):
    k=cosilab.load_camera(cosilab.DATA_ROOT,c,use_refit=False); dd=Data(a,c,1); dd.n_ref=sum(len(J) for J,*_ in dd.per_frame(k['rvec'],k['t_w2c'],k['K'],k['D']))
    cams[c]=(dd,k['K'],k['D'],*rel(c))
c4=cosilab.load_camera(cosilab.DATA_ROOT,4); RA0,CA0=c4['R_w2c'],-c4['R_w2c'].T@c4['t_w2c']
def poses(y):  # y: rotation (deg, about cam04's own axes), centre shift (m)
    RA=cv2.Rodrigues(np.radians(y[:3]))[0]@RA0; tA=-RA@(CA0+y[3:])
    out={}
    for c,(dd,K,D,Rr,tr) in cams.items():
        R=Rr@RA; out[c]=(cv2.Rodrigues(R)[0].ravel(), Rr@tA+tr)
    return out
def f(y):
    p=poses(y); return sum(cams[c][0].coarse_cost(*p[c],cams[c][1],cams[c][2],50.0) for c in cams)/2
def report(tag,y):
    p=poses(y); s={c:cams[c][0].score(*p[c],cams[c][1],cams[c][2]) for c in cams}
    print(f'{tag}: anchor rot {np.round(y[:3],2)} deg, shift {np.round(y[3:],2)} m; cost {f(y):.1f}; '+'; '.join(f'cam{c} u40 {s[c]["under_40px"]:.2f} med {s[c]["median_px"]:.1f} ({s[c]["n_under_40px"]})' for c in s),flush=True)
report('4TU cam04 (stored)',np.zeros(6))
starts=[np.r_[np.array([a_,b_,c_]),np.zeros(3)] for a_ in (-10,-5,0,5,10) for b_ in (-10,-5,0,5,10) for c_ in (-10,0,10)]
starts=sorted(starts,key=f)[:6]
best=None
for x in starts:
    r=minimize(f,x,method='Nelder-Mead',options={'maxiter':800,'xatol':1e-3,'fatol':1e-3,'initial_simplex':x+np.vstack([np.zeros(6),np.diag([2.0]*3+[0.2]*3)])})
    if best is None or r.fun<best.fun: best=r
report('anchor fit',best.x)
np.save(sys.argv[1] if len(sys.argv)>1 else '/dev/null',best.x)
