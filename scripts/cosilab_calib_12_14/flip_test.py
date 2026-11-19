import sys, types, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data
E='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
for cam in map(int,sys.argv[1:]):
    a=types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam',single_root=f'{E}/cosilab_smpl',kp_conf=0.3,det_dir=f'{E}/cosilab_vit003_extra',neighbours=None,start=6000,num_frames=600,every=10,all_people=False,occ_depth=0.3,max_occ=0.2)
    d=Data(a,cam,1); k=cosilab.load_camera(cosilab.DATA_ROOT,cam,use_refit=False)
    orig=[(J,kp.copy()) for J,kp in d.frames]; out=[]
    for name,fx,fy in [('as is',0,0),('rot180',1,1),('hflip',1,0),('vflip',0,1)]:
        d.frames=[]
        for J,kp in orig:
            kp=kp.copy()
            if fx: kp[...,0]=1919-kp[...,0]
            if fy: kp[...,1]=1079-kp[...,1]
            d.frames.append((J,kp))
        s=d.score(k['rvec'],k['t_w2c'],k['K'],k['D']); out.append(f'{name} {s["under_40px"]:.2f} (med {s["median_px"]:.0f})')
    print(f'cam{cam}: '+', '.join(out),flush=True)
