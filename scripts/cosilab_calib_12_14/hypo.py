import sys, json, types, numpy as np, cv2
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data
C='/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/idiap_params'
def rt(p):
    d=json.load(open(p)); return np.array(d['rvec'],float).reshape(3), np.array(d['tvec'],float).reshape(3)
tops={c:cosilab.load_camera(cosilab.DATA_ROOT,c) for c in (1,2,3,4)}
for cam in (12,14):
    a=types.SimpleNamespace(multi_root='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_multiview_4cam',single_root='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl',kp_conf=0.3,det_dir='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_vit003_extra',neighbours=None,start=6000,num_frames=600,every=10)
    data=Data(a,cam,1); st=cosilab.load_camera(cosilab.DATA_ROOT,cam,use_refit=False); K,D=st['K'],st['D']
    other=14 if cam==12 else 12
    H={}
    for c in (12,14):
        rr,tr=rt(f'{C}/relative_mingling_2_side/camera-{c}/extrinsic.json'); Rr=cv2.Rodrigues(rr)[0]
        for tc,tcam in tops.items():
            R=Rr@tcam['R_w2c']; t=Rr@tcam['t_w2c']+tr
            H[f'rel{c} via cam{tc:02d}']=(cv2.Rodrigues(R)[0].ravel(),t)
            # inverse relative (rel is c2w of side in top frame?)
            Ri=Rr.T; ti=-Rr.T@tr; R=Ri@tcam['R_w2c']; t=Ri@tcam['t_w2c']+ti
            H[f'inv rel{c} via cam{tc:02d}']=(cv2.Rodrigues(R)[0].ravel(),t)
        H[f'rel{c} as world']=(rr,tr)
        rs,ts=rt(f'{C}/mingling_2_side/camera_{c}/extrinsic.json'); H[f'stored {c}']=(rs,ts)
    res=sorted(((data.score(r,t,K,D)['under_40px'],n) for n,(r,t) in H.items()),reverse=True)
    print(f'cam{cam}: '+', '.join(f'{n} {s:.2f}' for s,n in res[:6]),flush=True)
