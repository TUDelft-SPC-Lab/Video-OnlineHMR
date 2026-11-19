import json, sys, numpy as np, cv2
sys.path.insert(0,'.')
from lib.utils import cosilab
P="/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/calibrator_projects/mingling_2_side_Extrinsic Parameters/"
proj, origin = sys.argv[1], int(sys.argv[2])
d=json.load(open(P+proj))
names=[int(d['fileInfo'][0][c]['filePath'].split('/')[-2].split('_')[1].split('-')[0]) for c in range(len(d['fileInfo'][0]))]
op=np.array([[p['x'],p['y'],p['z']] for p in d['calibration']['targets'][0]['objectPoints']])
def T(tr):
    r=tr['rotation']; t=tr['translation']
    return cv2.Rodrigues(np.array([r['rx'],r['ry'],r['rz']]))[0], np.array([t['x'],t['y'],t['z']])
o=cosilab.load_camera(cosilab.DATA_ROOT,origin)  # world->origin cam
Ro,to=o['R_w2c'],o['t_w2c']
for ci,cam in enumerate(names):
    k=cosilab.load_camera(cosilab.DATA_ROOT,cam,use_refit=False); K,D=k['K'],k['D']
    Rc,tc=T(d['calibration']['cameras'][ci]['transform'])  # origin -> cam (project frame)
    Xw,uv=[],[]
    for det in d['detections']:
        if det['cameraId']!=ci: continue
        Rp,tp=T(d['calibration']['poses'][det['poseId']]['transform'])  # board -> origin
        for fp in det['featurePoints']:
            Xo=Rp@op[fp['id']]+tp; Xw.append(Ro.T@(Xo-to)); uv.append([fp['point']['x'],fp['point']['y']])
    Xw,uv=np.array(Xw),np.array(uv)
    def res(x):
        p=cv2.fisheye.projectPoints(Xw.reshape(-1,1,3),x[:3],x[3:],K,D)[0].reshape(-1,2); return (p-uv).ravel()
    for label,(R,t) in [('project chain',(Rc@Ro,Rc@to+tc)),('stored world',(k['R_w2c'],k['t_w2c']))]:
        x=np.r_[cv2.Rodrigues(R)[0].ravel(),t]; e=np.linalg.norm(res(x).reshape(-1,2),axis=1)
        print(f'cam{cam:02d} {label:<14}: {len(uv)} corners, board reprojection median {np.median(e):.2f} px, max {e.max():.2f}')
    # Jacobian wrt (rotation about camera axes in deg, camera centre in m)
    R=k['R_w2c']; C=-R.T@k['t_w2c']
    def resC(y):
        Rn=cv2.Rodrigues(np.radians(y[:3]))[0]@R; return res(np.r_[cv2.Rodrigues(Rn)[0].ravel(),-Rn@(C+y[3:])])
    J=np.stack([(resC(np.eye(6)[i]*1e-3)-resC(-np.eye(6)[i]*1e-3))/2e-3 for i in range(6)],1)
    U,s,Vt=np.linalg.svd(J/np.sqrt(len(uv)),full_matrices=False)
    print(f'   rms px per unit (deg / m) along singular directions: {s.round(2)}; weakest direction (rot deg xyz, centre m xyz): {Vt[-1].round(2)}')
