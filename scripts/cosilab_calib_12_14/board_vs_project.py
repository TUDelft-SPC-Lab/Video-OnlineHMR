import json, sys, numpy as np, cv2
import os; sys.path.insert(0,'.'); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib.utils import cosilab
from board_fit import board_pose, OBJ
det={(int(c),int(i)):(tuple(s),np.asarray(u,float)) for c,i,s,u in np.load(sys.argv[1],allow_pickle=True)}
P="/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/calibrator_projects/mingling_2_side_Extrinsic Parameters/top_cam_04_side_cam_12_14.json"
d=json.load(open(P)); names=[4,12,14]
op=np.array([[p['x'],p['y'],p['z']] for p in d['calibration']['targets'][0]['objectPoints']])
print('project object points order (first 6):',op[:6,:2].round(2).tolist())
for det_ in d['detections']:
    cam=names[det_['cameraId']]; i=int(d['fileInfo'][det_['poseId']][det_['cameraId']]['filePath'].split('/')[-1][:-4])
    pj=np.array([[f['point']['x'],f['point']['y']] for f in det_['featurePoints']]); ids=[f['id'] for f in det_['featurePoints']]
    mine=det.get((cam,i))
    if mine is None: print(f'cam{cam:02d} frame {i}: project has {len(pj)} corners, mine: none'); continue
    s,u=mine; dmin=np.linalg.norm(pj[:,None]-u[None],axis=-1).min(1)
    k=cosilab.load_camera(cosilab.DATA_ROOT,cam,use_refit=False); R,t=board_pose(u,s,k['K'],k['D'])
    print(f'cam{cam:02d} frame {i}: size {s}, project corners matched by mine within 2 px {(dmin<2).sum()}/{len(pj)} (median {np.median(dmin):.1f} px); my board depth {np.linalg.norm(t):.2f} m vs project pose t {np.linalg.norm([d["calibration"]["poses"][det_["poseId"]]["transform"]["translation"][a] for a in "xyz"]):.2f} m (cam04 frame)')
