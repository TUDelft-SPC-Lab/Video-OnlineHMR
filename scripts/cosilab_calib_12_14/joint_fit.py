"""
Joint fit of a side camera's extrinsics to (a) its Calibrator board corners, board poses fixed in the project's origin
camera frame and mapped to world by that camera's 4TU pose, and (b) the people: Hungarian-matched joints of clearly
visible people (cosilab_refit_extrinsic.Data), IRLS weights on the people only. Board residuals are not robustified. Sweeps the board weight.
Usage: joint_fit.py cam out.json
"""
import json, sys, types
import numpy as np, cv2
from scipy.optimize import least_squares
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data

E = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
P = '/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/calibrator_projects/mingling_2_side_Extrinsic Parameters/'
PROJECT = {12: ('top_cam_04_side_cam_12_14.json', 4), 14: ('top_cam_04_side_cam_12_14.json', 4),
           13: ('top_cam_01_side_cam_13_15.json', 1), 15: ('top_cam_01_side_cam_13_15.json', 1)}
SCALE = 15.0  # px, robust scale of the people residuals


def T(tr):
    r, t = tr['rotation'], tr['translation']
    return cv2.Rodrigues(np.array([r['rx'], r['ry'], r['rz']]))[0], np.array([t['x'], t['y'], t['z']])


def board_points(cam):
    name, origin = PROJECT[cam]
    d = json.load(open(P + name))
    names = [int(d['fileInfo'][0][c]['filePath'].split('/')[-2].split('_')[1].split('-')[0]) for c in range(len(d['fileInfo'][0]))]
    ci = names.index(cam)
    op = np.array([[p['x'], p['y'], p['z']] for p in d['calibration']['targets'][0]['objectPoints']])
    o = cosilab.load_camera(cosilab.DATA_ROOT, origin)
    Xw, uv = [], []
    for det in d['detections']:
        if det['cameraId'] != ci:
            continue
        Rp, tp = T(d['calibration']['poses'][det['poseId']]['transform'])
        for fp in det['featurePoints']:
            Xw.append(o['R_w2c'].T @ (Rp @ op[fp['id']] + tp - o['t_w2c']))
            uv.append([fp['point']['x'], fp['point']['y']])
    return np.array(Xw), np.array(uv)


def proj(X, x, K, D):
    rvec, C = x[:3], x[3:]
    R = cv2.Rodrigues(rvec)[0]
    return cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rvec, -R @ C, K, D)[0].reshape(-1, 2)


def to_rt(x):
    R = cv2.Rodrigues(x[:3])[0]
    return x[:3].copy(), -R @ x[3:]


def main():
    cam, out_path = int(sys.argv[1]), sys.argv[2]
    k = cosilab.load_camera(cosilab.DATA_ROOT, cam, use_refit=False)
    K, D = k['K'], k['D']
    Xb, ub = board_points(cam)
    a = types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam', single_root=f'{E}/cosilab_smpl', kp_conf=0.3,
                              det_dir=f'{E}/cosilab_vit003_extra', neighbours=None, start=6000, num_frames=600, every=10,
                              all_people=False, occ_depth=0.3, max_occ=0.2)
    data = Data(a, cam, 1)
    data.n_ref = sum(len(J) for J, *_ in data.per_frame(k['rvec'], k['t_w2c'], K, D))
    x0 = np.r_[k['rvec'], -k['R_w2c'].T @ k['t_w2c']]

    def report(tag, x):
        eb = np.linalg.norm(proj(Xb, x, K, D) - ub, axis=1)
        s = data.score(*to_rt(x), K, D)
        R = cv2.Rodrigues(x[:3])[0]
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ k['R_w2c'].T)[0]))
        row = {'board_px_median': float(np.median(eb)), 'board_px_p90': float(np.percentile(eb, 90)),
               'people_u40': s['under_40px'], 'people_median_px': s['median_px'], 'people_n_u40': s['n_under_40px'],
               'moved_m': float(np.linalg.norm(x[3:] - x0[3:])), 'rotated_deg': float(dR), 'centre': x[3:].round(3).tolist()}
        print(f'cam{cam} {tag:<16} board median {row["board_px_median"]:6.2f} px (p90 {row["board_px_p90"]:6.2f}); people within 40 px '
              f'{row["people_u40"]:.2f} (median {row["people_median_px"]:6.1f}, n {row["people_n_u40"]}); moved {row["moved_m"]:.2f} m, '
              f'rotated {dR:.1f} deg', flush=True)
        return row

    rows = {'stored': report('stored', x0)}
    print(f'cam{cam}: {len(ub)} board corners', flush=True)
    x = x0.copy()
    for wb in [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]:  # board weight: one board corner counts as wb people joints
        for it, gate in enumerate([80, 60, 45, 35, 30, 30]):
            Xp, up, _ = data.match(*to_rt(x), K, D, gate)
            if Xp is None or len(Xp) < 30:
                break
            for _ in range(3):  # IRLS: robust (Cauchy-like) weights on the people only, the board term stays quadratic
                e = np.linalg.norm(proj(Xp, x, K, D) - up, axis=1)
                wp = np.repeat(1 / np.sqrt(1 + (e / SCALE) ** 2), 2)
                def res(y):
                    rp = (proj(Xp, y, K, D) - up).ravel() * np.sqrt(wp)
                    rb = (proj(Xb, y, K, D) - ub).ravel()
                    return np.r_[rp, np.sqrt(wb) * rb]
                x = least_squares(res, x, x_scale=np.r_[[0.01] * 3, [0.05] * 3], max_nfev=200).x
        rows[f'w_board {wb}'] = report(f'w_board {wb:g}', x)
        rows[f'w_board {wb}']['rvec'], rows[f'w_board {wb}']['tvec'] = [v.tolist() for v in to_rt(x)]
    json.dump(rows, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
