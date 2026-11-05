"""
Test of the board's 180-degree ambiguity: the 5x3 ChArUco board looks the same rotated by 180 deg, and far cameras can't
read the markers, so their corner numbering may be reversed relative to the origin camera's. Refit a side camera to its
Calibrator corners (board poses from the project, mapped to world by the origin camera's 4TU pose) with:
  - the corner order as in the project (the stored solution),
  - the order reversed in every frame,
  - per frame the better order, by RANSAC over the frames' single-frame PnP poses,
and score each pose against the people (clearly visible, within 40 px) like the other scripts.
Usage: python scripts/cosilab_calib_12_14/flip_boards.py out.json 12 14 13 15
"""
import json, sys, types
import numpy as np, cv2
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/cosilab_calib_12_14')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data
from joint_fit import P, PROJECT, T

E = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
INLIER_PX = 3.0


def per_view(cam):
    """[(poseId, world corners (15, 3) by id, px (15, 2) by id, ids)] of the camera's detections."""
    name, origin = PROJECT[cam]
    d = json.load(open(P + name))
    names = [int(d['fileInfo'][0][c]['filePath'].split('/')[-2].split('_')[1].split('-')[0]) for c in range(len(d['fileInfo'][0]))]
    ci = names.index(cam)
    op = np.array([[p['x'], p['y'], p['z']] for p in d['calibration']['targets'][0]['objectPoints']])
    o = cosilab.load_camera(cosilab.DATA_ROOT, origin)
    views = []
    for det in d['detections']:
        if det['cameraId'] != ci:
            continue
        Rp, tp = T(d['calibration']['poses'][det['poseId']]['transform'])
        ids = np.array([fp['id'] for fp in det['featurePoints']])
        uv = np.array([[fp['point']['x'], fp['point']['y']] for fp in det['featurePoints']])
        Xw = (o['R_w2c'].T @ ((op @ Rp.T + tp) - o['t_w2c']).T).T  # all 15 board corners in world
        views.append((det['poseId'], Xw, uv, ids))
    return views


def corners(view, flip):
    _, Xw, uv, ids = view
    j = (len(Xw) - 1 - ids) if flip else ids  # reversed numbering = 180 deg rotation of the 5x3 grid
    return Xw[j], uv


def pnp(X, uv, K, D, guess=None):
    n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    if guess is None:
        ok, rv, tv = cv2.solvePnP(X, n, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    else:
        ok, rv, tv = cv2.solvePnP(X, n, np.eye(3), None, guess[0].reshape(3, 1).copy(), guess[1].reshape(3, 1).copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return rv.ravel(), tv.ravel()


def err(X, uv, rv, tv, K, D):
    p = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rv, tv, K, D)[0].reshape(-1, 2)
    return np.linalg.norm(p - uv, axis=1)


def main():
    out, cams = sys.argv[1], [int(c) for c in sys.argv[2:]]
    res = {}
    for cam in cams:
        k = cosilab.load_camera(cosilab.DATA_ROOT, cam)
        K, D = k['K'], k['D']
        views = per_view(cam)
        a = types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam', single_root=f'{E}/cosilab_smpl', kp_conf=0.3,
                                  det_dir=f'{E}/cosilab_vit003_extra', neighbours=None, start=6000, num_frames=600, every=10,
                                  all_people=False, occ_depth=0.3, max_occ=0.2)
        data = Data(a, cam, 1)
        data.n_ref = sum(len(J) for J, *_ in data.per_frame(k['rvec'], k['t_w2c'], K, D))

        def fit(flips):
            X = np.concatenate([corners(v, f)[0] for v, f in zip(views, flips)])
            uv = np.concatenate([corners(v, f)[1] for v, f in zip(views, flips)])
            rv, tv = pnp(X, uv, K, D)
            rv, tv = pnp(X, uv, K, D, (rv, tv))
            return rv, tv, err(X, uv, rv, tv, K, D)

        def report(tag, rv, tv, e, flips):
            s = data.score(rv, tv, K, D)
            R = cv2.Rodrigues(rv)[0]
            C = -R.T @ tv
            dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ k['R_w2c'].T)[0]))
            print(f'cam{cam} {tag:<22} board median {np.median(e):6.2f} px (p90 {np.percentile(e, 90):6.2f}); people within '
                  f'40 px {s["under_40px"]:.2f} (median {s["median_px"]:.0f}); centre {C.round(2)}; {dR:.1f} deg from stored; '
                  f'reversed views {int(sum(flips))}/{len(flips)}', flush=True)
            return {'board_median_px': float(np.median(e)), 'people_u40': s['under_40px'], 'people_median_px': s['median_px'],
                    'centre': C.tolist(), 'rvec': rv.tolist(), 'tvec': tv.tolist(), 'reversed': [int(f) for f in flips],
                    'pose_ids': [int(v[0]) for v in views]}

        r = {}
        e0 = np.concatenate([err(*corners(v, False), k['rvec'], k['t_w2c'], K, D) for v in views])
        r['stored'] = report('stored', k['rvec'], k['t_w2c'], e0, [0] * len(views))
        for tag, flips in [('as project (refit)', [0] * len(views)), ('all reversed', [1] * len(views))]:
            rv, tv, e = fit(flips)
            r[tag] = report(tag, rv, tv, e, flips)
        # per-frame order by RANSAC over single-frame poses (both orders per frame)
        cand = []
        for v in views:
            for f in (0, 1):
                cand.append(pnp(*corners(v, f), K, D))
        best = None
        for rv, tv in cand:
            flips, n_in = [], 0
            for v in views:
                e_ = [np.median(err(*corners(v, f), rv, tv, K, D)) for f in (0, 1)]
                flips.append(int(np.argmin(e_)))
                n_in += min(e_) < INLIER_PX * 5
            if best is None or n_in > best[0]:
                best = (n_in, flips)
        rv, tv, e = fit(best[1])
        r['ransac'] = report('per-frame order (RANSAC)', rv, tv, e, best[1])
        res[cam] = r
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
