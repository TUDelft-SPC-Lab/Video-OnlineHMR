"""
Side-camera extrinsics fitted directly to board corners triangulated by the top cameras (01-04), no cam04 chaining.
Robust to several boards in use: top cameras must agree on the board (centre within --agree m), and each side-camera
frame gives its own pose candidate; RANSAC over frames keeps the consistent ones.
Usage: board_fit.py boards.npy out.json cam [cam ...]
"""
import json, sys, types
import numpy as np, cv2
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from lib.utils import cosilab

AGREE = 0.05      # m, top cameras' board centres must agree
INLIER_PX = 4.0   # px, RANSAC inlier threshold (median corner error of a frame)
SPACING = 0.09
OBJ = {(5, 3): np.array([[x * SPACING, y * SPACING, 0] for y in range(3) for x in range(5)], float),
       (3, 5): np.array([[x * SPACING, y * SPACING, 0] for y in range(5) for x in range(3)], float)}


def board_pose(uv, size, K, D):
    """Board -> camera (R, t) by planar PnP on undistorted points."""
    n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    ok, rv, tv = cv2.solvePnP(OBJ[size], n, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    return cv2.Rodrigues(rv)[0], tv.ravel()


def rays(uv, cam):
    n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), cam['K'], cam['D']).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))] @ cam['R_w2c']  # world directions
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def triangulate(obs):
    """obs: [(C (3,), dirs (N, 3))] -> least-squares closest points (N, 3)."""
    N = obs[0][1].shape[0]
    X = np.zeros((N, 3))
    for j in range(N):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for C, d in obs:
            P = np.eye(3) - np.outer(d[j], d[j])
            A += P; b += P @ C
        X[j] = np.linalg.solve(A, b)
    return X


def reproj(X, uv, rvec, tvec, K, D):
    p = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rvec, tvec, K, D)[0].reshape(-1, 2)
    return np.linalg.norm(p - uv, axis=1)


def main():
    boards, out_path, cams = sys.argv[1], sys.argv[2], [int(c) for c in sys.argv[3:]]
    det = {}
    for c, i, size, uv in np.load(boards, allow_pickle=True):
        det[(int(c), int(i))] = (tuple(size), np.asarray(uv, float))
    top = {c: cosilab.load_camera(cosilab.DATA_ROOT, c) for c in (1, 2, 3, 4)}
    for c in top:
        top[c]['C'] = -top[c]['R_w2c'].T @ top[c]['t_w2c']

    # 1. world board corners per frame, where >= 2 top cameras agree
    world = {}
    stats = {'top_frames': 0, 'agree': 0}
    for i in range(332):
        seen = [c for c in top if (c, i) in det]
        if len(seen) < 2:
            continue
        stats['top_frames'] += 1
        est = {}
        for c in seen:
            size, uv = det[(c, i)]
            R, t = board_pose(uv, size, top[c]['K'], top[c]['D'])
            Xc = OBJ[size] @ R.T + t
            est[c] = (Xc - top[c]['t_w2c']) @ top[c]['R_w2c']  # world corners in this camera's order
        # largest group of cameras whose board centres agree
        best = []
        for a in seen:
            grp = [b for b in seen if np.linalg.norm(est[a].mean(0) - est[b].mean(0)) < AGREE]
            best = grp if len(grp) > len(best) else best
        if len(best) < 2:
            continue
        ref = best[0]
        obs = []
        for c in best:  # reorder each camera's corners to the reference ordering (nearest world corner)
            order = np.argmin(np.linalg.norm(est[ref][:, None] - est[c][None], axis=-1), axis=1)
            if len(set(order.tolist())) < 15:
                break
            obs.append((top[c]['C'], rays(det[(c, i)][1][order], top[c])))
        else:
            X = triangulate(obs)
            gap = np.mean([np.linalg.norm(np.cross(X - C, d), axis=1).mean() for C, d in obs])
            world[i] = (X, best, gap)
            stats['agree'] += 1
    gaps = np.array([g for _, _, g in world.values()])
    print(f'top cameras: {stats["top_frames"]} frames with >= 2 detections, {stats["agree"]} agree on one board; '
          f'ray gap median {np.median(gaps) * 1000:.1f} mm', flush=True)

    results = {}
    for cam in cams:
        k = cosilab.load_camera(cosilab.DATA_ROOT, cam, use_refit=False)
        K, D = k['K'], k['D']
        # 2. per-frame candidates
        frames = []
        for i, (X, grp, _) in world.items():
            if (cam, i) not in det:
                continue
            size, uv = det[(cam, i)]
            best = None
            for Xo in (X, X[::-1]):  # the 5x3 board looks the same rotated by 180 deg
                n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
                ok, rv, tv = cv2.solvePnP(Xo, n, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    e = np.median(reproj(Xo, uv, rv.ravel(), tv.ravel(), K, D))
                    if best is None or e < best[0]:
                        best = (e, rv.ravel(), tv.ravel(), Xo)
            if best:
                frames.append((i, uv, best))
        if not frames:
            print(f'cam{cam}: no frames shared with the triangulated boards')
            continue
        # 3. RANSAC over frame candidates
        def inliers(rv, tv):
            out = []
            for i, uv, (_, _, _, Xo) in frames:
                e = min(np.median(reproj(Xo, uv, rv, tv, K, D)), np.median(reproj(Xo[::-1], uv, rv, tv, K, D)))
                out.append(e < INLIER_PX)
            return np.array(out)
        cand = [(inliers(rv, tv), rv, tv) for _, _, (_, rv, tv, _) in frames]
        inl, rv, tv = max(cand, key=lambda x: x[0].sum())
        Xs, us = [], []
        for keep, (i, uv, (_, _, _, Xo)) in zip(inl, frames):
            if keep:
                e1 = np.median(reproj(Xo, uv, rv, tv, K, D)); e2 = np.median(reproj(Xo[::-1], uv, rv, tv, K, D))
                Xs.append(Xo if e1 <= e2 else Xo[::-1]); us.append(uv)
        Xs, us = np.concatenate(Xs), np.concatenate(us)
        n = cv2.fisheye.undistortPoints(us.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        rv2, tv2 = cv2.solvePnPRefineLM(Xs, n, np.eye(3), None, rv.reshape(3, 1).copy(), tv.reshape(3, 1).copy())
        rv2, tv2 = rv2.ravel(), tv2.ravel()
        e = reproj(Xs, us, rv2, tv2, K, D)
        R = cv2.Rodrigues(rv2)[0]
        Rs = k['R_w2c']
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ Rs.T)[0]))
        Cn, Cs = -R.T @ tv2, -Rs.T @ k['t_w2c']
        stored_e = reproj(Xs, us, k['rvec'], k['t_w2c'], K, D)
        print(f'cam{cam}: {len(frames)} frames with a candidate, {int(inl.sum())} RANSAC inliers ({len(us)} corners); '
              f'corner error median {np.median(e):.2f} px (stored pose on the same corners {np.median(stored_e):.1f} px); '
              f'centre {Cn.round(2)} vs stored {Cs.round(2)} ({np.linalg.norm(Cn - Cs):.2f} m), rotation diff {dR:.1f} deg',
              flush=True)
        results[cam] = {'rvec': rv2.tolist(), 'tvec': tv2.tolist(), 'frames': int(len(frames)), 'inliers': int(inl.sum()),
                        'corner_px_median': float(np.median(e)), 'stored_corner_px_median': float(np.median(stored_e)),
                        'moved_m': float(np.linalg.norm(Cn - Cs)), 'rotated_deg': float(dR)}
    json.dump(results, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
