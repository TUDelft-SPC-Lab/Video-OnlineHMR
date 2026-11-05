"""
Re-estimate the extrinsics of a COSI-Lab camera from the people: the world joints of the multi-view bodies of the
neighbouring cameras vs the camera's detected VIT003 keypoints (scripts/run_cosilab_detect_vitpose.py). CPU only.
For side cameras 12 and 14, whose stored extrinsics (chained through camera 04) put the bodies 300+ px away from the
detected people; cameras 13 and 15 serve as the control (their stored extrinsics fit, so the refit should barely move).

  1. coarse, without a one-to-one matching: minimise the mean over projected person-frames of the median joint distance
     to the best detector box (truncated at --cap), Nelder-Mead from the best of a grid of starts around the stored pose.
  2. fine: per frame, Hungarian between projected people and detector boxes (median joint distance, gated), then
     RANSAC PnP + Levenberg-Marquardt on the undistorted matched joints; repeated with a tightening gate.
Only people clearly visible under the tested pose take part (unless --all_people): no other person more than
--occ_depth m closer covers more than --max_occ of their padded joint box. This drops the occluded people, whose
detected keypoints often belong to the person in front.
Score of a pose: the share of Hungarian-matched (visible) person-frames within 40 px. The best-scoring pose of all stages is kept, the stored one included.
With --passes > 1, both steps are repeated from the kept pose, with starts on a +-20 deg yaw/pitch/roll grid and
+-0.5 m centre offsets around it.
The intrinsics are kept.

python scripts/cosilab_refit_extrinsic.py --cam_batch cam12_batch01 --multi_root .../cosilab_multiview_4cam --start 6000 --num_frames 600
Output: <out_root>/mingling_<session>_side/camera_XX/extrinsic.json (rvec, tvec: world->camera, as the dataset), and
refit.json with the scores. lib/utils/cosilab.load_camera uses it for the cameras in cosilab.SIDE_EXTRINSIC_REFIT.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import types

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

from lib.models.smpl import SMPL
from lib.utils import cosilab
from cosilab_project_boxes import collect_bodies

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
IMG_W, IMG_H = 1920, 1080


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True)
    parser.add_argument('--neighbours', type=int, nargs='+', default=None)
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--single_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview_4cam')
    parser.add_argument('--det_dir', type=str, default=f'{EXP_DIR}/cosilab_vit003_extra')
    parser.add_argument('--out_root', type=str, default=cosilab.SIDE_EXTRINSIC_ROOT)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--num_frames', type=int, required=True)
    parser.add_argument('--every', type=int, default=5, help='use every n-th frame')
    parser.add_argument('--kp_conf', type=float, default=0.3)
    parser.add_argument('--cap', type=float, default=50.0, help='px, truncation of the coarse cost')
    parser.add_argument('--gates', type=float, nargs='+', default=[80, 50, 35, 25], help='px, per fine iteration')
    parser.add_argument('--global_search', action='store_true',
                        help='coarse starts from a grid of camera centres (--grid_step m, around the people, ceiling '
                             'height +-0.5 m) looking at the people, instead of around the stored pose')
    parser.add_argument('--grid_step', type=float, default=0.5)
    parser.add_argument('--passes', type=int, default=1,
                        help='later passes restart from the kept pose with a +-20 deg yaw/pitch/roll grid')
    parser.add_argument('--all_people', action='store_true', help='also use people occluded under the tested pose')
    parser.add_argument('--pids', type=int, nargs='*', default=None,
                        help='only these people (e.g. those whose bodies fit the well-calibrated side cameras)')
    parser.add_argument('--occ_depth', type=float, default=0.3, help='m closer before another person occludes')
    parser.add_argument('--max_occ', type=float, default=0.2, help='max covered share of a visible person\'s box')
    parser.add_argument('--no_save', action='store_true')
    return parser.parse_args(argv)


def project(X, rvec, tvec, K, D):
    """World points (N, 3) -> px (N, 2), in front of the camera (N,), distance to the camera (N,)."""
    R = cv2.Rodrigues(rvec)[0]
    Xc = X @ R.T + tvec
    uv = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rvec, tvec, K, D)[0].reshape(-1, 2)
    return uv, Xc[:, 2] > 0.1, np.linalg.norm(Xc, axis=1)


def clearly_visible(u, front, dist, occ_depth, max_occ, pad=0.15):
    """(P,) people of one frame not covered (> max_occ of their padded joint box) by people occ_depth m closer."""
    lo, hi = u.min(1), u.max(1)
    size = hi - lo
    lo, hi = lo - pad * size, hi + pad * size
    area = np.prod(hi - lo, -1).clip(min=1)
    inter = np.prod((np.minimum(hi[:, None], hi[None]) - np.maximum(lo[:, None], lo[None])).clip(min=0), -1)
    d = np.median(dist, 1)
    closer = d[None, :] < d[:, None] - occ_depth  # [p, q]: q in front of p
    covered = (np.where(closer, inter, 0) / area[:, None]).sum(1)
    centre = (lo + hi) / 2
    inside = (centre[:, 0] >= 0) & (centre[:, 0] < IMG_W) & (centre[:, 1] >= 0) & (centre[:, 1] < IMG_H)
    return front.all(1) & inside & (covered <= max_occ)


class Data:
    """Per used frame: world joints (P, 17, 3) and detected keypoints (B, 17, 3) of the detector boxes."""

    def __init__(self, args, cam, batch):
        a = types.SimpleNamespace(multi_root=args.multi_root, single_root=args.single_root, kp_conf=args.kp_conf)
        best = collect_bodies(a, args.neighbours or cosilab.NEIGHBOURS[cam], batch, args.start,
                              args.start + args.num_frames)
        pids = getattr(args, 'pids', None)
        keys = [k for k in sorted(best) if (k[0] - args.start) % args.every == 0 and (not pids or k[1] in pids)]
        rows = [best[k] for k in keys]
        with torch.no_grad():
            J = SMPL()(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for *_, d, i in rows]))[:, None],
                       body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for *_, d, i in rows]))[:, 1:],
                       betas=torch.tensor(np.stack([d['pred_shape'][i] for *_, d, i in rows])),
                       transl=torch.tensor(np.stack([d['world_trans'][i] for *_, d, i in rows])),
                       pose2rot=False).joints[:, cosilab.CONFLAB_TO_SPIN].numpy().astype(np.float64)
        fr = np.array([k[0] for k in keys])
        det = np.load(os.path.join(args.det_dir, f'cam{cam:02d}_batch{batch:02d}', 'detections.npz'))
        is_det = det['kind'] == 'detector'
        self.frames = []
        for f in np.unique(fr):
            kp = det['kp'][is_det & (det['frame'] == f)].astype(np.float64)
            if len(kp):
                self.frames.append((J[fr == f], kp))
        self.X = np.concatenate([j.reshape(-1, 3) for j, _ in self.frames])
        self.all_people, self.occ_depth, self.max_occ = args.all_people, args.occ_depth, args.max_occ
        self.n_ref = 0  # visible person-frames under the stored pose, set in main
        print(f'{len(self.frames)} frames, {sum(len(j) for j, _ in self.frames)} person-frames, '
              f'{sum(len(k) for _, k in self.frames)} detector boxes', flush=True)

    def per_frame(self, rvec, tvec, K, D):
        """Per frame: world joints, detected keypoints, projected joints and in-front mask of the people used."""
        uv, front, dist = project(self.X, rvec, tvec, K, D)
        s = 0
        for J, kp in self.frames:
            n = J.shape[0] * 17
            u, fr, di = uv[s:s + n].reshape(-1, 17, 2), front[s:s + n].reshape(-1, 17), dist[s:s + n].reshape(-1, 17)
            s += n
            keep = np.ones(len(J), bool) if self.all_people else clearly_visible(u, fr, di, self.occ_depth, self.max_occ)
            if keep.any():
                yield J[keep], kp, u[keep], fr[keep]

    def coarse_cost(self, rvec, tvec, K, D, cap):
        out = []
        for J, kp, u, fr in self.per_frame(rvec, tvec, K, D):
            ok = fr[:, None] & (kp[None, :, :, 2] > 0.3)
            e = np.linalg.norm(u[:, None] - kp[None, :, :, :2], axis=-1)
            with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore', RuntimeWarning)
                M = np.nanmedian(np.where(ok, e, np.nan), -1)
            M = np.where(ok.sum(-1) >= 6, M, np.inf)
            out.append(np.minimum(M.min(1), cap))
        if not out:
            return cap
        # people that leave the visible set count as misses, so a pose can't win by hiding everyone
        n = np.concatenate(out)
        return float((n.sum() + cap * max(self.n_ref - len(n), 0)) / max(len(n), self.n_ref))

    def match(self, rvec, tvec, K, D, gate):
        """Matched (world joint, px) pairs and the per person-frame median px error of the matches."""
        Xs, us, errs = [], [], []
        for J, kp, u, fr in self.per_frame(rvec, tvec, K, D):
            ok = fr[:, None] & (kp[None, :, :, 2] > 0.3)
            e = np.linalg.norm(u[:, None] - kp[None, :, :, :2], axis=-1)
            with np.errstate(all='ignore'):
                M = np.nanmedian(np.where(ok, e, np.nan), -1)
            M = np.where(ok.sum(-1) >= 6, M, 1e4)
            for a, b in zip(*linear_sum_assignment(M)):
                if M[a, b] < gate:
                    errs.append(M[a, b])
                    v = ok[a, b] & (e[a, b] < 2 * gate)
                    Xs.append(J[a][v])
                    us.append(kp[b][v, :2])
        if not Xs:
            return None, None, np.array([])
        return np.concatenate(Xs), np.concatenate(us), np.array(errs)

    def score(self, rvec, tvec, K, D):
        """Median px over all Hungarian matches (gate 1e3), the share under 40 px, and that share over n_ref people."""
        _, _, e = self.match(rvec, tvec, K, D, 1e3)
        if len(e) == 0:
            return {'median_px': np.inf, 'under_40px': 0.0, 'n': 0, 'n_under_40px': 0}
        return {'median_px': float(np.median(e)), 'under_40px': float((e < 40).mean()), 'n': int(len(e)),
                'n_under_40px': int((e < 40).sum())}


def pose_params(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    return np.r_[rvec, -R.T @ tvec]


def from_params(x):
    rvec = x[:3].copy()
    R = cv2.Rodrigues(rvec)[0]
    return rvec, -R @ x[3:]


def look_at(C, target, roll=0.0):
    """rvec, tvec of a camera at C looking at target, image down = world +z (down) projected, rolled by roll rad."""
    z = target - C
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, 1.0])
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = cv2.Rodrigues(np.array([0, 0, roll]))[0] @ np.stack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel(), -R @ C


def global_starts(data, f, step, keep=8):
    """Best `keep` start poses from a grid of centres around the people, each looking at the people's centroid."""
    X = data.X
    lo, hi = X.min(0), X.max(0)
    target = X.mean(0)
    xs = np.arange(lo[0] - 4, hi[0] + 4.01, step)
    ys = np.arange(lo[1] - 4, hi[1] + 4.01, step)
    scored = []
    for zc in (-0.5, 0.0, 0.5):  # the top cameras (world z = 0) are at the ceiling
        for xc in xs:
            for yc in ys:
                C = np.array([xc, yc, zc])
                if np.linalg.norm((C - target)[:2]) < 1.5:  # side cameras are at the edge of the room
                    continue
                scored.append((f(pose_params(*look_at(C, target))), C))
    scored.sort(key=lambda x: x[0])
    print(f'global search: {len(scored)} centres, best costs {[round(c, 1) for c, _ in scored[:keep]]}', flush=True)
    starts = []
    for _, C in scored[:keep]:
        for dt in [np.zeros(3), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]),
                   np.array([0, -1.0, 0])]:
            for roll in np.radians([-10, 0, 10]):
                starts.append(pose_params(*look_at(C, target + dt, roll)))
    return starts

def refine_pass(args, data, f, starts, K, D, tag=''):
    """Coarse (Nelder-Mead from the 5 best starts) then fine (PnP) steps. Returns [(name, rvec, tvec, score)]."""
    ranked = sorted(starts, key=f)[:5]
    best_x, best_c = None, np.inf
    for x in ranked:
        res = minimize(f, x, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-3,
                                                            'initial_simplex': x + np.vstack([np.zeros(6),
                                                                                              np.diag([0.05] * 3 + [0.3] * 3)])})
        if res.fun < best_c:
            best_x, best_c = res.x, res.fun
    rvec, tvec = from_params(best_x)
    coarse = data.score(rvec, tvec, K, D)
    print(f'{tag}coarse: cost {best_c:.1f} px, centre {best_x[3:].round(2)}, {coarse}', flush=True)
    poses = [(tag + 'coarse', rvec, tvec, coarse)]

    # 2. fine: Hungarian matches + PnP on undistorted points
    for gate in args.gates:
        X, u, e = data.match(rvec, tvec, K, D, gate)
        if X is None or len(X) < 50:
            print(f'{tag}gate {gate}: too few matches, stop')
            break
        n = cv2.fisheye.undistortPoints(u.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        ok, rv, tv, inl = cv2.solvePnPRansac(X, n, np.eye(3), None, rvec.copy(), tvec.copy(), useExtrinsicGuess=True,
                                             reprojectionError=gate / K[0, 0], iterationsCount=500)
        if not ok or inl is None:
            print(f'{tag}gate {gate}: PnP failed, stop')
            break
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(X[inl], n[inl], np.eye(3), None, rv, tv)
        rvec, tvec = rv.ravel(), tv.ravel()
        poses.append((f'{tag}gate {gate:.0f}', rvec, tvec, data.score(rvec, tvec, K, D)))
        print(f'{tag}gate {gate:4.0f} px: {len(e)} person-frames matched, {len(X)} joints, {len(inl)} inliers; '
              f'{poses[-1][3]}', flush=True)

    return poses


def main(args):
    cam, batch, session = cosilab.parse_cam_batch(args.cam_batch)
    stored = cosilab.load_camera(args.data_root, cam, use_refit=False)
    K, D = stored['K'], stored['D']
    data = Data(args, cam, batch)
    r0, t0 = stored['rvec'], stored['t_w2c']
    data.n_ref = sum(len(J) for J, *_ in data.per_frame(r0, t0, K, D))
    before = data.score(r0, t0, K, D)
    print(f'stored extrinsics: centre {(-stored["R_w2c"].T @ t0).round(2)}, {before}', flush=True)

    # 1. coarse: matching-free, several starts
    x0 = pose_params(r0, t0)
    R0 = stored['R_w2c']
    starts = []
    for yaw in [] if args.global_search else np.radians([-20, -10, 0, 10, 20]):
        for pitch in np.radians([-10, 0, 10]):
            dR = cv2.Rodrigues(np.array([pitch, yaw, 0.0]))[0]  # about the camera's own x / y axes
            rv = cv2.Rodrigues(dR @ R0)[0].ravel()
            for dC in [np.zeros(3), np.array([0, 1.0, 0]), np.array([0, 2.0, 0]), np.array([0, -1.0, 0])]:
                starts.append(np.r_[rv, x0[3:] + dC])

    def f(x):
        return data.coarse_cost(*from_params(x), K, D, args.cap)
    if args.global_search:
        starts = global_starts(data, f, args.grid_step)
    poses = [('stored', r0, t0, before)]
    for pas in range(args.passes):
        if pas > 0:  # restart around the best pose so far
            _, rb, tb, _ = max(poses, key=lambda p: p[3]['n_under_40px'])
            Rb, xb = cv2.Rodrigues(rb)[0], pose_params(rb, tb)
            starts = []
            for a in np.radians([-20, -10, 0, 10, 20]):
                for b in np.radians([-20, -10, 0, 10, 20]):
                    for c in np.radians([-20, -10, 0, 10, 20]):
                        rv = cv2.Rodrigues(cv2.Rodrigues(np.array([a, b, c]))[0] @ Rb)[0].ravel()
                        for dC in [np.zeros(3), np.array([0.5, 0, 0]), np.array([-0.5, 0, 0]),
                                   np.array([0, 0.5, 0]), np.array([0, -0.5, 0])]:
                            starts.append(np.r_[rv, xb[3:] + dC])
        poses += refine_pass(args, data, f, starts, K, D, tag=f'pass {pas + 1} ')

    name, rvec, tvec, after = max(poses, key=lambda p: p[3]['n_under_40px'])
    print(f'kept: {name}')
    R = cv2.Rodrigues(rvec)[0]
    dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R0.T)[0]))
    move = np.linalg.norm(-R.T @ tvec + R0.T @ t0)
    print(f'\nrefit: centre {(-R.T @ tvec).round(2)} (moved {move:.2f} m, rotated {dR:.1f} deg); {after} '
          f'(stored: {before})', flush=True)
    if not args.no_save:
        d = os.path.join(args.out_root, f'mingling_{session}_side', f'camera_{cam:02d}')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'extrinsic.json'), 'w') as fp:
            json.dump({'rvec': rvec.reshape(3, 1).tolist(), 'tvec': tvec.reshape(3, 1).tolist()}, fp, indent=1)
        with open(os.path.join(d, 'refit.json'), 'w') as fp:
            json.dump({'args': vars(args), 'kept': name, 'stages': {p[0]: p[3] for p in poses}, 'refit': after,
                       'moved_m': float(move), 'rotated_deg': float(dR)}, fp, indent=1)
        print(f'Saved to {d}')


if __name__ == '__main__':
    main(parse_args())
