"""
Calibration diagnostic for two cameras of a COSI-Lab session, from the ViTPose keypoints only (no video).
Track IDs are treated as shared person IDs. For each ID in both cameras, the keypoint rays are intersected
in the world frame, which measures how consistent the extrinsics are.

python scripts/cosilab_multiview_calib.py --cam_batches cam01_batch01 cam03_batch01
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse

import numpy as np
import torch

from lib.utils import cosilab, fisheye

JOINTS = ['head', 'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_foot', 'l_foot']  # ConfLab-17
CONF = 0.3


def load_view(root, cam_batch):
    camera, tracks = cosilab.load_results(os.path.join(root, cam_batch), 'hmr')
    tracks = {p: t for p, t in tracks.items() if p >= 0}  # -2: not a participant ID
    R = np.array(camera['R_w2c'], dtype=np.float64)
    t = np.array(camera['t_w2c'], dtype=np.float64).reshape(3)
    view = {'name': cam_batch, 'K': np.array(camera['K']), 'D': np.array(camera['D']), 'R': R, 't': t,
            'C': -R.T @ t, 'tracks': {}}
    for pid, d in tracks.items():
        kp = d['kp2d']
        T = kp.shape[0]
        rays = fisheye.unproject(kp[..., :2].reshape(-1, 2), view['K'], view['D']).reshape(T, 17, 3)
        view['tracks'][pid] = {'frames': d['session_frame_ids'], 'kp': kp, 'rays': rays @ R}  # world rays
    return view


def common(ta, tb):
    _, ia, ib = np.intersect1d(ta['frames'], tb['frames'], return_indices=True)
    return ia, ib


def ray_gap(ra, rb, baseline):
    n = np.cross(ra, rb)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
    return np.abs(n @ baseline)


def triangulate(ca, ra, cb, rb):
    """Midpoint of the closest approach of rays ca + s ra and cb + u rb. Returns the points and s, u."""
    w = cb - ca
    ab = (ra * rb).sum(-1)
    d = (1 - ab**2).clip(1e-9)
    s = ((ra @ w) - ab * (rb @ w)) / d
    u = (ab * (ra @ w) - (rb @ w)) / d
    return (ca + s[..., None] * ra + cb + u[..., None] * rb) / 2, s, u


def project(view, X):
    Xc = X @ view['R'].T + view['t']
    K = torch.tensor(view['K'], dtype=torch.float64)
    D = torch.tensor(view['D'], dtype=torch.float64)
    return fisheye.project_torch(torch.tensor(Xc), K, D).numpy()


def pair_stats(a, b, pid_a, pid_b):
    ta, tb = a['tracks'][pid_a], b['tracks'][pid_b]
    ia, ib = common(ta, tb)
    ka, kb = ta['kp'][ia], tb['kp'][ib]
    valid = (ka[..., 2] > CONF) & (kb[..., 2] > CONF)
    gap = ray_gap(ta['rays'][ia], tb['rays'][ib], b['C'] - a['C'])
    return {'ia': ia, 'ib': ib, 'valid': valid, 'gap': gap}


def pct(x):
    return f'{np.percentile(x, 25):.3f} / {np.median(x):.3f} / {np.percentile(x, 75):.3f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl')
    parser.add_argument('--cam_batches', type=str, nargs=2, required=True)
    args = parser.parse_args()

    a, b = [load_view(args.root, cb) for cb in args.cam_batches]
    shared = sorted(set(a['tracks']) & set(b['tracks']))
    print(f'baseline {np.linalg.norm(b["C"] - a["C"]):.3f} m, cameras at {a["C"].round(3)} and {b["C"].round(3)}')
    print(f'{a["name"]} IDs {sorted(a["tracks"])}\n{b["name"]} IDs {sorted(b["tracks"])}\nshared {shared}')

    # ID x ID ray gap over all frames
    pa, pb = sorted(a['tracks']), sorted(b['tracks'])
    M = np.full((len(pa), len(pb)), np.nan)
    for i, p in enumerate(pa):
        for j, q in enumerate(pb):
            s = pair_stats(a, b, p, q)
            if s['valid'].any():
                M[i, j] = np.median(s['gap'][s['valid']])
    print('\nmedian ray gap (m): each row is a track of the first camera')
    for i, p in enumerate(pa):
        o = np.argsort(M[i])
        same = f'{M[i, pb.index(p)]:.3f}' if p in pb else '-'
        print(f'  {p:3d}: best {pb[o[0]]:3d} {M[i, o[0]]:.3f}   2nd {pb[o[1]]:3d} {M[i, o[1]]:.3f}   same-ID {same}')

    # same-ID geometry
    rows = []
    for pid in shared:
        s = pair_stats(a, b, pid, pid)
        ta, tb = a['tracks'][pid], b['tracks'][pid]
        ra, rb = ta['rays'][s['ia']], tb['rays'][s['ib']]
        X, sa, sb = triangulate(a['C'], ra, b['C'], rb)
        ka, kb = ta['kp'][s['ia']], tb['kp'][s['ib']]
        ea = np.linalg.norm(project(a, X) - ka[..., :2], axis=-1)
        eb = np.linalg.norm(project(b, X) - kb[..., :2], axis=-1)
        angle = np.degrees(np.arccos((ra * rb).sum(-1).clip(-1, 1)))
        rows.append(dict(pid=pid, v=s['valid'], gap=s['gap'], ea=ea, eb=eb, X=X, sa=sa, sb=sb, angle=angle,
                         ka=ka, kb=kb))

    print('\nper ID: valid joints, ray gap (m), reprojection px (first / second camera), ray angle (deg), '
          'median depth from each camera (m)')
    for r in rows:
        v = r['v']
        print(f'  {r["pid"]:3d}: {v.sum():6d}  gap {np.median(r["gap"][v]):.3f}  px {np.median(r["ea"][v]):5.1f} / '
              f'{np.median(r["eb"][v]):5.1f}  angle {np.median(r["angle"][v]):5.1f}  '
              f'depth {np.median(r["sa"][v]):.2f} / {np.median(r["sb"][v]):.2f}')

    v = np.concatenate([r['v'].ravel() for r in rows])
    gap = np.concatenate([r['gap'].ravel() for r in rows])[v]
    ea = np.concatenate([r['ea'].ravel() for r in rows])[v]
    eb = np.concatenate([r['eb'].ravel() for r in rows])[v]
    ahead = (np.concatenate([r['sa'].ravel() for r in rows])[v] > 0) & (np.concatenate([r['sb'].ravel() for r in rows])[v] > 0)
    print(f'\nall shared IDs: {v.sum()} joint observations, in front of both cameras {ahead.mean():.3f}')
    print(f'  ray gap m, p25 / p50 / p75: {pct(gap)}')
    print(f'  reprojection px, first camera: {pct(ea)}, second camera: {pct(eb)}')

    print('\nper joint: median ray gap (m)')
    print('  ' + ', '.join(f'{n} {np.median(np.concatenate([r["gap"][:, j][r["v"][:, j]] for r in rows])):.3f}'
                           for j, n in enumerate(JOINTS)))

    print('\nray gap (m) by distance from the image centre (px), per camera')
    for name, key, view in [(a['name'], 'ka', a), (b['name'], 'kb', b)]:
        rad = np.concatenate([np.linalg.norm(r[key][..., :2] - view['K'][:2, 2], axis=-1).ravel() for r in rows])[v]
        edges = [0, 200, 400, 600, 800, 1200]
        print(f'  {name}: ' + ', '.join(
            f'{lo}-{hi}: {np.median(gap[(rad >= lo) & (rad < hi)]):.3f} (n={((rad >= lo) & (rad < hi)).sum()})'
            for lo, hi in zip(edges[:-1], edges[1:]) if ((rad >= lo) & (rad < hi)).any()))

    print('\nbody size from the triangulated joints, median over frames (m; adults expect shoulders ~0.35-0.45, '
          'hips ~0.25-0.35, thigh ~0.40-0.50)')
    for r in rows:
        def seg(i, j):
            m = r['v'][:, i] & r['v'][:, j]
            return np.median(np.linalg.norm(r['X'][m, i] - r['X'][m, j], axis=-1)) if m.any() else np.nan
        print(f'  {r["pid"]:3d}: shoulders {seg(3, 6):.2f}  hips {seg(9, 12):.2f}  thigh {seg(9, 10):.2f} / {seg(12, 13):.2f}  '
              f'shin {seg(10, 11):.2f} / {seg(13, 14):.2f}')


if __name__ == '__main__':
    main()
