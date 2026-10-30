"""
Compare two runs of scripts/run_cosilab_smpl.py with different ViTPose keypoints (e.g. the dataset ones and
those of scripts/run_cosilab_vitpose.py, both ConfLab-17) on the same boxes and frames. There is no keypoint
ground truth, so all numbers are proxies:
  1. old vs new keypoints (distance, confidence)
  2. keypoint jitter (2D acceleration)
  3. reprojection error of hmr / refined_old / refined_new against each keypoint set
  4. 3D smoothness and how far refinement moves the pose away from HMR
  5. cross-camera consistency: world pelvis distance between two cameras of the same session,
     on mutual-nearest pairs found once from hmr

python scripts/compare_cosilab_runs.py \
    --old_root /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl \
    --new_root /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl_vit003 \
    --cam_batches cam01_batch01 cam03_batch01
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse

import numpy as np
import torch

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye

JOINTS = ['head', 'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_foot', 'l_foot']  # ConfLab-17


def load_run(result_dir):
    camera, hmr = cosilab.load_results(result_dir, 'hmr')
    _, refined = cosilab.load_results(result_dir, 'refined')
    return {'camera': camera, 'hmr': hmr, 'refined': refined}


def smpl_forward(smpl, d):
    with torch.no_grad():
        rotmat = torch.tensor(d['pred_rotmat'])
        out = smpl(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:], betas=torch.tensor(d['pred_shape']),
                   transl=torch.tensor(d['pred_trans'][:, 0]), pose2rot=False)
    return out.joints


def consecutive_triples(frames):
    i = np.arange(1, len(frames) - 1)
    return i[(frames[i] - frames[i - 1] == 1) & (frames[i + 1] - frames[i] == 1)]


def fmt(x, p=3):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{p}f}'


def compare_cam_batch(smpl, old, new, kp_conf):
    hmr, ro, rn = old['hmr'], old['refined'], new['refined']
    pids = sorted(set(hmr) & set(rn) & set(ro))
    for pid in pids:  # same boxes -> the same person-frames in both runs
        assert np.array_equal(hmr[pid]['frame_ids'], new['hmr'][pid]['frame_ids']), pid
    K = torch.tensor(old['camera']['K'], dtype=torch.float32)
    D = torch.tensor(old['camera']['D'], dtype=torch.float32)
    kps = {'old': {p: hmr[p]['kp2d'] for p in pids}, 'new': {p: rn[p]['kp2d'] for p in pids}}

    # 1. old vs new keypoints
    a = np.concatenate([kps['old'][p] for p in pids])
    b = np.concatenate([kps['new'][p] for p in pids])
    both = (a[..., 2] > kp_conf) & (b[..., 2] > kp_conf)
    dist = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    print(f'\n1. keypoints, old vs new, {len(a)} person-frames, pixels at 1920x1080, valid = conf > {kp_conf}')
    print(f'   {"joint":<11} {"median dist":>11} {"conf old":>8} {"conf new":>8} {"valid old":>9} {"valid new":>9}')
    for j, name in enumerate(JOINTS):
        m = both[:, j]
        print(f'   {name:<11} {fmt(np.median(dist[m, j]) if m.any() else np.nan, 1):>11} '
              f'{a[:, j, 2].mean():8.3f} {b[:, j, 2].mean():8.3f} '
              f'{(a[:, j, 2] > kp_conf).mean():9.3f} {(b[:, j, 2] > kp_conf).mean():9.3f}')
    print(f'   {"all":<11} {fmt(np.median(dist[both]), 1):>11} {a[..., 2].mean():8.3f} {b[..., 2].mean():8.3f} '
          f'{(a[..., 2] > kp_conf).mean():9.3f} {(b[..., 2] > kp_conf).mean():9.3f}')

    # 2. jitter
    print('\n2. keypoint jitter: median |2D acceleration|, px/frame^2 (lower = smoother)')
    for name, kp in kps.items():
        acc = []
        for p in pids:
            x = kp[p]
            i = consecutive_triples(hmr[p]['frame_ids'])
            v = (x[i - 1, :, 2] > kp_conf) & (x[i, :, 2] > kp_conf) & (x[i + 1, :, 2] > kp_conf)
            acc.append(np.linalg.norm(x[i + 1, :, :2] - 2 * x[i, :, :2] + x[i - 1, :, :2], axis=-1)[v])
        print(f'   {name}: {np.median(np.concatenate(acc)):.2f}')

    # 3. reprojection, 4. 3D
    results = {'hmr': hmr, 'refined_old': ro, 'refined_new': rn}
    joints = {r: {p: smpl_forward(smpl, res[p]) for p in pids} for r, res in results.items()}
    print('\n3. reprojection error, median over valid keypoints, in box sizes (|err| / max(box w, h))')
    print(f'   {"result":<12} ' + ' '.join(f'{"vs " + n:>12}' for n in kps))
    for r in results:
        errs = []
        for kp in kps.values():
            e = []
            for p in pids:
                j2d = fisheye.project_torch(joints[r][p][:, cosilab.CONFLAB_TO_SPIN], K, D).numpy()
                box = hmr[p]['boxes']
                size = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]).clip(min=1)
                v = kp[p][..., 2] > kp_conf
                e.append((np.linalg.norm(j2d - kp[p][..., :2], axis=-1) / size[:, None])[v])
            errs.append(np.median(np.concatenate(e)))
        print(f'   {r:<12} ' + ' '.join(f'{e:12.4f}' for e in errs))

    print('\n4. 3D: median |joint acceleration| (25 OpenPose joints, mm/frame^2) and median per-frame mean '
          'body-joint rotation away from hmr (deg)')
    for r, res in results.items():
        acc, ang = [], []
        for p in pids:
            j3 = joints[r][p][:, :25].numpy()
            i = consecutive_triples(hmr[p]['frame_ids'])
            acc.append(np.linalg.norm(j3[i + 1] - 2 * j3[i] + j3[i - 1], axis=-1).ravel() * 1000)
            R = np.einsum('njab,njac->njbc', hmr[p]['pred_rotmat'][:, 1:], res[p]['pred_rotmat'][:, 1:])
            cos = ((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1)
            ang.append(np.degrees(np.arccos(cos)).mean(1))
        print(f'   {r:<12} accel {np.median(np.concatenate(acc)):6.2f}   rotation from hmr '
              f'{np.median(np.concatenate(ang)):6.2f}')
    return results


def world_pelvis(smpl, tracks):
    """{session frame: {pid: world pelvis (3,)}}"""
    out = {}
    for pid, d in tracks.items():
        J0 = fisheye.rest_pelvis(smpl, torch.tensor(d['pred_shape'])).numpy()
        for f, wt, j0 in zip(d['session_frame_ids'], d['world_trans'], J0):
            out.setdefault(int(f), {})[pid] = wt + j0
    return out


def cross_camera(smpl, res_a, res_b, max_dist):
    """5. mutual-nearest world pelvis pairs between two cameras, chosen on hmr, scored for every result."""
    pa, pb = world_pelvis(smpl, res_a['hmr']), world_pelvis(smpl, res_b['hmr'])
    pairs = []
    for f in sorted(set(pa) & set(pb)):
        ia, ib = list(pa[f]), list(pb[f])
        d = np.linalg.norm(np.stack([pa[f][i] for i in ia])[:, None] - np.stack([pb[f][i] for i in ib])[None], axis=-1)
        for r, c in enumerate(d.argmin(1)):
            if d[:, c].argmin() == r and d[r, c] < max_dist:
                pairs.append((f, ia[r], ib[c]))
    print(f'\n5. cross-camera world pelvis distance on {len(pairs)} mutual-nearest pairs (from hmr, < {max_dist} m), m')
    for r in res_a:
        wa, wb = world_pelvis(smpl, res_a[r]), world_pelvis(smpl, res_b[r])
        d = np.array([np.linalg.norm(wa[f][i] - wb[f][j]) for f, i, j in pairs])
        print(f'   {r:<12} median {np.median(d):.3f}   mean {d.mean():.3f}   <0.25 m {np.mean(d < 0.25):.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_root', type=str, required=True)
    parser.add_argument('--new_root', type=str, required=True)
    parser.add_argument('--cam_batches', type=str, nargs='+', required=True)
    parser.add_argument('--kp_conf', type=float, default=0.3)
    parser.add_argument('--max_match_dist', type=float, default=1.0)
    args = parser.parse_args()

    smpl = SMPL()
    all_results = {}
    for cb in args.cam_batches:
        old = load_run(os.path.join(args.old_root, cb))
        new = load_run(os.path.join(args.new_root, cb))
        print(f'\n===== {cb}: old {args.old_root}, new {args.new_root}')
        all_results[cb] = (compare_cam_batch(smpl, old, new, args.kp_conf), old['camera'])

    cbs = list(all_results)
    for i in range(len(cbs)):
        for j in range(i + 1, len(cbs)):
            (ra, ca), (rb, cb) = all_results[cbs[i]], all_results[cbs[j]]
            if ca['session'] == cb['session'] and ca['R_w2c'] is not None and cb['R_w2c'] is not None:
                print(f'\n===== {cbs[i]} vs {cbs[j]}')
                cross_camera(smpl, ra, rb, args.max_match_dist)


if __name__ == '__main__':
    main()
