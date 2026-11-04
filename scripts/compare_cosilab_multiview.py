"""
Compare multi-view SMPL (scripts/run_cosilab_multiview.py) with the single-view results of the same cameras,
on the people seen by both cameras. There is no ground truth, so all numbers are proxies:
  1. reprojection into each camera. Single-view bodies are also scored in the camera they did not see.
  2. distance to the triangulated joints
  3. floor: spread of the lowest body point around a plane fitted to it (not part of any loss)
  4. body height per person, and 3D acceleration and rotation from hmr

python scripts/compare_cosilab_multiview.py --cam_batches cam01_batch01 cam03_batch01
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse

import numpy as np
import torch

from lib.models.smpl import SMPL
from lib.utils import cosilab
from cosilab_multiview_calib import CONF, JOINTS, load_view, project, triangulate

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
FOOT_JOINTS = [11, 14, 19, 20, 21, 22, 23, 24]  # OpenPose ankles, toes and heels in the SPIN joints


def world_joints(smpl, d):
    """SMPL joints (T, 49, 3) in the world frame."""
    with torch.no_grad():
        rotmat = torch.tensor(d['pred_rotmat'])
        out = smpl(global_orient=torch.tensor(d['world_rotmat_root'])[:, None], body_pose=rotmat[:, 1:],
                   betas=torch.tensor(d['pred_shape']), transl=torch.tensor(d['world_trans']), pose2rot=False)
    return out.joints.numpy()


def rest_height(smpl, betas):
    with torch.no_grad():
        eye = torch.eye(3).expand(1, 24, 3, 3)
        v = smpl(global_orient=eye[:, :1], body_pose=eye[:, 1:], betas=torch.tensor(betas)[None],
                 pose2rot=False, default_smpl=True).vertices[0]
    return (v[:, 1].max() - v[:, 1].min()).item()


def reprojection(joints, view, pid):
    """Median-able errors (box sizes, px) of world joints (T, 17, 3) against one camera's keypoints."""
    t = view['tracks'][pid]
    kp = t['kp']
    box = view['boxes'][pid]
    size = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]).clip(min=1)
    err = np.linalg.norm(project(view, joints.astype(np.float64)) - kp[..., :2], axis=-1)
    valid = kp[..., 2] > CONF
    return (err / size[:, None])[valid], err[valid], err, valid


def floor_spread(points):
    """Robust plane fit: residual std (m) and plane normal, with two rounds of 3-sigma outlier removal."""
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        c = points[keep].mean(0)
        normal = np.linalg.svd(points[keep] - c, full_matrices=False)[2][-1]
        res = (points - c) @ normal
        keep = np.abs(res) < 3 * res[keep].std()
    return res[keep].std(), normal, keep.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batches', type=str, nargs=2, required=True)
    parser.add_argument('--single_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview')
    args = parser.parse_args()

    smpl = SMPL()
    a, b = [load_view(args.single_root, cb) for cb in args.cam_batches]
    na, nb = args.cam_batches
    for v in (a, b):
        v['boxes'] = {p: d['boxes'] for p, d in cosilab.load_results(os.path.join(args.single_root, v['name']), 'hmr')[1].items()}
    runs = {f'hmr {na[:5]}': cosilab.load_results(os.path.join(args.single_root, na), 'hmr')[1],
            f'hmr {nb[:5]}': cosilab.load_results(os.path.join(args.single_root, nb), 'hmr')[1],
            f'refined {na[:5]}': cosilab.load_results(os.path.join(args.single_root, na), 'refined')[1],
            f'refined {nb[:5]}': cosilab.load_results(os.path.join(args.single_root, nb), 'refined')[1],
            'multiview': cosilab.load_results(os.path.join(args.multi_root, na), 'refined')[1]}
    multi_b = cosilab.load_results(os.path.join(args.multi_root, nb), 'refined')[1]
    pids = sorted(runs['multiview'])
    assert all(np.array_equal(a['tracks'][p]['frames'], b['tracks'][p]['frames']) for p in pids), 'track frames differ'
    n_frames = sum(len(a['tracks'][p]['frames']) for p in pids)
    print(f'{len(pids)} people {pids}, {n_frames} person-frames')

    spin = cosilab.CONFLAB_TO_SPIN
    joints = {r: {p: world_joints(smpl, d[p]) for p in pids} for r, d in runs.items()}

    # the same world body must come out of both cameras' multi-view files
    max_diff = max(np.abs(joints['multiview'][p] - world_joints(smpl, multi_b[p])).max() for p in pids)
    print(f'multi-view body, cam-batch 1 vs 2 files: max difference {max_diff * 1000:.2f} mm')

    print(f'\n1. reprojection error (median, box sizes / px; keypoints with conf > {CONF})')
    print(f'   {"result":<16} {"into " + na[:5]:>18} {"into " + nb[:5]:>18}')
    for r in runs:
        cols = []
        for v in (a, b):
            s, px = zip(*[reprojection(joints[r][p][:, spin], v, p)[:2] for p in pids])
            cols.append(f'{np.median(np.concatenate(s)):.4f} / {np.median(np.concatenate(px)):5.1f}')
        print(f'   {r:<16} {cols[0]:>18} {cols[1]:>18}')

    print('\n   multiview, per joint median px, ' + f'{na[:5]} / {nb[:5]}')
    out = []
    for j, name in enumerate(JOINTS):
        e = []
        for v in (a, b):
            rows = [reprojection(joints['multiview'][p][:, spin], v, p) for p in pids]
            e.append(np.median(np.concatenate([x[2][:, j][x[3][:, j]] for x in rows])))
        out.append(f'{name} {e[0]:.1f}/{e[1]:.1f}')
    print('   ' + ', '.join(out))

    print('\n2. distance to the triangulated joints (m, joints valid in both cameras, median)')
    tri = {}
    for p in pids:
        ta, tb = a['tracks'][p], b['tracks'][p]
        tri[p] = (triangulate(a['C'], ta['rays'], b['C'], tb['rays'])[0],
                  (ta['kp'][..., 2] > CONF) & (tb['kp'][..., 2] > CONF))
    for r in runs:
        d = np.concatenate([np.linalg.norm(joints[r][p][:, spin] - tri[p][0], axis=-1)[tri[p][1]] for p in pids])
        pel = np.concatenate([np.linalg.norm(joints[r][p][:, 8] - tri[p][0][:, [9, 12]].mean(1), axis=-1)[tri[p][1][:, [9, 12]].all(1)]
                              for p in pids])
        print(f'   {r:<16} all joints {np.median(d):.3f}   pelvis {np.median(pel):.3f}')

    print('\n3. floor: lowest foot joint per person-frame, spread around a fitted plane (m, not used by any loss)')
    for r in runs:
        rough = np.concatenate([joints[r][p][:, FOOT_JOINTS].reshape(-1, 3) for p in pids])
        down = floor_spread(rough)[1]
        down = down if down[2] > 0 else -down  # the top-view cameras look along +z
        lows = []
        for p in pids:
            f = joints[r][p][:, FOOT_JOINTS]
            lows.append(np.take_along_axis(f, np.argmax(f @ down, 1)[:, None, None], 1)[:, 0])
        spread, normal, kept = floor_spread(np.concatenate(lows))
        print(f'   {r:<16} spread {spread:.3f}   kept {kept:.2f}   normal {normal.round(3)}')

    print('\n4. rest height (m) per person')
    print('   ' + ' '.join(f'{p:>6d}' for p in pids))
    for r in [f'refined {na[:5]}', f'refined {nb[:5]}', 'multiview']:
        print('   ' + ' '.join(f'{rest_height(smpl, runs[r][p]["pred_shape"].mean(0)):6.2f}' for p in pids) + f'   {r}')

    print('\n   median 3D acceleration (mm/frame^2) and mean body-joint rotation from hmr (deg)')
    base = runs[f'hmr {na[:5]}']
    for r in runs:
        acc, ang = [], []
        for p in pids:
            j3 = joints[r][p][:, :25]
            acc.append(np.linalg.norm(j3[2:] - 2 * j3[1:-1] + j3[:-2], axis=-1).ravel() * 1000)
            R = np.einsum('njab,njac->njbc', base[p]['pred_rotmat'][:, 1:], runs[r][p]['pred_rotmat'][:, 1:])
            ang.append(np.degrees(np.arccos(((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1))).mean(1))
        print(f'   {r:<16} accel {np.median(np.concatenate(acc)):6.2f}   rotation {np.median(np.concatenate(ang)):6.2f}')


if __name__ == '__main__':
    main()
