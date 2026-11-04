"""
Compare multi-view SMPL (scripts/run_cosilab_multiview.py) with the single-view results of the same cameras, on the people
of the multi-view fit. Works for any number of cameras. There is no ground truth, so all numbers are proxies:
  1. reprojection into every camera that sees the person (single-view bodies also into the cameras they did not see)
  2. distance to the joints triangulated from each pair of cameras with dataset keypoints
  3. floor: spread of the lowest body point around a plane fitted to it (not part of any loss)
  4. rest height per person, and 3D acceleration and rotation from the first camera's hmr
Single-view `refined` rows only for the cameras that have `refined/`. With --ref_multi_root (e.g. the 2-camera fit),
that fit is added as a row, and every table is also printed on its people only.

python scripts/compare_cosilab_multiview.py --cam_batches cam01_batch01 cam03_batch01
python scripts/compare_cosilab_multiview.py --cam_batches cam01_batch01 cam02_batch01 cam03_batch01 cam04_batch01 \\
    --multi_root .../cosilab_multiview_4cam --ref_multi_root .../cosilab_multiview
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
from itertools import combinations

import numpy as np
import torch

from lib.models.smpl import SMPL
from lib.utils import cosilab
from cosilab_multiview_calib import CONF, JOINTS, load_view, project, triangulate

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
FOOT_JOINTS = [11, 14, 19, 20, 21, 22, 23, 24]  # OpenPose ankles, toes and heels in the SPIN joints
SPIN = cosilab.CONFLAB_TO_SPIN


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


def bodies(smpl, track_sets):
    """
    One body table per person from one or more {pid: npz dict} (the cam-batch files of one fit hold the same world body
    on different frames): {pid: {'frames': session frames, 'joints': (T, 49, 3), 'rotmat': (T, 24, 3, 3), 'betas': (10,)}}.
    """
    out = {}
    for tracks in track_sets:
        for pid, d in tracks.items():
            if pid < 0:
                continue
            o = out.setdefault(pid, {'frames': [], 'joints': [], 'rotmat': [], 'betas': []})
            o['frames'].append(d['session_frame_ids'])
            o['joints'].append(world_joints(smpl, d))
            o['rotmat'].append(d['pred_rotmat'])
            o['betas'].append(d['pred_shape'])
    for o in out.values():
        f, idx = np.unique(np.concatenate(o['frames']), return_index=True)
        o['frames'] = f
        o['joints'] = np.concatenate(o['joints'])[idx]
        o['rotmat'] = np.concatenate(o['rotmat'])[idx]
        o['betas'] = np.concatenate(o['betas']).mean(0)
    return out


def on_frames(body, frames):
    """Rows of the body table at the given session frames, and a mask of the frames it has."""
    i = np.searchsorted(body['frames'], frames).clip(max=len(body['frames']) - 1)
    return i, body['frames'][i] == frames


def reprojection(body, view, pid):
    """Errors (box sizes, px) of a body against one camera's keypoints of the same person, on their common frames."""
    t = view['tracks'][pid]
    i, has = on_frames(body, t['frames'])
    kp = t['kp'][has]
    box = view['boxes'][pid][has]
    size = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]).clip(min=1)
    err = np.linalg.norm(project(view, body['joints'][i[has]][:, SPIN].astype(np.float64)) - kp[..., :2], axis=-1)
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


def report(runs, views, tri_pairs, pids, smpl, label):
    print(f'\n=========== {label}: {len(pids)} people {pids}')
    names = [v['name'][:5] for v in views]
    have = {r: [p for p in pids if p in b] for r, b in runs.items()}

    print(f'\n1. reprojection error (median px, box sizes in brackets; keypoints with conf > {CONF}; '
          f'n = people of the row seen by that camera)')
    print(f'   {"result":<18} ' + ' '.join(f'{"into " + n:>22}' for n in names))
    for r, b in runs.items():
        cols = []
        for v in views:
            ps = [p for p in have[r] if p in v['tracks']]
            if not ps:
                cols.append(f'{"-":>22}')
                continue
            s, px = zip(*[reprojection(b[p], v, p)[:2] for p in ps])
            s, px = np.concatenate(s), np.concatenate(px)
            cols.append(f'{np.median(px):6.1f} ({np.median(s):.3f}) n{len(ps):3d}' if len(px) else f'{"-":>22}')
        print(f'   {r:<18} ' + ' '.join(f'{c:>22}' for c in cols))

    mv = [r for r in runs if r.startswith('multiview')]
    for r in mv:
        print(f'\n   {r}, per joint median px, ' + ' / '.join(names))
        out = []
        for j, name in enumerate(JOINTS):
            e = []
            for v in views:
                rows = [reprojection(runs[r][p], v, p) for p in have[r] if p in v['tracks']]
                x = np.concatenate([x[2][:, j][x[3][:, j]] for x in rows]) if rows else np.array([])
                e.append(f'{np.median(x):.1f}' if len(x) else '-')
            out.append(f'{name} {"/".join(e)}')
        print('   ' + ', '.join(out))

    print('\n2. distance to the triangulated joints (m, joints valid in both cameras, median)')
    for a, b in tri_pairs:
        print(f'   pair {a["name"][:5]}-{b["name"][:5]}')
        tri = {}
        for p in pids:
            ta, tb = a['tracks'].get(p), b['tracks'].get(p)
            if ta is None or tb is None:
                continue
            f, ia, ib = np.intersect1d(ta['frames'], tb['frames'], return_indices=True)
            X = triangulate(a['C'], ta['rays'][ia], b['C'], tb['rays'][ib])[0]
            tri[p] = (f, X, (ta['kp'][ia][..., 2] > CONF) & (tb['kp'][ib][..., 2] > CONF))
        for r, bod in runs.items():
            d, pel = [], []
            for p, (f, X, v) in tri.items():
                if p not in bod:
                    continue
                i, has = on_frames(bod[p], f)
                J = bod[p]['joints'][i[has]]
                d.append(np.linalg.norm(J[:, SPIN] - X[has], axis=-1)[v[has]])
                hip = v[has][:, [9, 12]].all(1)
                pel.append(np.linalg.norm(J[:, 8] - X[has][:, [9, 12]].mean(1), axis=-1)[hip])
            if d and len(np.concatenate(d)):
                print(f'     {r:<18} all joints {np.median(np.concatenate(d)):.3f}   pelvis {np.median(np.concatenate(pel)):.3f}'
                      f'   ({len(d)} people)')

    print('\n3. floor: lowest foot joint per person-frame, spread around a fitted plane (m, not used by any loss)')
    for r, b in runs.items():
        if not have[r]:
            continue
        rough = np.concatenate([b[p]['joints'][:, FOOT_JOINTS].reshape(-1, 3) for p in have[r]])
        down = floor_spread(rough)[1]
        down = down if down[2] > 0 else -down  # the top-view cameras look along +z
        lows = []
        for p in have[r]:
            f = b[p]['joints'][:, FOOT_JOINTS]
            lows.append(np.take_along_axis(f, np.argmax(f @ down, 1)[:, None, None], 1)[:, 0])
        spread, normal, kept = floor_spread(np.concatenate(lows))
        print(f'   {r:<18} spread {spread:.3f}   kept {kept:.2f}   normal {normal.round(3)}   ({len(have[r])} people)')

    print('\n4. rest height (m) per person')
    print(f'   {"":<18} ' + ' '.join(f'{p:>5d}' for p in pids))
    for r, b in runs.items():
        if r.startswith('hmr'):
            continue
        print(f'   {r:<18} ' + ' '.join(f'{rest_height(smpl, b[p]["betas"]):5.2f}' if p in b else f'{"-":>5}' for p in pids))

    base_name = next(r for r in runs if r.startswith('hmr'))
    base = runs[base_name]
    print(f'\n   median 3D acceleration (mm/frame^2) and mean body-joint rotation from {base_name} (deg, common frames)')
    for r, b in runs.items():
        acc, ang = [], []
        for p in have[r]:
            j3 = b[p]['joints'][:, :25]
            if len(j3) > 2:
                acc.append(np.linalg.norm(j3[2:] - 2 * j3[1:-1] + j3[:-2], axis=-1).ravel() * 1000)
            if p in base:
                i, has = on_frames(b[p], base[p]['frames'])
                if has.any():
                    R = np.einsum('njab,njac->njbc', base[p]['rotmat'][has][:, 1:], b[p]['rotmat'][i[has]][:, 1:])
                    ang.append(np.degrees(np.arccos(((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1))).mean(1))
        a = f'{np.median(np.concatenate(acc)):6.2f}' if acc else '     -'
        g = f'{np.median(np.concatenate(ang)):6.2f}' if ang else '     -'
        print(f'   {r:<18} accel {a}   rotation {g}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batches', type=str, nargs='+', required=True)
    parser.add_argument('--single_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview')
    parser.add_argument('--ref_multi_root', type=str, default=None,
                        help='another multi-view fit to compare with (e.g. the 2-camera one), on its own cam-batches')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    args = parser.parse_args()

    smpl = SMPL()
    views = [load_view(args.single_root, cb) for cb in args.cam_batches]
    for v in views:
        v['boxes'] = {p: d['boxes'] for p, d in cosilab.load_results(os.path.join(args.single_root, v['name']), 'hmr')[1].items()}

    runs = {}
    for cb in args.cam_batches:
        runs[f'hmr {cb[:5]}'] = bodies(smpl, [cosilab.load_results(os.path.join(args.single_root, cb), 'hmr')[1]])
    for cb in args.cam_batches:
        if os.path.isdir(os.path.join(args.single_root, cb, 'refined')):
            runs[f'refined {cb[:5]}'] = bodies(smpl, [cosilab.load_results(os.path.join(args.single_root, cb), 'refined')[1]])
    multi = [cosilab.load_results(os.path.join(args.multi_root, cb), 'refined')[1] for cb in args.cam_batches]
    # the same world body must come out of every camera's multi-view file
    max_diff = 0
    for a, b in combinations(multi, 2):
        for p in set(a) & set(b):
            _, ia, ib = np.intersect1d(a[p]['session_frame_ids'], b[p]['session_frame_ids'], return_indices=True)
            if len(ia):
                keys = ['pred_rotmat', 'world_rotmat_root', 'pred_shape', 'world_trans']
                ja = world_joints(smpl, {k: a[p][k][ia] for k in keys})
                jb = world_joints(smpl, {k: b[p][k][ib] for k in keys})
                max_diff = max(max_diff, np.abs(ja - jb).max())
    print(f'multi-view body, max difference between the cam-batch files on common frames: {max_diff * 1000:.2f} mm')
    runs[f'multiview {len(args.cam_batches)}cam'] = bodies(smpl, multi)

    ref_pids = None
    if args.ref_multi_root:
        ref_cbs = [cb for cb in args.cam_batches if os.path.isdir(os.path.join(args.ref_multi_root, cb, 'refined'))]
        ref = bodies(smpl, [cosilab.load_results(os.path.join(args.ref_multi_root, cb), 'refined')[1] for cb in ref_cbs])
        runs[f'multiview {len(ref_cbs)}cam ref'] = ref
        ref_pids = sorted(ref)

    has_dataset_kp = [v for v in views
                      if os.path.exists(cosilab.keypoints_path(args.data_root, v['name']))]
    tri_pairs = list(combinations(has_dataset_kp, 2))
    pids = sorted(runs[f'multiview {len(args.cam_batches)}cam'])
    print(f'people per camera: ' + ', '.join(f'{v["name"][:5]} {len(v["tracks"])}' for v in views))
    print(f'triangulation pairs (dataset keypoints): {[(a["name"][:5], b["name"][:5]) for a, b in tri_pairs]}')
    report(runs, views, tri_pairs, pids, smpl, 'people of the multi-view fit')
    if ref_pids is not None:
        report(runs, views, tri_pairs, [p for p in ref_pids if p in pids], smpl, 'people of the reference fit only')


if __name__ == '__main__':
    main()
