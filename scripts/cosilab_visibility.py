"""
Which joints of each person a camera actually sees, i.e. not hidden behind another person, from the SMPL bodies of
the neighbouring cameras (same bodies as scripts/cosilab_project_boxes.py: multi-view body, else single-view refined).
Meant for the side cameras, where people often stand behind each other. CPU only.

Per frame, every person's SMPL vertices are projected into the camera and splatted into a coarse z-buffer
(--cell px cells; depth = distance from the camera centre). Per cell and person: the nearest vertex of the person.
  - joint occluded: in at least half of the 3x3 cells around the joint's projection, another person is more than
    --margin m closer to the camera than the joint. Self-occlusion is not counted.
  - visible fraction of a person: the share of the cells the person covers where no other person is more than
    --margin m in front of the person's own surface.
Non-person occluders and people without a body are not modelled.

python scripts/cosilab_visibility.py --cam_batch cam12_batch01 --multi_root .../cosilab_multiview_4cam --start 6000 --num_frames 600
Output: <out_dir>/<cam_batch>/visibility.npz with frame (batch frame), pid, joint_vis (N, 17) bool, vis_frac (N,),
joint_uv (N, 17, 2) px, source.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from collections import defaultdict

import numpy as np
import torch

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye
from cosilab_project_boxes import collect_bodies

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
IMG_W, IMG_H = 1920, 1080


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='e.g. cam12_batch01')
    parser.add_argument('--neighbours', type=int, nargs='+', default=None,
                        help='cameras whose bodies are used (default: cosilab.NEIGHBOURS)')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--single_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview')
    parser.add_argument('--out_dir', type=str, default=f'{EXP_DIR}/cosilab_vit003_extra')
    parser.add_argument('--start', type=int, required=True, help='first batch frame')
    parser.add_argument('--num_frames', type=int, required=True)
    parser.add_argument('--kp_conf', type=float, default=0.3)
    parser.add_argument('--cell', type=int, default=8, help='z-buffer cell size (px)')
    parser.add_argument('--margin', type=float, default=0.2, help='m in front before another person occludes')
    return parser.parse_args(argv)


def frame_visibility(uv, depth, juv, jdepth, cell, margin):
    """
    One frame. uv (P, V, 2), depth (P, V) of the vertices (nan behind the camera), juv (P, 17, 2), jdepth (P, 17).
    Returns joint_vis (P, 17) bool and vis_frac (P,).
    """
    P = uv.shape[0]
    gw, gh = -(-IMG_W // cell), -(-IMG_H // cell)
    near = np.full((P, gh * gw), np.inf)  # nearest vertex of each person per cell
    for p in range(P):
        ok = np.isfinite(depth[p]) & (uv[p, :, 0] >= 0) & (uv[p, :, 0] < IMG_W) & (uv[p, :, 1] >= 0) & (uv[p, :, 1] < IMG_H)
        c = (uv[p, ok, 1] // cell).astype(int) * gw + (uv[p, ok, 0] // cell).astype(int)
        np.minimum.at(near[p], c, depth[p, ok])
    # nearest *other* person per cell: the two smallest depths over people
    order = np.sort(near, axis=0)
    first, second = order[0], order[1] if P > 1 else np.full(gh * gw, np.inf)
    other = np.where(near == first[None], second[None], first[None])  # (P, cells)

    covered = np.isfinite(near)
    hidden = covered & (other < near - margin)
    vis_frac = np.where(covered.sum(1) > 0, 1 - hidden.sum(1) / covered.sum(1).clip(min=1), 0.0)

    joint_vis = np.zeros((P, 17), dtype=bool)
    cx, cy = np.floor(juv[..., 0] / cell).astype(int), np.floor(juv[..., 1] / cell).astype(int)
    occ = np.zeros((P, 17))
    n = np.zeros((P, 17))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = cx + dx, cy + dy
            inside = (x >= 0) & (x < gw) & (y >= 0) & (y < gh)
            idx = np.where(inside, y.clip(0, gh - 1) * gw + x.clip(0, gw - 1), 0)
            o = np.take_along_axis(other, idx, axis=1)
            occ += inside & (o < jdepth - margin)
            n += inside
    in_img = (juv[..., 0] >= 0) & (juv[..., 0] < IMG_W) & (juv[..., 1] >= 0) & (juv[..., 1] < IMG_H) & np.isfinite(jdepth)
    joint_vis = in_img & (occ < 0.5 * n.clip(min=1))
    return joint_vis, vis_frac


def main(args):
    torch.set_grad_enabled(False)
    cam, batch, _ = cosilab.parse_cam_batch(args.cam_batch)
    neighbours = args.neighbours or cosilab.NEIGHBOURS[cam]
    camera = cosilab.load_camera(args.data_root, cam)
    R = torch.tensor(camera['R_w2c'], dtype=torch.float32)
    t = torch.tensor(camera['t_w2c'], dtype=torch.float32)
    K = torch.tensor(camera['K'], dtype=torch.float32)
    D = torch.tensor(camera['D'], dtype=torch.float32)
    start, end = args.start, args.start + args.num_frames

    best = collect_bodies(args, neighbours, batch, start, end)
    by_frame = defaultdict(list)
    for f, p in sorted(best):
        by_frame[f].append(p)
    smpl = SMPL()
    out = {'frame': [], 'pid': [], 'joint_vis': [], 'vis_frac': [], 'joint_uv': [], 'source': []}
    for f in sorted(by_frame):
        rows = [best[(f, p)] for p in by_frame[f]]
        res = smpl(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for *_, d, i in rows]))[:, None],
                   body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for *_, d, i in rows]))[:, 1:],
                   betas=torch.tensor(np.stack([d['pred_shape'][i] for *_, d, i in rows])),
                   transl=torch.tensor(np.stack([d['world_trans'][i] for *_, d, i in rows])), pose2rot=False)
        Xv = res.vertices @ R.T + t
        Xj = res.joints[:, cosilab.CONFLAB_TO_SPIN] @ R.T + t
        uv, juv = fisheye.project_torch(Xv, K, D).numpy(), fisheye.project_torch(Xj, K, D).numpy()
        depth = np.where(Xv[..., 2].numpy() > 0, Xv.norm(dim=-1).numpy(), np.nan)
        jdepth = np.where(Xj[..., 2].numpy() > 0, Xj.norm(dim=-1).numpy(), np.nan)
        jv, vf = frame_visibility(uv, depth, juv, jdepth, args.cell, args.margin)
        for k, p in enumerate(by_frame[f]):
            out['frame'].append(f)
            out['pid'].append(p)
            out['joint_vis'].append(jv[k])
            out['vis_frac'].append(vf[k])
            out['joint_uv'].append(juv[k])
            out['source'].append(rows[k][2])

    out = {k: np.array(v) for k, v in out.items()}
    jv, vf = out['joint_vis'], out['vis_frac']
    print(f'{args.cam_batch}: {len(vf)} person-frames of {len(set(out["pid"].tolist()))} people; joints visible '
          f'{jv.mean():.3f}; visible fraction p10 / median / p90 {np.percentile(vf, 10):.2f} / {np.median(vf):.2f} / '
          f'{np.percentile(vf, 90):.2f}; person-frames under 0.3 visible {(vf < 0.3).mean():.3f}')
    for p in sorted(set(out['pid'].tolist())):
        m = out['pid'] == p
        print(f'  person {p:3d}: {m.sum():4d} frames, joints visible {jv[m].mean():.2f}, '
              f'visible fraction median {np.median(vf[m]):.2f}, under 0.3 {(vf[m] < 0.3).mean():.2f}')
    path = os.path.join(args.out_dir, args.cam_batch, 'visibility.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, start=start, num_frames=args.num_frames, cell=args.cell, margin=args.margin, **out)
    print(f'Saved to {path}')


if __name__ == '__main__':
    main(parse_args())
