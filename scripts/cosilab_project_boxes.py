"""
Assumed person boxes in a camera without keypoints, projected from the SMPL bodies of its neighbouring cameras.
Detectors miss many people in these top-down fisheye views, so every person a neighbour sees gets a box.

Per (frame, person seen by a neighbour) the body is, best source first: the multi-view body (scripts/run_cosilab_multiview.py,
people seen by two cameras), otherwise the single-view refined body of the neighbour (scripts/run_cosilab_smpl.py --refine);
with two neighbours, the one with the more confident keypoints in that frame. The box is the fisheye projection of the
SMPL vertices into the new camera, padded and clipped to the image. Bodies with under half of their vertices in the
image (or in front of the camera) are skipped.

python scripts/cosilab_project_boxes.py --cam_batch cam02_batch01 --start 6000 --num_frames 600
Output: <out_dir>/<cam_batch>/proposals.npz with per box frame, pid, box (xyxy, full-res), source, inside, src_conf.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
from collections import Counter

import numpy as np
import torch

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
IMG_W, IMG_H = 1920, 1080


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='the new camera, e.g. cam02_batch01')
    parser.add_argument('--neighbours', type=int, nargs='+', default=None,
                        help='neighbouring cameras with keypoints (default: cosilab.NEIGHBOURS)')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--single_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview')
    parser.add_argument('--out_dir', type=str, default=f'{EXP_DIR}/cosilab_vit003_extra')
    parser.add_argument('--start', type=int, required=True, help='first batch frame')
    parser.add_argument('--num_frames', type=int, required=True)
    parser.add_argument('--pad', type=float, default=0.1, help='box growth, as a fraction of its width and height')
    parser.add_argument('--min_inside', type=float, default=0.5, help='min fraction of vertices in the image')
    parser.add_argument('--kp_conf', type=float, default=0.3)
    return parser.parse_args(argv)


def frame_conf(kp, thresh):
    """Per-frame keypoint confidence of a track (T, 17, 3): the mean confidence, counting joints below thresh as 0."""
    c = kp[..., 2]
    return np.where(c > thresh, c, 0).mean(-1)


def collect_bodies(args, neighbours, batch, start, end):
    """
    {(frame, pid): (source, conf, params)} with the best body per person-frame. params index into a
    list of per-track npz dicts: (track dict, row).
    """
    best = {}
    for cam in neighbours:
        nb = f'cam{cam:02d}_batch{batch:02d}'
        multi_dir = os.path.join(args.multi_root, nb)
        multi = cosilab.load_results(multi_dir, 'refined')[1] if os.path.isdir(os.path.join(multi_dir, 'refined')) else {}
        single_dir = os.path.join(args.single_root, nb)  # cameras without dataset keypoints have no refined bodies
        single = cosilab.load_results(single_dir, 'refined')[1] if os.path.isdir(os.path.join(single_dir, 'refined')) else {}
        print(f'{nb}: {len(multi)} multi-view bodies {sorted(multi)}, {len(single)} single-view {sorted(single)}')
        for kind, tracks in [('multiview', multi), (f'single:{nb}', single)]:
            rank = 1 if kind == 'multiview' else 0  # multi-view wins over any single-view body
            for pid, d in tracks.items():
                if pid < 0:  # -2: not a participant
                    continue
                conf = frame_conf(d['kp2d'], args.kp_conf)
                for i, f in enumerate(d['frame_ids']):
                    f = int(f)
                    if not start <= f < end:
                        continue
                    key = (f, pid)
                    cand = (rank, conf[i], kind, d, i)
                    if key not in best or cand[:2] > best[key][:2]:
                        best[key] = cand
    return best


def main(args):
    torch.set_grad_enabled(False)
    cam, batch, session = cosilab.parse_cam_batch(args.cam_batch)
    neighbours = args.neighbours or cosilab.NEIGHBOURS[cam]
    camera = cosilab.load_camera(args.data_root, cam)
    R = torch.tensor(camera['R_w2c'], dtype=torch.float32)
    t = torch.tensor(camera['t_w2c'], dtype=torch.float32)
    K = torch.tensor(camera['K'], dtype=torch.float32)
    D = torch.tensor(camera['D'], dtype=torch.float32)
    start, end = args.start, args.start + args.num_frames

    best = collect_bodies(args, neighbours, batch, start, end)
    keys = sorted(best)
    print(f'{args.cam_batch}: {len(keys)} person-frames from cameras {neighbours}, '
          f'{len({p for _, p in keys})} people', flush=True)

    smpl = SMPL()
    out = {'frame': [], 'pid': [], 'box': [], 'source': [], 'inside': [], 'src_conf': []}
    skipped = Counter()
    for s in range(0, len(keys), 512):
        chunk = keys[s:s + 512]
        rows = [best[k] for k in chunk]
        rotmat = torch.tensor(np.stack([d['pred_rotmat'][i] for _, _, _, d, i in rows]))
        root = torch.tensor(np.stack([d['world_rotmat_root'][i] for _, _, _, d, i in rows]))
        betas = torch.tensor(np.stack([d['pred_shape'][i] for _, _, _, d, i in rows]))
        trans = torch.tensor(np.stack([d['world_trans'][i] for _, _, _, d, i in rows]))
        verts = smpl(global_orient=root[:, None], body_pose=rotmat[:, 1:], betas=betas, transl=trans,
                     pose2rot=False).vertices
        Xc = verts @ R.T + t
        uv = fisheye.project_torch(Xc, K, D).numpy()
        front = Xc[..., 2].numpy() > 0
        inside = front & (uv[..., 0] >= 0) & (uv[..., 0] < IMG_W) & (uv[..., 1] >= 0) & (uv[..., 1] < IMG_H)
        for j, ((f, pid), (_, conf, kind, _, _)) in enumerate(zip(chunk, rows)):
            frac = inside[j].mean()
            if frac < args.min_inside:
                skipped[kind.split(':')[0]] += 1
                continue
            p = uv[j][front[j]]
            x1, y1 = p.min(0)
            x2, y2 = p.max(0)
            dx, dy = (x2 - x1) * args.pad / 2, (y2 - y1) * args.pad / 2
            box = np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy]).clip([0, 0, 0, 0], [IMG_W - 1, IMG_H - 1] * 2)
            out['frame'].append(f)
            out['pid'].append(pid)
            out['box'].append(box)
            out['source'].append(kind)
            out['inside'].append(frac)
            out['src_conf'].append(conf)

    out = {k: np.array(v) for k, v in out.items()}
    out['box'] = out['box'].reshape(-1, 4).astype(np.float32)
    n_src = Counter(out['source'].tolist())
    print(f'{len(out["frame"])} boxes kept; by source {dict(n_src)}; skipped (< {args.min_inside:.0%} in the image) '
          f'{dict(skipped)}')
    per_frame = np.bincount(out['frame'] - start, minlength=args.num_frames)
    print(f'boxes per frame: mean {per_frame.mean():.1f}, min {per_frame.min()}, max {per_frame.max()}')
    for pid in sorted(set(out['pid'].tolist())):
        m = out['pid'] == pid
        w = out['box'][m, 2] - out['box'][m, 0]
        h = out['box'][m, 3] - out['box'][m, 1]
        print(f'  person {pid:3d}: {m.sum():4d} frames, sources {dict(Counter(out["source"][m].tolist()))}, '
              f'median box {np.median(w):.0f} x {np.median(h):.0f} px, inside {np.median(out["inside"][m]):.2f}')

    out_dir = os.path.join(args.out_dir, args.cam_batch)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'proposals.npz')
    np.savez(path, neighbours=np.array(neighbours), start=start, num_frames=args.num_frames, **out)
    print(f'Saved to {path}')


if __name__ == '__main__':
    main(parse_args())
