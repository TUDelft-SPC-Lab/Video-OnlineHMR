"""
Multi-view SMPL refinement for COSI-Lab: one world-frame body per person, fitted to the ViTPose keypoints
of all its cameras at once. Needs the HMR results of scripts/run_cosilab_smpl.py for each cam-batch, and
assumes track IDs are shared person IDs between the cameras (see scripts/cosilab_multiview_calib.py).

Each person starts from the HMR pose of their most confident camera, shifted to the triangulated joints. The output has the layout of run_cosilab_smpl.py, per cam-batch, so the overlay works on it.
Frames where fewer than two cameras have usable keypoints of a person keep that person's depth in the best
camera close to the start, like the single-view refinement.
By default the start body is smoothed in time (Hampel + zero-phase Gaussian, lib/utils/temporal_smooth.py, as in
sam-body4d Stage 3) and the loss has a high-pass term on the rotations and translation (--smooth pre --w_highpass 1;
chosen on batch01, 8 cameras: better reprojection in every camera and ~2.5x less 3D acceleration). The fit without
them: --smooth none --w_highpass 0. People seen by one camera only are not refined
here; their single-view result stays in --hmr_root.

python scripts/run_cosilab_multiview.py --cam_batches cam01_batch01 cam03_batch01
python scripts/run_cosilab_multiview.py --cam_batches cam01_batch01 cam02_batch01 cam03_batch01 cam04_batch01 \
    --out_dir .../cosilab_multiview_4cam   # cam02/cam04: keypoints of scripts/cosilab_assign_ids.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import shutil
import time

import numpy as np
import torch
from scipy.ndimage import median_filter

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye
from lib.utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
from lib.utils.temporal_smooth import SmoothConfig, gaussian_lowpass_torch, smooth_body
from cosilab_multiview_calib import load_view, triangulate
from compare_cosilab_multiview import FOOT_JOINTS, floor_spread
from run_cosilab_smpl import contiguous_chunks, gmof_keypoint_loss, save_track, smoothness_loss

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batches', type=str, nargs='+', required=True)
    parser.add_argument('--hmr_root', type=str, default=f'{EXP_DIR}/cosilab_smpl')
    parser.add_argument('--out_dir', type=str, default=f'{EXP_DIR}/cosilab_multiview')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--kp_conf', type=float, default=0.3, help='ignore keypoints below this confidence')
    parser.add_argument('--refine_window', type=int, default=1000, help='frames optimised together')
    parser.add_argument('--refine_iters', type=int, default=600,
                        help='Adam steps per window; 150 left slow, localised fixes (e.g. one arm) unfinished')
    parser.add_argument('--refine_lr', type=float, default=0.01)
    parser.add_argument('--gmof_sigma', type=float, default=0.3, help='robust scale, in box-size units')
    parser.add_argument('--w_pose', type=float, default=0.01, help='stay close to the HMR body pose')
    parser.add_argument('--w_beta', type=float, default=1e-3)
    parser.add_argument('--w_smooth', type=float, default=10.0, help='3D joint acceleration')
    parser.add_argument('--w_depth', type=float, default=0.1,
                        help='on single-camera frames: stay close to the start depth in that camera')
    parser.add_argument('--view_weight', type=str, nargs='*', default=[],
                        help='keypoint weight per cam-batch, e.g. cam02_batch01=0.5 (default 1 for every view)')
    parser.add_argument('--min_view_kp', type=int, default=3,
                        help='confident keypoints a camera needs in a frame to count as seeing the person')
    # temporal smoothing as in sam-body4d Stage 3 (lib/utils/temporal_smooth.py)
    parser.add_argument('--smooth', type=str, default='pre', choices=['none', 'pre', 'post'],
                        help='Hampel + zero-phase Gaussian of the body: pre = of the start body, before the refinement '
                             '(sam-body4d order); post = of the refined body')
    parser.add_argument('--w_highpass', type=float, default=1.0,
                        help='in the refinement loss: |x - G*x|^2 of the 6D rotations and the translation (m), i.e. '
                             'what the Gaussian smoothing would remove')
    parser.add_argument('--sigma_pose', type=float, default=2.0, help='frames, body joint rotations')
    parser.add_argument('--sigma_rot', type=float, default=3.0, help='frames, root rotation')
    parser.add_argument('--sigma_trans', type=float, default=4.0, help='frames, translation')
    parser.add_argument('--hampel_half_window', type=int, default=3)
    parser.add_argument('--hampel_n_sigma', type=float, default=3.0)
    parser.add_argument('--w_floor', type=float, default=0.0,
                        help='in the refinement loss: Huber of the lowest foot joint\'s distance to a floor plane fitted '
                             'to all people\'s start bodies (sam-body4d ground step)')
    parser.add_argument('--floor_huber', type=float, default=0.10, help='m, Huber threshold of the floor term')
    parser.add_argument('--exclude', type=str, nargs='*', default=[],
                        help='person/camera pairs left out, e.g. cam12_batch01:14 (from a manual check of the overlays, '
                             'e.g. a person occluded by someone the visibility term does not model)')
    return parser.parse_args(argv)


def load_views(args):
    views = []
    for cam_batch in args.cam_batches:
        view = load_view(args.hmr_root, cam_batch)
        camera, tracks = cosilab.load_results(os.path.join(args.hmr_root, cam_batch), 'hmr')
        view.update(camera=camera, hmr=tracks, batch=camera['batch'], weight=args.view_weights.get(cam_batch, 1.0))
        for pid in args.excluded.get(cam_batch, []):
            if view['tracks'].pop(pid, None) is None:
                raise ValueError(f'--exclude {cam_batch}:{pid}: no such person in that view')
            tracks.pop(pid, None)
            print(f'{cam_batch}: person {pid} left out (--exclude)')
        for key, to_tensor in [('K', view['K']), ('D', view['D']), ('R', view['R']), ('t', view['t'])]:
            view['torch_' + key] = torch.tensor(to_tensor, dtype=torch.float32, device=args.device)
        views.append(view)
    return views


def triangulated_offset(views, pid, frames, joints, kp_conf, min_joints=3):
    """
    Per-frame shift (T, 3) that moves world joints (T, 17, 3) onto the joints triangulated from every pair of
    cameras: the median over the joints seen in both views, median-filtered over time and interpolated over
    frames with fewer than min_joints such joints. None if under 10 frames have enough.
    """
    shifts = {}
    for i, a in enumerate(views):
        for b in views[i + 1:]:
            ta, tb = a['tracks'].get(pid), b['tracks'].get(pid)
            if ta is None or tb is None:
                continue
            _, ia, ib = np.intersect1d(ta['frames'], tb['frames'], return_indices=True)
            valid = (ta['kp'][ia][..., 2] > kp_conf) & (tb['kp'][ib][..., 2] > kp_conf)
            X, _, _ = triangulate(a['C'], ta['rays'][ia], b['C'], tb['rays'][ib])
            idx = np.searchsorted(frames, ta['frames'][ia])
            for k in np.nonzero(valid.sum(1) >= min_joints)[0]:
                shifts.setdefault(idx[k], []).append(np.median(X[k][valid[k]] - joints[idx[k]][valid[k]], 0))
    if len(shifts) < 10:
        return None
    known = np.array(sorted(shifts))
    shift = median_filter(np.stack([np.mean(shifts[k], 0) for k in known]), size=(31, 1), mode='nearest')
    return np.stack([np.interp(np.arange(len(frames)), known, shift[:, i]) for i in range(3)], 1), len(known)


def init_person(smpl, views, pid, args):
    """World-frame start of one person, on the union of the frames of its cameras."""
    device = args.device
    seen = [v for v in views if pid in v['hmr']]
    frames = np.unique(np.concatenate([v['hmr'][pid]['session_frame_ids'] for v in seen]))

    def mean_conf(v):
        conf = v['hmr'][pid]['kp2d'][..., 2]
        return conf[conf > args.kp_conf].sum() / conf.size
    seen = sorted(seen, key=mean_conf)  # the most confident camera last, so it wins

    rotmat = np.zeros((len(frames), 24, 3, 3), dtype=np.float32)
    trans = np.zeros((len(frames), 3), dtype=np.float32)
    for v in seen:
        d = v['hmr'][pid]
        idx = np.searchsorted(frames, d['session_frame_ids'])
        rotmat[idx] = d['pred_rotmat']
        rotmat[idx, 0] = d['world_rotmat_root']
        trans[idx] = d['world_trans']
    betas = np.concatenate([v['hmr'][pid]['pred_shape'] for v in seen]).mean(0)

    betas = torch.tensor(betas, dtype=torch.float32, device=device)[None]
    rotmat, trans = torch.tensor(rotmat, device=device), torch.tensor(trans, device=device)
    joints = smpl(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:], betas=betas.expand(len(frames), -1),
                  transl=trans, pose2rot=False).joints[:, cosilab.CONFLAB_TO_SPIN].cpu().numpy()
    shift = triangulated_offset(views, pid, frames, joints, args.kp_conf)
    if shift is None:
        print(f'[Warning] person {pid}: too few triangulated frames, starting from the HMR translation')
    else:
        trans = trans + torch.tensor(shift[0], dtype=torch.float32, device=device)
        print(f'person {pid}: start shifted by a mean {np.linalg.norm(shift[0], axis=1).mean():.2f} m '
              f'({shift[1]} triangulated frames)', flush=True)
    if args.smooth == 'pre':
        rotmat, trans = smooth_body(rotmat, trans, frames, args.smooth_cfg)
    return {'frames': frames, 'best_cam': seen[-1]['name'],
            'rotmat': rotmat, 'betas': betas, 'trans': trans}


def foot_low(joints, down):
    """Per frame, the foot joint furthest along `down` (T, 3); joints are SMPL joints (T, 49, 3)."""
    f = joints[:, FOOT_JOINTS]
    return torch.gather(f, 1, (f @ down).argmax(1)[:, None, None].expand(-1, 1, 3))[:, 0]


def fit_floor(smpl, starts):
    """
    Floor plane (point c, unit normal pointing down) from the lowest foot joint of every start body and frame, as in
    compare_cosilab_multiview.py (robust plane fit with outlier removal). Returns (c, n) tensors.
    """
    lows, rough = [], []
    with torch.no_grad():
        for st in starts.values():
            J = smpl(global_orient=st['rotmat'][:, :1], body_pose=st['rotmat'][:, 1:],
                     betas=st['betas'].expand(len(st['frames']), -1), transl=st['trans'], pose2rot=False).joints
            rough.append(J[:, FOOT_JOINTS].reshape(-1, 3))
            lows.append(J)
    rough = torch.cat(rough).cpu().numpy().astype(np.float64)
    down = floor_spread(rough)[1]
    down = down if down[2] > 0 else -down  # the top-view cameras look along +z
    pts = torch.cat([foot_low(J, torch.tensor(down, dtype=J.dtype, device=J.device)) for J in lows])
    pts = pts.cpu().numpy().astype(np.float64)
    spread, normal, kept = floor_spread(pts)
    normal = normal if normal @ down > 0 else -normal
    keep = np.abs((pts - pts.mean(0)) @ normal) < 3 * spread
    c = pts[keep].mean(0)
    print(f'floor from the start bodies: spread {spread:.3f} m, kept {kept:.2f}, normal (down) {normal.round(3)}, '
          f'point {c.round(3)}', flush=True)
    return torch.tensor(c, dtype=torch.float32), torch.tensor(normal, dtype=torch.float32)


def view_targets(view, pid, frames, device):
    """Keypoints (T, 17, 3) and box sizes (T,) of one camera on the person's frames. Confidence 0 where unseen."""
    kp2d = np.zeros((len(frames), 17, 3), dtype=np.float32)
    box_size = np.ones(len(frames), dtype=np.float32)
    d = view['hmr'][pid]
    idx = np.searchsorted(frames, d['session_frame_ids'])
    kp2d[idx] = d['kp2d']
    box = d['boxes']
    box_size[idx] = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]).clip(min=1)
    return torch.tensor(kp2d, device=device), torch.tensor(box_size, device=device)


def project_view(joints, view):
    Xc = joints @ view['torch_R'].T + view['torch_t']
    return fisheye.project_torch(Xc, view['torch_K'], view['torch_D'])


def refine_person(args, smpl, views, pid, start):
    """Fit one world-frame body to the keypoints of every camera, per contiguous chunk and window."""
    seen = [v for v in views if pid in v['hmr']]
    frames = start['frames']
    targets = [view_targets(v, pid, frames, args.device) for v in seen]
    # cameras with usable keypoints per frame; with fewer than two, depth is anchored in the best one
    counts = torch.stack([(kp[..., 2] > args.kp_conf).sum(-1) for kp, _ in targets])
    single = (counts >= args.min_view_kp).sum(0) < 2
    best = counts.argmax(0)
    R_best = torch.stack([v['torch_R'] for v in seen])[best]
    t_best = torch.stack([v['torch_t'] for v in seen])[best]
    print(f'person {pid}: {int(single.sum())} of {len(frames)} frames with fewer than two usable cameras', flush=True)
    rotmat = start['rotmat'].clone()
    trans = start['trans'].clone()
    shape = start['betas'].expand(len(frames), -1).clone()

    windows = []
    for s, e in contiguous_chunks(frames, 1):
        n_win = int(np.ceil((e - s) / args.refine_window))
        bounds = np.linspace(s, e, n_win + 1).round().astype(int)
        windows += list(zip(bounds[:-1], bounds[1:]))

    kp_to_spin = torch.tensor(cosilab.CONFLAB_TO_SPIN, device=args.device)
    for s, e in windows:
        T = e - s
        init_body = rotmat[s:e, 1:].clone()
        pose6d = matrix_to_rotation_6d(rotmat[s:e]).clone().requires_grad_(True)
        betas = shape[s:e].mean(0, keepdim=True).clone().requires_grad_(True)
        init_trans = trans[s:e].clone()
        transl = init_trans.clone().requires_grad_(True)
        w_single = single[s:e].float()

        def highpass(x, sigma):  # what a Gaussian of sigma frames would remove, per frame
            return (x - gaussian_lowpass_torch(x, sigma)).pow(2).flatten(1).sum(-1).mean()

        def depth(x):  # depth of a translation in the best camera of each frame
            return (R_best[s:e, 2] * x).sum(-1) + t_best[s:e, 2]
        init_depth = depth(init_trans)
        window_targets = [(kp[s:e, :, :2], kp[s:e, :, 2] * (kp[s:e, :, 2] > args.kp_conf), size[s:e, None, None])
                          for kp, size in targets]
        optim = torch.optim.Adam([pose6d, betas, transl], lr=args.refine_lr)

        with torch.enable_grad():
            for _ in range(args.refine_iters):
                R = rotation_6d_to_matrix(pose6d)
                out = smpl(global_orient=R[:, :1], body_pose=R[:, 1:], betas=betas.expand(T, -1),
                           transl=transl, pose2rot=False)
                joints = out.joints[:, kp_to_spin]
                kp_sum, conf_sum = 0, 0
                for view, (target, conf, box_size) in zip(seen, window_targets):
                    gmof = gmof_keypoint_loss(project_view(joints, view), target, box_size, args.gmof_sigma)
                    kp_sum = kp_sum + view['weight'] * (conf * gmof).sum()
                    conf_sum = conf_sum + view['weight'] * conf.sum()
                l_kp = kp_sum / conf_sum.clamp(min=1)
                l_pose = (R[:, 1:] - init_body).pow(2).sum((-1, -2)).mean()
                l_beta = betas.pow(2).mean()
                l_depth = (w_single * (depth(transl) - init_depth).pow(2)).sum() / w_single.sum().clamp(min=1)
                loss = (l_kp + args.w_pose * l_pose + args.w_beta * l_beta + args.w_depth * l_depth
                        + args.w_smooth * smoothness_loss(out.joints[:, :25]))
                if args.w_highpass > 0:
                    loss = loss + args.w_highpass * (highpass(pose6d[:, :1], args.sigma_rot)
                                                     + highpass(pose6d[:, 1:], args.sigma_pose)
                                                     + highpass(transl, args.sigma_trans))
                if args.w_floor > 0:
                    dist = ((foot_low(out.joints, args.floor[1]) - args.floor[0]) * args.floor[1]).sum(-1)
                    loss = loss + args.w_floor * torch.nn.functional.huber_loss(
                        dist, torch.zeros_like(dist), delta=args.floor_huber)
                optim.zero_grad()
                loss.backward()
                optim.step()

        rotmat[s:e] = rotation_6d_to_matrix(pose6d)
        shape[s:e] = betas.expand(T, -1)
        trans[s:e] = transl

    if args.smooth == 'post':
        rotmat, trans = smooth_body(rotmat, trans, frames, args.smooth_cfg)
    return {'frames': frames, 'rotmat': rotmat, 'shape': shape, 'trans': trans}


def camera_result(smpl, body, view, pid):
    """The world-frame body on the frames of one camera's track, in that camera's frame too."""
    d = view['hmr'][pid]
    idx = torch.as_tensor(np.searchsorted(body['frames'], d['session_frame_ids']), device=body['rotmat'].device)
    rotmat = body['rotmat'][idx].clone()
    shape, trans = body['shape'][idx], body['trans'][idx]
    root_w = rotmat[:, 0].clone()
    J0 = fisheye.rest_pelvis(smpl, shape)
    root_c, trans_c = fisheye.transform_smpl(root_w, trans, J0, view['torch_R'], view['torch_t'])
    rotmat[:, 0] = root_c
    return {'pred_rotmat': rotmat, 'pred_shape': shape, 'pred_trans': trans_c,
            'world_rotmat_root': root_w, 'world_trans': trans}


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    args.view_weights = {k: float(v) for k, v in (w.split('=') for w in args.view_weight)}
    unknown = set(args.view_weights) - set(args.cam_batches)
    if unknown:
        raise ValueError(f'--view_weight for cam-batches not in --cam_batches: {sorted(unknown)}')
    args.excluded = {}
    for e in args.exclude:
        cb, pid = e.split(':')
        if cb not in args.cam_batches:
            raise ValueError(f'--exclude for a cam-batch not in --cam_batches: {e}')
        args.excluded.setdefault(cb, []).append(int(pid))
    args.smooth_cfg = SmoothConfig(args.sigma_pose, args.sigma_rot, args.sigma_trans, args.hampel_half_window,
                                   args.hampel_n_sigma)
    views = load_views(args)
    print('keypoint weight per view: ' + ', '.join(f'{v["name"]} {v["weight"]:g}' for v in views))
    smpl = SMPL().to(args.device)

    ids = []  # in order of the first camera that sees them
    for v in views:
        ids += [p for p in sorted(v['tracks']) if p not in ids and sum(p in u['tracks'] for u in views) >= 2]
    print(f'{args.cam_batches}: {len(ids)} people in at least two cameras: {ids}', flush=True)

    for view in views:
        out_dir = os.path.join(args.out_dir, view['name'])
        shutil.rmtree(out_dir, ignore_errors=True)  # never mix tracks of different runs
        os.makedirs(os.path.join(out_dir, 'refined'))
        shutil.copytree(os.path.join(args.hmr_root, view['name'], 'hmr'), os.path.join(out_dir, 'hmr'))
        shutil.copy(os.path.join(args.hmr_root, view['name'], 'camera.json'), out_dir)
        with open(os.path.join(out_dir, 'run_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    starts = {pid: init_person(smpl, views, pid, args) for pid in ids}
    if args.w_floor > 0:
        c, n = fit_floor(smpl, starts)
        args.floor = (c.to(args.device), n.to(args.device))
    for pid in ids:
        t0 = time.time()
        start = starts[pid]
        body = refine_person(args, smpl, views, pid, start)
        for view in views:
            if pid not in view['hmr']:
                continue
            d = view['hmr'][pid]
            track = {'frames': d['frame_ids'], 'boxes': d['boxes'], 'kp2d': d['kp2d']}
            save_track(os.path.join(args.out_dir, view['name'], 'refined', f'track_{pid}.npz'),
                       camera_result(smpl, body, view, pid), d, track, view['batch'])
        print(f'person {pid} ({len(body["frames"])} frames, start from {start["best_cam"]}) '
              f'in {time.time() - t0:.0f} s', flush=True)

    for view in views:
        with open(os.path.join(args.out_dir, view['name'], 'matches.json'), 'w') as f:
            json.dump({'people': ids, 'start_camera': {p: s['best_cam'] for p, s in starts.items()},
                       'cam_batches': args.cam_batches}, f, indent=2)
    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main(parse_args())
