"""
Run only the OnlineHMR (SMPL) part of the pipeline on COSI-Lab videos, using the given ViTPose
tracks and the fisheye calibration of the static cameras. No detection, tracking, SLAM or depth.

Each person is cropped through a virtual pinhole camera looking at them (lib/utils/fisheye.py),
OnlineHMR runs per track in temporal order with its KV cache, and the result is rotated back into
the real camera frame and, when extrinsics exist, into the session world frame. With --refine, the
SMPL parameters are also fitted to the ViTPose keypoints.

python scripts/run_cosilab_smpl.py --cam_batch cam01_batch01 [--refine] [--num_frames 600]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from lib.core import constants
from lib.models import get_hmr_vimo
from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye
from lib.utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

CROP_SIZE = 256


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='e.g. cam01_batch01')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--out_dir', type=str,
                        default='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl')
    parser.add_argument('--checkpoint', type=str, default='./results/onlinehmr/checkpoint.pth.tar')
    parser.add_argument('--start', type=int, default=None, help='first batch frame (default: first annotated)')
    parser.add_argument('--num_frames', type=int, default=None, help='number of frames (default: all)')
    parser.add_argument('--chunk', type=int, default=64, help='crops per HMR forward')
    parser.add_argument('--min_track_len', type=int, default=3, help='drop track segments shorter than this')
    parser.add_argument('--extrinsic_c2w', action='store_true',
                        help='read rvec/tvec as camera->world instead of OpenCV world->camera')
    parser.add_argument('--device', type=str, default='cuda')
    # keypoint refinement
    parser.add_argument('--refine', action='store_true', help='also fit SMPL to the ViTPose keypoints')
    parser.add_argument('--kp_conf', type=float, default=0.3, help='ignore keypoints below this confidence')
    parser.add_argument('--refine_window', type=int, default=1000, help='frames optimised together')
    parser.add_argument('--refine_iters', type=int, default=150)
    parser.add_argument('--refine_lr', type=float, default=0.01)
    parser.add_argument('--gmof_sigma', type=float, default=0.3, help='robust scale, in box-size units')
    parser.add_argument('--w_pose', type=float, default=0.01, help='stay close to the HMR pose')
    parser.add_argument('--w_beta', type=float, default=1e-3)
    parser.add_argument('--w_smooth', type=float, default=10.0, help='3D joint acceleration')
    parser.add_argument('--w_depth', type=float, default=0.1, help='stay close to the HMR depth')
    return parser.parse_args(argv)


def contiguous_chunks(frames, min_len):
    """Split sorted frame ids into runs of consecutive frames. Returns (start, end) index pairs."""
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(frames)]])
    return [(s, e) for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= min_len]


def prepare_tracks(tracks, start, end, min_len):
    """Keep the frames in [start, end) of each track and split them into contiguous chunks."""
    out = {}
    for pid, t in tracks.items():
        keep = (t['frames'] >= start) & (t['frames'] < end)
        t = {k: v[keep] for k, v in t.items()}
        chunks = contiguous_chunks(t['frames'], min_len)
        if not chunks:
            continue
        keep = np.concatenate([np.arange(s, e) for s, e in chunks])
        t = {k: v[keep] for k, v in t.items()}
        t['chunks'] = contiguous_chunks(t['frames'], 1)
        out[pid] = t
    return out


class TrackState:
    def __init__(self):
        self.crops, self.focals, self.rots = [], [], []
        self.cache = None
        self.results = defaultdict(list)


def run_hmr(hmr_model, state, device):
    """Run OnlineHMR on the buffered crops of one track, continuing from its cache."""
    if not state.crops:
        return
    crops = torch.stack(state.crops)
    T = crops.shape[0]
    f_v = torch.tensor(state.focals, dtype=torch.float32, device=device)
    batch = {'img': crops,
             'center': torch.full((T, 2), CROP_SIZE / 2, device=device),
             'scale': torch.full((T,), CROP_SIZE / 200, device=device),
             'img_focal': f_v,
             'img_center': torch.full((T, 2), CROP_SIZE / 2, device=device)}
    out, _, state.cache = hmr_model.inference_forward(batch, cache=state.cache, device=device)

    state.results['pred_rotmat'].append(out['pred_rotmat'].float().cpu())
    state.results['pred_shape'].append(out['pred_shape'].float().cpu())
    state.results['pred_cam'].append(out['pred_cam'].float().cpu())
    state.results['trans_virtual'].append(out['trans_full'][:, 0].float().cpu())
    state.results['virtual_focal'].append(f_v.cpu())
    state.results['virtual_R'].append(torch.tensor(np.stack(state.rots), dtype=torch.float32))
    state.crops, state.focals, state.rots = [], [], []


def run_tracks(args, hmr_model, tracks, camera, video, start):
    """One sequential pass over the video, cropping every tracked person and running OnlineHMR."""
    device = args.device
    K = torch.tensor(camera['K'], dtype=torch.float32, device=device)
    D = torch.tensor(camera['D'], dtype=torch.float32, device=device)
    mean = torch.tensor(constants.IMG_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(constants.IMG_NORM_STD, device=device).view(1, 3, 1, 1)

    frame_index = defaultdict(list)
    chunk_starts = set()
    for pid, t in tracks.items():
        for i, key in enumerate(t['frames']):
            frame_index[int(key)].append((pid, i))
        chunk_starts.update((pid, int(t['frames'][s])) for s, _ in t['chunks'])

    first, last = min(frame_index), max(frame_index)
    reader = cosilab.VideoReader(video, cosilab.session_frame(args.batch, first), last - first + 1)
    states = {pid: TrackState() for pid in tracks}

    n_crops, t0 = 0, time.time()
    key = first - 1
    for key, frame in zip(range(first, last + 1), reader):
        entries = frame_index.get(key)
        if not entries:
            continue
        img = torch.from_numpy(frame).to(device).permute(2, 0, 1).flip(0).float() / 255  # BGR -> RGB
        boxes = np.stack([tracks[pid]['boxes'][i] for pid, i in entries])
        R_v, f_v = fisheye.setup_virtual_cameras(boxes, camera['K'], camera['D'], CROP_SIZE)
        crops = fisheye.render_crops(img, torch.tensor(R_v, dtype=torch.float32, device=device),
                                     torch.tensor(f_v, dtype=torch.float32, device=device), K, D, CROP_SIZE)
        crops = (crops - mean) / std

        for j, (pid, i) in enumerate(entries):
            state = states[pid]
            if (pid, key) in chunk_starts:  # the track was lost, start a new temporal context
                run_hmr(hmr_model, state, device)
                state.cache = None
            state.crops.append(crops[j])
            state.focals.append(float(f_v[j]))
            state.rots.append(R_v[j])
            if len(state.crops) >= args.chunk:
                run_hmr(hmr_model, state, device)

        n_crops += len(entries)
        if (key - first) % 1000 == 0:
            el = time.time() - t0
            print(f'frame {key} ({key - first + 1}/{last - first + 1}), {n_crops} crops, '
                  f'{n_crops / max(el, 1e-6):.1f} crops/s, {el / 60:.1f} min', flush=True)

    if key != last:
        raise RuntimeError(f'Video ended at batch frame {key}, expected {last}')
    for state in states.values():
        run_hmr(hmr_model, state, device)
    print(f'HMR done: {n_crops} crops in {(time.time() - t0) / 60:.1f} min', flush=True)

    return {pid: {k: torch.cat(v) for k, v in s.results.items()} for pid, s in states.items()}


def to_camera_and_world(smpl, res, camera, device):
    """Rotate the HMR output from the virtual cameras into the real camera and the world frame."""
    out = {}
    rotmat = res['pred_rotmat'].to(device)
    shape = res['pred_shape'].to(device)
    J0 = batched(lambda b: fisheye.rest_pelvis(smpl, b), shape)
    root_c, trans_c = fisheye.transform_smpl(rotmat[:, 0], res['trans_virtual'].to(device), J0,
                                             res['virtual_R'].to(device).transpose(1, 2))
    rotmat = rotmat.clone()
    rotmat[:, 0] = root_c
    out['pred_rotmat'] = rotmat
    out['pred_shape'] = shape
    out['pred_trans'] = trans_c
    out.update(world_frame(rotmat[:, 0], trans_c, J0, camera))
    return out


def world_frame(root_c, trans_c, J0, camera):
    N = root_c.shape[0]
    if camera['R_w2c'] is None:
        nan = torch.full((N, 3), float('nan'), device=root_c.device)
        return {'world_rotmat_root': nan[:, :, None].expand(N, 3, 3), 'world_trans': nan}
    R_c2w = torch.tensor(camera['R_w2c'].T, dtype=torch.float32, device=root_c.device)
    t_c2w = -R_c2w @ torch.tensor(camera['t_w2c'], dtype=torch.float32, device=root_c.device)
    root_w, trans_w = fisheye.transform_smpl(root_c, trans_c, J0, R_c2w, t_c2w)
    return {'world_rotmat_root': root_w, 'world_trans': trans_w}


def batched(fn, x, bs=2048):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, x.shape[0], bs)])


def refine_track(args, smpl, cam_res, track, camera):
    """
    SMPLify-style fit to the ViTPose keypoints, initialised from HMR, per contiguous chunk and window.
    The camera-frame parameters are optimised and projected with the real fisheye model.
    """
    device = args.device
    K = torch.tensor(camera['K'], dtype=torch.float32, device=device)
    D = torch.tensor(camera['D'], dtype=torch.float32, device=device)
    kp2d = torch.tensor(track['kp2d'], dtype=torch.float32, device=device)
    boxes = torch.tensor(track['boxes'], dtype=torch.float32, device=device)
    box_size = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).clamp(min=1)
    coco = torch.tensor(cosilab.COCO_TO_SPIN, device=device)

    rotmat = cam_res['pred_rotmat'].clone()
    shape = cam_res['pred_shape'].clone()
    trans = cam_res['pred_trans'].clone()

    windows = []
    for s, e in track['chunks']:
        n_win = int(np.ceil((e - s) / args.refine_window))
        bounds = np.linspace(s, e, n_win + 1).round().astype(int)
        windows += list(zip(bounds[:-1], bounds[1:]))

    for s, e in windows:
        T = e - s
        init_rot = rotmat[s:e]
        init_trans = trans[s:e]
        pose6d = matrix_to_rotation_6d(init_rot).clone().requires_grad_(True)
        betas = shape[s:e].mean(0, keepdim=True).clone().requires_grad_(True)
        transl = init_trans.clone().requires_grad_(True)
        target = kp2d[s:e, :, :2]
        conf = kp2d[s:e, :, 2] * (kp2d[s:e, :, 2] > args.kp_conf)
        optim = torch.optim.Adam([pose6d, betas, transl], lr=args.refine_lr)

        with torch.enable_grad():
            for _ in range(args.refine_iters):
                R = rotation_6d_to_matrix(pose6d)
                out = smpl(global_orient=R[:, :1], body_pose=R[:, 1:], betas=betas.expand(T, -1),
                           transl=transl, pose2rot=False)
                j2d = fisheye.project_torch(out.joints[:, coco], K, D)
                res = ((j2d - target) / box_size[s:e, None, None]).pow(2).sum(-1)
                gmof = args.gmof_sigma**2 * res / (args.gmof_sigma**2 + res)
                l_kp = (conf * gmof).sum() / conf.sum().clamp(min=1)
                l_pose = (R - init_rot).pow(2).sum((-1, -2)).mean()
                l_beta = betas.pow(2).mean()
                l_depth = (transl[:, 2] - init_trans[:, 2]).pow(2).mean()
                j3d = out.joints[:, :25]
                l_smooth = (j3d[2:] - 2 * j3d[1:-1] + j3d[:-2]).pow(2).sum(-1).mean() if T > 2 else 0
                loss = (l_kp + args.w_pose * l_pose + args.w_beta * l_beta
                        + args.w_depth * l_depth + args.w_smooth * l_smooth)
                optim.zero_grad()
                loss.backward()
                optim.step()

        with torch.no_grad():
            rotmat[s:e] = rotation_6d_to_matrix(pose6d)
            shape[s:e] = betas.expand(T, -1)
            trans[s:e] = transl

    with torch.no_grad():
        J0 = batched(lambda b: fisheye.rest_pelvis(smpl, b), shape)
        out = {'pred_rotmat': rotmat, 'pred_shape': shape, 'pred_trans': trans}
        out.update(world_frame(rotmat[:, 0], trans, J0, camera))
    return out


def save_track(path, cam_res, hmr_res, track, batch):
    data = {
        'frame_ids': track['frames'],
        'session_frame_ids': cosilab.session_frame(batch, track['frames']),
        'pred_rotmat': cam_res['pred_rotmat'],
        'pred_shape': cam_res['pred_shape'],
        'pred_trans': cam_res['pred_trans'][:, None],
        'world_rotmat_root': cam_res['world_rotmat_root'],
        'world_trans': cam_res['world_trans'],
        'pred_cam': hmr_res['pred_cam'],
        'virtual_R': hmr_res['virtual_R'],
        'virtual_focal': hmr_res['virtual_focal'],
        'boxes': track['boxes'],
        'kp2d': track['kp2d'],
    }
    data = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in data.items()}
    np.savez(path, **data)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    args.cam, args.batch, args.session = cosilab.parse_cam_batch(args.cam_batch)

    camera = cosilab.load_camera(args.data_root, args.cam, args.extrinsic_c2w)
    if camera['R_w2c'] is None:
        print(f'[Warning] camera {args.cam:02d} has no extrinsics, world_* outputs are NaN')
    video = cosilab.video_path(args.data_root, args.cam)
    tracks = cosilab.load_tracks(cosilab.keypoints_path(args.data_root, args.cam_batch))

    first = min(int(t['frames'].min()) for t in tracks.values())
    start = first if args.start is None else args.start
    end = np.inf if args.num_frames is None else start + args.num_frames
    tracks = prepare_tracks(tracks, start, end, args.min_track_len)
    print(f'{args.cam_batch}: {len(tracks)} tracks, {sum(len(t["frames"]) for t in tracks.values())} '
          f'person-frames, video {video}', flush=True)

    out_dir = os.path.join(args.out_dir, args.cam_batch)
    for sub in ['hmr', 'refined']:  # never mix tracks of different runs
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    os.makedirs(os.path.join(out_dir, 'hmr'), exist_ok=True)
    with open(os.path.join(out_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(out_dir, 'camera.json'), 'w') as f:
        json.dump({
            'cam': args.cam, 'batch': args.batch, 'session': args.session,
            'K': camera['K'].tolist(), 'D': camera['D'].tolist(),
            'R_w2c': None if camera['R_w2c'] is None else camera['R_w2c'].tolist(),
            't_w2c': None if camera['t_w2c'] is None else camera['t_w2c'].tolist(),
            'extrinsic_c2w': args.extrinsic_c2w,
            'video': video, 'data_root': args.data_root,
            'session_frame_offset': cosilab.session_frame(args.batch, 0),
            'fps': cosilab.FPS,
            'note': 'pred_* are in the camera frame (OpenCV axes), world_* in the session world frame '
                    '(camera 06 for session 1, camera 01 for session 2). X_c = R_w2c X_w + t_w2c.',
        }, f, indent=2)

    hmr_model = get_hmr_vimo(checkpoint=args.checkpoint, device=args.device)
    smpl = SMPL().to(args.device)

    hmr_results = run_tracks(args, hmr_model, tracks, camera, video, start)
    del hmr_model
    torch.cuda.empty_cache()

    if args.refine:
        os.makedirs(os.path.join(out_dir, 'refined'), exist_ok=True)
    for pid, track in tracks.items():
        cam_res = to_camera_and_world(smpl, hmr_results[pid], camera, args.device)
        save_track(os.path.join(out_dir, 'hmr', f'track_{pid}.npz'), cam_res, hmr_results[pid], track, args.batch)
        if args.refine:
            t0 = time.time()
            ref_res = refine_track(args, smpl, cam_res, track, camera)
            save_track(os.path.join(out_dir, 'refined', f'track_{pid}.npz'), ref_res, hmr_results[pid], track, args.batch)
            print(f'refined track {pid} ({len(track["frames"])} frames) in {time.time() - t0:.0f} s', flush=True)
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    main(parse_args())
