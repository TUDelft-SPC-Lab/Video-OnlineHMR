"""
Re-run top-down ViTPose on the COSI-Lab videos with a custom mmpose checkpoint, on the person boxes of
the dataset ViTPose results. The checkpoint must output the ConfLab-17 keypoints, like the dataset. The output
has the same layout as the dataset json (960x540 pixels), plus an "info" entry, so
scripts/run_cosilab_smpl.py --keypoints_root can use it directly.

The model is ViTPose-B 256x192 with the simple heatmap head, evaluated as in mmpose 0.24 with
use_udp=True, flip_test=True and DARK-UDP decoding (modulate_kernel=11), without mmpose itself.

python scripts/run_cosilab_vitpose.py --cam_batch cam01_batch01 --start 6000 --num_frames 600
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core import constants
from lib.models.vit import ViT, make_deconv
from lib.utils import cosilab

VIT003 = ('/tudelft.net/staff-umbrella/neon/experiments/VIT003/output/'
          'ViTPose_coco_plus_conflab_w_bg_256x192/best_AP_epoch_1.pth')
IMG_W, IMG_H = 192, 256  # model input
PADDING = 1.25  # mmpose box padding


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='e.g. cam01_batch01')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--checkpoint', type=str, default=VIT003)
    parser.add_argument('--out_dir', type=str,
                        default='/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_vitpose_vit003')
    parser.add_argument('--start', type=int, default=None, help='first batch frame (default: first annotated)')
    parser.add_argument('--num_frames', type=int, default=None, help='number of frames (default: all)')
    parser.add_argument('--batch', type=int, default=64, help='crops per forward (before flipping)')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args(argv)


class ViTPose(nn.Module):
    """mmpose TopDown with a ViT-B backbone and TopdownHeatmapSimpleHead; same state_dict keys."""

    def __init__(self, num_joints=17):
        super().__init__()
        self.backbone = ViT(img_size=(IMG_H, IMG_W), patch_size=16, embed_dim=768, depth=12, num_heads=12,
                            ratio=1, mlp_ratio=4, qkv_bias=True)
        self.keypoint_head = nn.Module()
        self.keypoint_head.deconv_layers = make_deconv(in_channels=768, out_channels=256, num_layers=2)
        self.keypoint_head.final_layer = nn.Conv2d(256, num_joints, kernel_size=1)

    def forward(self, x):
        return self.keypoint_head.final_layer(self.keypoint_head.deconv_layers(self.backbone(x)))


def load_model(path, device):
    model = ViTPose()
    state_dict = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def box_to_cs(boxes):
    """xywh boxes (N, 4) -> mmpose center (N, 2) and size in pixels (N, 2), at the model aspect ratio."""
    center = boxes[:, :2] + boxes[:, 2:] / 2
    w, h = boxes[:, 2], boxes[:, 3]
    aspect = IMG_W / IMG_H
    w, h = np.where(w > aspect * h, w, h * aspect), np.where(w > aspect * h, w / aspect, h)
    return center, np.stack([w, h], 1) * PADDING


def crop(img, center, size):
    """
    UDP affine crops (no rotation) of a (3, H, W) image: output pixel u maps to c - s/2 + u * s / (W_out - 1),
    like cv2.warpAffine with mmpose get_warp_matrix. Returns (N, 3, IMG_H, IMG_W).
    """
    H, W = img.shape[1:]
    u = torch.arange(IMG_W, device=img.device, dtype=torch.float32) / (IMG_W - 1)
    v = torch.arange(IMG_H, device=img.device, dtype=torch.float32) / (IMG_H - 1)
    x = center[:, 0, None] - size[:, 0, None] / 2 + u[None] * size[:, 0, None]  # (N, IMG_W)
    y = center[:, 1, None] - size[:, 1, None] / 2 + v[None] * size[:, 1, None]  # (N, IMG_H)
    grid = torch.stack([x[:, None, :].expand(-1, IMG_H, -1) / (W - 1),
                        y[:, :, None].expand(-1, -1, IMG_W) / (H - 1)], -1) * 2 - 1
    return F.grid_sample(img[None].expand(len(center), -1, -1, -1), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


def gaussian_blur(heatmaps, kernel=11):
    """cv2.GaussianBlur(hm, (k, k), 0) per heatmap: sigma from the kernel size, BORDER_REFLECT_101."""
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    r = torch.arange(kernel, device=heatmaps.device, dtype=heatmaps.dtype) - (kernel - 1) / 2
    g = torch.exp(-r**2 / (2 * sigma**2))
    g = g / g.sum()
    N, K, H, W = heatmaps.shape
    x = F.pad(heatmaps.reshape(N * K, 1, H, W), [kernel // 2] * 4, mode='reflect')
    x = F.conv2d(x, g.view(1, 1, 1, -1))
    x = F.conv2d(x, g.view(1, 1, -1, 1))
    return x.reshape(N, K, H, W)


def decode_udp(heatmaps, center, size, kernel=11):
    """
    mmpose keypoints_from_heatmaps with use_udp=True and GaussianHeatmap: argmax, then post_dark_udp
    (one Newton step on the log of the blurred heatmap), then back to image pixels.
    Returns (N, K, 2) keypoints and (N, K) confidences (the heatmap maxima).
    """
    N, K, H, W = heatmaps.shape
    maxvals, idx = heatmaps.reshape(N, K, -1).max(-1)
    px, py = idx % W, idx // W

    hm = gaussian_blur(heatmaps, kernel).clamp(0.001, 50).log()
    hm = F.pad(hm, [1, 1, 1, 1], mode='replicate')  # np.pad(mode='edge')
    x, y = px + 1, py + 1

    def at(dx, dy):
        return hm.reshape(N, K, -1).gather(-1, ((y + dy) * (W + 2) + x + dx)[..., None])[..., 0]

    i_ = at(0, 0)
    dx = 0.5 * (at(1, 0) - at(-1, 0))
    dy = 0.5 * (at(0, 1) - at(0, -1))
    dxx = at(1, 0) - 2 * i_ + at(-1, 0)
    dyy = at(0, 1) - 2 * i_ + at(0, -1)
    dxy = 0.5 * (at(1, 1) - at(1, 0) - at(0, 1) + 2 * i_ - at(-1, 0) - at(0, -1) + at(-1, -1))
    hessian = torch.stack([dxx, dxy, dxy, dyy], -1).reshape(N, K, 2, 2)
    hessian = hessian + torch.finfo(torch.float32).eps * torch.eye(2, device=hm.device)
    coords = torch.stack([px, py], -1).float() - (torch.linalg.inv(hessian)
                                                  @ torch.stack([dx, dy], -1)[..., None])[..., 0]

    scale = size[:, None] / torch.tensor([W - 1, H - 1], device=hm.device, dtype=torch.float32)
    return coords * scale + center[:, None] - size[:, None] / 2, maxvals


@torch.no_grad()
def predict(model, img, boxes, flip_pairs, device):
    """Keypoints (N, K, 3) in image pixels for xywh boxes (N, 4) of a (3, H, W) RGB image in [0, 1]."""
    center, size = box_to_cs(boxes)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    size = torch.tensor(size, dtype=torch.float32, device=device)
    mean = torch.tensor(constants.IMG_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(constants.IMG_NORM_STD, device=device).view(1, 3, 1, 1)
    x = (crop(img, center, size) - mean) / std

    heatmaps = model(torch.cat([x, x.flip(3)])).float()
    hm, hm_flip = heatmaps.chunk(2)
    perm = list(range(hm.shape[1]))
    for a, b in flip_pairs:
        perm[a], perm[b] = b, a
    hm = (hm + hm_flip[:, perm].flip(3)) / 2  # flip_test with shift_heatmap=False

    kp, conf = decode_udp(hm, center, size)
    return torch.cat([kp, conf[..., None]], -1).cpu().numpy()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    cam, batch, _ = cosilab.parse_cam_batch(args.cam_batch)
    video = cosilab.video_path(args.data_root, cam)
    src = cosilab.keypoints_path(args.data_root, args.cam_batch)
    flip_pairs = cosilab.CONFLAB_FLIP_PAIRS

    with open(src) as f:
        ann = json.load(f)['annotations']
    keys = sorted(map(int, ann))
    start = keys[0] if args.start is None else args.start
    end = keys[-1] + 1 if args.num_frames is None else start + args.num_frames
    # the same person-frames as cosilab.load_tracks: boxes that also have keypoints
    frames = {k: {pid: box for pid, box in ann[str(k)]['bbox'].items() if pid in ann[str(k)]['keypoints']}
              for k in keys if start <= k < end}
    frames = {k: v for k, v in frames.items() if v}
    del ann
    first, last = min(frames), max(frames)
    print(f'{args.cam_batch}: {len(frames)} frames, {sum(map(len, frames.values()))} boxes, '
          f'video {video}, checkpoint {args.checkpoint}', flush=True)

    model = load_model(args.checkpoint, args.device)
    reader = cosilab.VideoReader(video, cosilab.session_frame(batch, first), last - first + 1)

    out = {}
    n_crops, t0 = 0, time.time()
    key = first - 1
    for key, frame in zip(range(first, last + 1), reader):
        boxes = frames.get(key)
        if not boxes:
            continue
        img = torch.from_numpy(frame).to(args.device).permute(2, 0, 1).flip(0).float() / 255  # BGR -> RGB
        pids = list(boxes)
        xywh = np.array([boxes[p] for p in pids], dtype=np.float64) * cosilab.KP_SCALE
        kp = np.concatenate([predict(model, img, xywh[i:i + args.batch], flip_pairs, args.device)
                             for i in range(0, len(pids), args.batch)])
        kp[..., :2] /= cosilab.KP_SCALE
        out[str(key)] = {'bbox': boxes, 'keypoints': {p: kp[j].tolist() for j, p in enumerate(pids)}}

        n_crops += len(pids)
        if (key - first) % 1000 == 0:
            el = time.time() - t0
            print(f'frame {key} ({key - first + 1}/{last - first + 1}), {n_crops} crops, '
                  f'{n_crops / max(el, 1e-6):.1f} crops/s, {el / 60:.1f} min', flush=True)
    if key != last:
        raise RuntimeError(f'Video ended at batch frame {key}, expected {last}')
    print(f'ViTPose done: {n_crops} crops in {(time.time() - t0) / 60:.1f} min', flush=True)

    conf = np.array([kp for v in out.values() for kp in v['keypoints'].values()])[..., 2]
    if not np.isfinite(conf).all():
        raise RuntimeError('Non-finite keypoints')
    print('confidence per keypoint (mean, fraction > 0.3):')
    for j in range(conf.shape[1]):
        print(f'  {j:2d}: {conf[:, j].mean():.3f}  {(conf[:, j] > 0.3).mean():.3f}')

    out_path = cosilab.keypoints_path(args.data_root, args.cam_batch, args.out_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'info': {'checkpoint': args.checkpoint,
                            'boxes_from': src, 'start': start, 'num_frames': args.num_frames},
                   'annotations': out}, f)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main(parse_args())
