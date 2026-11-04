"""
Person boxes and VIT003 keypoints for a COSI-Lab camera without dataset keypoints (e.g. cam02, cam04).
Two kinds of box per frame, in one pass over the (raw) video:
  detector:  ViTDet-H (detectron2, as in scripts/run_custom.py), class person, score >= --det_score
  projected: the boxes of scripts/cosilab_project_boxes.py (neighbouring cameras' bodies projected into this one)
VIT003 (scripts/run_cosilab_vitpose.py) runs on both. No IDs are final here: scripts/cosilab_assign_ids.py
gives the detections person IDs on the CPU.

python scripts/run_cosilab_detect_vitpose.py --cam_batch cam02_batch01 --start 6000 --num_frames 600
Output: <out_dir>/<cam_batch>/detections.npz with per box frame, box (xyxy, full-res), kind (detector / projected),
score (detector, else nan), pid (projected, else -1), source (projected), kp (17, 3) in full-res pixels.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import time

import numpy as np
import torch

from lib.utils import cosilab
from run_cosilab_vitpose import VIT003, load_model, predict

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
DETECTOR_CFG = './data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
DETECTOR_WEIGHTS = ('https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/'
                    'model_final_f05665.pkl')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='e.g. cam02_batch01')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--checkpoint', type=str, default=VIT003)
    parser.add_argument('--out_dir', type=str, default=f'{EXP_DIR}/cosilab_vit003_extra',
                        help='holds <cam_batch>/proposals.npz; detections.npz is written next to it')
    parser.add_argument('--start', type=int, required=True, help='first batch frame')
    parser.add_argument('--num_frames', type=int, required=True)
    parser.add_argument('--det_score', type=float, default=0.5)
    parser.add_argument('--batch', type=int, default=64, help='ViTPose crops per forward (before flipping)')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args(argv)


def init_detector():
    from detectron2.config import LazyConfig
    from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
    cfg = LazyConfig.load(DETECTOR_CFG)
    cfg.train.init_checkpoint = DETECTOR_WEIGHTS
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(cfg)
    detector.model.eval()
    return detector


def iou_matrix(a, b):
    """IoU of xyxy boxes (N, 4) x (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clip(min=1e-9)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    cam, batch, _ = cosilab.parse_cam_batch(args.cam_batch)
    out_dir = os.path.join(args.out_dir, args.cam_batch)
    prop = dict(np.load(os.path.join(out_dir, 'proposals.npz')))
    start, end = args.start, args.start + args.num_frames
    print(f'{args.cam_batch}: frames {start}-{end - 1}, {len(prop["frame"])} projected boxes, '
          f'video {cosilab.video_source(args.data_root, cam)}', flush=True)

    detector = init_detector()
    model = load_model(args.checkpoint, args.device)
    reader = cosilab.open_video(args.data_root, cam, cosilab.session_frame(batch, start), args.num_frames)
    if isinstance(reader, cosilab.RawVideoReader):
        print(f'raw frames: {[s[:3] for s in reader.segments()]}, max |dt| {np.abs(reader.dt).max():.4f} s, '
              f'repeated {int((np.diff(reader.raw_frame) == 0).sum())}, skipped {int((np.diff(reader.raw_frame) == 2).sum())}',
              flush=True)

    out = {k: [] for k in ['frame', 'box', 'kind', 'score', 'pid', 'source', 'kp']}
    stats = []
    t0 = time.time()
    key = start - 1
    for key, frame in zip(range(start, end), reader):
        inst = detector(frame)['instances']  # BGR, like run_custom.py
        keep = (inst.pred_classes == 0) & (inst.scores >= args.det_score)
        det = inst.pred_boxes.tensor[keep].cpu().numpy().astype(np.float64)
        score = inst.scores[keep].cpu().numpy()
        pm = prop['frame'] == key
        proj = prop['box'][pm].astype(np.float64)

        boxes = np.concatenate([det, proj]).reshape(-1, 4)
        if len(boxes):
            img = torch.from_numpy(frame).to(args.device).permute(2, 0, 1).flip(0).float() / 255  # BGR -> RGB
            xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
            kp = np.concatenate([predict(model, img, xywh[i:i + args.batch], cosilab.CONFLAB_FLIP_PAIRS, args.device)
                                 for i in range(0, len(boxes), args.batch)])
            n = len(det)
            out['frame'] += [key] * len(boxes)
            out['box'].append(boxes)
            out['kind'] += ['detector'] * n + ['projected'] * len(proj)
            out['score'] += score.tolist() + [np.nan] * len(proj)
            out['pid'] += [-1] * n + prop['pid'][pm].tolist()
            out['source'] += [''] * n + prop['source'][pm].tolist()
            out['kp'].append(kp)
        iou = iou_matrix(proj, det)
        stats.append((len(det), len(proj), int((iou.max(1) >= 0.5).sum()) if iou.size else 0))
        if (key - start) % 100 == 0:
            el = time.time() - t0
            print(f'frame {key} ({key - start + 1}/{args.num_frames}): {len(det)} detections, {len(proj)} projected, '
                  f'{el / 60:.1f} min', flush=True)
    if key != end - 1:
        raise RuntimeError(f'Video ended at batch frame {key}, expected {end - 1}')
    print(f'done in {(time.time() - t0) / 60:.1f} min', flush=True)

    out['box'] = np.concatenate(out['box']).astype(np.float32)
    out['kp'] = np.concatenate(out['kp']).astype(np.float32)
    out = {k: np.asarray(v) for k, v in out.items()}
    if not np.isfinite(out['kp']).all():
        raise RuntimeError('Non-finite keypoints')

    s = np.array(stats)
    print(f'\ndetector boxes per frame: mean {s[:, 0].mean():.1f}, min {s[:, 0].min()}, max {s[:, 0].max()}; '
          f'people projected from the neighbours: mean {s[:, 1].mean():.1f}, min {s[:, 1].min()}, max {s[:, 1].max()}')
    print(f'projected boxes with a detector box at IoU >= 0.5: {s[:, 2].sum()} of {s[:, 1].sum()} '
          f'({s[:, 2].sum() / max(s[:, 1].sum(), 1):.3f})')
    for kind in ['detector', 'projected']:
        conf = out['kp'][out['kind'] == kind][..., 2]
        if len(conf):
            print(f'VIT003 confidence on {kind} boxes ({len(conf)}): mean {conf.mean():.3f}, '
                  f'fraction > 0.3 {(conf > 0.3).mean():.3f}, joints > 0.3 per box (median) {np.median((conf > 0.3).sum(1)):.0f}')

    path = os.path.join(out_dir, 'detections.npz')
    np.savez(path, start=start, num_frames=args.num_frames, **out)
    with open(os.path.join(out_dir, 'detect_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'Saved to {path}')


if __name__ == '__main__':
    main(parse_args())
