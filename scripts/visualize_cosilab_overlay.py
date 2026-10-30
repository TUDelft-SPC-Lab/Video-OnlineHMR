"""
Overlay the SMPL results of scripts/run_cosilab_smpl.py on the original fisheye video.
The meshes are projected with the real fisheye model, so no undistortion is needed.
SMPL joints are drawn in the track colour, the input ViTPose keypoints in white.

python scripts/visualize_cosilab_overlay.py \
    --result_dir /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl/cam01_batch01 \
    --which hmr --start 6000 --num_frames 300 --draw_mesh
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye


def draw_skeleton(img, kp, color, skeleton, conf_thresh=None, radius=4, thickness=2):
    """kp: (17, 2) or (17, 3) keypoints, skeleton: list of index pairs into them."""
    valid = np.ones(len(kp), dtype=bool) if conf_thresh is None else kp[:, 2] > conf_thresh
    pts = kp[:, :2].astype(int)
    for a, b in skeleton:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for p, v in zip(pts, valid):
        if v:
            cv2.circle(img, tuple(p), radius, color, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Overlay COSI-Lab SMPL results on the fisheye video')
    parser.add_argument('--result_dir', type=str, required=True, help='<out_dir>/camXX_batchYY')
    parser.add_argument('--which', type=str, default='hmr', choices=['hmr', 'refined'])
    parser.add_argument('--start', type=int, default=None, help='first batch frame (default: first result)')
    parser.add_argument('--num_frames', type=int, default=300)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--draw_mesh', action='store_true', help='draw filled mesh silhouettes')
    parser.add_argument('--no_kp', action='store_true', help='do not draw the input ViTPose keypoints')
    parser.add_argument('--kp_conf', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.5, help='mesh opacity')
    parser.add_argument('--resize', type=float, default=0.5, help='output scale')
    parser.add_argument('--video', type=str, default=None, help='default: the video in camera.json')
    parser.add_argument('--out', type=str, default=None,
                        help='default: <result_dir>/overlay_<which>_<start>.mp4')
    args = parser.parse_args()

    camera, tracks = cosilab.load_results(args.result_dir, args.which)
    index = cosilab.frame_index(tracks)
    K = torch.tensor(camera['K'], dtype=torch.float32)
    D = torch.tensor(camera['D'], dtype=torch.float32)
    start = min(index) if args.start is None else args.start
    video = args.video or camera['video']
    out_path = args.out or os.path.join(args.result_dir, f'overlay_{args.which}_{start}.mp4')

    smpl = SMPL()
    faces = np.asarray(smpl.faces, dtype=np.int64)
    kp_to_spin, skeleton = cosilab.CONFLAB_TO_SPIN, cosilab.CONFLAB_SKELETON

    reader = cosilab.VideoReader(video, camera['session_frame_offset'] + start, args.num_frames)
    out_size = (int(reader.width * args.resize), int(reader.height * args.resize))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cosilab.FPS / args.stride, out_size)

    for n, frame in enumerate(tqdm(reader, total=args.num_frames)):
        if n % args.stride:
            continue
        key = start + n
        img = frame.copy()
        entries = index.get(key, [])

        if entries:
            rotmat = torch.tensor(np.stack([tracks[p]['pred_rotmat'][i] for p, i in entries]))
            betas = torch.tensor(np.stack([tracks[p]['pred_shape'][i] for p, i in entries]))
            transl = torch.tensor(np.stack([tracks[p]['pred_trans'][i, 0] for p, i in entries]))
            with torch.no_grad():
                out = smpl(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:], betas=betas,
                           transl=transl, pose2rot=False)
                verts2d = fisheye.project_torch(out.vertices, K, D).numpy()
                joints2d = fisheye.project_torch(out.joints[:, kp_to_spin], K, D).numpy()
            depth = out.vertices[..., 2].numpy()

            if args.draw_mesh:
                overlay = img.copy()
                for j in np.argsort(-depth.mean(1)):  # far to near, so near people cover far ones
                    if (depth[j] <= 0).any():
                        continue
                    pid = entries[j][0]
                    polys = np.round(verts2d[j][faces]).astype(np.int32)
                    cv2.fillPoly(overlay, list(polys), cosilab.track_color(pid)[::-1], cv2.LINE_AA)
                img = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)

            for j, (pid, i) in enumerate(entries):
                color = cosilab.track_color(pid)[::-1]
                if not args.no_kp:
                    draw_skeleton(img, tracks[pid]['kp2d'][i], (255, 255, 255), skeleton, args.kp_conf,
                                  radius=3, thickness=1)
                draw_skeleton(img, joints2d[j], color, skeleton)
                x1, y1 = tracks[pid]['boxes'][i][:2].astype(int)
                cv2.putText(img, str(pid), (x1, max(y1 - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(img, f'{args.which} frame {key}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        writer.write(cv2.resize(img, out_size, interpolation=cv2.INTER_AREA))

    writer.release()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
