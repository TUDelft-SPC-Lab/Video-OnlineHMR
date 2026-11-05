"""
Per person: how well the projected 4-camera body matches the detected keypoints in a side camera (stored extrinsics,
clearly visible, Hungarian per frame), against where the person stands (world x, y) and how many top cameras saw them.
If only the people at one end of the room fail, the bodies there are the problem, not the side-camera calibration.
Usage: python scripts/cosilab_calib_12_14/per_person_error.py [--batch 1] [--multi_root ...] [--valid_json out.json] 12 14 13 15
--valid_json: people whose body fits a control camera (--controls, default 13 15) with >= 0.6 within 40 px over >= 20
person-frames, for cosilab_refit_extrinsic.py --pids_json. Use 4-camera bodies, which never saw the side cameras.
"""
import argparse, json, os, sys, types
from collections import defaultdict
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from lib.models.smpl import SMPL
from lib.utils import cosilab
from cosilab_project_boxes import collect_bodies
from cosilab_refit_extrinsic import project, clearly_visible

E = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cams', type=int, nargs='+')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--start', type=int, default=6000)
    ap.add_argument('--multi_root', default=f'{E}/cosilab_multiview_4cam')
    ap.add_argument('--valid_json', default=None)
    ap.add_argument('--controls', type=int, nargs='*', default=[13, 15])
    args = ap.parse_args()
    cams, batch = args.cams, args.batch
    a = types.SimpleNamespace(multi_root=args.multi_root, single_root=f'{E}/cosilab_smpl', kp_conf=0.3)
    best = collect_bodies(a, (1, 2, 3, 4), batch, args.start, args.start + 600)
    valid = set()
    keys = [k for k in sorted(best) if k[0] % 5 == 0]
    rows = [best[k] for k in keys]
    with torch.no_grad():
        J = SMPL()(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for *_, d, i in rows]))[:, None],
                   body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for *_, d, i in rows]))[:, 1:],
                   betas=torch.tensor(np.stack([d['pred_shape'][i] for *_, d, i in rows])),
                   transl=torch.tensor(np.stack([d['world_trans'][i] for *_, d, i in rows])),
                   pose2rot=False).joints[:, cosilab.CONFLAB_TO_SPIN].numpy().astype(float)
    fr = np.array([k[0] for k in keys])
    pid = np.array([k[1] for k in keys])
    # top cameras with keypoints of the person (from the HMR results' tracks)
    seen_by = defaultdict(set)
    for c in (1, 2, 3, 4):
        _, tracks = cosilab.load_results(os.path.join(E, 'cosilab_smpl', f'cam{c:02d}_batch{batch:02d}'), 'hmr')
        for p in tracks:
            seen_by[p].add(c)
    for cam in cams:
        k = cosilab.load_camera(cosilab.DATA_ROOT, cam)
        C = -k['R_w2c'].T @ k['t_w2c']
        uv, front, dist = project(J.reshape(-1, 3), k['rvec'], k['t_w2c'], k['K'], k['D'])
        uv, front, dist = uv.reshape(-1, 17, 2), front.reshape(-1, 17), dist.reshape(-1, 17)
        det = np.load(f'{E}/cosilab_vit003_extra/cam{cam}_batch{batch:02d}/detections.npz')
        m = det['kind'] == 'detector'
        per = defaultdict(list)
        for f in np.unique(fr):
            s = np.nonzero(fr == f)[0]
            vis = clearly_visible(uv[s], front[s], dist[s], 0.3, 0.2)
            s = s[vis]
            kp = det['kp'][m & (det['frame'] == f)]
            if not len(s) or not len(kp):
                continue
            ok = front[s][:, None] & (kp[None, :, :, 2] > 0.3)
            e = np.linalg.norm(uv[s][:, None] - kp[None, :, :, :2], axis=-1)
            with np.errstate(all='ignore'):
                M = np.nanmedian(np.where(ok, e, np.nan), -1)
            M = np.where(ok.sum(-1) >= 6, M, 1e4)
            for r, c in zip(*linear_sum_assignment(M)):
                per[int(pid[s[r]])].append(M[r, c])
        print(f'\ncam{cam} (centre x {C[0]:+.1f}, y {C[1]:+.1f}): person, position (world x, y), distance, top cameras, '
              f'share within 40 px, median px', flush=True)
        out = []
        for p, e in per.items():
            e = np.array(e)
            pos = J[pid == p][:, 2].mean(0)  # neck
            out.append((pos[1], p, pos, np.linalg.norm(pos[:2] - C[:2]), (e < 40).mean(), np.median(e), len(e)))
        for y, p, pos, dd, u40, med, n in sorted(out):
            if cam in args.controls and u40 >= 0.6 and n >= 20:
                valid.add(p)
            print(f'  {p:3d}: ({pos[0]:+.1f}, {pos[1]:+.1f}) {dd:4.1f} m  cams {sorted(seen_by[p])}  '
                  f'{u40:.2f}  {med:6.0f} px  (n {n})', flush=True)

    if args.valid_json:
        d = json.load(open(args.valid_json)) if os.path.exists(args.valid_json) else {}
        d[str(batch)] = sorted(int(p) for p in valid)
        json.dump(d, open(args.valid_json, 'w'), indent=1)
        print(f'\nbatch{batch:02d}: {len(valid)} validated people {sorted(valid)} -> {args.valid_json}')


if __name__ == '__main__':
    main()
