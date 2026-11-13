"""
Jumps at the --refine_window chunk boundaries of a multi-view fit (scripts/run_cosilab_multiview.py): per person, the
frame-to-frame step of the world translation (mm), of the root rotation and of the body joint rotations (deg), at the
chunk boundaries vs at all other frames. With independent windows (the fit before ec4a3f5) a boundary can jump; with
one optimisation over all frames, boundary steps should look like any other step.
Also the body shape (betas) spread over the person's frames (0 if one shape per person).

python scripts/cosilab_window_check.py --multi_root .../cosilab_multiview_8cam_2min --cam_batches cam01_batch01 ...
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
sys.path.insert(0, os.path.dirname(__file__))

import argparse

import numpy as np

from lib.utils import cosilab
from run_cosilab_smpl import contiguous_chunks


def angle_deg(Ra, Rb):
    """Rotation angle (deg) of Ra^T Rb, elementwise over leading dims."""
    tr = np.einsum('...ji,...ji->...', Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--multi_root', required=True)
    p.add_argument('--cam_batches', nargs='+', required=True)
    p.add_argument('--refine_window', type=int, default=1000)
    a = p.parse_args()

    bodies = {}  # pid -> {frame: (world_trans, world root, body rotmats, betas)}
    for cb in a.cam_batches:
        d = os.path.join(a.multi_root, cb)
        if not os.path.isdir(os.path.join(d, 'refined')):
            continue
        for pid, t in cosilab.load_results(d, 'refined')[1].items():
            b = bodies.setdefault(pid, {})
            for i, f in enumerate(t['session_frame_ids']):
                b[int(f)] = (t['world_trans'][i], t['world_rotmat_root'][i], t['pred_rotmat'][i, 1:], t['pred_shape'][i])

    rows, at_b, other = [], {'trans': [], 'root': [], 'body': []}, {'trans': [], 'root': [], 'body': []}
    for pid in sorted(bodies):
        fr = np.array(sorted(bodies[pid]))
        tr = np.stack([bodies[pid][f][0] for f in fr])
        root = np.stack([bodies[pid][f][1] for f in fr])
        body = np.stack([bodies[pid][f][2] for f in fr])
        betas = np.stack([bodies[pid][f][3] for f in fr])
        boundary = np.zeros(len(fr) - 1, dtype=bool)  # step i is between frames i and i+1
        for s, e in contiguous_chunks(fr, 1):
            n = int(np.ceil((e - s) / a.refine_window))
            for b in np.linspace(s, e, n + 1).round().astype(int)[1:-1]:
                boundary[b - 1] = True
        gap = np.diff(fr) != 1
        steps = {'trans': np.linalg.norm(np.diff(tr, axis=0), axis=1) * 1000,
                 'root': angle_deg(root[:-1], root[1:]),
                 'body': angle_deg(body[:-1], body[1:]).mean(1)}
        for k, v in steps.items():
            at_b[k].append(v[boundary & ~gap])
            other[k].append(v[~boundary & ~gap])
        if boundary.any():
            rows.append((pid, len(fr), int(boundary.sum()),
                         *[(steps[k][boundary & ~gap].max(), np.percentile(steps[k][~boundary & ~gap], 99)) for k in steps],
                         float(np.abs(betas - betas[0]).max())))

    print(f'{a.multi_root}: {len(bodies)} people, chunk boundaries every <= {a.refine_window} frames')
    print('per person: boundaries; max step at a boundary vs p99 of all other steps (translation mm, root deg, '
          'mean body joint deg); max betas change over the frames')
    for pid, n, nb, t, r, bd, db in rows:
        print(f'  {pid:3d}: {n} frames, {nb} boundaries; trans {t[0]:6.1f} vs {t[1]:6.1f}; root {r[0]:5.2f} vs {r[1]:5.2f}; '
              f'body {bd[0]:5.2f} vs {bd[1]:5.2f}; betas {db:.2e}')
    print('all people: median / p99 / max step')
    for k, u in (('trans', 'mm'), ('root', 'deg'), ('body', 'deg')):
        b, o = np.concatenate(at_b[k]), np.concatenate(other[k])
        print(f'  {k:<5} at boundaries (n {len(b)}): {np.median(b):.2f} / {np.percentile(b, 99):.2f} / {b.max():.2f} {u};'
              f'  elsewhere (n {len(o)}): {np.median(o):.2f} / {np.percentile(o, 99):.2f} / {o.max():.2f} {u}')


if __name__ == '__main__':
    main()
