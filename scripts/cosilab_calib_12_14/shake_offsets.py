"""
Is the camera shaking? Per frame, the 2D shift that best aligns the projected 4-camera bodies (clearly visible people,
stored extrinsics) with the detected VIT003 keypoints: a vote over all same-joint displacements within --radius px
(2D histogram peak, then the median of the displacements near the peak). A shaking camera gives a shift that changes
over time, common to all people of a frame; a still, well-calibrated one a small constant shift.
Usage (repo root, container): python scripts/cosilab_calib_12_14/shake_offsets.py out.npz 13 15 12 14 [--batch 2]
"""
import sys, types
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data

E = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
RADIUS, BIN, NEAR = 300.0, 8.0, 15.0


def frame_shift(u, kp):
    """u (P, 17, 2) projected, kp (B, 17, 3) detected -> (shift (2,), support, aligned share within 40 px)."""
    d = []
    for j in range(17):
        ok = kp[:, j, 2] > 0.3
        if ok.any():
            dj = (kp[None, ok, j, :2] - u[:, None, j]).reshape(-1, 2)
            d.append(dj[np.linalg.norm(dj, axis=1) < RADIUS])
    d = np.concatenate(d) if d else np.zeros((0, 2))
    if len(d) < 20:
        return None
    edges = np.arange(-RADIUS, RADIUS + BIN, BIN)
    H, _, _ = np.histogram2d(d[:, 0], d[:, 1], bins=[edges, edges])
    H = H + np.roll(H, 1, 0) + np.roll(H, -1, 0)  # light smoothing
    H = H + np.roll(H, 1, 1) + np.roll(H, -1, 1)
    i, k = np.unravel_index(np.argmax(H), H.shape)
    peak = np.array([edges[i] + BIN / 2, edges[k] + BIN / 2])
    near = d[np.linalg.norm(d - peak, axis=1) < NEAR * 2]
    shift = np.median(near, 0) if len(near) else peak
    support = int((np.linalg.norm(d - shift, axis=1) < NEAR).sum())
    # share of visible people within 40 px of a detection once shifted (median joint distance, best box)
    e = np.linalg.norm((u + shift)[:, None] - kp[None, :, :, :2], axis=-1)
    ok = kp[None, :, :, 2] > 0.3
    with np.errstate(all='ignore'):
        M = np.nanmedian(np.where(ok, e, np.nan), -1)
    M = np.where(ok.sum(-1) >= 6, M, np.inf).min(1)
    M0 = np.nanmedian(np.where(ok, np.linalg.norm(u[:, None] - kp[None, :, :, :2], axis=-1), np.nan), -1)
    M0 = np.where(ok.sum(-1) >= 6, M0, np.inf).min(1)
    return shift, support, float((M < 40).mean()), float((M0 < 40).mean())


def main():
    batch = 1
    if '--batch' in sys.argv:
        i = sys.argv.index('--batch'); batch = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
    out, cams = sys.argv[1], [int(c) for c in sys.argv[2:]]
    res = {}
    for cam in cams:
        a = types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam', single_root=f'{E}/cosilab_smpl', kp_conf=0.3,
                                  det_dir=f'{E}/cosilab_vit003_extra', neighbours=None, start=6000, num_frames=600, every=1,
                                  all_people=False, occ_depth=0.3, max_occ=0.2)
        data = Data(a, cam, batch)
        k = cosilab.load_camera(cosilab.DATA_ROOT, cam)
        frames = sorted({0})  # placeholder to keep order
        rows = []
        for fi, (J, kp, u, fr) in enumerate(data.per_frame(k['rvec'], k['t_w2c'], k['K'], k['D'])):
            r = frame_shift(u, kp)
            if r is not None:
                rows.append((fi, *r[0], r[1], r[2], r[3]))
        rows = np.array(rows)
        s = rows[:, 1:3]
        step = np.linalg.norm(np.diff(s, axis=0), axis=1)
        mag = np.linalg.norm(s, axis=1)
        print(f'cam{cam} batch{batch:02d}: {len(rows)} frames; shift median ({np.median(s[:, 0]):+.0f}, {np.median(s[:, 1]):+.0f}) px, '
              f'|shift| p10/50/90 {np.percentile(mag, 10):.0f}/{np.median(mag):.0f}/{np.percentile(mag, 90):.0f}; '
              f'spread around its median (p90 of distance) {np.percentile(np.linalg.norm(s - np.median(s, 0), axis=1), 90):.0f} px; '
              f'frame-to-frame step median {np.median(step):.1f}, p90 {np.percentile(step, 90):.1f} px; '
              f'support median {np.median(rows[:, 3]):.0f}; visible people within 40 px: as stored {rows[:, 5].mean():.2f}, '
              f'with the per-frame shift {rows[:, 4].mean():.2f}, with one constant shift: see below', flush=True)
        # one constant shift for the whole clip: how much does the per-frame part add?
        c = np.median(s, 0)
        cst = []
        for fi, (J, kp, u, fr) in enumerate(data.per_frame(k['rvec'], k['t_w2c'], k['K'], k['D'])):
            ok = kp[None, :, :, 2] > 0.3
            with np.errstate(all='ignore'):
                M = np.nanmedian(np.where(ok, np.linalg.norm((u + c)[:, None] - kp[None, :, :, :2], axis=-1), np.nan), -1)
            cst.append((np.where(ok.sum(-1) >= 6, M, np.inf).min(1) < 40).mean())
        print(f'cam{cam} batch{batch:02d}:   with one constant shift {np.mean(cst):.2f}; shift every 30 frames: ' +
              ' '.join(f'({x:+.0f},{y:+.0f})' for x, y in s[::30]), flush=True)
        res[f'cam{cam}'] = rows
    np.savez(out, **res)


if __name__ == '__main__':
    main()
