"""
Rotation-only refit of a side camera: the camera centre stays where the Calibrator boards put it (stored extrinsics),
only the 3 rotation parameters are fitted to the people (validated people of several clips, Hungarian-matched joints of
clearly visible people, IRLS weights, tightening gate). Motivation: cam14 shows the same ~30 px image shift against the
people in batch01, 02 and 03 (shake_offsets.py --batch), cam12 ~15 px in batch03, i.e. a small, constant rotation
error rather than a wrong camera position. Reports the board error, the people score per clip, and the rotation.
Usage: rot_fit.py cam out.json --pids_json validated_people.json --clips 1:6000 2:6000 3:6000
"""
import argparse, json, sys, types
import numpy as np, cv2
from scipy.optimize import least_squares
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/cosilab_calib_12_14')
from lib.utils import cosilab
from cosilab_refit_extrinsic import Data
from joint_fit import board_points, proj, to_rt, SCALE

E = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cam', type=int)
    ap.add_argument('out')
    ap.add_argument('--pids_json', default=None)
    ap.add_argument('--clips', nargs='+', default=['1:6000', '2:6000', '3:6000'])
    o = ap.parse_args()
    cam = o.cam
    k = cosilab.load_camera(cosilab.DATA_ROOT, cam, use_refit=False)
    K, D = k['K'], k['D']
    Xb, ub = board_points(cam)
    clips = [tuple(map(int, c.split(':'))) for c in o.clips]

    def make(cl):
        a = types.SimpleNamespace(multi_root=f'{E}/cosilab_multiview_4cam', single_root=f'{E}/cosilab_smpl', kp_conf=0.3,
                                  det_dir=f'{E}/cosilab_vit003_extra', neighbours=None, start=6000, num_frames=600,
                                  every=10, all_people=False, occ_depth=0.3, max_occ=0.2, pids=None,
                                  pids_json=o.pids_json, clips=cl)
        return Data(a, cam, cl[0][0])

    data = make(clips)
    per_clip = {c: make([c]) for c in clips}
    x0 = np.r_[k['rvec'], -k['R_w2c'].T @ k['t_w2c']]
    C = x0[3:].copy()

    def report(tag, x):
        eb = np.linalg.norm(proj(Xb, x, K, D) - ub, axis=1)
        R = cv2.Rodrigues(x[:3])[0]
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ k['R_w2c'].T)[0]))
        row = {'board_px_median': float(np.median(eb)), 'board_px_p90': float(np.percentile(eb, 90)),
               'rotated_deg': float(dR), 'rvec': x[:3].tolist(), 'tvec': to_rt(x)[1].tolist(), 'clips': {}}
        s = data.score(*to_rt(x), K, D)
        row['people'] = s
        txt = []
        for c, d in per_clip.items():
            sc = d.score(*to_rt(x), K, D)
            row['clips'][f'{c[0]}:{c[1]}'] = sc
            txt.append(f'b{c[0]} {sc["under_40px"]:.2f} ({sc["median_px"]:.0f} px, n {sc["n"]})')
        print(f'cam{cam} {tag:<10} board median {row["board_px_median"]:6.2f} px (p90 {row["board_px_p90"]:6.2f}); rotated '
              f'{dR:.2f} deg; people within 40 px {s["under_40px"]:.2f} (median {s["median_px"]:.0f} px, n {s["n"]}); '
              f'per clip: ' + ', '.join(txt), flush=True)
        return row

    rows = {'stored': report('stored', x0)}
    r = x0[:3].copy()
    for gate in [80, 60, 45, 35, 30, 30, 30]:
        x = np.r_[r, C]
        Xp, up, _ = data.match(*to_rt(x), K, D, gate)
        if Xp is None or len(Xp) < 30:
            break
        for _ in range(3):  # IRLS
            e = np.linalg.norm(proj(Xp, np.r_[r, C], K, D) - up, axis=1)
            w = np.repeat(1 / np.sqrt(1 + (e / SCALE) ** 2), 2)
            r = least_squares(lambda y: (proj(Xp, np.r_[y, C], K, D) - up).ravel() * np.sqrt(w), r,
                              x_scale=0.01, max_nfev=200).x
        print(f'  gate {gate}: {len(Xp)} matched joints', flush=True)
    rows['rotation_only'] = report('rot only', np.r_[r, C])
    json.dump(rows, open(o.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
