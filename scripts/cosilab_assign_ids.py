"""
Person IDs for the boxes of scripts/run_cosilab_detect_vitpose.py in a camera without dataset keypoints, from the
keypoints of its neighbouring cameras (which carry the dataset person IDs). CPU only.

Cost of a box vs person p in frame f: per neighbour that sees p in f, the median ray gap (m, scripts/cosilab_multiview_calib.py)
between the box's VIT003 keypoint rays and the neighbour's keypoint rays of p, over joints with conf > --kp_conf in both
(at least --min_joints); the mean over those neighbours. Joints whose rays only meet behind a camera count as a 1 m gap. It doesn't depend on any HMR depth.
  1. detector boxes are linked into tracklets (greedy IoU between consecutive frames); a tracklet's cost for p is the
     median over its frames. Per frame, Hungarian over the tracklets present, gated at --max_gap and at --best_margin over
     the tracklet's best ID (so duplicate boxes don't take a second-best ID). Unmatched boxes are dropped.
  2. fill-in: a (frame, person) a neighbour sees but no detector box got takes that person's projected box
     (scripts/cosilab_project_boxes.py) if its own keypoints pass the same gate, unless it overlaps (IoU >= 0.5) a detector box
     assigned to someone else.
  3. runs shorter than --min_track_len frames are dropped.
Then, as an ID check that doesn't use the new view's fit, the 2-camera multi-view bodies are projected into this camera and
compared with the assigned keypoints of the same ID.

python scripts/cosilab_assign_ids.py --cam_batch cam02_batch01
Output in <out_dir>/<cam_batch>/: vitpose_keypoints.json (dataset layout, 960x540) for run_cosilab_smpl.py --keypoints_root,
and id_assignment.json.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from lib.models.smpl import SMPL
from lib.utils import cosilab, fisheye
from cosilab_multiview_calib import load_view, ray_gap, triangulate
from run_cosilab_detect_vitpose import iou_matrix

EXP_DIR = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR'
IMG_W, IMG_H = 1920, 1080
BEHIND_GAP = 1.0  # m, the gap of joints whose rays meet behind either camera


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_batch', type=str, required=True, help='e.g. cam02_batch01')
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--hmr_root', type=str, default=f'{EXP_DIR}/cosilab_smpl', help='neighbour results (keypoints)')
    parser.add_argument('--multi_root', type=str, default=f'{EXP_DIR}/cosilab_multiview', help='for the ID check')
    parser.add_argument('--out_dir', type=str, default=f'{EXP_DIR}/cosilab_vit003_extra')
    parser.add_argument('--kp_conf', type=float, default=0.3)
    parser.add_argument('--min_joints', type=int, default=3)
    parser.add_argument('--max_gap', type=float, default=0.15, help='gate on the ray-gap cost (m)')
    parser.add_argument('--w_body', type=float, default=0.1,
                        help='score = ray gap + w_body * body distance (m), for ranking IDs')
    parser.add_argument('--max_body_dist', type=float, default=1.0,
                        help='gate on the body distance (m): triangulated joints vs the neighbour\'s 3D body')
    parser.add_argument('--max_body_dist_mv', type=float, default=0.5,
                        help='the same gate for people with a multi-view body')
    parser.add_argument('--best_margin', type=float, default=0.03,
                        help='a tracklet only takes IDs within this score of its best ID; duplicate boxes of a person '
                             'then get no ID instead of their second-best one')
    parser.add_argument('--link_iou', type=float, default=0.3, help='IoU linking detector boxes of consecutive frames')
    parser.add_argument('--overlap_iou', type=float, default=0.5,
                        help='a fill-in box is skipped if it overlaps a detector box of another person this much')
    parser.add_argument('--min_track_len', type=int, default=3)
    return parser.parse_args(argv)


def world_rays(kp, K, D, R):
    """Keypoints (N, 17, 3) -> world rays (N, 17, 3) from the camera centre."""
    N = kp.shape[0]
    return fisheye.unproject(np.ascontiguousarray(kp[..., :2]).reshape(-1, 2), K, D).reshape(N, 17, 3) @ R


def neighbour_index(view):
    """{session frame: (pids (P,), kp (P, 17, 3), rays (P, 17, 3))} of one neighbour."""
    per = defaultdict(list)
    for pid, t in view['tracks'].items():
        for i, f in enumerate(t['frames']):
            per[int(f)].append((pid, i))
    out = {}
    for f, rows in per.items():
        tr = view['tracks']
        out[f] = (np.array([p for p, _ in rows]), np.stack([tr[p]['kp'][i] for p, i in rows]),
                  np.stack([tr[p]['rays'][i] for p, i in rows]))
    return out


def multiview_pids(args):
    """People with a multi-view body under --multi_root."""
    pids = set()
    for d in sorted(os.listdir(args.multi_root)) if os.path.isdir(args.multi_root) else []:
        if os.path.isdir(os.path.join(args.multi_root, d, 'refined')):
            pids |= {p for p in cosilab.load_results(os.path.join(args.multi_root, d), 'refined')[1] if p >= 0}
    return pids


def neighbour_bodies(args, nb_name, smpl):
    """
    {session frame: {pid: ConfLab joints (17, 3)}} of a neighbour's people in the world frame: the multi-view body
    (--multi_root) where one exists, otherwise the neighbour's single-view refined body (--hmr_root/<nb>/refined).
    """
    multi = {}
    for d in sorted(os.listdir(args.multi_root)) if os.path.isdir(args.multi_root) else []:
        if os.path.isdir(os.path.join(args.multi_root, d, 'refined')):
            for pid, t in cosilab.load_results(os.path.join(args.multi_root, d), 'refined')[1].items():
                multi.setdefault(pid, []).append(t)
    single = cosilab.load_results(os.path.join(args.hmr_root, nb_name), 'refined')[1]
    out = defaultdict(dict)
    for pid in set(single) | set(multi):
        if pid < 0:
            continue
        for t in multi[pid] if pid in multi else [single[pid]]:
            with torch.no_grad():
                J = smpl(global_orient=torch.tensor(t['world_rotmat_root'])[:, None],
                         body_pose=torch.tensor(t['pred_rotmat'])[:, 1:], betas=torch.tensor(t['pred_shape']),
                         transl=torch.tensor(t['world_trans']), pose2rot=False).joints[:, cosilab.CONFLAB_TO_SPIN].numpy()
            for f, j in zip(t['session_frame_ids'], J):
                out[int(f)][pid] = j
    return out


def box_costs(kp, rays, C, neighbours, sframe, args):
    """
    Costs (N, P) of N boxes of one frame vs every person seen by a neighbour in that frame, nan without evidence.
    Also the body distance (N, P): the median distance (m) of the triangulated joints to the neighbour's 3D body of
    the person (neighbour_bodies), mean over the neighbours, nan without a body. It separates people that the ray gap
    can't, those in one plane through the (collinear) camera centres.
    Returns (pids (P,), cost, n_joints (N, P), body distance).
    """
    seen = sorted({int(p) for nb in neighbours if sframe in nb['index'] for p in nb['index'][sframe][0]})
    col = {p: j for j, p in enumerate(seen)}
    total = np.zeros((len(kp), len(seen)))
    count = np.zeros((len(kp), len(seen)))
    joints = np.zeros((len(kp), len(seen)), dtype=int)
    btotal = np.zeros((len(kp), len(seen)))
    bcount = np.zeros((len(kp), len(seen)))
    for nb in neighbours:
        if sframe not in nb['index']:
            continue
        pids, nkp, nrays = nb['index'][sframe]
        valid = (kp[:, None, :, 2] > args.kp_conf) & (nkp[None, :, :, 2] > args.kp_conf)  # (N, P', 17)
        gap = ray_gap(rays[:, None], nrays[None], nb['C'] - C)  # (N, P', 17)
        # ray_gap is the distance between lines: rays that only meet behind a camera get BEHIND_GAP instead
        X, s, u = triangulate(C, rays[:, None], nb['C'], nrays[None])
        gap = np.where((s > 0) & (u > 0), gap, BEHIND_GAP)
        gap = np.where(valid, gap, np.nan)
        ok = valid.sum(-1) >= args.min_joints
        body = nb['bodies'].get(sframe, {})
        J = np.stack([body.get(int(p), np.full((17, 3), np.nan)) for p in pids])  # (P', 17, 3)
        bd = np.where(valid, np.linalg.norm(X - J[None], axis=-1), np.nan)
        with warnings.catch_warnings():  # all-nan slices: fewer than min_joints, masked by ok
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(np.where(ok[..., None], gap, np.nan), -1)
            bmed = np.nanmedian(np.where(ok[..., None], bd, np.nan), -1)
        for jj, p in enumerate(pids):
            c = col[int(p)]
            total[:, c] += np.where(ok[:, jj], med[:, jj], 0)
            count[:, c] += ok[:, jj]
            joints[:, c] = np.maximum(joints[:, c], valid[:, jj].sum(-1))
            hb = ok[:, jj] & np.isfinite(bmed[:, jj])
            btotal[:, c] += np.where(hb, bmed[:, jj], 0)
            bcount[:, c] += hb
    with np.errstate(all='ignore'):
        cost = np.where(count > 0, total / count.clip(min=1), np.nan)
        bdist = np.where(bcount > 0, btotal / bcount.clip(min=1), np.nan)
    return np.array(seen, dtype=int), cost, joints, bdist


def passes(c, args):
    """
    The gate on a cost dict: ray gap <= --max_gap and body distance <= --max_body_dist (if there is a body), or
    --max_body_dist_mv for people with a multi-view body (accurate, so a tighter gate).
    """
    limit = args.max_body_dist_mv if c['mv'] else args.max_body_dist
    return c['gap'] <= args.max_gap and not c['body'] > limit


def link_tracklets(frames, boxes, link_iou):
    """Greedy IoU linking of boxes between consecutive frames. Returns a tracklet id per box."""
    tid = -np.ones(len(frames), dtype=int)
    by_frame = defaultdict(list)
    for i, f in enumerate(frames):
        by_frame[int(f)].append(i)
    nxt = 0
    prev = []
    for f in sorted(by_frame):
        cur = by_frame[f]
        if prev and f - 1 in by_frame:
            iou = iou_matrix(boxes[prev], boxes[cur])
            pairs = sorted(((iou[a, b], a, b) for a in range(len(prev)) for b in range(len(cur))
                            if iou[a, b] >= link_iou), reverse=True)
            used_a, used_b = set(), set()
            for _, a, b in pairs:
                if a in used_a or b in used_b:
                    continue
                tid[cur[b]] = tid[prev[a]]
                used_a.add(a)
                used_b.add(b)
        for i in cur:
            if tid[i] < 0:
                tid[i], nxt = nxt, nxt + 1
        prev = cur
    return tid


def runs(frames):
    """Contiguous runs (start, end) of sorted unique frames."""
    if len(frames) == 0:
        return []
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    return list(zip(np.r_[0, breaks], np.r_[breaks, len(frames)]))


def id_check(args, assigned, camera, batch, neighbours_names):
    """Project the 2-camera multi-view bodies into this camera; px against the assigned keypoints of the same ID."""
    bodies = {}
    for nb in neighbours_names:
        d = os.path.join(args.multi_root, nb, 'refined')
        if os.path.isdir(d):
            for pid, t in cosilab.load_results(os.path.dirname(d), 'refined')[1].items():
                bodies.setdefault(pid, {})
                for i, f in enumerate(t['frame_ids']):
                    bodies[pid][int(f)] = (t, i)
    if not bodies:
        print('ID check: no multi-view bodies')
        return {}
    smpl = SMPL()
    R = torch.tensor(camera['R_w2c'], dtype=torch.float32)
    t_ = torch.tensor(camera['t_w2c'], dtype=torch.float32)
    K = torch.tensor(camera['K'], dtype=torch.float32)
    D = torch.tensor(camera['D'], dtype=torch.float32)
    rows = [(f, p, kind, kp) for (f, p), (kind, kp, _) in assigned.items() if p in bodies and f in bodies[p]]
    if not rows:
        print('ID check: no assigned person-frames with a multi-view body')
        return {}
    err = []
    with torch.no_grad():
        for s in range(0, len(rows), 1024):
            ch = rows[s:s + 1024]
            src = [bodies[p][f] for f, p, _, _ in ch]
            out = smpl(global_orient=torch.tensor(np.stack([d['world_rotmat_root'][i] for d, i in src]))[:, None],
                       body_pose=torch.tensor(np.stack([d['pred_rotmat'][i] for d, i in src]))[:, 1:],
                       betas=torch.tensor(np.stack([d['pred_shape'][i] for d, i in src])),
                       transl=torch.tensor(np.stack([d['world_trans'][i] for d, i in src])), pose2rot=False)
            uv = fisheye.project_torch(out.joints[:, cosilab.CONFLAB_TO_SPIN] @ R.T + t_, K, D).numpy()
            for (f, p, kind, kp), u in zip(ch, uv):
                v = kp[:, 2] > args.kp_conf
                if v.sum() >= args.min_joints:
                    err.append((p, kind, np.median(np.linalg.norm(u[v] - kp[v, :2], axis=-1))))
    report = {}
    print(f'\nID check: 2-camera multi-view bodies projected into {args.cam_batch} vs the assigned keypoints '
          f'(median px per person-frame; correct IDs ~15-25 px, swaps far more)')
    for kind in ['detector', 'fill-in']:
        e = np.array([x for _, k, x in err if k == kind])
        if len(e):
            report[kind] = {'n': int(len(e)), 'p25': float(np.percentile(e, 25)), 'median': float(np.median(e)),
                            'p75': float(np.percentile(e, 75)), 'frac_over_60px': float((e > 60).mean())}
            print(f'  {kind:<9} n {len(e):5d}: p25 / median / p75 {np.percentile(e, 25):.1f} / {np.median(e):.1f} / '
                  f'{np.percentile(e, 75):.1f} px, > 60 px {(e > 60).mean():.3f}')
    for p in sorted({p for p, _, _ in err}):
        parts = []
        for kind in ['detector', 'fill-in']:
            e = [x for q, k, x in err if q == p and k == kind]
            parts.append(f'{kind} {np.median(e):6.1f} px ({len(e):3d})' if e else f'{kind}      -       ')
        report.setdefault('per_person', {})[int(p)] = parts
        print(f'  person {p:3d}: ' + '   '.join(parts))
    return report


def main(args):
    cam, batch, session = cosilab.parse_cam_batch(args.cam_batch)
    camera = cosilab.load_camera(args.data_root, cam)
    K, D, R = camera['K'], camera['D'], camera['R_w2c']
    C = -R.T @ camera['t_w2c']
    out_dir = os.path.join(args.out_dir, args.cam_batch)
    det = dict(np.load(os.path.join(out_dir, 'detections.npz')))
    nb_names = [f'cam{c:02d}_batch{batch:02d}' for c in cosilab.NEIGHBOURS[cam]]
    smpl = SMPL()
    neighbours = []
    for name in nb_names:
        v = load_view(args.hmr_root, name)
        v['index'] = neighbour_index(v)
        v['bodies'] = neighbour_bodies(args, name, smpl)
        neighbours.append(v)
        print(f'{name}: {len(v["tracks"])} people, camera {np.linalg.norm(v["C"] - C):.2f} m away')

    mv_pids = multiview_pids(args)
    print(f'people with a multi-view body (body-distance gate {args.max_body_dist_mv} m, else {args.max_body_dist} m): '
          f'{sorted(mv_pids)}')
    frames = det['frame'].astype(int)
    sframes = cosilab.session_frame(batch, frames)
    rays = world_rays(det['kp'], K, D, R)
    is_det = det['kind'] == 'detector'

    # costs per box (vs the people of its frame)
    costs = {}
    for f in np.unique(sframes):
        idx = np.nonzero(sframes == f)[0]
        pids, cost, nj, bdist = box_costs(det['kp'][idx], rays[idx], C, neighbours, int(f), args)
        for k, i in enumerate(idx):
            costs[i] = (pids, cost[k], nj[k], bdist[k])

    # 1. detector tracklets, Hungarian per frame on tracklet costs
    di = np.nonzero(is_det)[0]
    tid = np.full(len(frames), -1)
    tid[di] = link_tracklets(frames[di], det['box'][di], args.link_iou)
    tracklet_cost = {}
    for t in np.unique(tid[di]):
        members = np.nonzero(tid == t)[0]
        per_p = defaultdict(list)
        for i in members:
            for p, c, b in zip(costs[i][0], costs[i][1], costs[i][3]):
                if np.isfinite(c):
                    per_p[int(p)].append((c, b))
        tracklet_cost[int(t)] = {}
        for p, v in per_p.items():
            if len(v) < min(3, len(members)):
                continue
            c, b = np.array(v).T
            gap = float(np.median(c))
            body = float(np.nanmedian(b)) if np.isfinite(b).any() else np.nan
            score = gap + args.w_body * (body if np.isfinite(body) else args.max_body_dist)
            tracklet_cost[int(t)][p] = {'gap': gap, 'body': body, 'score': score, 'mv': p in mv_pids}
    assigned = {}  # (frame, pid) -> (kind, kp, box index)
    det_pid = np.full(len(frames), -1)
    for f in np.unique(frames[di]):
        idx = di[frames[di] == f]
        ts = [int(tid[i]) for i in idx]
        people = sorted({p for t in ts for p in tracklet_cost[t]})
        if not people:
            continue
        M = np.full((len(idx), len(people)), 1e3)
        for a, t in enumerate(ts):
            tc = tracklet_cost[t]
            best_t = min((x['score'] for x in tc.values()), default=1e3)
            for b, p in enumerate(people):
                if p in tc and passes(tc[p], args) and tc[p]['score'] <= best_t + args.best_margin:
                    M[a, b] = tc[p]['score']
        for a, b in zip(*linear_sum_assignment(M)):
            if M[a, b] < 1e3:
                det_pid[idx[a]] = people[b]
                assigned[(int(f), people[b])] = ('detector', det['kp'][idx[a]], int(idx[a]))

    # 2. fill-in from the projected boxes
    fill_stats = Counter()
    rejected = defaultdict(int)
    pi = np.nonzero(~is_det)[0]
    for i in pi:
        f, p = int(frames[i]), int(det['pid'][i])
        src = 'multiview' if det['source'][i] == 'multiview' else 'single-view'
        if (f, p) in assigned:
            fill_stats[f'{src}: detector box exists'] += 1
            continue
        pids, cost, _, bdist = costs[i]
        c, b = cost[pids == p], bdist[pids == p]
        if len(c) == 0 or not np.isfinite(c[0]) or not passes({'gap': c[0], 'body': b[0], 'mv': p in mv_pids}, args):
            fill_stats[f'{src}: failed gate'] += 1
            rejected[p] += 1
            continue
        same_frame = di[(frames[di] == f) & (det_pid[di] >= 0) & (det_pid[di] != p)]
        if len(same_frame) and iou_matrix(det['box'][i:i + 1], det['box'][same_frame]).max() >= args.overlap_iou:
            fill_stats[f'{src}: overlaps another person'] += 1
            continue
        fill_stats[f'{src}: accepted'] += 1
        assigned[(f, p)] = ('fill-in', det['kp'][i], int(i))

    # 3. drop short runs
    per_pid = defaultdict(list)
    for f, p in assigned:
        per_pid[p].append(f)
    dropped = 0
    for p, fs in per_pid.items():
        fs = np.array(sorted(fs))
        for s, e in runs(fs):
            if e - s < args.min_track_len:
                for f in fs[s:e]:
                    del assigned[(int(f), p)]
                    dropped += 1

    # report
    n_det = int(is_det.sum())
    print(f'\n{args.cam_batch}: {n_det} detector boxes in {len(np.unique(tid[di]))} tracklets; '
          f'{int((det_pid >= 0).sum())} got an ID ({(det_pid >= 0).sum() / max(n_det, 1):.3f})')
    for k in sorted(fill_stats):
        print(f'  fill-in {k}: {fill_stats[k]}')
    for src in ['multiview', 'single-view']:
        acc, fail = fill_stats[f'{src}: accepted'], fill_stats[f'{src}: failed gate']
        if acc + fail:
            print(f'  fill-in gate pass rate, {src}: {acc / (acc + fail):.3f}')
    print(f'  dropped {dropped} person-frames in runs shorter than {args.min_track_len}')

    tracklets = []
    for t, tc in sorted(tracklet_cost.items()):
        members = np.nonzero(tid == t)[0]
        got = Counter(det_pid[members][det_pid[members] >= 0].tolist())
        order = sorted(tc.items(), key=lambda x: x[1]['score'])
        tracklets.append({'tracklet': t, 'n_frames': int(len(members)),
                          'frames': [int(frames[members].min()), int(frames[members].max())],
                          'ids': {int(k): int(v) for k, v in got.items()},
                          'candidates': [{'pid': p, **c} for p, c in order[:4]]})
    long_t = [x for x in tracklets if x['n_frames'] >= 30 and x['candidates']]

    def margin(x):
        c = x['candidates']
        return c[1]['score'] - c[0]['score'] if len(c) > 1 else np.inf
    margins = np.array([margin(x) for x in long_t])
    if len(long_t):
        print(f'  tracklets >= 30 frames: {len(long_t)}, best score median '
              f'{np.median([x["candidates"][0]["score"] for x in long_t]):.3f}, margin to the 2nd-best ID median '
              f'{np.median(margins):.3f}, under 0.05: {int((margins < 0.05).sum())}')
        for x in long_t:
            c = x['candidates']
            if margin(x) < 0.05 and passes(c[0], args):
                print(f'    small margin: tracklet {x["tracklet"]} ({x["n_frames"]} frames {x["frames"]}, got {x["ids"]}): '
                      + ', '.join(f'{y["pid"]} gap {y["gap"]:.3f} body {y["body"]:.2f}' for y in c[:2]))

    per_id = {}
    print('\nper ID: person-frames from detector / fill-in, fill-ins rejected by the gate, frames seen by a neighbour')
    nb_frames = Counter()
    for nb in neighbours:
        for f, (pids, _, _) in nb['index'].items():
            fb = f - cosilab.session_frame(batch, 0)
            if det['start'] <= fb < det['start'] + det['num_frames']:
                for p in set(pids.tolist()):
                    nb_frames[(p, fb)] += 1
    seen_by_nb = Counter(p for p, _ in nb_frames)
    for p in sorted(set(seen_by_nb) | {p for _, p in assigned}):
        n_d = sum(1 for (f, q), v in assigned.items() if q == p and v[0] == 'detector')
        n_f = sum(1 for (f, q), v in assigned.items() if q == p and v[0] == 'fill-in')
        per_id[int(p)] = {'detector': n_d, 'fill_in': n_f, 'fill_in_rejected': int(rejected[p]),
                          'neighbour_frames': int(seen_by_nb[p])}
        print(f'  {p:3d}: {n_d:4d} / {n_f:4d}   rejected {rejected[p]:4d}   seen {seen_by_nb[p]:4d}')

    # outputs
    ann = {}
    for (f, p), (kind, kp, i) in sorted(assigned.items()):
        x1, y1, x2, y2 = det['box'][i].astype(float)
        e = ann.setdefault(str(f), {'bbox': {}, 'keypoints': {}})
        e['bbox'][str(p)] = [x1 / cosilab.KP_SCALE, y1 / cosilab.KP_SCALE,
                             (x2 - x1) / cosilab.KP_SCALE, (y2 - y1) / cosilab.KP_SCALE]
        k = kp.astype(float).copy()
        k[:, :2] /= cosilab.KP_SCALE
        e['keypoints'][str(p)] = k.tolist()
    with open(os.path.join(out_dir, 'vitpose_keypoints.json'), 'w') as fp:
        json.dump({'info': {'source': 'scripts/cosilab_assign_ids.py', 'neighbours': nb_names, 'args': vars(args)},
                   'annotations': ann}, fp)

    check = id_check(args, assigned, camera, batch, nb_names)
    with open(os.path.join(out_dir, 'id_assignment.json'), 'w') as fp:
        json.dump({'args': vars(args), 'neighbours': nb_names, 'tracklets': tracklets, 'per_id': per_id,
                   'fill_in': dict(fill_stats), 'id_check': check}, fp, indent=1, default=float)
    print(f'\n{len(assigned)} person-frames, {len({p for _, p in assigned})} IDs. Saved to {out_dir}')


if __name__ == '__main__':
    main(parse_args())
