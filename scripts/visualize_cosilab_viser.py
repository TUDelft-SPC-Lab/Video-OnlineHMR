"""
Show the SMPL results of scripts/run_cosilab_smpl.py for one cam-batch in 3D with viser:
all tracks over time, the static camera(s) and a floor grid.

--result_dir can also be a top folder (e.g. cosilab_multiview_8cam_smooth_prehp1) holding several
camXX_batchYY folders of one batch. Each person is then shown once, in the world frame: frames come from
the camera that has the most frames of that person, and the other cameras fill in the frames it misses.
For multi-view `refined` results the per-camera copies are the same body; for single-view results (or
`hmr`) they are separate fits, so the printed overlap disagreement is large and the body can jump where
another camera fills in.

The world frame is the frame of camera 06 (session 1) or camera 01 (session 2). The floor grid is a robust
plane fitted to the people's lowest points, and the up direction is its normal. SMPL meshes are computed per timestep.

python scripts/visualize_cosilab_viser.py \
    --result_dir /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl/cam01_batch01 \
    --which hmr --all_cameras

python scripts/visualize_cosilab_viser.py \
    --result_dir /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_multiview_8cam_smooth_prehp1 \
    --which refined --all_cameras
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time

import numpy as np
import torch
import viser
import viser.transforms as vtf

from lib.models.smpl import SMPL
from lib.utils import cosilab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_cosilab_multiview import floor_spread

# Start view for session 2 in the world frame, copied from the viser client's URL parameters
INITIAL_CAMERA = {'position': (0.311, 2.754, 1.765), 'look_at': (0.202, 1.331, 2.124),
                  'up': (0.001, -0.117, -0.993), 'fov': 1.3090, 'near': 0.01, 'far': 1000.0}


def rotation_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def merge_cam_batches(root, which, batch=None):
    """
    Load all camXX_batchYY folders under root (one session and batch) and merge each person into one
    world-frame track. Returns (list of camera dicts, {pid: track dict}).
    """
    found = {}
    for name in sorted(os.listdir(root)):
        m = re.fullmatch(r'cam(\d+)_batch(\d+)', name)
        if m and os.path.isfile(os.path.join(root, name, 'camera.json')) \
                and os.path.isdir(os.path.join(root, name, which)):
            found.setdefault(int(m.group(2)), []).append(name)
    if not found:
        raise SystemExit(f'No camXX_batchYY folders with camera.json and {which}/ in {root}')
    if batch is None:
        if len(found) > 1:
            raise SystemExit(f'{root} has batches {sorted(found)}; pick one with --batch')
        batch = next(iter(found))
    if batch not in found:
        raise SystemExit(f'No batch {batch} in {root} (batches: {sorted(found)})')

    cameras, per_cam = [], {}
    for name in found[batch]:
        camera, tracks = cosilab.load_results(os.path.join(root, name), which)
        if camera['R_w2c'] is None:
            print(f'{name}: no extrinsics, skipped')
            continue
        cameras.append(camera)
        for pid, t in tracks.items():
            per_cam.setdefault(pid, []).append((name, t))
    sessions = {c['session'] for c in cameras}
    if len(sessions) > 1:
        raise SystemExit(f'{root} batch {batch} mixes sessions {sorted(sessions)}')
    print(f'{root}: batch {batch}, {which}, cameras {[c["cam"] for c in cameras]}')

    keys = ['frame_ids', 'pred_rotmat', 'pred_shape', 'world_rotmat_root', 'world_trans']
    merged, gaps = {}, []
    for pid, items in per_cam.items():
        items = sorted(items, key=lambda it: -len(it[1]['frame_ids']))  # most frames first
        rows = {}  # batch frame -> (camera name, row)
        for name, t in items:
            for i, k in enumerate(t['frame_ids']):
                rows.setdefault(int(k), (name, i))
        tracks = dict(items)
        frames = sorted(rows)
        merged[pid] = {key: np.stack([tracks[rows[k][0]][key][rows[k][1]] for k in frames]) for key in keys}
        # Disagreement of the world translation where several cameras have this person
        ref = items[0][1]
        ref_row = {int(k): i for i, k in enumerate(ref['frame_ids'])}
        for name, t in items[1:]:
            common = [(ref_row[int(k)], i) for i, k in enumerate(t['frame_ids']) if int(k) in ref_row]
            if common:
                a, b = np.array(common).T
                gaps.append(np.linalg.norm(ref['world_trans'][a] - t['world_trans'][b], axis=-1))
        print(f'  person {pid}: {len(frames)} frames from ' +
              ', '.join(f'{name} ({len(t["frame_ids"])})' for name, t in items))
    if gaps:
        gaps = np.concatenate(gaps)
        print(f'world translation of the same person in two cameras on shared frames: median '
              f'{np.median(gaps) * 1000:.2f} mm, max {gaps.max() * 1000:.2f} mm ({len(gaps)} pairs)')
        if np.median(gaps) > 0.01:
            print('  -> the per-camera bodies are separate fits, not one shared body')
    return cameras, merged


def main():
    parser = argparse.ArgumentParser(description='3D view of COSI-Lab SMPL results')
    parser.add_argument('--result_dir', type=str, required=True, help='<out_dir>/camXX_batchYY, or <out_dir> to merge all its cameras')
    parser.add_argument('--which', type=str, default='hmr', choices=['hmr', 'refined'])
    parser.add_argument('--frame', type=str, default='world', choices=['world', 'camera'])
    parser.add_argument('--start', type=int, default=None, help='first batch frame (default: first result)')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=2, help='timeline step in video frames')
    parser.add_argument('--all_cameras', action='store_true', help='also show the other session cameras')
    parser.add_argument('--fps', type=int, default=60, help='initial playback speed (1-60)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--batch', type=int, default=None,
                        help='with a top folder holding several batches: the batch to show')
    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.result_dir, 'camera.json')):
        camera, tracks = cosilab.load_results(args.result_dir, args.which)
        own_cams = [camera]
    else:
        if args.frame != 'world':
            raise SystemExit('A top folder merges cameras, so it needs --frame world')
        own_cams, tracks = merge_cam_batches(args.result_dir, args.which, args.batch)
        camera = own_cams[0]
    if args.frame == 'world' and camera['R_w2c'] is None:
        print(f'Camera {camera["cam"]:02d} has no extrinsics, showing the camera frame')
        args.frame = 'camera'
    world = args.frame == 'world'
    for t in tracks.values():
        t['root'] = t['world_rotmat_root'] if world else t['pred_rotmat'][:, 0]
        t['trans'] = t['world_trans'] if world else t['pred_trans'][:, 0]

    index = cosilab.frame_index(tracks)
    start = min(index) if args.start is None else args.start
    end = max(index) + 1 if args.num_frames is None else start + args.num_frames
    timeline = [k for k in range(start, end, args.stride) if k in index]
    if not timeline:
        print('No results in the requested range')
        return

    smpl = SMPL()
    faces = np.asarray(smpl.faces, dtype=np.int32)

    def bodies(entries):
        rotmat = torch.tensor(np.stack([tracks[p]['pred_rotmat'][i] for p, i in entries]))
        rotmat[:, 0] = torch.tensor(np.stack([tracks[p]['root'][i] for p, i in entries]))
        with torch.no_grad():
            out = smpl(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:],
                       betas=torch.tensor(np.stack([tracks[p]['pred_shape'][i] for p, i in entries])),
                       transl=torch.tensor(np.stack([tracks[p]['trans'][i] for p, i in entries])),
                       pose2rot=False, default_smpl=True)
        return out.vertices.numpy()

    # Floor: a robust plane through the lowest body point of sampled person-frames (as floor_spread in
    # compare_cosilab_multiview.py). The mean SMPL +y axis of the people only picks "lowest" and the normal's sign,
    # and is the fallback normal when the lowest points don't span an area (e.g. one person standing still).
    roots = np.concatenate([t['root'] for t in tracks.values()])
    body_up = roots[:, :, 1].mean(0)
    body_up /= np.linalg.norm(body_up)
    rng = np.random.default_rng(0)
    sample = [(p, i) for p, t in tracks.items() for i in range(len(t['frame_ids']))]
    sample = [sample[j] for j in rng.choice(len(sample), min(1000, len(sample)), replace=False)]
    low = []
    for k in range(0, len(sample), 250):
        verts = bodies(sample[k:k + 250])
        low.append(verts[np.arange(len(verts)), (verts @ body_up).argmin(1)])
    low = np.concatenate(low).astype(np.float64)
    in_plane = np.linalg.svd(low - low.mean(0), compute_uv=False)[1] / np.sqrt(len(low))
    if in_plane > 0.3:
        spread, up, kept = floor_spread(low)
        up = up if up @ body_up > 0 else -up
        keep = np.abs((low - low.mean(0)) @ up) < 3 * spread
        angle = np.degrees(np.arccos(np.clip(up @ body_up, -1, 1)))
        print(f'floor: plane fitted to the lowest point of {len(low)} person-frames, spread {spread:.3f} m '
              f'(kept {kept:.2f}), up {np.round(up, 3)}, {angle:.1f} deg from the mean body up')
    else:
        up, keep = body_up, np.ones(len(low), dtype=bool)
        print(f'floor: lowest points span only {in_plane:.2f} m, using the mean body up {np.round(up, 3)} as the normal')
    floor_point = np.median(low[keep], 0)
    centre = np.concatenate([t['trans'] for t in tracks.values()]).mean(0)
    floor_centre = centre - ((centre - floor_point) @ up) * up

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(tuple(up))
    if world and camera['session'] == 2:
        for name, value in INITIAL_CAMERA.items():
            setattr(server.initial_camera, name, value)
    server.scene.add_grid('/floor', width=12, height=12, plane='xy',
                          wxyz=vtf.SO3.from_matrix(rotation_between(np.array([0., 0., 1.]), up)).wxyz,
                          position=floor_centre)

    # Cameras
    if world:
        cams = cosilab.session_cameras(camera['data_root'], camera['session'], camera['extrinsic_c2w']) \
            if args.all_cameras else []
        own_ids = {c['cam'] for c in own_cams}
        cams = [{'cam': c['cam'], 'K': np.array(c['K']), 'R_w2c': np.array(c['R_w2c']),
                 't_w2c': np.array(c['t_w2c'])} for c in own_cams] + [c for c in cams if c['cam'] not in own_ids]
    else:
        own_ids = {camera['cam']}
        cams = [{'cam': camera['cam'], 'K': np.array(camera['K']), 'R_w2c': np.eye(3), 't_w2c': np.zeros(3)}]
    for c in cams:
        R_c2w = c['R_w2c'].T
        pos = -R_c2w @ c['t_w2c']
        fov = 2 * np.arctan(540 / c['K'][1, 1])
        color = (255, 80, 40) if c['cam'] in own_ids else (60, 60, 60)
        server.scene.add_camera_frustum(f'/cameras/cam{c["cam"]:02d}', fov=fov, aspect=16 / 9, scale=0.3,
                                        wxyz=vtf.SO3.from_matrix(R_c2w).wxyz, position=pos, color=color)
        server.scene.add_label(f'/cameras/cam{c["cam"]:02d}/label', f'cam{c["cam"]:02d}')

    # Trajectories of the SMPL translation, one polyline per track
    traj_handles = []
    for pid, t in tracks.items():
        keep = (t['frame_ids'] >= start) & (t['frame_ids'] < end)
        pts = t['trans'][keep][::args.stride]
        consecutive = np.diff(t['frame_ids'][keep][::args.stride]) <= args.stride
        if len(pts) < 2 or not consecutive.any():
            continue
        segs = np.stack([pts[:-1], pts[1:]], 1)[consecutive].astype(np.float32)
        traj_handles.append(server.scene.add_line_segments(
            f'/trajectories/{pid}', points=segs, colors=cosilab.track_color(pid), thickness=0.01))

    with server.gui.add_folder('Playback'):
        gui_show_traj = server.gui.add_checkbox('Show trajectories', True)
        gui_step = server.gui.add_slider('Timestep', min=0, max=len(timeline) - 1, step=1, initial_value=0)
        gui_frame = server.gui.add_text('Batch frame', str(timeline[0]), disabled=True)
        gui_playing = server.gui.add_checkbox('Playing', True)
        gui_fps = server.gui.add_slider('FPS', min=1, max=60, step=1, initial_value=min(max(args.fps, 1), 60))

    meshes = {}

    def update(step):
        key = timeline[step]
        entries = index[key]
        verts = bodies(entries)
        shown = set()
        for (pid, _), v in zip(entries, verts):
            if pid not in meshes:
                meshes[pid] = server.scene.add_mesh_simple(f'/humans/{pid}', vertices=v, faces=faces,
                                                           color=cosilab.track_color(pid), side='double')
            else:
                meshes[pid].vertices = v
                meshes[pid].visible = True
            shown.add(pid)
        for pid, m in meshes.items():
            if pid not in shown:
                m.visible = False
        gui_frame.value = str(key)

    @gui_step.on_update
    def _(_):
        update(gui_step.value)

    @gui_show_traj.on_update
    def _(_):
        for h in traj_handles:
            h.visible = gui_show_traj.value

    update(0)
    print(f'{len(tracks)} tracks, {len(timeline)} timesteps from batch frame {timeline[0]} to {timeline[-1]}')
    while True:
        if gui_playing.value:
            gui_step.value = (gui_step.value + 1) % len(timeline)
        time.sleep(1.0 / gui_fps.value)


if __name__ == '__main__':
    main()
