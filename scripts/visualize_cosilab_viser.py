"""
Show the SMPL results of scripts/run_cosilab_smpl.py for one cam-batch in 3D with viser:
all tracks over time, the static camera(s) and a floor grid.

The world frame is the frame of camera 06 (session 1) or camera 01 (session 2). The up direction
and the floor height are estimated from the people. SMPL meshes are computed per timestep.

python scripts/visualize_cosilab_viser.py \
    --result_dir /tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_smpl/cam01_batch01 \
    --which hmr --all_cameras
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np
import torch
import viser
import viser.transforms as vtf

from lib.models.smpl import SMPL
from lib.utils import cosilab


def rotation_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main():
    parser = argparse.ArgumentParser(description='3D view of COSI-Lab SMPL results')
    parser.add_argument('--result_dir', type=str, required=True, help='<out_dir>/camXX_batchYY')
    parser.add_argument('--which', type=str, default='hmr', choices=['hmr', 'refined'])
    parser.add_argument('--frame', type=str, default='world', choices=['world', 'camera'])
    parser.add_argument('--start', type=int, default=None, help='first batch frame (default: first result)')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=2, help='timeline step in video frames')
    parser.add_argument('--all_cameras', action='store_true', help='also show the other session cameras')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    camera, tracks = cosilab.load_results(args.result_dir, args.which)
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

    # Up direction: the SMPL +y axis of the people (they mostly stand). Floor: their lowest points.
    roots = np.concatenate([t['root'] for t in tracks.values()])
    up = roots[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    rng = np.random.default_rng(0)
    sample = [(p, i) for p, t in tracks.items() for i in range(len(t['frame_ids']))]
    sample = [sample[j] for j in rng.choice(len(sample), min(300, len(sample)), replace=False)]
    verts = bodies(sample)
    floor_h = np.median((verts @ up).min(1))
    centre = np.concatenate([t['trans'] for t in tracks.values()]).mean(0)
    floor_centre = centre - (centre @ up - floor_h) * up
    print(f'up {np.round(up, 3)}, floor at {floor_h:.2f} m along up')

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(tuple(up))
    server.scene.add_grid('/floor', width=12, height=12, plane='xy',
                          wxyz=vtf.SO3.from_matrix(rotation_between(np.array([0., 0., 1.]), up)).wxyz,
                          position=floor_centre)

    # Cameras
    if world:
        cams = cosilab.session_cameras(camera['data_root'], camera['session'], camera['extrinsic_c2w']) \
            if args.all_cameras else []
        cams = [c for c in cams if c['cam'] != camera['cam']]
        own = {'cam': camera['cam'], 'K': np.array(camera['K']),
               'R_w2c': np.array(camera['R_w2c']), 't_w2c': np.array(camera['t_w2c'])}
        cams = [own] + cams
    else:
        cams = [{'cam': camera['cam'], 'K': np.array(camera['K']), 'R_w2c': np.eye(3), 't_w2c': np.zeros(3)}]
    for c in cams:
        R_c2w = c['R_w2c'].T
        pos = -R_c2w @ c['t_w2c']
        fov = 2 * np.arctan(540 / c['K'][1, 1])
        color = (255, 80, 40) if c['cam'] == camera['cam'] else (60, 60, 60)
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
        gui_fps = server.gui.add_slider('FPS', min=1, max=60, step=1, initial_value=15)

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
