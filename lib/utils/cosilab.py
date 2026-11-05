"""
COSI-Lab dataset helpers: camera calibration, ViTPose tracks and session video frames.

Layout under DATA_ROOT (the `processed/` folder of the 4TU upload):
    keypoints/vitpose_results/camXX_batchYY/vitpose_keypoints.json
    videos/mingling_session_videos/camXX.mp4
    videos/camera_calibration/idiap_params/{intrinsics,mingling_1_top,mingling_2_top}/
The side cameras 12-15 are not in the 4TU upload. Their raw GoPro files and calibration are next to it, under
cosilab_project/{raw_sensor_data,processed_data}/gopro_data/ (see SIDE_CAMS).
"""

import json
import os
import re
import subprocess

import cv2
import numpy as np

DATA_ROOT = '/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload/processed'
FPS = 60000 / 1001
BATCH_FRAMES = 18000  # each batch is 5 min of the session video
KP_SCALE = 2.0  # the ViTPose results are in 960x540, the videos are 1920x1080
# Side cameras: in both sessions, with the same numbers. A cam number alone does not give the session, so the
# side cameras are taken from SIDE_SESSION (only session 2 is processed so far).
SIDE_CAMS = (12, 13, 14, 15)
SIDE_SESSION = 2
# The stored extrinsics of side cameras 12 and 14 (chained through camera 04) don't fit the people (300+ px), see
# scripts/cosilab_calib_12_14/FINDINGS.md. load_camera(..., use_refit=True) reads re-estimated extrinsics for them
# (scripts/cosilab_refit_extrinsic.py); no usable refit exists yet, so it is off by default.
SIDE_EXTRINSIC_REFIT = (12, 14)
SIDE_EXTRINSIC_ROOT = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_side_calibration'
# Their timestamp CSVs (not in the dataset), written by scripts/cosilab_side_timestamps.py
SIDE_TIMESTAMP_ROOT = '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_side_timestamps'


def cam_session(cam):
    """Session of a camera: top 01-05 -> 2, top 06-10 -> 1, side 12-15 -> SIDE_SESSION."""
    if cam in SIDE_CAMS:
        return SIDE_SESSION
    return 1 if 6 <= cam <= 10 else 2


def project_root(data_root):
    """cosilab_project/, two levels above the 4TU processed/ folder."""
    return os.path.dirname(os.path.dirname(os.path.normpath(data_root)))

# The ViTPose results (dataset and scripts/run_cosilab_vitpose.py) use the ConfLab-17 layout, not COCO: head, nose,
# neck, R/L shoulder-elbow-wrist, R/L hip-knee-ankle, R/L foot. Indices into the 49 SPIN joints of
# lib.models.smpl.SMPL: head -> H36M head, feet -> big toes.
CONFLAB_TO_SPIN = [43, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 22, 19]
CONFLAB_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (7, 8), (3, 9), (6, 12), (9, 12),
                    (9, 10), (10, 11), (11, 15), (12, 13), (13, 14), (14, 16)]
CONFLAB_FLIP_PAIRS = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14), (15, 16)]


def parse_cam_batch(cam_batch):
    """'cam01_batch03' -> (cam=1, batch=3, session=2). Session 1 has top cameras 06-10, session 2 has 01-05."""
    m = re.fullmatch(r'cam(\d+)_batch(\d+)', cam_batch)
    if m is None:
        raise ValueError(f'Expected camXX_batchYY, got {cam_batch}')
    cam, batch = int(m.group(1)), int(m.group(2))
    return cam, batch, cam_session(cam)


def session_frame(batch, key):
    """Frame index in the session video of frame `key` of a batch (see keypoints/README.md)."""
    return (batch - 1) * BATCH_FRAMES + key


def keypoints_path(data_root, cam_batch, root=None):
    """The dataset ViTPose results, or those under `root` (same <cam_batch>/vitpose_keypoints.json layout)."""
    root = root or os.path.join(data_root, 'keypoints', 'vitpose_results')
    return os.path.join(root, cam_batch, 'vitpose_keypoints.json')


def video_path(data_root, cam):
    return os.path.join(data_root, 'videos', 'mingling_session_videos', f'cam{cam:02d}.mp4')


def calib_root(data_root, cam=None):
    if cam in SIDE_CAMS:
        return os.path.join(project_root(data_root), 'processed_data', 'gopro_data', 'side_view_camera_calibration',
                            'idiap_params')
    return os.path.join(data_root, 'videos', 'camera_calibration', 'idiap_params')


def load_camera(data_root, cam, extrinsic_c2w=False, use_refit=False):
    """
    Load the fisheye intrinsics and the session extrinsics of a camera.
    The extrinsics are read as OpenCV world->camera (X_c = R X_w + t) unless extrinsic_c2w is set.
    The world frame is the frame of camera 06 (session 1) or camera 01 (session 2), for the side cameras too.
    Camera 05 has no extrinsics; R_w2c and t_w2c are then None.
    With use_refit, cameras in SIDE_EXTRINSIC_REFIT read their extrinsics from SIDE_EXTRINSIC_ROOT.
    """
    root = calib_root(data_root, cam)
    with open(os.path.join(root, 'intrinsics', f'camera_{cam:02d}', 'intrinsic.json')) as f:
        intr = json.load(f)
    session = cam_session(cam)
    # the session 1 folders use 'camera-XX', the session 2 folders 'camera_XX'
    sep = '-' if session == 1 else '_'
    view = 'side' if cam in SIDE_CAMS else 'top'
    extr_file = os.path.join(root, f'mingling_{session}_{view}', f'camera{sep}{cam:02d}', 'extrinsic.json')
    if use_refit and cam in SIDE_EXTRINSIC_REFIT:
        extr_file = os.path.join(SIDE_EXTRINSIC_ROOT, f'mingling_{session}_side', f'camera_{cam:02d}', 'extrinsic.json')
        if not os.path.exists(extr_file):
            raise FileNotFoundError(f'{extr_file}: run scripts/cosilab_refit_extrinsic.py first')

    camera = {
        'cam': cam,
        'session': session,
        'K': np.array(intr['intrinsic'], dtype=np.float64),
        'D': np.array(intr['distortion_coefficients'], dtype=np.float64).reshape(4),
        'R_w2c': None,
        't_w2c': None,
        'rvec': None,
        'tvec': None,
        'extrinsic_c2w': extrinsic_c2w,
    }
    if os.path.exists(extr_file):
        with open(extr_file) as f:
            extr = json.load(f)
        rvec = np.array(extr['rvec'], dtype=np.float64).reshape(3)
        tvec = np.array(extr['tvec'], dtype=np.float64).reshape(3)
        R = cv2.Rodrigues(rvec)[0]
        if extrinsic_c2w:
            R, tvec_w2c = R.T, -R.T @ tvec
        else:
            tvec_w2c = tvec
        camera.update(R_w2c=R, t_w2c=tvec_w2c, rvec=rvec, tvec=tvec)
    return camera


def session_cameras(data_root, session, extrinsic_c2w=False):
    """All top cameras of a session that have extrinsics."""
    cams = range(6, 11) if session == 1 else range(1, 6)
    cameras = [load_camera(data_root, c, extrinsic_c2w) for c in cams]
    return [c for c in cameras if c['R_w2c'] is not None]


def load_tracks(json_path, kp_scale=KP_SCALE, img_size=(1920, 1080)):
    """
    Load the ViTPose results as tracks, in full-resolution pixels.
    Returns {pid: {'frames': (N,) int, 'boxes': (N, 4) xyxy, 'kp2d': (N, 17, 3) x, y, conf}}.
    Keypoints outside the image get confidence 0.
    """
    with open(json_path) as f:
        ann = json.load(f)['annotations']

    tracks = {}
    for key in sorted(ann, key=int):
        entry = ann[key]
        for pid, box in entry['bbox'].items():
            kp = entry['keypoints'].get(pid)
            if kp is None:
                continue
            t = tracks.setdefault(int(pid), {'frames': [], 'boxes': [], 'kp2d': []})
            x, y, w, h = box
            t['frames'].append(int(key))
            t['boxes'].append([x, y, x + w, y + h])
            t['kp2d'].append(kp)

    W, H = img_size
    for t in tracks.values():
        t['frames'] = np.array(t['frames'], dtype=np.int64)
        t['boxes'] = np.array(t['boxes'], dtype=np.float64) * kp_scale
        kp = np.array(t['kp2d'], dtype=np.float64)
        kp[..., :2] *= kp_scale
        outside = (kp[..., 0] < 0) | (kp[..., 0] >= W) | (kp[..., 1] < 0) | (kp[..., 1] >= H)
        kp[outside, 2] = 0
        t['kp2d'] = kp
    return tracks


class VideoReader:
    """
    Read consecutive frames of a video with ffmpeg, starting at an exact frame index.
    Yields BGR uint8 arrays of shape (H, W, 3).
    """
    def __init__(self, path, start_frame, num_frames, fps=FPS):
        self.path = path
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.fps = fps
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
             '-of', 'csv=p=0', path], capture_output=True, text=True, check=True)
        self.width, self.height = map(int, probe.stdout.strip().split(',')[:2])

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{self.start_frame / self.fps:.6f}',
               '-i', self.path, '-frames:v', str(self.num_frames),
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
        frame_bytes = self.width * self.height * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        try:
            for _ in range(self.num_frames):
                buf = bytearray(frame_bytes)
                if proc.stdout.readinto(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()


REFERENCE_CAM = {1: 6, 2: 1}  # the camera whose world_time.csv maps session frames to world time
# Cameras without dataset keypoints -> their neighbours in the camera row that have them. Camera 05 is left out (bad camera).
# Side cameras use all the top cameras of the session.
NEIGHBOURS = {2: (1, 3), 4: (3,), 7: (6, 8), 9: (8, 10), **{c: (1, 2, 3, 4) for c in SIDE_CAMS}}


def raw_video_dir(data_root, cam):
    """Folder of the raw GoPro files GH0*.MP4 of a camera (raw/ is next to processed/)."""
    session = cam_session(cam)
    if cam in SIDE_CAMS:  # not in the 4TU upload
        return os.path.join(project_root(data_root), 'raw_sensor_data', 'gopro_data',
                            f'side_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')
    return os.path.join(os.path.dirname(os.path.normpath(data_root)), 'raw', 'video',
                        f'top_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')


def _seconds(stamps):
    """'HH:MM:SS.ffffff' strings -> seconds of the day."""
    return np.array([int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:]) for s in stamps])


def _timecode_frames(tc):
    """'HH:MM:SS:FF' (60 frames per timecode second) -> frames since midnight."""
    h, m, s, f = map(int, tc.split(':'))
    return ((h * 60 + m) * 60 + s) * 60 + f


def _format_timecode(n):
    s, f = divmod(int(n), 60)
    return f'{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}'


def _format_timestamp(n):
    s, f = divmod(int(n), 60)
    return f'{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}.{round(f * 1e6 / 60):06d}'


def tag_timestamps(video_dir):
    """
    Per-file timestamp rows of a raw GoPro folder, from the MP4 timecode tags, as in the 4TU
    processed/videos/timestamps CSVs: [(name.csv, [(original_timecode, world_timecode, world_timestamp), ...])].
    The rule reproduces the 4TU CSVs of cameras 01-04 row for row: original = file tag + row (60 per timecode
    second); world = first file's tag + round-half-even(global frame * 1.001) ticks of 1/60 s (59.94 fps drift).
    Stops at the first file ffprobe can't read.
    """
    names = sorted(n for n in os.listdir(video_dir) if re.fullmatch(r'GH\d+\.MP4', n))
    out, g, prev_end, start = [], 0, None, None
    for name in names:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames:stream_tags=timecode', '-of', 'json', os.path.join(video_dir, name)],
            capture_output=True, text=True)
        if probe.returncode != 0:  # e.g. cam01 GH080189.MP4, also left out of the 4TU CSVs
            print(f'[Warning] {video_dir}/{name} is unreadable; it and later files get no timestamps', flush=True)
            break
        stream = json.loads(probe.stdout)['streams'][0]
        n0, count = _timecode_frames(stream['tags']['timecode']), int(stream['nb_frames'])
        if prev_end is not None and n0 != prev_end:
            raise RuntimeError(f'{video_dir}/{name}: timecode tag does not continue the previous file')
        start = n0 if start is None else start
        world = start + np.round((g + np.arange(count)) * 1001 / 1000).astype(np.int64)
        out.append((name.replace('.MP4', '.csv'),
                    [(_format_timecode(n0 + i), _format_timecode(w), _format_timestamp(w)) for i, w in enumerate(world)]))
        g, prev_end = g + count, n0 + count
    return out


def timestamp_dir(data_root, cam, session=None):
    """The per-file timestamp CSVs: the 4TU ones, or for the side cameras those of scripts/cosilab_side_timestamps.py."""
    session = session or cam_session(cam)
    sub = os.path.join(f'mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')
    if cam in SIDE_CAMS:
        return os.path.join(SIDE_TIMESTAMP_ROOT, sub)
    return os.path.join(data_root, 'videos', 'timestamps', sub)


def raw_frame_table(data_root, cam, session=None):
    """
    For every session frame, the raw GoPro frame of camera `cam` nearest in world time.
    Session frame -> world time from the reference camera's keypoints/vitpose_dataframe/camXX_world_time.csv,
    world time -> raw frame from processed/videos/timestamps/mingle_session_S/camera_XX-*/GH0*.csv (one row per frame;
    side cameras: SIDE_TIMESTAMP_ROOT, made from the timecode tags by scripts/cosilab_side_timestamps.py).
    Returns {'files': [GH0*.MP4 names], 'file': (N,) index into files, 'raw_frame': (N,), 'dt': (N,) s}.
    """
    session = session or cam_session(cam)
    ref = os.path.join(data_root, 'keypoints', 'vitpose_dataframe', f'cam{REFERENCE_CAM[session]:02d}_world_time.csv')
    with open(ref) as f:
        header = f.readline().strip().split(',')
        rows = [line.rstrip('\n').split(',') for line in f]
    assert [int(r[0]) for r in rows[:2]] == [0, 1], ref
    session_time = _seconds([r[header.index('world_timestamp')] for r in rows])

    ts_dir = timestamp_dir(data_root, cam, session)
    names = sorted(n for n in os.listdir(ts_dir) if re.fullmatch(r'GH\d+\.csv', n))
    times, file_idx, frame_idx = [], [], []
    for i, name in enumerate(names):
        with open(os.path.join(ts_dir, name)) as f:
            col = f.readline().strip().split(',').index('world_timestamp')
            t = _seconds([line.rstrip('\n').split(',')[col] for line in f if line.strip()])
        times.append(t)
        file_idx.append(np.full(len(t), i))
        frame_idx.append(np.arange(len(t)))
    times, file_idx, frame_idx = map(np.concatenate, (times, file_idx, frame_idx))
    order = np.argsort(times, kind='stable')
    times, file_idx, frame_idx = times[order], file_idx[order], frame_idx[order]

    j = np.searchsorted(times, session_time).clip(1, len(times) - 1)
    j = np.where(np.abs(times[j - 1] - session_time) <= np.abs(times[j] - session_time), j - 1, j)
    return {'files': [n.replace('.csv', '.MP4') for n in names], 'file': file_idx[j], 'raw_frame': frame_idx[j],
            'dt': times[j] - session_time}


class RawVideoReader:
    """
    Frames of a camera without a session video, read from its raw GoPro files at the session frame indices
    (see raw_frame_table). Same interface as VideoReader: yields BGR uint8 (H, W, 3) per session frame.
    Raw frames are repeated or skipped where the world clock has gaps.
    """
    def __init__(self, data_root, cam, start_frame, num_frames, fps=FPS, table=None):
        self.cam, self.start_frame, self.num_frames, self.fps = cam, start_frame, num_frames, fps
        self.dir = raw_video_dir(data_root, cam)
        table = table or raw_frame_table(data_root, cam)
        sl = slice(start_frame, start_frame + num_frames)
        self.files = table['files']
        self.file, self.raw_frame, self.dt = table['file'][sl], table['raw_frame'][sl], table['dt'][sl]
        if len(self.file) < num_frames:
            raise ValueError(f'cam{cam:02d}: the session has {len(table["file"])} frames, asked up to {sl.stop}')
        self.path = self.dir
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
             '-of', 'csv=p=0', os.path.join(self.dir, self.files[self.file[0]])],
            capture_output=True, text=True, check=True)
        self.width, self.height = map(int, probe.stdout.strip().split(',')[:2])

    def __len__(self):
        return self.num_frames

    def segments(self):
        """(file name, first raw frame, raw frames to decode, wanted raw frames per session frame) per file."""
        out = []
        breaks = np.nonzero(np.diff(self.file) != 0)[0] + 1
        for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(self.file)]):
            raw = self.raw_frame[s:e]
            if np.any(np.diff(raw) < 0):
                raise RuntimeError(f'cam{self.cam:02d}: raw frames go backwards in {self.files[self.file[s]]}')
            out.append((self.files[self.file[s]], int(raw[0]), int(raw[-1] - raw[0] + 1), raw - raw[0]))
        return out

    def __iter__(self):
        frame_bytes = self.width * self.height * 3
        for name, first, count, wanted in self.segments():
            cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{first / self.fps:.6f}',
                   '-i', os.path.join(self.dir, name), '-frames:v', str(count),
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
            try:
                pos, frame = -1, None
                for w in wanted:
                    while pos < w:
                        buf = bytearray(frame_bytes)
                        if proc.stdout.readinto(buf) < frame_bytes:
                            return
                        frame, pos = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3), pos + 1
                    yield frame
            finally:
                proc.stdout.close()
                proc.kill()
                proc.wait()


def video_source(data_root, cam):
    """The session video of a camera if it exists, otherwise its raw GoPro folder."""
    path = video_path(data_root, cam)
    return path if os.path.exists(path) else raw_video_dir(data_root, cam)


def open_video(data_root, cam, start_frame, num_frames):
    """Reader of session frames [start_frame, start_frame + num_frames): the session video, else the raw files."""
    path = video_path(data_root, cam)
    if os.path.exists(path):
        return VideoReader(path, start_frame, num_frames)
    return RawVideoReader(data_root, cam, start_frame, num_frames)


def load_results(result_dir, which='hmr'):
    """
    Load the output of scripts/run_cosilab_smpl.py for one cam-batch.
    Returns (camera dict from camera.json, {pid: dict of npz arrays}).
    """
    with open(os.path.join(result_dir, 'camera.json')) as f:
        camera = json.load(f)
    track_dir = os.path.join(result_dir, which)
    tracks = {}
    for name in sorted(os.listdir(track_dir)):
        m = re.fullmatch(r'track_(-?\d+)\.npz', name)
        if m:
            with np.load(os.path.join(track_dir, name)) as data:
                tracks[int(m.group(1))] = dict(data)
    return camera, tracks


def frame_index(tracks):
    """{batch frame: [(pid, row in the track arrays)]} for loaded result tracks."""
    index = {}
    for pid, t in tracks.items():
        for i, key in enumerate(t['frame_ids']):
            index.setdefault(int(key), []).append((pid, i))
    return index


def track_color(pid):
    """Distinct RGB colour (0-255) per track id."""
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
               (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
               (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
               (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]
    return palette[pid % len(palette)]
