"""
COSI-Lab dataset helpers: camera calibration, ViTPose tracks and session video frames.

Layout under DATA_ROOT (the `processed/` folder of the 4TU upload):
    keypoints/vitpose_results/camXX_batchYY/vitpose_keypoints.json
    videos/mingling_session_videos/camXX.mp4
    videos/camera_calibration/idiap_params/{intrinsics,mingling_1_top,mingling_2_top}/
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

# The ViTPose results (dataset and scripts/run_cosilab_vitpose.py) use the ConfLab-17 layout, not COCO: head, nose,
# neck, R/L shoulder-elbow-wrist, R/L hip-knee-ankle, R/L foot. Indices into the 49 SPIN joints of
# lib.models.smpl.SMPL: head -> H36M head, feet -> big toes.
CONFLAB_TO_SPIN = [43, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 22, 19]
CONFLAB_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (7, 8), (3, 9), (6, 12), (9, 12),
                    (9, 10), (10, 11), (11, 15), (12, 13), (13, 14), (14, 16)]
CONFLAB_FLIP_PAIRS = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14), (15, 16)]


def parse_cam_batch(cam_batch):
    """'cam01_batch03' -> (cam=1, batch=3, session=2). Session 1 has cameras 06-10, session 2 has 01-05."""
    m = re.fullmatch(r'cam(\d+)_batch(\d+)', cam_batch)
    if m is None:
        raise ValueError(f'Expected camXX_batchYY, got {cam_batch}')
    cam, batch = int(m.group(1)), int(m.group(2))
    return cam, batch, 1 if cam >= 6 else 2


def session_frame(batch, key):
    """Frame index in the session video of frame `key` of a batch (see keypoints/README.md)."""
    return (batch - 1) * BATCH_FRAMES + key


def keypoints_path(data_root, cam_batch, root=None):
    """The dataset ViTPose results, or those under `root` (same <cam_batch>/vitpose_keypoints.json layout)."""
    root = root or os.path.join(data_root, 'keypoints', 'vitpose_results')
    return os.path.join(root, cam_batch, 'vitpose_keypoints.json')


def video_path(data_root, cam):
    return os.path.join(data_root, 'videos', 'mingling_session_videos', f'cam{cam:02d}.mp4')


def calib_root(data_root):
    return os.path.join(data_root, 'videos', 'camera_calibration', 'idiap_params')


def load_camera(data_root, cam, extrinsic_c2w=False):
    """
    Load the fisheye intrinsics and the session extrinsics of a camera.
    The extrinsics are read as OpenCV world->camera (X_c = R X_w + t) unless extrinsic_c2w is set.
    The world frame is the frame of camera 06 (session 1) or camera 01 (session 2).
    Camera 05 has no extrinsics; R_w2c and t_w2c are then None.
    """
    root = calib_root(data_root)
    with open(os.path.join(root, 'intrinsics', f'camera_{cam:02d}', 'intrinsic.json')) as f:
        intr = json.load(f)
    session = 1 if cam >= 6 else 2
    # the session 1 folders use 'camera-XX', the session 2 folders 'camera_XX'
    sep = '-' if session == 1 else '_'
    extr_file = os.path.join(root, f'mingling_{session}_top', f'camera{sep}{cam:02d}', 'extrinsic.json')

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
    """All cameras of a session that have extrinsics."""
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
NEIGHBOURS = {2: (1, 3), 4: (3,), 7: (6, 8), 9: (8, 10)}


def raw_video_dir(data_root, cam):
    """Folder of the raw GoPro files GH0*.MP4 of a camera (raw/ is next to processed/)."""
    session = 1 if cam >= 6 else 2
    return os.path.join(os.path.dirname(os.path.normpath(data_root)), 'raw', 'video',
                        f'top_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')


def _seconds(stamps):
    """'HH:MM:SS.ffffff' strings -> seconds of the day."""
    return np.array([int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:]) for s in stamps])


def raw_frame_table(data_root, cam, session=None):
    """
    For every session frame, the raw GoPro frame of camera `cam` nearest in world time.
    Session frame -> world time from the reference camera's keypoints/vitpose_dataframe/camXX_world_time.csv,
    world time -> raw frame from processed/videos/timestamps/mingle_session_S/camera_XX-*/GH0*.csv (one row per frame).
    Returns {'files': [GH0*.MP4 names], 'file': (N,) index into files, 'raw_frame': (N,), 'dt': (N,) s}.
    """
    session = session or (1 if cam >= 6 else 2)
    ref = os.path.join(data_root, 'keypoints', 'vitpose_dataframe', f'cam{REFERENCE_CAM[session]:02d}_world_time.csv')
    with open(ref) as f:
        header = f.readline().strip().split(',')
        rows = [line.rstrip('\n').split(',') for line in f]
    assert [int(r[0]) for r in rows[:2]] == [0, 1], ref
    session_time = _seconds([r[header.index('world_timestamp')] for r in rows])

    ts_dir = os.path.join(data_root, 'videos', 'timestamps', f'mingle_session_{session}',
                          f'camera_{cam:02d}-mingle_session_{session}')
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
