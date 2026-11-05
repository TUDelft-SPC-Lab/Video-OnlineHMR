"""
When was each calibration frame really taken? Find the raw GoPro frame that a calib_frames PNG equals, by mean absolute
pixel difference (grey, 1/4 size) against the raw frames in a window around the expected raw timecode. Numbers only.
calib frame i was extracted at timecode 14:47:14:00 + 30 i in the (timecode-synchronized) videos.
Usage: python scripts/cosilab_calib_12_14/calib_frame_times.py out.json --frames 25 56 76 293 306 --cams 1 4 12 13 14 15
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, '.')
from lib.utils import cosilab

TOP = '/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload/processed/videos/camera_calibration/calib_frames/top_view_camera-mingle_session_2'
SIDE = '/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/calib_frames/side_view_camera-mingle_session_2'
START = ((14 * 60 + 47) * 60 + 14) * 60  # 14:47:14:00 in timecode frames (60 per second)


def raw_index(cam):
    """original timecode (frames since midnight) -> (file, raw frame), from the per-file timestamp CSVs."""
    d = cosilab.timestamp_dir(cosilab.DATA_ROOT, cam)
    out = {}
    for name in sorted(n for n in os.listdir(d) if n.endswith('.csv')):
        with open(os.path.join(d, name)) as f:
            next(f)
            for r, line in enumerate(f):
                tc = line.split(',')[0]
                out[cosilab._timecode_frames(tc)] = (name.replace('.csv', '.MP4'), r)
    return out


def decode(path, first, count, scale=4):
    w, h = 1920 // scale, 1080 // scale
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{first / cosilab.FPS:.6f}', '-i', path, '-frames:v', str(count),
           '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(buf, np.uint8).reshape(-1, h, w).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--frames', type=int, nargs='+', default=[25, 56, 76, 293, 306])
    p.add_argument('--cams', type=int, nargs='+', default=[1, 4, 12, 13, 14, 15])
    p.add_argument('--before', type=int, default=150, help='raw frames searched before the expected timecode')
    p.add_argument('--after', type=int, default=60)
    a = p.parse_args()
    res = {}
    for cam in a.cams:
        idx = raw_index(cam)
        vdir = cosilab.raw_video_dir(cosilab.DATA_ROOT, cam)
        res[cam] = {}
        for i in a.frames:
            png = f'{TOP if cam < 10 else SIDE}/camera_{cam:02d}-mingle_session_2/{i:09d}.png'
            img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32)
            t = START + 30 * i
            if t - a.before not in idx:
                print(f'cam{cam:02d} frame {i}: timecode outside the raw files')
                continue
            name, r0 = idx[t - a.before]
            frames = decode(os.path.join(vdir, name), r0, a.before + a.after + 1)
            mad = np.abs(frames - img[None]).mean((1, 2))
            k = int(np.argmin(mad))
            second = np.partition(mad, 1)[1] if len(mad) > 1 else np.inf
            offset = k - a.before  # raw timecode of the matched frame minus the expected one
            res[cam][i] = {'offset_frames': offset, 'mad': float(mad[k]), 'mad_next_best': float(np.sort(mad)[1]),
                           'mad_median': float(np.median(mad))}
            print(f'cam{cam:02d} calib frame {i:3d}: matches raw timecode {offset:+d} frames from 14:47:14:00+30i '
                  f'(MAD {mad[k]:.2f}, next best {np.sort(mad)[1]:.2f}, median {np.median(mad):.2f})', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
