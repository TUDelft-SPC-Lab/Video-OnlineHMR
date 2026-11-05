"""
Timestamp CSVs of the COSI-Lab side cameras (12-15), which the dataset doesn't have, from the timecode tags of their
raw GoPro files, with the rule that makes the 4TU processed/videos/timestamps CSVs (cosilab.tag_timestamps).
--check first rebuilds the CSVs of top cameras from their raw files and compares them with the 4TU ones row by row.

    python scripts/cosilab_side_timestamps.py --check 1 2 3 4 --cams 12 13 14 15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils import cosilab  # noqa: E402


def top_raw_dir(data_root, cam):
    """The raw top-camera files under raw_sensor_data (same files as 4TU_upload/raw)."""
    session = cosilab.cam_session(cam)
    return os.path.join(cosilab.project_root(data_root), 'raw_sensor_data', 'gopro_data',
                        f'top_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=cosilab.DATA_ROOT)
    parser.add_argument('--cams', type=int, nargs='*', default=list(cosilab.SIDE_CAMS))
    parser.add_argument('--check', type=int, nargs='*', default=[], help='top cameras to compare with the 4TU CSVs')
    parser.add_argument('--out_root', type=str, default=cosilab.SIDE_TIMESTAMP_ROOT)
    args = parser.parse_args()

    for cam in args.check:
        ref_dir = cosilab.timestamp_dir(args.data_root, cam)
        bad = rows = 0
        for name, made in cosilab.tag_timestamps(top_raw_dir(args.data_root, cam)):
            with open(os.path.join(ref_dir, name)) as f:
                ref = [tuple(line.rstrip('\n').split(',')) for line in f.readlines()[1:] if line.strip()]
            bad += sum(a != b for a, b in zip(made, ref)) + abs(len(made) - len(ref))
            rows += len(ref)
        print(f'cam{cam:02d}: {bad} of {rows} rows differ from the 4TU CSVs', flush=True)
        if bad:
            sys.exit(1)

    for cam in args.cams:
        assert cam in cosilab.SIDE_CAMS, cam
        session = cosilab.cam_session(cam)
        out_dir = os.path.join(args.out_root, f'mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')
        os.makedirs(out_dir, exist_ok=True)
        files = cosilab.tag_timestamps(cosilab.raw_video_dir(args.data_root, cam))
        for name, rows in files:
            with open(os.path.join(out_dir, name), 'w') as f:
                f.write('original_timecode,world_timecode,world_timestamp\n')
                f.writelines(','.join(r) + '\n' for r in rows)
        print(f'cam{cam:02d}: {len(files)} files, {sum(len(r) for _, r in files)} frames, first world time '
              f'{files[0][1][0][2]}, last {files[-1][1][-1][2]} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
