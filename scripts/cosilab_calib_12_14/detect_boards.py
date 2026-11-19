"""Detect the 5x3 inner-corner board in every calibration frame of cams 01-04 and 12-15 (numbers only)."""
import os, sys, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
TOP='/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload/processed/videos/camera_calibration/calib_frames/top_view_camera-mingle_session_2'
SIDE='/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/calib_frames/side_view_camera-mingle_session_2'
def path(cam, i):
    root = TOP if cam < 10 else SIDE
    return f'{root}/camera_{cam:02d}-mingle_session_2/{i:09d}.png'
def detect(args):
    cam, i = args
    img = cv2.imread(path(cam, i), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return cam, i, None
    for size in [(5, 3), (3, 5)]:
        ok, c = cv2.findChessboardCornersSB(img, size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return cam, i, (size, c.reshape(-1, 2))
    return cam, i, None
if __name__ == '__main__':
    jobs = [(c, i) for c in (1, 2, 3, 4, 12, 13, 14, 15) for i in range(332)]
    out = {}
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 16) as ex:
        for n, (cam, i, r) in enumerate(ex.map(detect, jobs, chunksize=4)):
            if r is not None:
                out[(cam, i)] = r
            if n % 332 == 331:
                print(f'cam{cam:02d}: {sum(1 for k in out if k[0] == cam)} frames with a board', flush=True)
    np.save(sys.argv[1], np.array([(c, i, s, x) for (c, i), (s, x) in out.items()], dtype=object), allow_pickle=True)
