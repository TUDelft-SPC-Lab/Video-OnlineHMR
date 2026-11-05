# COSI-Lab side cameras 12 and 14: calibration does not fit the people (open, 2026-11-05)

Session 2, batch01 frames 6000–6600 (world time 14:53:40–14:53:50). Branch `cosilab-side-calib-12-14`.
Result files (logs, JSON) are in `/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_calib_12_14/`.

## The problem
- **What is compared:** the 4-camera multi-view bodies (`cosilab_multiview_4cam`, cams 01–04) are projected into each side
  camera and compared with the ViTDet + VIT003 detections in that camera
  (`cosilab_vit003_extra/cam1X_batch01/detections.npz`, job 102220).
- **How it is scored:** Hungarian matching per frame, median joint error per person.
- **Who counts:** only people clearly visible under the tested pose, i.e. no other person 0.3 m or more closer covers
  more than 20 % of their joint box (`clearly_visible` in `scripts/cosilab_refit_extrinsic.py`).

| camera | calibrated with | within 40 px | median |
|---|---|---|---|
| cam13 | cam01 (`top_cam_01_side_cam_13_15`) | **0.77** | 21 px |
| cam15 | cam01 | (about 20 px median with all people) | |
| cam12 | cam04 (`top_cam_04_side_cam_12_14`), chained | 0.07 | 117 px |
| cam14 | cam04, chained | 0.20 | 79 px |

With all people counted, not only the clearly visible ones, the medians are 59 px (cam13) vs 343 px (cam12) and
303 px (cam14).

## Facts established
- **Timestamps:** side-camera timestamps come from the raw MP4 timecode tags, with the 4TU rule
  (`cosilab.tag_timestamps`).
  - The rule is: world tick = first file's tag + round-half-even(frame × 1.001). It reproduces the 4TU CSVs of
    cams 01–04 row for row.
  - The `timecode_fixed/` videos are wrong (user) and are not used.
  - Per the user, timing between cameras is good to 11 frames or better.
- **Cameras didn't move:** they hang from the ceiling (user).
- **Intrinsics:** the cam12/14 intrinsics are good (user).
- **Rough geometry:** the side cameras are roughly symmetric, up to about 1.5 m differences, and on one plane at
  ceiling height (user).
- **The stored extrinsics are exactly the documented chain.** World→cam12 = relative pose (Calibrator, w2c, from cam04)
  ∘ cam04's 4TU w2c pose (`calib_diag.py`).
  - The idiap relative files equal the Calibrator export
    (`parameters-camera-12-and14-relative-to-04.json`).
- **The Calibrator board fit is self-consistent and tight** (`board_diag.py`).
  - Board residuals at the stored poses: cam04 0.21 px, cam12 0.46 px, cam14 0.39 px. This also verifies that the
    conventions are read correctly: board pose = board→origin camera, camera transform = origin→camera.
  - The weakest direction of the board-residual Jacobian is a ~1° rotation traded against a ~8 cm shift, costing
    only 1.4 px (cam12) or 1.2 px (cam14). cam13/15 are similar (2.5–2.7 px), so the boards forbid large changes.
  - Coverage is sparse:
    - cam12: 12 board poses (10 shared with cam04), all in a ~2 × 1.4 m patch.
    - cam14: only 3 board poses (1 shared with cam04, 2 with cam12).
    - Compare cam13 and cam15: 23 and 17 poses, spread widely.
  - The board is 5×3 inner corners at 9 cm, ChArUco. Several identical boards were in use, with different markers
    that are not detectable at these distances.
- **Not a flip:** rotating or mirroring the image coordinates makes everything worse, all at 0 % within 40 px
  (`flip_test.py`). The raw videos have no rotation metadata.

## What was tried (all fail to give cam12/14 a pose that fits)
1. **Other chaining conventions** (inverse, chain through cam01/02/03, 12↔14 swap, relative pose as world): 6 % or
   less (cam12) and 11 % or less (cam14) within 40 px (`hypo.py`, `calib_diag.py`).
2. **People-only refit** (`scripts/cosilab_refit_extrinsic.py`): a coarse truncated cost, then Hungarian + RANSAC PnP,
   with per-pose visibility. `--passes 3` adds a ±20° yaw/pitch/roll restart grid.
   - **Local:**
     - cam12 reaches 0.25 within 40 px (centre moved 2.3 m, 9°).
     - cam14 reaches 0.41 (0.35 m, 9°).
     - The cam13 control stays put (0.2 m, 3°, 0.77 → 0.81).
     - Passes 2 and 3 add nothing.
   - **Global** (`--global_search`, a grid of look-at poses): fails its control and doesn't find cam13's pose.
3. **One shared anchor:** cam04's pose at calibration time, shared by cam12/14 with their relative poses fixed by the
   boards (`anchor_fit.py`).
   - Best fit: 12° rotation; cam12 0.18 and cam14 0.45 within 40 px.
   - cam04's own 4TU pose is consistent with the session (38 px against the cam01+cam03 bodies), and cam04 didn't move.
4. **Joint boards + people** (`joint_fit.py`): the board residuals are quadratic and weighted by `w_board`; the people
   get IRLS Cauchy weights at a 15 px scale. Sweep `w_board` 100 → 0.1.

   | | board ≤ 2.5–4 px (w_board ≥ 1) | board 20–33 px (board given up) |
   |---|---|---|
   | cam12 | people 0.07 (no change) | people ≤ 0.21 |
   | cam14 | people 0.20–0.22 (board ≤ 6 px) | people ≤ 0.37 |
   | cam13 control | people 0.77 → 0.81, board 4.2 → 7 px | – |

   No pose satisfies both: the boards and the people disagree for cam12/14.
5. **Own board detection** in the calibration frames (`detect_boards.py`, `board_fit.py`, `board_dbg.py`,
   `board_vs_project.py`), to triangulate boards with cams 01–04 and fit cam12/14 directly.
   - `findChessboardCornersSB(5×3)` locks onto shifted 5×3 patches made of the ChArUco marker patterns: usually
     only 1–5 of 15 corners match the Calibrator corners, and the rest are one square (10–20 px) off.
   - Top cameras agree on the board centre (within 5 cm) in only 1 of 1343 pairs. Unusable.

## Open hypothesis (not tested)
The board frames in `calib_frames/` were extracted from the timecode-synchronized videos (side calibration
README), i.e. the `timecode_fixed` ones the user says are wrong.
- **Mechanism:** if a frame number isn't simultaneous in cam04 and cam12/14 while the board is handheld and moving,
  the relative pose is biased.
- **Why the residuals stay small anyway:** there are few board poses, all in a small patch, and each board pose is
  free.
- **Size of the error:**
  - `timecode_fixed` − raw tag, per camera: cam01 +34, cam04 +40, cam12 +49, cam14 +60, cam13 +80, cam15 +75 frames.
  - Relative to cam04 that leaves cam12 9 frames and cam14 20 frames off.
  - cam13/15 vs cam01 would be 46/41 frames off, but they fit. Their board may have been held still, or their many
    spread-out poses average it out.
- **Cheap test:**
  1. Map the calibration frame numbers to raw-tag times per camera, with 14:47:14 + 30·i in timecode_fixed time.
  2. Measure how much the board moves in cam04 between those times (frame differencing on cam04's raw video around
     the board, numbers only).
- **Fix, if it is confirmed:** re-extract the cam04/12/14 calibration frames from the raw videos at the right times
  and redo the board corners (Calibrator, or track the project's corners into the re-timed frames), then redo the
  extrinsic fit.

## Scripts (run from the repo root inside the container, CPU)
| script | what |
|---|---|
| `scripts/cosilab_refit_extrinsic.py` | people-based refit (local / `--global_search` / `--passes`), per-pose visibility, scores |
| `scripts/cosilab_calib_12_14/calib_diag.py` | chaining conventions, centre-shift grid vs people |
| `scripts/cosilab_calib_12_14/hypo.py` | swap / other top cameras / relative pose as world |
| `scripts/cosilab_calib_12_14/board_diag.py` | Calibrator board residuals and pose sensitivity (Jacobian SVD) |
| `scripts/cosilab_calib_12_14/anchor_fit.py` | shared cam04 anchor for cam12 + cam14 |
| `scripts/cosilab_calib_12_14/joint_fit.py` | joint board + people fit, `w_board` sweep |
| `scripts/cosilab_calib_12_14/flip_test.py` | image flips vs the stored poses |
| `scripts/cosilab_calib_12_14/detect_boards.py`, `board_fit.py`, `board_dbg.py`, `board_vs_project.py` | own board detection + direct fit (failed, see 5) |
| `scripts/cosilab_calib_12_14/time_check.py` | ±5-frame timing check (flat) |
| `scripts/cosilab_calib_12_14/time_scan.py` | ±5 s timing scan (written, never completed; the user checked timing instead) |
| `scripts/cosilab_calib_12_14/tc_rule.py` | finds the timecode → 4TU CSV rounding rule |
| `scripts/cosilab_calib_12_14/prop_stats.py` | proposal box and timing stats per side camera |

`lib/utils/cosilab.load_camera(..., use_refit=True)` reads refit extrinsics for cams 12/14 from
`.../Video-OnlineHMR/cosilab_side_calibration/` (off by default; no refit is saved).
