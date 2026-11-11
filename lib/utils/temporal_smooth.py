"""
Zero-phase temporal smoothing of SMPL bodies, ported from sam-body4d Stage 3 (smoothing/temporal.py; Notion
"SAM4D for keypoints", Step 1). Per run of consecutive frames:

  body joint rotations   Hampel filter (isolated spikes -> local median), then Gaussian of sigma_pose frames,
                         on sign-aligned unit quaternions per joint, renormalised
  root rotation (world)  the same with sigma_rot
  translation (world)    the same on xyz with sigma_trans (sam-body4d smooths depth and image ray separately to
                         keep one camera's image position; a multi-view world-frame body has no single camera)

At ~60 fps a Gaussian of sigma frames keeps motion up to ~8/sigma Hz: sigma 2 ~4 Hz (gestures), 3 ~2.7 Hz, 4 ~2 Hz.
"""
from dataclasses import dataclass

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d

from lib.utils.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


@dataclass
class SmoothConfig:
    sigma_pose: float = 2.0  # body joint rotations, frames
    sigma_rot: float = 3.0  # root rotation
    sigma_trans: float = 4.0  # translation
    hampel_half_window: int = 3  # spike window 2h+1 frames
    hampel_n_sigma: float = 3.0  # spike: further than this many robust stds from the local median


def runs(frames):
    """(start, end) index pairs of the runs of consecutive frame ids in sorted `frames`."""
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(frames)]])
    return list(zip(bounds[:-1], bounds[1:]))


def hampel(x, half_window, n_sigma):
    """Replace samples further than n_sigma robust stds from the local median by that median, per dimension. x: (L, D)."""
    L = x.shape[0]
    if half_window <= 0 or L < 2 * half_window + 1:
        return x
    padded = np.pad(x, ((half_window, half_window), (0, 0)), mode='edge')
    win = sliding_window_view(padded, 2 * half_window + 1, axis=0)  # (L, D, w)
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    out = x.copy()
    spike = np.abs(x - med) > n_sigma * np.maximum(mad, 1e-9)
    out[spike] = med[spike]
    return out


def lowpass(x, sigma, cfg):
    """Hampel, then a zero-phase Gaussian along axis 0. x: (L, D)."""
    x = hampel(x, cfg.hampel_half_window, cfg.hampel_n_sigma)
    return gaussian_filter1d(x, sigma, axis=0, mode='nearest') if sigma > 0 else x


def smooth_rotmats(R, sigma, cfg):
    """(L, J, 3, 3) rotations, each joint smoothed as sign-aligned unit quaternions."""
    L, J = R.shape[:2]
    q = matrix_to_quaternion(torch.as_tensor(R, dtype=torch.float64)).numpy()  # (L, J, 4)
    flip = np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1.0, 1.0)  # (L-1, J)
    q[1:] *= np.cumprod(flip, axis=0)[..., None]
    q = lowpass(q.reshape(L, J * 4), sigma, cfg).reshape(L, J, 4)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return quaternion_to_matrix(torch.from_numpy(q)).numpy()


def smooth_body(rotmat, trans, frames, cfg):
    """
    Smoothed copies of a body's rotations (T, 24, 3, 3) (root first) and translation (T, 3), torch tensors, per run of
    consecutive `frames`. Runs shorter than 2 frames are left as they are.
    """
    R = rotmat.detach().cpu().double().numpy().copy()
    t = trans.detach().cpu().double().numpy().copy()
    for s, e in runs(frames):
        if e - s < 2:
            continue
        R[s:e, :1] = smooth_rotmats(R[s:e, :1], cfg.sigma_rot, cfg)
        R[s:e, 1:] = smooth_rotmats(R[s:e, 1:], cfg.sigma_pose, cfg)
        t[s:e] = lowpass(t[s:e], cfg.sigma_trans, cfg)
    return (torch.tensor(R, dtype=rotmat.dtype, device=rotmat.device),
            torch.tensor(t, dtype=trans.dtype, device=trans.device))


def gaussian_lowpass_torch(x, sigma):
    """Zero-phase Gaussian along dim 0 of (T, ...) x, replicate padding, differentiable (no Hampel)."""
    if sigma <= 0 or x.shape[0] < 2:
        return x
    r = int(4 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    shape = x.shape
    y = x.reshape(shape[0], -1).T[:, None]  # (D, 1, T)
    y = torch.nn.functional.pad(y, (r, r), mode='replicate')
    return torch.nn.functional.conv1d(y, k)[:, 0].T.reshape(shape)
