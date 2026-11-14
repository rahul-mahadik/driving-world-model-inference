"""Array-backed proxy distances for the trusted evaluator, never agent code.

No thresholds are selected here. LPIPS is explicitly supplied by the trusted
runner and its asset/source identity must be frozen in the scoring policy.
Missing or invalid perceptual measurements fail closed. Only future frames are
compared. These distances do not establish driving safety or visual acceptance.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from .vista_agent import file_hash


METRIC_DEFINITION = {'version': 1, 'frame_selection': 'exclude_conditioning_prefix',
                     'mae': 'mean_absolute_RGB_error',
                     'temporal_derivative_mae': 'mean_absolute_error_of_first_temporal_differences',
                     'lpips': 'mean_framewise_alex_LPIPS; frozen_weights_and_version_required',
                     'quality_preserving_claim': False}


def measure_pair(reference_path, candidate_path, *, reference_sha256, candidate_sha256,
                 conditioning_frames=1, perceptual_metric):
    from drivebench.evaluation.metrics import compare_videos
    if type(conditioning_frames) is not int or conditioning_frames < 0:
        raise ValueError('invalid conditioning frame count')
    paths = [Path(reference_path), Path(candidate_path)]
    expected = [reference_sha256, candidate_sha256]
    if any(path.is_symlink() or not path.is_file() or file_hash(path) != sha for path, sha in zip(paths, expected)):
        raise ValueError('raw output hash mismatch')
    arrays = [np.load(path, mmap_mode='r', allow_pickle=False) for path in paths]
    if any(array.dtype != np.float32 or array.ndim != 4 or array.shape[0] < conditioning_frames + 2 for array in arrays):
        raise ValueError('float32 RGB future sequences with >=2 future frames required')
    future = [array[conditioning_frames:] for array in arrays]
    compared = compare_videos(*future)
    lpips = perceptual_metric.compare(*future)
    if type(lpips) not in (float, int) or not math.isfinite(lpips) or lpips < 0:
        raise ValueError('perceptual metric unavailable or invalid; no fallback score')
    if any(file_hash(path) != sha for path, sha in zip(paths, expected)):
        raise ValueError('raw output changed during metric execution')
    return {'mae': compared['mae'], 'temporal_derivative_mae': compared['temporal_derivative_mae'],
            'lpips': float(lpips)}


def negative_control(video, control_id):
    """Prespecified corruptions; references must already be validated float32 RGB.

    Controls are scored on development data only and preserve the conditioning
    frame. Store control arrays and hashes before measuring them. A control that
    the proposed gate misses makes calibration inconclusive, not acceptable.
    """
    from drivebench.evaluation.metrics import as_video
    value = as_video(video).copy()
    if value.shape[0] < 3:
        raise ValueError('at least two future frames required')
    if control_id == 'freeze_future':
        value[1:] = value[1]
    elif control_id == 'black_future':
        value[1:] = 0
    elif control_id == 'reverse_future':
        value[1:] = value[1:][::-1]
    else:
        raise ValueError('unknown prespecified negative control')
    return value
