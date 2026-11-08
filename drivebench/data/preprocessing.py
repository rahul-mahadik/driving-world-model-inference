"""Image-sequence preprocessing matching pinned Vista sample.load_img."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
from PIL import Image


PREPROCESS_VERSION = "vista-sample-rgb-center-crop-lanczos-minus1plus1-v1"


def read_preprocess(observed_frames: Sequence[str | Path], height: int, width: int) -> np.ndarray:
    """Decode only supplied observed images to contiguous float32 [T,C,H,W].

    Integer crop rounding, RGB conversion order, Lanczos resize and scaling match
    Vista sample.py's load_img. Callers must pass observed paths, never a clip's
    complete frame list. Resolution bucket constraints belong to the model.
    """
    if height < 1 or width < 1 or not observed_frames:
        raise ValueError("Positive dimensions and at least one observed frame are required")
    frames = []
    for path in observed_frames:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            original_width, original_height = image.size
            if original_width / original_height > width / height:
                cropped_width = int(width / height * original_height)
                image = image.crop(((original_width - cropped_width) // 2, 0,
                                    (original_width + cropped_width) // 2, original_height))
            elif original_width / original_height < width / height:
                cropped_height = int(height / width * original_width)
                image = image.crop((0, (original_height - cropped_height) // 2,
                                    original_width, (original_height + cropped_height) // 2))
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
            frame = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
            frame = frame / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
            frames.append(frame)
    return np.ascontiguousarray(np.stack(frames), dtype=np.float32)
