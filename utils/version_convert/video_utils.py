from pathlib import Path

import decord
import numpy as np
import torch
from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchcodec,
    decode_video_frames_torchvision,
    get_safe_default_codec,
)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    elif backend == "decord":
        return decode_video_frames_decord(video_path, timestamps)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_decord(
    video_path: Path | str,
    timestamps: list[float],
) -> torch.Tensor:
    video_path = str(video_path)
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
    indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
    frames = vr.get_batch(indices)

    frames_tensor = torch.tensor(frames.asnumpy()).type(torch.float32).permute(0, 3, 1, 2) / 255
    return frames_tensor
