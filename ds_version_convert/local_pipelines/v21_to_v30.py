from __future__ import annotations

from pathlib import Path

from ..v21_to_v30.convert_dataset_v21_to_v30 import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_info,
)
from .utils import ensure_destination


def convert_v21_to_v30_local(
    source_root: Path,
    dest_root: Path,
    *,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    overwrite: bool = False,
) -> None:
    """Convert a local v2.1 dataset into the v3.0 layout."""

    ensure_destination(dest_root, overwrite=overwrite)
    dest_root.mkdir(parents=True, exist_ok=True)

    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    convert_info(source_root, dest_root, data_file_size_in_mb, video_file_size_in_mb)
    convert_tasks(source_root, dest_root)
    episodes_metadata = convert_data(source_root, dest_root, data_file_size_in_mb)
    episodes_videos_metadata = convert_videos(source_root, dest_root, video_file_size_in_mb)
    convert_episodes_metadata(source_root, dest_root, episodes_metadata, episodes_videos_metadata)
