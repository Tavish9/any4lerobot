from __future__ import annotations

from pathlib import Path

from lerobot.datasets.utils import load_info

from ..v30_to_v21.convert_dataset_v30_to_v21 import (
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
    copy_ancillary_directories,
    load_episode_records,
)
from .utils import ensure_destination


def _list_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def convert_v30_to_v21_local(
    source_root: Path,
    dest_root: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Convert a local v3.0 dataset into the legacy v2.1 layout."""

    ensure_destination(dest_root, overwrite=overwrite)
    dest_root.mkdir(parents=True, exist_ok=True)

    episode_records = load_episode_records(source_root)
    video_keys = _list_video_keys(source_root)

    convert_info(source_root, dest_root, episode_records, video_keys)
    convert_tasks(source_root, dest_root)
    convert_data(source_root, dest_root, episode_records)
    convert_videos(source_root, dest_root, episode_records, video_keys)
    convert_episodes_metadata(dest_root, episode_records)
    copy_ancillary_directories(source_root, dest_root)
