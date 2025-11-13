from __future__ import annotations

from pathlib import Path

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import load_info, write_info, write_stats

from ..v21_to_v20_filter_v30_fields.convert_dataset_v21_to_v20 import (
    V20,
    V21,
    _format_stats_for_v20,
    _remove_episode_stats_files,
    _resolve_episodes_stats,
)
from .utils import clone_tree


def convert_v21_to_v20_filtered_local(
    source_root: Path,
    dest_root: Path,
    *,
    delete_episode_stats: bool = False,
    overwrite: bool = False,
    inplace: bool = False,
) -> None:
    """Convert a v2.1 dataset to v2.0 using the filtered stats pipeline."""

    target_root = source_root if inplace else dest_root
    if not inplace:
        clone_tree(source_root, target_root, overwrite=overwrite)
    else:
        if dest_root != source_root:
            raise ValueError("When inplace=True, dest_root must equal source_root.")

    info = load_info(target_root)
    version = info.get("codebase_version")
    if version != V21:
        raise ValueError(f"Expected source dataset to be '{V21}', found '{version}'")

    episodes_stats = _resolve_episodes_stats(target_root)
    if not episodes_stats:
        raise RuntimeError("No per-episode stats available; cannot aggregate stats.json")

    aggregated = aggregate_stats(list(episodes_stats.values()))
    legacy_stats = _format_stats_for_v20(aggregated)
    write_stats(legacy_stats, target_root)

    info["codebase_version"] = V20
    write_info(info, target_root)

    if delete_episode_stats:
        _remove_episode_stats_files(target_root)
