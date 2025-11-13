from __future__ import annotations

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import LEGACY_EPISODES_STATS_PATH, STATS_PATH, write_info

from ..v20_to_v21.convert_dataset_v20_to_v21 import V20, V21
from ..v20_to_v21.convert_stats import convert_stats, check_aggregate_stats
from .utils import clone_tree


def convert_v20_to_v21_local(
    source_root: Path,
    dest_root: Path,
    *,
    num_workers: int = 4,
    overwrite: bool = False,
    inplace: bool = False,
) -> None:
    """Convert a local v2.0 dataset into a v2.1 dataset copy."""

    target_root = source_root if inplace else dest_root
    if not inplace:
        clone_tree(source_root, target_root, overwrite=overwrite)
    else:
        if dest_root != source_root:
            raise ValueError("When inplace=True, dest_root must be the same as source_root.")

    dataset = LeRobotDataset("local", root=target_root, revision=V20)

    ep_stats = target_root / LEGACY_EPISODES_STATS_PATH
    if ep_stats.is_file():
        ep_stats.unlink()

    convert_stats(dataset, num_workers=num_workers)
    ref_stats = dataset.meta.stats
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = V21
    write_info(dataset.meta.info, target_root)

    if (target_root / STATS_PATH).is_file():
        # Stats already updated; nothing to remove
        pass
