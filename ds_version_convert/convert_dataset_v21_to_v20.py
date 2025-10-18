"""
This script converts a LeRobot dataset from codebase version 2.1 back to 2.0. It reverses the
changes introduced in ``convert_dataset_v20_to_v21.py`` by aggregating the per-episode statistics
into the legacy ``meta/stats.json`` file and updating the dataset metadata accordingly.

The script always downloads a fresh copy of the dataset metadata from the Hub into a user-provided
output directory before applying the conversion locally, and it never pushes updates back to the
Hub.

Typical usage:

```bash
python ds_version_convert/convert_dataset_v21_to_v20.py \
    --repo-id=aliberts/koch_tutorial \
    --output-dir=/tmp/lerobot_datasets \
    --delete-episodes-stats
```
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import cast_stats_to_numpy, load_info, write_info, write_stats

try:  # pragma: no cover - numpy is expected but we guard for minimal setups
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:  # pragma: no cover - compatibility with older package structure
    from lerobot.datasets.v21.convert_dataset_v20_to_v21 import V20, V21
except ImportError:  # pragma: no cover
    V20 = "v2.0"
    V21 = "v2.1"

try:  # pragma: no cover - new codebase renamed the constant
    from lerobot.datasets.utils import EPISODES_STATS_PATH
except ImportError:  # pragma: no cover
    try:
        from lerobot.datasets.utils import LEGACY_EPISODES_STATS_PATH as EPISODES_STATS_PATH
    except ImportError:
        EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"


def _load_episodes_stats_from_file(root: Path) -> dict[int, dict[str, Any]]:
    """Load per-episode stats from ``meta/episodes_stats.jsonl``.

    This is a lightweight fallback when ``LeRobotDatasetMetadata`` does not expose
    ``episodes_stats`` directly (e.g. on newer codebase versions).
    """

    path = root / EPISODES_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot locate '{EPISODES_STATS_PATH}' under '{root}'. The dataset must contain per-episode stats."
        )

    episodes_stats: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ep_index = int(record["episode_index"])
            episodes_stats[ep_index] = cast_stats_to_numpy(record["stats"])

    if not episodes_stats:
        raise ValueError(f"'{EPISODES_STATS_PATH}' is empty; cannot build legacy stats.json.")

    return episodes_stats


def _resolve_episodes_stats(root: Path) -> dict[int, dict[str, Any]]:
    # Try to rely on helper available in older package versions.
    try:  # pragma: no cover
        from lerobot.datasets.utils import load_episodes_stats as _load_episodes_stats
    except ImportError:
        _load_episodes_stats = None

    if _load_episodes_stats is not None:
        try:
            loaded = _load_episodes_stats(root)
        except (FileNotFoundError, OSError):
            loaded = {}
        if loaded:
            return loaded

    return _load_episodes_stats_from_file(root)


def _remove_episode_stats_files(root: Path) -> None:
    path = root / EPISODES_STATS_PATH
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


LEGACY_STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def _to_python_scalar(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, np.ndarray):
            return _to_python_scalar(value.tolist())

    if isinstance(value, (list, tuple)):
        return [_to_python_scalar(v) for v in value]

    if isinstance(value, dict):
        return {k: _to_python_scalar(v) for k, v in value.items()}

    if isinstance(value, (int, float)):
        return float(value)

    return value


def _format_stat_value(value: Any) -> Any:
    converted = _to_python_scalar(value)
    if isinstance(converted, list):
        return converted
    return [converted]


def _format_stats_for_v20(aggregated_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    legacy_stats: dict[str, dict[str, Any]] = {}
    for feature_name, stats in aggregated_stats.items():
        if not isinstance(stats, dict):
            continue

        formatted_stats = {}
        for key in LEGACY_STATS_KEYS:
            if key not in stats:
                continue
            formatted_stats[key] = _format_stat_value(stats[key])

        if formatted_stats:
            legacy_stats[feature_name] = formatted_stats

    return legacy_stats


def _prepare_dataset_root(repo_id: str, output_dir: str) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_root = output_path / repo_id.replace("/", "-")
    if dataset_root.exists():
        logging.info("Removing existing directory %s", dataset_root)
        shutil.rmtree(dataset_root)

    return dataset_root


def convert_dataset(
    repo_id: str,
    output_dir: str,
    delete_episodes_stats: bool = False,
    download_media: bool = False,
) -> None:
    dataset_root = _prepare_dataset_root(repo_id, output_dir)

    allow_patterns = ["meta/**"]
    if download_media:
        allow_patterns.extend(["data/**", "videos/**", "images/**"])

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    info = load_info(dataset_root)
    info_version = info.get("codebase_version")
    if info_version != V21:
        raise ValueError(
            f"Dataset '{repo_id}' is marked as '{info_version}', expected '{V21}'. "
            "This script only handles conversions from v2.1 to v2.0."
        )

    episodes_stats = _resolve_episodes_stats(dataset_root)
    if not episodes_stats:
        raise RuntimeError("No per-episode stats found; cannot reconstruct legacy stats.json")

    aggregated_stats = aggregate_stats(list(episodes_stats.values()))
    legacy_formatted_stats = _format_stats_for_v20(aggregated_stats)
    write_stats(legacy_formatted_stats, dataset_root)

    info["codebase_version"] = V20
    write_info(info, dataset_root)

    if delete_episodes_stats:
        _remove_episode_stats_files(dataset_root)

    logging.info("Conversion completed. Output available at %s", dataset_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Directory where the dataset will be downloaded and converted. Existing data in "
            "`<output-dir>/<repo-id>` is removed before download."
        ),
    )
    parser.add_argument(
        "--delete-episodes-stats",
        action="store_true",
        help="Remove the per-episode stats file after generating the legacy stats.json.",
    )
    parser.add_argument(
        "--download-media",
        action="store_true",
        help="Also download episode data (parquet), videos, and images alongside metadata.",
    )
    args = parser.parse_args()
    convert_dataset(**vars(args))
