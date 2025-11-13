from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Iterable

import datasets
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import jsonlines

from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_TASKS_PATH,
    INFO_PATH,
    STATS_PATH,
    flatten_dict,
    load_json,
    unflatten_dict,
    write_json,
)
from lerobot.datasets.video_utils import get_video_info
from safetensors.torch import load_file

from .utils import ensure_destination

V16 = "v1.6"
V20 = "v2.0"
V1_INFO_PATH = "meta_data/info.json"
V1_STATS_PATH = "meta_data/stats.safetensors"
V1_VIDEO_FILE = "{video_key}_episode_{episode_index:06d}.mp4"


def _build_features(dataset: datasets.arrow_dataset.Dataset) -> dict:
    info = load_json((Path(dataset.cache_files[0]["filename"]).parents[2] / INFO_PATH))
    return info["features"]


def add_task_index_by_episodes(dataset: datasets.Dataset, tasks_by_episodes: dict) -> tuple[datasets.Dataset, list[str]]:
    df = dataset.to_pandas()
    tasks = list(set(tasks_by_episodes.values()))
    tasks_to_task_index = {task: task_idx for task_idx, task in enumerate(tasks)}
    episodes_to_task_index = {ep_idx: tasks_to_task_index[task] for ep_idx, task in tasks_by_episodes.items()}
    df["task_index"] = df["episode_index"].map(episodes_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = datasets.Dataset.from_pandas(df, features=features, split="train")
    return dataset, tasks


def add_task_index_from_tasks_col(dataset: datasets.Dataset, tasks_col: str):
    df = dataset.to_pandas()
    prefix_to_clean = "tf.Tensor(b'"
    suffix_to_clean = "', shape=(), dtype=string)"
    df[tasks_col] = df[tasks_col].str.removeprefix(prefix_to_clean).str.removesuffix(suffix_to_clean)

    tasks_by_episode = df.groupby("episode_index")[tasks_col].unique().apply(lambda x: x.tolist()).to_dict()
    tasks = df[tasks_col].unique().tolist()
    tasks_to_task_index = {task: idx for idx, task in enumerate(tasks)}
    df["task_index"] = df[tasks_col].map(tasks_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = datasets.Dataset.from_pandas(df, features=features, split="train")
    dataset = dataset.remove_columns(tasks_col)

    return dataset, tasks, tasks_by_episode


def split_parquet_by_episodes(dataset: datasets.Dataset, total_episodes: int, total_chunks: int, output_dir: Path):
    table = dataset.data.table
    episode_lengths: list[int] = []
    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)
        chunk_dir = "/".join(DEFAULT_DATA_PATH.split("/")[:-1]).format(episode_chunk=ep_chunk)
        (output_dir / chunk_dir).mkdir(parents=True, exist_ok=True)
        for ep_idx in range(ep_chunk_start, ep_chunk_end):
            ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
            episode_lengths.insert(ep_idx, len(ep_table))
            output_file = output_dir / DEFAULT_DATA_PATH.format(episode_chunk=ep_chunk, episode_index=ep_idx)
            pq.write_table(ep_table, output_file)

    return episode_lengths


def _default_video_dirs(source_root: Path) -> Iterable[Path]:
    for folder in sorted(source_root.glob("videos*")):
        if folder.is_dir():
            yield folder


def _build_video_info(source_root: Path, video_key: str) -> dict:
    for folder in _default_video_dirs(source_root):
        sample = folder / V1_VIDEO_FILE.format(video_key=video_key, episode_index=0)
        if sample.exists():
            return get_video_info(sample)
    raise FileNotFoundError(f"Unable to locate sample video for key '{video_key}'")


def _copy_videos(source_root: Path, dest_root: Path, video_keys: list[str], total_episodes: int) -> dict[str, dict]:
    videos_info: dict[str, dict] = {}
    for video_key in video_keys:
        videos_info[video_key] = _build_video_info(source_root, video_key)

    for vid_key in video_keys:
        for episode_index in range(total_episodes):
            filename = V1_VIDEO_FILE.format(video_key=vid_key, episode_index=episode_index)
            source_path = None
            for folder in _default_video_dirs(source_root):
                candidate = folder / filename
                if candidate.exists():
                    source_path = candidate
                    break
            if source_path is None:
                raise FileNotFoundError(f"Video file '{filename}' not found for key '{vid_key}'")

            chunk_index = episode_index // DEFAULT_CHUNK_SIZE
            dest_path = dest_root / DEFAULT_VIDEO_PATH.format(
                episode_chunk=chunk_index,
                video_key=vid_key,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)

    return videos_info


def convert_v16_to_v20_local(
    source_root: Path,
    dest_root: Path,
    *,
    single_task: str | None = None,
    tasks_path: Path | None = None,
    tasks_col: str | None = None,
    robot_config=None,
    overwrite: bool = False,
) -> None:
    """Convert a local v1.6 dataset directory into a v2.0 layout."""

    ensure_destination(dest_root, overwrite=overwrite)
    dest_root.mkdir(parents=True, exist_ok=True)

    metadata_v1 = load_json(source_root / V1_INFO_PATH)
    dataset = datasets.load_dataset("parquet", data_dir=source_root / "data", split="train")
    features = _build_features(dataset, robot_config)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]

    if single_task and "language_instruction" in dataset.column_names:
        logging.warning("'single_task' provided but language instructions already exist; using column instead.")
        single_task = None
        tasks_col = "language_instruction"

    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    total_chunks = math.ceil(total_episodes / DEFAULT_CHUNK_SIZE) if total_episodes else 0

    if single_task:
        tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_path:
        tasks_by_episodes = load_json(tasks_path)
        tasks_by_episodes = {int(ep_idx): task for ep_idx, task in tasks_by_episodes.items()}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_col:
        dataset, tasks, tasks_by_episodes = add_task_index_from_tasks_col(dataset, tasks_col)
    else:
        raise ValueError("One of single_task, tasks_path, or tasks_col must be provided")

    tasks_json = [{"task_index": idx, "task": task} for idx, task in enumerate(tasks)]
    with jsonlines.open(dest_root / LEGACY_TASKS_PATH, mode="w") as writer:
        for record in tasks_json:
            writer.write(record)

    if video_keys:
        dataset = dataset.remove_columns(video_keys)
        videos_info = _copy_videos(source_root, dest_root, video_keys, total_episodes)
        for key in video_keys:
            ft = features[key]
            ft["shape"] = (
                videos_info[key]["video.height"],
                videos_info[key]["video.width"],
                videos_info[key]["video.channels"],
            )
            ft["video_info"] = videos_info[key]
    else:
        videos_info = None

    episode_lengths = split_parquet_by_episodes(dataset, total_episodes, total_chunks, dest_root)
    episodes = [
        {"episode_index": ep_idx, "tasks": tasks_by_episodes[ep_idx], "length": episode_lengths[ep_idx]}
        for ep_idx in episode_indices
    ]
    with jsonlines.open(dest_root / LEGACY_EPISODES_PATH, mode="w") as writer:
        for record in episodes:
            writer.write(record)

    metadata_v2 = {
        "codebase_version": V20,
        "robot_type": metadata_v1.get("robot_type", "unknown"),
        "total_episodes": total_episodes,
        "total_frames": len(dataset),
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": metadata_v1.get("fps"),
        "splits": metadata_v1.get("splits", {"train": f"0:{total_episodes}"}),
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,
    }
    write_json(metadata_v2, dest_root / INFO_PATH)

    convert_stats_to_json(source_root, dest_root)


def convert_stats_to_json(source_root: Path, dest_root: Path) -> None:
    safetensor_path = source_root / V1_STATS_PATH
    if not safetensor_path.exists():
        raise FileNotFoundError(f"Missing legacy stats file at {safetensor_path}")

    stats = load_file(safetensor_path)
    serialized = {key: value.tolist() for key, value in stats.items()}
    serialized = unflatten_dict(serialized)

    json_path = dest_root / STATS_PATH
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=4)

    with open(json_path, encoding="utf-8") as f:
        stats_json = json.load(f)
    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])
