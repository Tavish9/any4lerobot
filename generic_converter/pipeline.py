import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import get_random_str, get_timestamp
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

from .adapter import BaseAdapter
from .utils import (
    ConversionTask,
    setup_logger,
    unique_strings,
)


class SaveLeRobotDataset(PipelineStep):
    name = "Save Temp LeRobotDataset"

    def __init__(self, tasks: list[ConversionTask], adapter: BaseAdapter):
        super().__init__()
        self.tasks = tasks
        self.adapter = adapter
        self.type = f"{adapter.dataset_type}2lerobot"

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = setup_logger()
        task = self.tasks[rank]

        if task.output_path.exists():
            shutil.rmtree(task.output_path)

        dataset = self.adapter.create_dataset(task)

        logger.info(
            f"start processing for {task.input_path}, saving to {task.output_path}"
        )
        raw_dataset = self.adapter.load_subset(task)
        saved_episodes = 0
        for episode_index, episode_data in enumerate(raw_dataset):
            with self.track_time("saving episode"):
                saved = self.adapter.save_episode(
                    dataset,
                    episode_data,
                    task,
                )
                status = "skipped" if saved is False else "process done"
                logger.info(
                    f"{status} for {dataset.repo_id}, episode {episode_index}, "
                    f"len {self.adapter.get_episode_length(episode_data)}"
                )
                if saved is not False:
                    saved_episodes += 1
        dataset.finalize()
        if saved_episodes == 0:
            logger.info(
                f"no episodes saved for {dataset.repo_id}; deleting temp output"
            )
            shutil.rmtree(task.output_path, ignore_errors=True)


def run_converter(
    adapter: BaseAdapter,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    resume_dir: str | None = None,
    debug: bool = False,
    local_repo_id: str | None = None,
    hub_repo_id: str | None = None,
    push_to_hub: bool = False,
    cleanup_temp: bool = True,
    extra_tags: Sequence[str] | None = None,
) -> Path:
    tasks = adapter.load_tasks()
    output_path = adapter.output_path

    if not tasks:
        raise ValueError(
            "No conversion tasks found. Provide a non-empty tasks file or matching source files."
        )
    if cpus_per_task < 1:
        raise ValueError("--cpus-per-task must be >= 1")

    output_path.mkdir(parents=True, exist_ok=True)

    if debug:
        executor = "local"
        workers = 1
        tasks = tasks[:2]
        push_to_hub = False

    match executor:
        case "local":
            from datatrove.executor import LocalPipelineExecutor

            resolved_workers = (
                max(1, (os.cpu_count() or 1) // cpus_per_task)
                if workers == -1
                else workers
            )
            executor_cls, executor_config = (
                LocalPipelineExecutor,
                {
                    "tasks": len(tasks),
                    "workers": resolved_workers,
                },
            )
        case "ray":
            import ray
            from datatrove.executor import RayPipelineExecutor
            from ray.runtime_env import RuntimeEnv

            runtime_env = RuntimeEnv(env_vars=_build_ray_env_vars())
            ray.init(runtime_env=runtime_env)
            executor_cls, executor_config = (
                RayPipelineExecutor,
                {
                    "tasks": len(tasks),
                    "workers": workers,
                    "cpus_per_task": cpus_per_task,
                    "tasks_per_job": tasks_per_job,
                },
            )
        case _:
            raise ValueError(f"Executor {executor} not supported")

    if resume_dir:
        logging_dir = str(resume_dir)
    else:
        logging_dir = str(Path.cwd() / "logs" / f"{get_timestamp()}_{get_random_str()}")

    executor_cls(
        pipeline=[SaveLeRobotDataset(tasks, adapter)],
        **executor_config,
        logging_dir=logging_dir,
    ).run()
    aggregate_tasks(
        tasks,
        output_path,
        aggr_repo_id=local_repo_id,
    )

    if cleanup_temp:
        logger = setup_logger()
        logger.info("Delete temp data_dir")
        shutil.rmtree(adapter.temp_output_path, ignore_errors=True)

    if push_to_hub:
        if hub_repo_id is None:
            raise ValueError("--repo-id is required when --push-to-hub is set")

        tags = unique_strings(
            [
                "LeRobot",
                adapter.dataset_type,
                adapter.robot_type,
                *adapter.tags,
                *(extra_tags or []),
            ]
        )
        LeRobotDataset(
            repo_id=hub_repo_id,
            root=output_path,
        ).push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=False,
        )

    return output_path


def _build_ray_env_vars() -> dict[str, str]:
    env_vars = {
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
        "SVT_LOG": "1",
    }
    pythonpath = _build_ray_pythonpath()
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath
    return env_vars


def _build_ray_pythonpath() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    paths: list[str] = []

    def add_path(path_value: str | Path):
        path = Path(path_value).expanduser()
        try:
            path = path.resolve()
        except OSError:
            return
        if not path.exists():
            return
        path_str = str(path)
        if path_str not in paths:
            paths.append(path_str)

    add_path(repo_root)
    add_path(Path.cwd())
    for path in sys.path:
        if path:
            add_path(path)
    for path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if path:
            add_path(path)

    return os.pathsep.join(paths)


def aggregate_tasks(
    tasks: list[ConversionTask],
    output_dir: Path,
    aggr_repo_id: str | None = None,
):
    logger = setup_logger()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    roots = [task.output_path for task in tasks if task.output_path.exists()]
    if not roots:
        raise ValueError("No temporary datasets were produced; nothing to aggregate.")

    resolved_aggr_repo_id = aggr_repo_id or output_dir.name

    logger.info(
        f"aggregate {len(roots)} temporary datasets into {output_dir} as {resolved_aggr_repo_id}"
    )
    _aggregate_datasets_with_normalized_arrays(
        repo_ids=[None] * len(roots),
        roots=roots,
        aggr_repo_id=resolved_aggr_repo_id,
        aggr_root=output_dir,
    )
    logger.info(f"aggregation complete: {output_dir}")


def _aggregate_datasets_with_normalized_arrays(**kwargs) -> None:
    from lerobot.datasets import aggregate as aggregate_module

    original_aggregate_videos = aggregate_module.aggregate_videos
    original_read_parquet = aggregate_module.pd.read_parquet
    original_writer = aggregate_module.to_parquet_one_row_group_per_episode
    original_update_meta_data = aggregate_module.update_meta_data

    def read_normalized_arrays(*args, **kwargs):
        return _normalize_array_values(original_read_parquet(*args, **kwargs))

    def write_normalized_arrays(df, path):
        return original_writer(_normalize_array_values(df), path)

    aggregate_module.aggregate_videos = _aggregate_videos_by_key_parallel
    aggregate_module.pd.read_parquet = read_normalized_arrays
    aggregate_module.to_parquet_one_row_group_per_episode = write_normalized_arrays
    aggregate_module.update_meta_data = _update_meta_data_without_fragmenting
    try:
        aggregate_datasets(**kwargs)
    finally:
        aggregate_module.aggregate_videos = original_aggregate_videos
        aggregate_module.pd.read_parquet = original_read_parquet
        aggregate_module.to_parquet_one_row_group_per_episode = original_writer
        aggregate_module.update_meta_data = original_update_meta_data


def _aggregate_videos_by_key_parallel(
    src_meta,
    dst_meta,
    videos_idx,
    video_files_size_in_mb,
    chunk_size,
    concatenate_videos=True,
):
    from concurrent.futures import ThreadPoolExecutor

    for video_idx in videos_idx.values():
        video_idx["episode_duration"] = 0
        video_idx["src_to_offset"] = {}
        video_idx["src_to_dst"] = {}
        if "dst_file_durations" not in video_idx:
            video_idx["dst_file_durations"] = {}

    def aggregate_key(key):
        return (
            key,
            _aggregate_video_key(
                key,
                src_meta,
                dst_meta,
                videos_idx[key],
                video_files_size_in_mb,
                chunk_size,
                concatenate_videos,
            ),
        )

    keys = list(videos_idx)
    if not keys:
        return videos_idx

    max_workers = min(len(keys), os.cpu_count() or len(keys))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, video_idx in executor.map(aggregate_key, keys):
            videos_idx[key] = video_idx

    return videos_idx


def _aggregate_video_key(
    key,
    src_meta,
    dst_meta,
    video_idx,
    video_files_size_in_mb,
    chunk_size,
    concatenate_videos,
):
    from lerobot.datasets import aggregate as aggregate_module

    unique_chunk_file_pairs = {
        (chunk, file)
        for chunk, file in zip(
            src_meta.episodes[f"videos/{key}/chunk_index"],
            src_meta.episodes[f"videos/{key}/file_index"],
            strict=False,
        )
    }
    unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

    chunk_idx = video_idx["chunk"]
    file_idx = video_idx["file"]
    dst_file_durations = video_idx["dst_file_durations"]

    for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
        src_path = src_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
            video_key=key,
            chunk_index=src_chunk_idx,
            file_index=src_file_idx,
        )
        dst_path = dst_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
            video_key=key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

        src_duration = aggregate_module.get_video_duration_in_s(src_path)
        dst_key = (chunk_idx, file_idx)

        if not dst_path.exists():
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_path), str(dst_path))
            dst_file_durations[dst_key] = src_duration
            video_idx["episode_duration"] += src_duration
            continue

        src_size = aggregate_module.get_file_size_in_mb(src_path)
        dst_size = aggregate_module.get_file_size_in_mb(dst_path)

        if not concatenate_videos or dst_size + src_size >= video_files_size_in_mb:
            chunk_idx, file_idx = aggregate_module.update_chunk_file_indices(
                chunk_idx, file_idx, chunk_size
            )
            dst_key = (chunk_idx, file_idx)
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            dst_path = dst_meta.root / aggregate_module.DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_path), str(dst_path))
            dst_file_durations[dst_key] = src_duration
        else:
            current_dst_duration = dst_file_durations.get(dst_key, 0)
            video_idx["src_to_offset"][(src_chunk_idx, src_file_idx)] = (
                current_dst_duration
            )
            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
            aggregate_module.concatenate_video_files(
                [dst_path, src_path],
                dst_path,
                compatibility_check=True,
            )
            dst_file_durations[dst_key] = current_dst_duration + src_duration

        video_idx["episode_duration"] += src_duration

    video_idx["chunk"] = chunk_idx
    video_idx["file"] = file_idx

    return video_idx


def _update_meta_data_without_fragmenting(df, dst_meta, meta_idx, data_idx, videos_idx):
    import pandas as pd

    df["meta/episodes/chunk_index"] = (
        df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    )
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]

    data_src_to_dst = data_idx.get("src_to_dst", {})
    if data_src_to_dst:
        orig_data_chunk = df["data/chunk_index"].copy()
        orig_data_file = df["data/file_index"].copy()
        mapping_index = pd.MultiIndex.from_tuples(
            list(data_src_to_dst.keys()),
            names=["chunk_index", "file_index"],
        )
        mapping_df = pd.DataFrame(
            list(data_src_to_dst.values()),
            index=mapping_index,
            columns=["dst_chunk", "dst_file"],
        )
        row_index = pd.MultiIndex.from_arrays(
            [orig_data_chunk, orig_data_file],
            names=["chunk_index", "file_index"],
        )
        reindexed = mapping_df.reindex(row_index)
        reindexed[["dst_chunk", "dst_file"]] = reindexed[
            ["dst_chunk", "dst_file"]
        ].fillna({"dst_chunk": data_idx["chunk"], "dst_file": data_idx["file"]})
        df["data/chunk_index"] = reindexed["dst_chunk"].to_numpy()
        df["data/file_index"] = reindexed["dst_file"].to_numpy()
    else:
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]

    for key, video_idx in videos_idx.items():
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        orig_chunks = df[orig_chunk_col].copy()
        orig_files = df[orig_file_col].copy()

        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})
        row_index = pd.MultiIndex.from_arrays(
            [orig_chunks, orig_files],
            names=["chunk_index", "file_index"],
        )

        if src_to_dst:
            src_keys = list(src_to_dst)
            mapping_index = pd.MultiIndex.from_tuples(
                src_keys,
                names=["chunk_index", "file_index"],
            )
            mapping_df = pd.DataFrame(
                [
                    (
                        *src_to_dst[src_key],
                        src_to_offset.get(src_key, 0.0),
                    )
                    for src_key in src_keys
                ],
                index=mapping_index,
                columns=["dst_chunk", "dst_file", "offset"],
            )
            reindexed = mapping_df.reindex(row_index)
            df[orig_chunk_col] = (
                reindexed["dst_chunk"]
                .fillna(video_idx["chunk"])
                .astype(orig_chunks.dtype, copy=False)
                .to_numpy()
            )
            df[orig_file_col] = (
                reindexed["dst_file"]
                .fillna(video_idx["file"])
                .astype(orig_files.dtype, copy=False)
                .to_numpy()
            )
            offsets = reindexed["offset"].fillna(0.0).to_numpy(dtype=float)
            df[f"videos/{key}/from_timestamp"] += offsets
            df[f"videos/{key}/to_timestamp"] += offsets
        elif src_to_offset:
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            mapping_series = pd.Series(src_to_offset, dtype=float)
            offsets = mapping_series.reindex(row_index).fillna(0.0).to_numpy()
            df[f"videos/{key}/from_timestamp"] += offsets
            df[f"videos/{key}/to_timestamp"] += offsets
        else:
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = (
                df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]
            )

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info.total_frames
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info.total_frames
    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes

    return df


def _normalize_array_values(df):
    import pandas as pd

    df = df.copy()
    for column in df.columns:
        if _has_array_values(df[column]):
            df[column] = pd.Series(
                [_normalize_array_value(value) for value in df[column]],
                dtype=object,
                index=df.index,
            )
    return df


def _normalize_array_value(value):
    import numpy as np

    if isinstance(value, np.ndarray) and value.ndim > 1:
        return [_normalize_array_value(item) for item in value]
    return value


def _has_array_values(series) -> bool:
    import numpy as np

    for value in series.head(32):
        if isinstance(value, np.ndarray):
            return True
    return False
