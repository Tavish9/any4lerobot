import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import ray
from datatrove.executor import LocalPipelineExecutor, RayPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from lerobot.datasets.aggregate import (
    aggregate_data,
    aggregate_metadata,
    aggregate_stats,
    aggregate_videos,
    validate_all_metadata,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    write_info,
    write_stats,
    write_tasks,
)
from libero_utils.config import LIBERO_FEATURES
from libero_utils.libero_utils import load_local_episodes
from ray.runtime_env import RuntimeEnv
from tqdm import tqdm


def setup_logger():
    import sys

    from datatrove.utils.logging import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    return logger


class SaveLerobotDataset(PipelineStep):
    name = "Save Temp LerobotDataset"
    type = "libero2lerobot"

    def __init__(self, tasks: list[tuple[Path, Path, str]]):
        super().__init__()
        self.tasks = tasks

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = setup_logger()

        input_h5, output_path, task_instruction = self.tasks[rank]

        if output_path.exists():
            shutil.rmtree(output_path)

        dataset = LeRobotDataset.create(
            repo_id=f"{input_h5.parent.name}/{input_h5.name}",
            root=output_path,
            fps=20,
            robot_type="franka",
            features=LIBERO_FEATURES,
        )

        logger.info(f"start processing for {input_h5}, saving to {output_path}")

        raw_dataset = load_local_episodes(input_h5)
        for episode_index, episode_data in enumerate(raw_dataset):
            with self.track_time("saving episode"):
                for frame_data in episode_data:
                    frame_data["task"] = task_instruction
                    dataset.add_frame(frame_data)
                dataset.save_episode()
                logger.info(f"process done for {dataset.repo_id}, episode {episode_index}, len {len(episode_data)}")


def create_aggr_dataset(raw_dirs: list[Path], aggregated_dir: Path):
    logger = setup_logger()

    all_metadata = [LeRobotDatasetMetadata("", root=raw_dir) for raw_dir in raw_dirs]

    fps, robot_type, features = validate_all_metadata(all_metadata)

    if aggregated_dir.exists():
        shutil.rmtree(aggregated_dir)

    aggr_meta = LeRobotDatasetMetadata.create(
        repo_id=f"{aggregated_dir.parent.name}/{aggregated_dir.name}",
        root=aggregated_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
    )

    video_keys = [key for key in features if features[key]["dtype"] == "video"]
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    aggr_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys}

    aggr_meta.episodes = {}

    for src_meta in tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(
            src_meta, aggr_meta, videos_idx, DEFAULT_VIDEO_FILE_SIZE_IN_MB, DEFAULT_CHUNK_SIZE
        )
        data_idx = aggregate_data(src_meta, aggr_meta, data_idx, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_CHUNK_SIZE)

        meta_idx = aggregate_metadata(src_meta, aggr_meta, meta_idx, data_idx, videos_idx)

        aggr_meta.info["total_episodes"] += src_meta.total_episodes
        aggr_meta.info["total_frames"] += src_meta.total_frames

    logger.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logger.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logger.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)


def delete_temp_data(temp_dirs: list[Path]):
    logger = setup_logger()
    logger.info("Delete temp data_dir")
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)


def main(
    src_paths: list[Path],
    output_path: Path,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    resume_dir: Path = None,
    debug: bool = False,
    repo_id: str = None,
    push_to_hub: bool = False,
):
    tasks = []
    pattern1 = re.compile(r"_SCENE\d+_(.*?)_demo\.hdf5")
    pattern2 = re.compile(r"(.*?)_demo\.hdf5")
    for src_path in src_paths:
        for input_h5 in src_path.glob("*.hdf5"):
            match = pattern1.search(input_h5.name)
            if match is None:
                match = pattern2.search(input_h5.name)
                if match is None:
                    continue
            tasks.append(
                (
                    input_h5,
                    (output_path / (src_path.name + "_temp") / input_h5.stem).resolve(),
                    match.group(1).replace("_", " "),
                )
            )
    if len(src_paths) > 1:
        aggregate_output_path = output_path / ("_".join([src_path.name for src_path in src_paths]) + "_aggregated_lerobot")
    else:
        aggregate_output_path = output_path / f"{src_paths[0].name}_lerobot"
    aggregate_output_path = aggregate_output_path.resolve()

    if debug:
        executor = "local"
        workers = 1
        tasks = tasks[:2]
        push_to_hub = False

    match executor:
        case "local":
            workers = os.cpu_count() // cpus_per_task if workers == -1 else workers
            executor = LocalPipelineExecutor
        case "ray":
            runtime_env = RuntimeEnv(
                env_vars={
                    "HDF5_USE_FILE_LOCKING": "FALSE",
                    "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                    "SVT_LOG": "1",
                },
            )
            ray.init(runtime_env=runtime_env)
            executor = RayPipelineExecutor
        case _:
            raise ValueError(f"Executor {executor} not supported")

    executor_config = {
        "tasks": len(tasks),
        "workers": workers,
        **({"cpus_per_task": cpus_per_task, "tasks_per_job": tasks_per_job} if executor is RayPipelineExecutor else {}),
    }

    executor(pipeline=[SaveLerobotDataset(tasks)], **executor_config, logging_dir=resume_dir).run()
    create_aggr_dataset([task[1] for task in tasks], aggregate_output_path)
    delete_temp_data([task[1] for task in tasks])

    for task in tasks:
        shutil.rmtree(task[1].parent, ignore_errors=True)

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", "libero", "franka"]
        tags.extend([src_path.name for src_path in src_paths])
        LeRobotDataset(
            repo_id=repo_id,
            root=aggregate_output_path,
        ).push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--executor", type=str, choices=["local", "ray"], default="local")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--tasks-per-job", type=int, default=1, help="number of concurrent tasks per job, only used for ray")
    parser.add_argument("--workers", type=int, default=-1, help="number of concurrent jobs to run")
    parser.add_argument("--resume-dir", type=Path, help="logs directory to resume")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-id", type=str, help="required when push-to-hub is True")
    parser.add_argument("--push-to-hub", action="store_true", help="upload to hub")
    args = parser.parse_args()

    main(**vars(args))
