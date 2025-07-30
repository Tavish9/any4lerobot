import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from datatrove.executor import LocalPipelineExecutor, RayPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
    write_task,
)
from libero_utils.config import LIBERO_FEATURES
from libero_utils.lerobot_utils import validate_all_metadata
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
                    dataset.add_frame(
                        frame_data,
                        task=task_instruction,
                    )
                dataset.save_episode()
                logger.info(f"process done for {dataset.repo_id}, episode {episode_index}, len {len(episode_data)}")


class AggregateDatasets(PipelineStep):
    name = "Aggregate Datasets"
    type = "libero2lerobot"

    def __init__(self, raw_dirs: list[Path], aggregated_dir: Path):
        super().__init__()
        self.raw_dirs = raw_dirs
        self.aggregated_dir = aggregated_dir

        self.create_aggr_dataset()

    def create_aggr_dataset(self):
        logger = setup_logger()

        all_metadata = [LeRobotDatasetMetadata("", root=raw_dir) for raw_dir in self.raw_dirs]

        fps, robot_type, features = validate_all_metadata(all_metadata)

        if self.aggregated_dir.exists():
            shutil.rmtree(self.aggregated_dir)

        aggr_meta = LeRobotDatasetMetadata.create(
            repo_id=f"{self.aggregated_dir.parent.name}/{self.aggregated_dir.name}",
            root=self.aggregated_dir,
            fps=fps,
            robot_type=robot_type,
            features=features,
        )

        datasets_task_index_to_aggr_task_index = {}
        aggr_task_index = 0
        for dataset_index, meta in enumerate(tqdm(all_metadata, desc="Aggregate tasks index")):
            task_index_to_aggr_task_index = {}

            for task_index, task in meta.tasks.items():
                if task not in aggr_meta.task_to_task_index:
                    # add the task to aggr tasks mappings
                    aggr_meta.tasks[aggr_task_index] = task
                    aggr_meta.task_to_task_index[task] = aggr_task_index
                    aggr_task_index += 1

                task_index_to_aggr_task_index[task_index] = aggr_meta.task_to_task_index[task]

            datasets_task_index_to_aggr_task_index[dataset_index] = task_index_to_aggr_task_index

        datasets_aggr_episode_index_shift = {}
        datasets_aggr_index_shift = {}
        for dataset_index, meta in enumerate(tqdm(all_metadata, desc="Aggregate episodes and global index")):
            datasets_aggr_episode_index_shift[dataset_index] = aggr_meta.total_episodes
            datasets_aggr_index_shift[dataset_index] = aggr_meta.total_frames

            # populate episodes
            for episode_index, episode_dict in meta.episodes.items():
                aggr_episode_index = episode_index + aggr_meta.total_episodes
                episode_dict["episode_index"] = aggr_episode_index
                aggr_meta.episodes[aggr_episode_index] = episode_dict

            # populate episodes_stats
            for episode_index, episode_stats in meta.episodes_stats.items():
                aggr_episode_index = episode_index + aggr_meta.total_episodes
                episode_stats["index"].update(
                    {
                        "min": episode_stats["index"]["min"] + aggr_meta.total_frames,
                        "max": episode_stats["index"]["max"] + aggr_meta.total_frames,
                        "mean": episode_stats["index"]["mean"] + aggr_meta.total_frames,
                    }
                )
                episode_stats["episode_index"].update(
                    {
                        "min": np.array([aggr_episode_index]),
                        "max": np.array([aggr_episode_index]),
                        "mean": np.array([aggr_episode_index]),
                    }
                )
                df = pd.read_parquet(meta.root / meta.get_data_file_path(episode_index))
                df["task_index"] = df["task_index"].map(datasets_task_index_to_aggr_task_index[dataset_index])
                episode_stats["task_index"].update(
                    {
                        "min": np.array([df["task_index"].min()]),
                        "max": np.array([df["task_index"].max()]),
                        "mean": np.array([df["task_index"].mean()]),
                        "std": np.array([df["task_index"].std()]),
                    }
                )
                aggr_meta.episodes_stats[aggr_episode_index] = episode_stats

            # populate info
            aggr_meta.info["total_episodes"] += meta.total_episodes
            aggr_meta.info["total_frames"] += meta.total_frames
            aggr_meta.info["total_videos"] += len(aggr_meta.video_keys) * meta.total_episodes

        logger.info("Write meta data")
        aggr_meta.info["total_tasks"] = len(aggr_meta.tasks)
        aggr_meta.info["total_chunks"] = aggr_meta.get_episode_chunk(aggr_meta.total_episodes - 1) + 1
        aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.info['total_episodes']}"}

        # create a new episodes jsonl with updated episode_index using write_episode
        for episode_dict in tqdm(aggr_meta.episodes.values(), desc="Write episodes info"):
            write_episode(episode_dict, aggr_meta.root)

        # create a new episode_stats jsonl with updated episode_index using write_episode_stats
        for episode_index, episode_stats in tqdm(aggr_meta.episodes_stats.items(), desc="Write episodes stats info"):
            write_episode_stats(episode_index, episode_stats, aggr_meta.root)

        # create a new task jsonl with updated episode_index using write_task
        for task_index, task in tqdm(aggr_meta.tasks.items(), desc="Write tasks info"):
            write_task(task_index, task, aggr_meta.root)

        write_info(aggr_meta.info, aggr_meta.root)

        self.datasets_task_index_to_aggr_task_index = datasets_task_index_to_aggr_task_index
        self.datasets_aggr_episode_index_shift = datasets_aggr_episode_index_shift
        self.datasets_aggr_index_shift = datasets_aggr_index_shift

        logger.info("Meta data done writing")

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = setup_logger()

        dataset_index = rank
        aggr_meta = LeRobotDatasetMetadata("", root=self.aggregated_dir)
        meta = LeRobotDatasetMetadata("", root=self.raw_dirs[dataset_index])
        aggr_episode_index_shift = self.datasets_aggr_episode_index_shift[dataset_index]
        aggr_index_shift = self.datasets_aggr_index_shift[dataset_index]
        task_index_to_aggr_task_index = self.datasets_task_index_to_aggr_task_index[dataset_index]

        with self.track_time("aggregating dataset"):
            logger.info("Copy data")
            for episode_index in range(meta.total_episodes):
                aggr_episode_index = episode_index + aggr_episode_index_shift
                data_path = meta.root / meta.get_data_file_path(episode_index)
                aggr_data_path = aggr_meta.root / aggr_meta.get_data_file_path(aggr_episode_index)
                aggr_data_path.parent.mkdir(parents=True, exist_ok=True)

                # update index, episode_index and task_index
                df = pd.read_parquet(data_path)
                df["index"] += aggr_index_shift
                df["episode_index"] += aggr_episode_index_shift
                df["task_index"] = df["task_index"].map(task_index_to_aggr_task_index)
                df.to_parquet(aggr_data_path)

            logger.info("Copy videos")
            for episode_index in range(meta.total_episodes):
                aggr_episode_index = episode_index + aggr_episode_index_shift
                for vid_key in meta.video_keys:
                    video_path = meta.root / meta.get_video_file_path(episode_index, vid_key)
                    aggr_video_path = aggr_meta.root / aggr_meta.get_video_file_path(aggr_episode_index, vid_key)
                    aggr_video_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(video_path, aggr_video_path)


class DeleteTempData(PipelineStep):
    name = "Delete Temp Data"
    type = "libero2lerobot"

    def __init__(self, temp_dirs: list[Path]):
        super().__init__()
        self.temp_dirs = temp_dirs

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = setup_logger()

        logger.info(f"Delete temp data {self.temp_dirs[rank]}")
        shutil.rmtree(self.temp_dirs[rank])


def main(
    src_paths: list[Path],
    output_path: Path,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    resume_from_save: Path = None,
    resume_from_aggregate: Path = None,
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

    executor(pipeline=[SaveLerobotDataset(tasks)], **executor_config, logging_dir=resume_from_save).run()
    executor(
        pipeline=[DeleteTempData([task[1] for task in tasks])],
        **executor_config,
        depends=executor(
            pipeline=[AggregateDatasets([task[1] for task in tasks], aggregate_output_path)],
            **executor_config,
            logging_dir=resume_from_aggregate,
        ),
    ).run()

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
    parser.add_argument("--resume-from-save", type=Path, help="logs directory to resume from save step")
    parser.add_argument("--resume-from-aggregate", type=Path, help="logs directory to resume from aggregate step")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-id", type=str, help="required when push-to-hub is True")
    parser.add_argument("--push-to-hub", action="store_true", help="upload to hub")
    args = parser.parse_args()

    main(**vars(args))
