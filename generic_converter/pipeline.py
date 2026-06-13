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

        dataset = LeRobotDataset.create(
            repo_id=task.local_repo_id,
            root=task.output_path,
            fps=self.adapter.fps,
            robot_type=self.adapter.robot_type,
            features=self.adapter.features,
        )

        logger.info(
            f"start processing for {task.input_path}, saving to {task.output_path}"
        )
        raw_dataset = self.adapter.load_subset(task)
        for episode_index, episode_data in enumerate(raw_dataset):
            with self.track_time("saving episode"):
                for frame in episode_data:
                    dataset.add_frame(frame)
                dataset.save_episode()
                logger.info(
                    f"process done for {dataset.repo_id}, episode {episode_index}, len {len(episode_data)}"
                )
        dataset.finalize()


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
            executor_cls, executor_config = LocalPipelineExecutor, {
                "tasks": len(tasks),
                "workers": resolved_workers,
            }
        case "ray":
            import ray
            from datatrove.executor import RayPipelineExecutor
            from ray.runtime_env import RuntimeEnv

            runtime_env = RuntimeEnv(env_vars=_build_ray_env_vars())
            ray.init(runtime_env=runtime_env)
            executor_cls, executor_config = RayPipelineExecutor, {
                "tasks": len(tasks),
                "workers": workers,
                "cpus_per_task": cpus_per_task,
                "tasks_per_job": tasks_per_job,
            }
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
    aggregate_tasks(tasks, output_path, aggr_repo_id=local_repo_id)

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

    roots = [task.output_path for task in tasks]
    resolved_aggr_repo_id = aggr_repo_id or output_dir.name

    logger.info(
        f"aggregate {len(tasks)} temporary datasets into {output_dir} as {resolved_aggr_repo_id}"
    )
    aggregate_datasets(
        repo_ids=[None] * len(tasks),
        roots=roots,
        aggr_repo_id=resolved_aggr_repo_id,
        aggr_root=output_dir,
    )
    logger.info(f"aggregation complete: {output_dir}")
