import argparse
import gc
import shutil
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np
import ray
import torch
from agibot_utils.agibot_utils import get_task_info, load_local_dataset
from agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot_utils.lerobot_utils import compute_episode_stats, generate_features_from_config
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec
from lerobot.common.robot_devices.robots.utils import Robot
from ray.runtime_env import RuntimeEnv


class AgiBotDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        action_config: list[dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            "action_config": action_config,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class AgiBotDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = AgiBotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, action_config: list, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        self._save_episode_table(episode_buffer, episode_index)

        # `meta.save_episode` be executed after encoding the videos
        # add action_config to current episode
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, action_config)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()


def get_all_tasks(src_path: Path, output_path: Path):
    json_files = src_path.glob("task_info/*.json")
    for json_file in json_files:
        local_dir = output_path / "agibotworld" / json_file.stem
        yield (json_file, local_dir.resolve())


def save_as_lerobot_dataset(agibot_world_config, task: tuple[Path, Path], num_threads, save_depth, debug):
    json_file, local_dir = task
    print(f"processing {json_file.stem}, saving to {local_dir}")
    src_path = json_file.parent.parent
    task_info = get_task_info(json_file)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    task_id = json_file.stem.split("_")[-1]
    task_info = {episode["episode_id"]: episode for episode in task_info}

    features = generate_features_from_config(agibot_world_config)

    if local_dir.exists():
        shutil.rmtree(local_dir)

    if not save_depth:
        features.pop("observation.images.head_depth")

    dataset = AgiBotDataset.create(
        repo_id=json_file.stem,
        root=local_dir,
        fps=30,
        robot_type="a2d",
        features=features,
    )

    all_subdir = [f.as_posix() for f in src_path.glob(f"observations/{task_id}/*") if f.is_dir()]

    all_subdir_eids = sorted([int(Path(path).name) for path in all_subdir])

    if debug or not save_depth:
        for eid in all_subdir_eids:
            if eid not in task_info:
                print(f"{json_file.stem}, episode_{eid} not in task_info.json, skipping...")
                continue
            try:
                action_config = task_info[eid]["label_info"]["action_config"]
                raw_dataset = load_local_dataset(
                    eid,
                    src_path=src_path,
                    task_id=task_id,
                    task_instruction=task_instruction,
                    save_depth=save_depth,
                    AgiBotWorld_CONFIG=agibot_world_config,
                )
                _, frames, videos = raw_dataset
                if not all([video_path.exists() for video_path in videos.values()]):
                    print(f"{json_file.stem}, episode_{eid}: some of the videos does not exist, skipping...")
                    continue

                for frame_data in frames:
                    dataset.add_frame(frame_data)
                dataset.save_episode(videos=videos, action_config=action_config)
            except Exception as e:
                raise Exception(f"{json_file.stem}, {eid}") from e
            gc.collect()
            print(f"process done for {json_file.stem}, episode_id {eid}, len {len(frames)}")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for eid in all_subdir_eids:
                if eid not in task_info:
                    print(f"{json_file.stem}, episode_{eid} not in task_info.json, skipping...")
                    continue
                futures.append(
                    executor.submit(
                        load_local_dataset,
                        eid,
                        src_path=src_path,
                        task_id=task_id,
                        task_instruction=task_instruction,
                        save_depth=save_depth,
                        AgiBotWorld_CONFIG=agibot_world_config,
                    )
                )

            for raw_dataset in as_completed(futures):
                eid, frames, videos = raw_dataset.result()
                action_config = task_info[eid]["label_info"]["action_config"]
                for frame_data in frames:
                    dataset.add_frame(frame_data)
                dataset.save_episode(videos=videos, action_config=action_config)
                gc.collect()


def main(
    src_path: str,
    output_path: str,
    eef_type: str,
    task_ids: list,
    cpus_per_task: int,
    num_threads_per_task: int,
    save_depth: bool,
    debug: bool = False,
):
    tasks = get_all_tasks(src_path, output_path)

    agibot_world_config, type_task_ids = (
        AgiBotWorld_TASK_TYPE[eef_type]["task_config"],
        AgiBotWorld_TASK_TYPE[eef_type]["task_ids"],
    )

    if eef_type == "gripper":
        remaining_ids = AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"] + AgiBotWorld_TASK_TYPE["tactile"]["task_ids"]
        tasks = filter(lambda task: task[0].stem not in remaining_ids, tasks)
    else:
        tasks = filter(lambda task: task[0].stem in type_task_ids, tasks)

    if task_ids:
        tasks = filter(lambda task: task[0].stem in task_ids, tasks)

    if debug:
        save_as_lerobot_dataset(agibot_world_config, next(tasks), num_threads_per_task, save_depth, debug)
    else:
        runtime_env = RuntimeEnv(
            env_vars={
                "HDF5_USE_FILE_LOCKING": "FALSE",
                "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
                "LD_PRELOAD": str(Path(__file__).resolve().parent / "libtcmalloc.so.4.5.3"),
            }
        )
        ray.init(runtime_env=runtime_env)
        resources = ray.available_resources()
        cpus = int(resources["CPU"])

        print(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}")

        remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)
        futures = []
        for task in tasks:
            futures.append(
                (task[0].stem, remote_task.remote(agibot_world_config, task, num_threads_per_task, save_depth, debug))
            )

        for task, future in futures:
            try:
                ray.get(future)
            except Exception as e:
                print(f"Exception occurred for {task}")
                with open("output.txt", "a") as f:
                    f.write(f"{task}, exception details: {str(e)}\n")

        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile"], default="gripper")
    parser.add_argument("--task-ids", type=str, nargs="+", help="task_327 task_351 ...", default=[])
    parser.add_argument("--cpus-per-task", type=int, default=3)
    parser.add_argument("--num-threads-per-task", type=int, default=2)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
