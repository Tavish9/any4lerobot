import argparse
import gc
import json
import logging
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import ray
import torch
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    get_hf_features_from_features,
    hf_transform_to_torch,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec
from lerobot.common.robot_devices.robots.utils import Robot
from ray.runtime_env import RuntimeEnv
from robomind_uitls.configs import ROBOMIND_CONFIG
from robomind_uitls.lerobot_uitls import compute_episode_stats, generate_features_from_config
from robomind_uitls.robomind_uitls import load_local_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RoboMINDDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        split,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        action_config: dict[str, str | dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        if split == "train":
            self.info["splits"]["train"] = f"0:{self.info['total_episodes']}"
            self.train_count = self.info["total_episodes"]
        elif "val" in split:
            self.info["splits"]["validation"] = f"{self.train_count}:{self.info['total_episodes']}"
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            **({"action_config": action_config} if action_config else {}),
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class RoboMINDDataset(LeRobotDataset):
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
        obj.meta = RoboMINDDatasetMetadata.create(
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

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

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

        validate_frame(frame, self.features)

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

            if self.features[key]["dtype"] in ["video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(value, img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(
        self, split, action_config: dict, episode_data: dict | None = None, keep_images: bool = False
    ) -> None:
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

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(split, episode_index, episode_length, episode_tasks, ep_stats, action_config)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not keep_images:
            # delete images
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()


def get_all_tasks(src_path: Path, output_path: Path, embodiment: str):
    output_path = output_path / src_path.name / embodiment
    src_path = src_path / f"h5_{embodiment}"

    if src_path.exists():
        df = pd.read_csv(src_path.parent.parent / "RoboMIND_v1_2_instr.csv", index_col=0).drop_duplicates()
        instruction_dict = df.set_index("task")["instruction"].to_dict()
        for task_type in src_path.iterdir():
            yield (
                task_type.name,
                {"train": task_type / "success_episodes" / "train", "val": task_type / "success_episodes" / "val"},
                (output_path / task_type.name).resolve(),
                instruction_dict[task_type.name],
            )


def save_as_lerobot_dataset(task: tuple[dict, Path, str], src_path, benchmark, embodiment, save_depth):
    task_type, splits, local_dir, task_instruction = task

    config = ROBOMIND_CONFIG[embodiment]
    # HACK: not consistent image shape...
    if "1_0" in benchmark:
        match embodiment:
            case "tienkung_gello_1rgb":
                if task_type in (
                    "clean_table_2_241211",
                    "clean_table_3_241210",
                    "clean_table_3_241211",
                    "place_paper_cup_dustbin_241212",
                    "place_plate_table_241211",
                    "place_plate_table_241211_12",
                    "place_plate_table_241212",
                ):
                    for value in config["images"].values():
                        value["shape"] = (720, 1280) + (value["shape"][2],)

            case "tienkung_xsens_1rgb":
                if task_type == "switch_manipulation":
                    for value in config["images"].values():
                        value["shape"] = (720, 1280) + (value["shape"][2],)

    features = generate_features_from_config(config)

    if local_dir.exists():
        shutil.rmtree(local_dir)

    if not save_depth:
        features = dict(filter(lambda item: "depth" not in item[0], features.items()))

    dataset: RoboMINDDataset = RoboMINDDataset.create(
        repo_id=f"{embodiment}/{local_dir.name}",
        root=local_dir,
        fps=30,
        robot_type=embodiment,
        features=features,
    )

    logging.info(f"start processing for {benchmark}, {embodiment}, {task_type}, saving to {local_dir}")
    for split, path in splits.items():
        action_config_path = src_path / "language_description_annotation_json" / f"h5_{embodiment}.json"
        if action_config_path.exists():
            action_config = json.load(open(action_config_path))
            action_config = {
                Path(config["id"]).parent.name: config["response"]
                for config in action_config
                if local_dir.name in config["id"] and split in config["id"]
            }
        else:
            action_config = {}
        for episode_path in path.glob("**/trajectory.hdf5"):
            status, raw_dataset, err = load_local_dataset(episode_path, config, save_depth)
            if status and len(raw_dataset) >= 50:
                for frame_data in raw_dataset:
                    frame_data.update({"task": task_instruction})
                    dataset.add_frame(frame_data)
                dataset.save_episode(split, action_config.get(episode_path.parent.parent.name, {}))
                logging.info(f"process done for {path}, len {len(raw_dataset)}")
            else:
                logging.warning(f"Skipped {episode_path}: len of dataset:{len(raw_dataset)} or {str(err)}")
            gc.collect()

    del dataset


def main(
    src_path: Path,
    output_path: Path,
    benchmark: str,
    embodiments: list[str],
    cpus_per_task: int,
    save_depth: bool,
    debug: bool = False,
):
    if debug:
        tasks = get_all_tasks(src_path / benchmark, output_path, embodiments[0])
        save_as_lerobot_dataset(next(tasks), src_path, benchmark, embodiments[0], save_depth)
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

        logging.info(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}")
        remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)

        futures = []
        for embodiment in embodiments:
            tasks = get_all_tasks(src_path / benchmark, output_path, embodiment)
            for task in tasks:
                futures.append((task[1], remote_task.remote(task, src_path, benchmark, embodiment, save_depth)))

        for task_path, future in futures:
            try:
                ray.get(future)
            except Exception as e:
                logging.error(f"Exception occurred for {task_path['train']}")
                with open("output.txt", "a") as f:
                    f.write(f"{task_path['train']}, exception details: {str(e)}\n")
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["benchmark1_0_release", "benchmark1_1_release", "benchmark1_2_release"],
        default="benchmark1_1_release",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--embodiments",
        type=str,
        nargs="+",
        help=str(
            [
                "agilex_3rgb",
                "franka_1rgb",
                "franka_3rgb",
                "franka_fr3_dual",
                "tienkung_gello_1rgb",
                "tienkung_prod1_gello_1rgb",
                "tienkung_xsens_1rgb",
                "ur_1rgb",
            ]
        ),
        default=["agilex_3rgb"],
    )
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
