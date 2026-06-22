import argparse
import importlib
import inspect
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.feature_utils import (
    get_hf_features_from_features,
    validate_episode_buffer,
    validate_frame,
)
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

AGIBOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGIBOT_DIR.parent
for import_path in (REPO_ROOT, AGIBOT_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from agibot_utils.agibot_utils import (  # noqa: E402
    get_episode_ids,
    get_task_id,
    get_task_info,
    has_episode_videos,
    load_local_dataset,
)
from agibot_utils.config import AgiBotWorld_TASK_TYPE  # noqa: E402
from agibot_utils.lerobot_utils import (  # noqa: E402
    compute_episode_stats,
    generate_features_from_config,
)

from generic_converter import BaseAdapter, ConversionTask, run_converter  # noqa: E402


class AgiBotDatasetMetadata(LeRobotDatasetMetadata):
    def _flush_metadata_buffer(self) -> None:
        """Write all buffered episode metadata to parquet file."""
        if not hasattr(self, "_metadata_buffer") or len(self._metadata_buffer) == 0:
            return

        combined_dict = {}
        for episode_dict in self._metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                # Extract value and serialize numpy arrays
                # because PyArrow's from_pydict function doesn't support numpy arrays
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(
                    val.tolist() if isinstance(val, np.ndarray) else val
                )

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        schema = None if not self._pq_writer else self._pq_writer.schema
        table = pa.Table.from_pydict(combined_dict, schema=schema)

        if not self._pq_writer:
            path = Path(
                self.root
                / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()


class AgiBotDatasetWriter(DatasetWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_features = get_hf_features_from_features(self._meta.features)

    def add_frame(self, frame: dict) -> None:
        """
        Add a single frame to the current episode buffer.

        Apart from images written to a temporary directory, nothing is written to disk
        until ``save_episode()`` is called.

        The caller must provide all user-defined features plus ``"task"``, and must
        not provide ``"timestamp"`` or ``"frame_index"``; those are computed
        automatically.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {
            key: value
            for key, value in self._meta.features.items()
            if key in self.hf_features
        }  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self._create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame_index / self._meta.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key not in self._meta.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self._meta.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(
        self,
        videos: dict,
        action_config: list,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        """Save the current episode in self.episode_buffer to disk."""
        episode_buffer = (
            episode_data if episode_data is not None else self.episode_buffer
        )

        validate_episode_buffer(
            episode_buffer, self._meta.total_episodes, self._meta.features
        )

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self._meta.total_frames, self._meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self._meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array(
            [self._meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self._meta.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in [
                "video"
            ]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self._meta.video_keys:
            episode_buffer[key] = str(videos[key])

        ep_stats = compute_episode_stats(episode_buffer, self._meta.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self._meta.video_keys) > 0
        use_batched_encoding = self._batch_encoding_size > 1

        self.current_videos = videos
        if has_video_keys and not use_batched_encoding:
            for video_key in self._meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        ep_metadata.update({"action_config": action_config})
        self._meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, ep_metadata
        )

        if has_video_keys and use_batched_encoding:
            self._episodes_since_last_encoding += 1
            if self._episodes_since_last_encoding == self._batch_encoding_size:
                start_ep = self._meta.total_episodes - self._batch_encoding_size
                end_ep = self._meta.total_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self._episodes_since_last_encoding = 0

        if not episode_data:
            self.clear_episode_buffer(delete_images=len(self._meta.image_keys) > 0)

    def _encode_temporary_episode_video(
        self, video_key: str, episode_index: int
    ) -> Path:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        temp_path = (
            Path(tempfile.mkdtemp(dir=self._root))
            / f"{video_key}_{episode_index:03d}.mp4"
        )
        shutil.copy(self.current_videos[video_key], temp_path)
        return temp_path


class AgiBotDataset(LeRobotDataset):
    @classmethod
    def create(cls, *args, **kwargs) -> "AgiBotDataset":
        sig = inspect.signature(super().create)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        obj = super().create(*args, **kwargs)

        shutil.rmtree(params["root"], ignore_errors=True)
        obj.meta: AgiBotDatasetMetadata = AgiBotDatasetMetadata.create(
            repo_id=params["repo_id"],
            fps=params["fps"],
            robot_type=params["robot_type"],
            features=params["features"],
            root=params["root"],
            use_videos=params["use_videos"],
            metadata_buffer_size=params["metadata_buffer_size"],
            video_files_size_in_mb=params["video_files_size_in_mb"],
            data_files_size_in_mb=params["data_files_size_in_mb"],
        )
        obj.writer: AgiBotDatasetWriter = AgiBotDatasetWriter(
            meta=obj.meta,
            root=obj.root,
            camera_encoder=obj.writer._camera_encoder,
            encoder_threads=obj.writer._encoder_threads,
            batch_encoding_size=obj.writer._batch_encoding_size,
            streaming_encoder=obj.writer._streaming_encoder,
        )
        return obj

    def save_episode(
        self,
        videos: dict,
        action_config: list,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        self._require_writer("save_episode")
        self.writer.save_episode(videos, action_config, episode_data, parallel_encoding)


class AgiBotAdapter(BaseAdapter):
    dataset_type = "agibot"
    fps = 30
    robot_type = "a2d"
    tags = ("agibot-world", "a2d")

    def __init__(
        self,
        src_path: Path,
        output_path: Path,
        eef_type: str,
        task_ids: Sequence[str],
        save_depth: bool,
        episodes_per_task: int,
    ):
        super().__init__(output_path)
        if episodes_per_task < 1:
            raise ValueError("--episodes-per-task must be >= 1")
        self.src_path = src_path.expanduser().resolve()
        self.eef_type = eef_type
        self.task_ids = set(task_ids)
        self.save_depth = save_depth
        self.episodes_per_task = episodes_per_task
        self.agibot_world_config = AgiBotWorld_TASK_TYPE[eef_type]["task_config"]
        self.type_task_ids = set(AgiBotWorld_TASK_TYPE[eef_type]["task_ids"])
        self.features = generate_features_from_config(self.agibot_world_config)
        if not save_depth:
            self.features.pop("observation.images.head_depth", None)

    def load_tasks(self) -> list[ConversionTask]:
        tasks = []
        for json_file in sorted(self.src_path.glob("task_info/*.json")):
            if not self._include_task(json_file.stem):
                continue
            _, _, _, task_info = self._load_task_context(json_file)
            episode_ids = sorted(task_info)

            for chunk_index, chunk_episode_ids in enumerate(
                self._chunk_episode_ids(episode_ids)
            ):
                task_name = self._format_task_name(
                    json_file.stem, chunk_index, chunk_episode_ids
                )
                tasks.append(
                    ConversionTask(
                        input_path=json_file.resolve(),
                        output_path=(
                            self.temp_output_path / "agibotworld" / task_name
                        ).resolve(),
                        local_repo_id=task_name,
                        metadata={"episode_ids": tuple(chunk_episode_ids)},
                    )
                )
        return tasks

    def load_subset(self, task: ConversionTask):
        json_file = task.input_path
        print(f"processing {json_file.stem}, saving to {task.output_path}")
        src_path, task_id, task_instruction, task_info = self._load_task_context(
            json_file
        )

        task_episode_ids = task.metadata.get("episode_ids")
        if task_episode_ids is None:
            task_episode_ids = get_episode_ids(src_path, task_id)

        for eid in task_episode_ids:
            if not self._is_convertible_episode(
                json_file.stem, src_path, task_id, eid, task_info
            ):
                continue
            action_config = task_info[eid]["label_info"]["action_config"]
            raw_dataset = load_local_dataset(
                eid,
                src_path=src_path,
                task_id=task_id,
                save_depth=self.save_depth,
                AgiBotWorld_CONFIG=self.agibot_world_config,
            )
            if raw_dataset is None:
                continue
            _, frames, videos = raw_dataset

            for frame_data in frames:
                frame_data["task"] = task_instruction

            yield {
                "episode_id": eid,
                "frames": frames,
                "videos": videos,
                "action_config": action_config,
            }

    def create_dataset(self, task: ConversionTask) -> AgiBotDataset:
        return AgiBotDataset.create(
            repo_id=task.local_repo_id,
            root=task.output_path,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
        )

    def save_episode(
        self,
        dataset: AgiBotDataset,
        episode_data: dict[str, Any],
        task: ConversionTask,
    ) -> bool:
        for frame_data in episode_data["frames"]:
            dataset.add_frame(frame_data)
        try:
            dataset.save_episode(
                videos=episode_data["videos"],
                action_config=episode_data["action_config"],
            )
        except Exception as e:
            print(
                f"{task.input_path.stem}, episode_{episode_data['episode_id']}: "
                f"there are some corrupted mp4s\nException details: {str(e)}"
            )
            dataset.clear_episode_buffer(delete_images=False)
            return False
        return True

    def get_episode_length(self, episode_data: dict[str, Any]) -> int:
        return len(episode_data["frames"])

    def _chunk_episode_ids(self, episode_ids: list[int]):
        for start in range(0, len(episode_ids), self.episodes_per_task):
            yield episode_ids[start : start + self.episodes_per_task]

    def _format_task_name(
        self, task_name: str, chunk_index: int, episode_ids: Sequence[int]
    ) -> str:
        if len(episode_ids) == 1:
            return f"{task_name}_episode_{episode_ids[0]}"
        return (
            f"{task_name}_chunk_{chunk_index:06d}_episodes_"
            f"{episode_ids[0]}_{episode_ids[-1]}"
        )

    def _load_task_context(
        self, json_file: Path
    ) -> tuple[Path, str, str, dict[int, dict[str, Any]]]:
        src_path = json_file.parent.parent
        task_id = get_task_id(json_file)
        task_info = get_task_info(json_file)
        task_name = task_info[0]["task_name"]
        task_init_scene = task_info[0]["init_scene_text"]
        task_instruction = f"{task_name} | {task_init_scene}"
        task_info_by_episode = {episode["episode_id"]: episode for episode in task_info}
        return src_path, task_id, task_instruction, task_info_by_episode

    def _is_convertible_episode(
        self,
        task_name: str,
        src_path: Path,
        task_id: str,
        episode_id: int,
        task_info: dict[int, dict[str, Any]],
    ) -> bool:
        if episode_id not in task_info:
            print(
                f"{task_name}, episode_{episode_id} not in task_info.json, skipping..."
            )
            return False
        if not has_episode_videos(
            src_path, task_id, episode_id, self.agibot_world_config
        ):
            print(
                f"{task_name}, episode_{episode_id}: "
                "some of the videos does not exist, skipping..."
            )
            return False
        return True

    def _include_task(self, task_id: str) -> bool:
        if self.task_ids and task_id not in self.task_ids:
            return False
        if self.eef_type == "gripper":
            remaining_ids = set(AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"])
            remaining_ids.update(AgiBotWorld_TASK_TYPE["tactile"]["task_ids"])
            return task_id not in remaining_ids
        return task_id in self.type_task_ids


def main(
    src_path: Path,
    output_path: Path,
    eef_type: str,
    task_ids: list[str],
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    save_depth: bool,
    episodes_per_task: int,
    resume_dir: Path | None = None,
    debug: bool = False,
    repo_id: str | None = None,
    push_to_hub: bool = False,
):
    adapter = AgiBotAdapter(
        src_path=src_path,
        output_path=output_path,
        eef_type=eef_type,
        task_ids=task_ids,
        save_depth=save_depth,
        episodes_per_task=episodes_per_task,
    )

    run_converter(
        adapter=adapter,
        executor=executor,
        cpus_per_task=cpus_per_task,
        tasks_per_job=tasks_per_job,
        workers=workers,
        resume_dir=resume_dir,
        debug=debug,
        local_repo_id=repo_id,
        hub_repo_id=repo_id,
        push_to_hub=push_to_hub,
        extra_tags=[eef_type],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--eef-type",
        type=str,
        choices=["gripper", "dexhand", "tactile"],
        default="gripper",
    )
    parser.add_argument(
        "--task-ids", type=str, nargs="+", help="task_327 task_351 ...", default=[]
    )
    parser.add_argument("--executor", type=str, choices=["local", "ray"], default="ray")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--tasks-per-job",
        type=int,
        default=1,
        help="number of concurrent tasks per job, only used for ray",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=10,
        help="number of AgiBot episodes grouped into one conversion task",
    )
    parser.add_argument(
        "--workers", type=int, default=-1, help="number of concurrent jobs to run"
    )
    parser.add_argument("--resume-dir", type=Path, help="logs directory to resume")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-id", type=str, help="required when push-to-hub is True")
    parser.add_argument("--push-to-hub", action="store_true", help="upload to hub")
    return parser.parse_args()


def cli():
    args = parse_args()
    module_name = "agibot2lerobot.agibot_h5"
    module = importlib.import_module(module_name)
    module.main(**vars(args))


if __name__ == "__main__":
    cli()
