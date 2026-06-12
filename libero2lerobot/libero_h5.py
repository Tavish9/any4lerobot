import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from h5py import File

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generic_converter import BaseAdapter, ConversionTask, run_converter  # noqa: E402


class LiberoAdapter(BaseAdapter):
    dataset_type = "libero"
    fps = 20
    robot_type = "franka"
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "rgb"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "rgb"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]},
        },
        "observation.states.ee_state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]},
        },
        "observation.states.joint_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        },
        "observation.states.gripper_state": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["gripper", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
        },
    }
    tags = ["libero", "franka"]

    def __init__(self, src_paths: list[Path], output_path: Path):
        super().__init__(output_path)
        self.src_paths = src_paths

    def load_tasks(self) -> list[ConversionTask]:
        tasks = []
        for src_path in self.src_paths:
            for input_h5 in src_path.glob("*.hdf5"):
                pattern1 = re.compile(r"_SCENE\d+_(.*?)_demo\.hdf5")
                pattern2 = re.compile(r"(.*?)_demo\.hdf5")

                match = pattern1.search(input_h5.name)
                if match is None:
                    match = pattern2.search(input_h5.name)
                if match is None:
                    continue
                else:
                    task_instruction = match.group(1).replace("_", " ")

                tasks.append(
                    ConversionTask(
                        input_path=input_h5.resolve(),
                        output_path=(
                            self.temp_output_path
                            / f"{src_path.name}"
                            / input_h5.stem
                        ).resolve(),
                        local_repo_id=f"{input_h5.parent.name}/{input_h5.name}",
                        metadata={"task": task_instruction},
                    )
                )
        return tasks

    def load_subset(self, task: ConversionTask) -> Iterable[Sequence[dict]]:
        input_h5 = task.input_path
        task_instruction = task.metadata.get("task")
        with File(input_h5, "r") as f:
            for demo in f["data"].values():
                demo_len = len(demo["obs/agentview_rgb"])
                # (-1: open, 1: close) -> (0: close, 1: open)
                action = np.array(demo["actions"])
                action = np.concatenate(
                    [
                        action[:, :6],
                        (1 - np.clip(action[:, -1], 0, 1))[:, None],
                    ],
                    axis=1,
                )
                state = np.concatenate(
                    [
                        np.array(demo["obs/ee_states"]),
                        np.array(demo["obs/gripper_states"]),
                    ],
                    axis=1,
                )
                episode = {
                    "observation.images.image": np.array(demo["obs/agentview_rgb"]),
                    "observation.images.wrist_image": np.array(demo["obs/eye_in_hand_rgb"]),
                    "observation.state": np.array(state, dtype=np.float32),
                    "observation.states.ee_state": np.array(demo["obs/ee_states"], dtype=np.float32),
                    "observation.states.joint_state": np.array(demo["obs/joint_states"], dtype=np.float32),
                    "observation.states.gripper_state": np.array(demo["obs/gripper_states"], dtype=np.float32),
                    "action": np.array(action, dtype=np.float32),
                }
                yield [{**{k: v[i] for k, v in episode.items()}, "task": task_instruction} for i in range(demo_len)]


def main(
    src_paths: list[Path],
    output_path: Path,
    executor: str,
    cpus_per_task: int,
    tasks_per_job: int,
    workers: int,
    resume_dir: Path | None = None,
    debug: bool = False,
    repo_id: str | None = None,
    push_to_hub: bool = False,
):
    adapter = LiberoAdapter(src_paths, output_path)

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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--executor", type=str, choices=["local", "ray"], default="local")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--tasks-per-job", type=int, default=1, help="number of concurrent tasks per job, only used for ray"
    )
    parser.add_argument("--workers", type=int, default=-1, help="number of concurrent jobs to run")
    parser.add_argument("--resume-dir", type=Path, help="logs directory to resume")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-id", type=str, help="required when push-to-hub is True")
    parser.add_argument("--push-to-hub", action="store_true", help="upload to hub")
    args = parser.parse_args()

    main(**vars(args))
