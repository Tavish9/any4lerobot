from pathlib import Path

import h5py
import numpy as np
from agibot_utils.config import AgiBotWorld_TASK_TYPE


def get_all_tasks(src_path: Path):
    json_files = src_path.glob("task_info/*.json")
    for json_file in json_files:
        yield json_file


def load_local_dataset(
    episode_id: int, src_path: str, task_id: int, task_name: str, save_depth: bool, AgiBotWorld_CONFIG: dict
) -> tuple[list, dict]:
    """Load local dataset and return a dict with observations and actions"""
    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    state = {}
    action = {}
    with h5py.File(proprio_dir / "proprio_stats.h5", "r") as f:
        for key in AgiBotWorld_CONFIG["states"]:
            state[f"observation.states.{key}"] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)

        num_frames = len(next(iter(state.values())))
        index = np.array(f["action/joint/index"])
        if len(index) != num_frames:
            return 1
        return 0


agibot_beta = Path("/oss/vla_next/DATA/AgiBotWorld-Beta/")

all_tasks = get_all_tasks(agibot_beta)


count = {}

for task in all_tasks:
    task_id = task.stem.split("_")[-1]
    src_path = task.parent.parent
    all_subdir = [f.as_posix() for f in agibot_beta.glob(f"observations/{task_id}/*") if f.is_dir()]
    all_subdir_eids = [int(Path(path).name) for path in all_subdir]
    count[task_id] = {"missing": 0, "total": len(all_subdir_eids)}
    for eid in all_subdir_eids:
        count[task_id]["missing"] += load_local_dataset(
            eid,
            src_path=src_path,
            task_id=task_id,
            task_name="",
            save_depth=False,
            AgiBotWorld_CONFIG=AgiBotWorld_TASK_TYPE["gripper"]["task_config"],
        )
    print(count)
