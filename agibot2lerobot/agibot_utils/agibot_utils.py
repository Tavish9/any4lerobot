import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def get_task_instruction(task_json_path: str) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    return task_instruction


def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32)[:, :, None] / 1000 for f in all_imgs]


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
        for key in AgiBotWorld_CONFIG["actions"]:
            action[f"actions.{key}"] = np.array(f["action/" + key.replace(".", "/")], dtype=np.float32)

        # HACK: agibot team forgot to pad some of the values
        num_frames = len(next(iter(state.values())))
        for action_key, action_value in action.items():
            if action_value.size and len(action_value) != num_frames:
                state_key = action_key.replace("actions", "state").replace(".", "/")
                new_action_value = np.array(f[state_key], dtype=np.float32).copy()
                action_index_key = "/".join(list(action_key.replace("actions", "action").split(".")[:-1]) + ["index"])
                action_index = np.array(f[action_index_key])
                # agibot lost end index, replace it with joint
                if not action_index.size:
                    action_index_key = action_index_key.replace("end", "joint")
                    action_index = np.array(f[action_index_key])
                new_action_value[action_index] = action_value
                action[action_key] = new_action_value

    if save_depth:
        depth_imgs = load_depths(ob_dir / "depth", "head_depth")
        assert num_frames == len(depth_imgs), "Number of images and states are not equal"

    state_key_prefix_len = len("observation.states.")
    action_key_prefix_len = len("actions.")
    frames = [
        {
            **({"observation.images.head_depth": depth_imgs[i]} if save_depth else {}),
            **{
                key: value[i]
                if value.size
                else np.zeros(
                    AgiBotWorld_CONFIG["states"][key[state_key_prefix_len:]]["shape"],
                    dtype=AgiBotWorld_CONFIG["states"][key[state_key_prefix_len:]]["dtype"],
                )
                for key, value in state.items()
            },
            **{
                key: value[i]
                if value.size
                else np.zeros(
                    AgiBotWorld_CONFIG["actions"][key[action_key_prefix_len:]]["shape"],
                    dtype=AgiBotWorld_CONFIG["actions"][key[action_key_prefix_len:]]["dtype"],
                )
                for key, value in action.items()
            },
            "task": task_name,
        }
        for i in range(num_frames)
    ]

    videos = {
        f"observation.images.{key}": ob_dir / "videos" / f"{key}_color.mp4"
        if "sensor" not in key
        else ob_dir / "tactile" / f"{key}.mp4"  # HACK: handle tactile videos
        for key in AgiBotWorld_CONFIG["images"]
        if "depth" not in key
    }
    return frames, videos
