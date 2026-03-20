import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def main(raw_dir: Path, repo_id: str, local_dir: Path):
    if local_dir.exists():
        shutil.rmtree(local_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="PandaOmron",
        root=local_dir,
        fps=20,
        features={
            "observation.images.robot0_agentview_right": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.robot0_agentview_left": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["actions"],
            },
        },
    )

    for dataset_path in raw_dir.glob("**/*.hdf5"):
         with h5py.File(dataset_path, "r") as raw_dataset:
            demos = raw_dataset["data"].keys()
            for demo in tqdm(demos):
                demo_length = len(raw_dataset["data"][demo]["actions"])
                demo_data = raw_dataset["data"][demo]

                left_images = demo_data["obs"]["robot0_agentview_left_image"][:]
                right_images = demo_data["obs"]["robot0_agentview_right_image"][:]
                wrist_images = demo_data["obs"]["robot0_eye_in_hand_image"][:]
                states = np.concatenate(
                    (
                        demo_data["obs"]["robot0_base_to_eef_pos"][:],
                        demo_data["obs"]["robot0_base_to_eef_quat"][:],
                        demo_data["obs"]["robot0_gripper_qpos"][:],
                    ),
                    axis=1,
                )
                actions = demo_data["actions"][:]
                for i in range(demo_length):
                    ep_meta = demo_data.attrs["ep_meta"]
                    ep_meta = json.loads(ep_meta)
                    lang = ep_meta["lang"]
                    dataset.add_frame(
                        {
                            "observation.images.robot0_agentview_right": right_images[i],
                            "observation.images.robot0_agentview_left": left_images[i],
                            "observation.images.robot0_eye_in_hand": wrist_images[i],
                            "observation.state": states[i].astype(np.float32),
                            "action": actions[i].astype(np.float32),
                            "task": lang,
                        },
                    )
                dataset.save_episode()
    dataset.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, repo_id=args.repo_id, local_dir=args.local_dir)
