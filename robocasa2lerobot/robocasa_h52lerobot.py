"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import h5py
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List


REPO_NAME = "your-username/your-repo-name" # E.x: "binhng/panda-kitchen-100demos-lerobot"
DATA_DIR = "direction/to/your/hdf5/files"
RAW_DATASET_NAME = [
    f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5")
]
print(RAW_DATASET_NAME)
OUT_DATA_DIR = "direction/to/your/output/dataset" # E.x: OUT_DATA_DIR="/home/binhng/Workspace/Custom_pi0_lerobot/robocasa_output/"

def main(data_dir: str = DATA_DIR, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    # output_path = LEROBOT_HOME / REPO_NAME
    output_path = os.path.join(OUT_DATA_DIR, REPO_NAME)
    # if output_path.exists():
    #     shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=20,
        features={
            "observation.right_image": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.left_image": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.wrist_image": {
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
        image_writer_threads=10,
        image_writer_processes=5,
        video_backend="torchcodec"
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for dataset_name in RAW_DATASET_NAME:
        raw_dataset = h5py.File(os.path.join(data_dir, dataset_name), "r")
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
                        "observation.right_image": right_images[i],
                        "observation.left_image": left_images[i],
                        "observation.wrist_image": wrist_images[i],
                        "observation.state": states[i].astype(np.float32),
                        "action": actions[i].astype(np.float32),
                    },
                    task=lang,
                )

            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        from huggingface_hub import login
        
        # HF_TOKEN=hf_token

        login(token="your_hf_token")
        dataset.push_to_hub(
            tags=["robocasa", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )



if __name__ == "__main__":
    # tyro.cli(main)
    main(push_to_hub=True)
    