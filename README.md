<h1 align="center">
    <p>LeRobot: State-of-the-art AI for real-world robotics</p>
</h1>

> [!NOTE]  
> This repository supports converting datasets from OpenX format to LeRobot V2.0 dataset format.

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **OXE Standard Transformations** 🔄: We have integrated OXE's standard transformations to ensure uniformity across data.
- **Alignment of State and Action Information** 🤖: State and action information are now perfectly aligned, enhancing the clarity and coherence of the dataset.
- **Robot Type and Control Frequency** 📊: Annotations have been added for robot type and control frequency to improve dataset comprehensibility.
- **Joint Information** 🦾: Joint-specific details have been included to assist with fine-grained understanding.

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v2.0", // lastest lerobot format
  "robot_type": "franka", // specific robot type, unknown if not provided
  "fps": 3, // control frequency, 10 if not provided
  // will add an additional key "control_frequency"
  "features": {
    "observation.images.image_key": {
      "dtype": "video",
      "shape": [128, 128, 3],
      "names": ["height", "width", "rgb"], // bgr to rgb if needed
      "info": {
        "video.fps": 3.0,
        "video.height": 128,
        "video.width": 128,
        "video.channels": 3,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [8],
      "names": {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"] 
        // unified 8-dim vector: [xyz, state type, gripper], motor_x if not provided
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [7],
      "names": {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"] 
        // unified 7-dim vector: [xyz, action type, gripper], motor_x if not provided
      }
    }
  }
}
```

## Installation

Download lerobot code:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install 🤗 LeRobot:

```bash
pip install -e .
```

## Get started

> [!IMPORTANT]  
> 1.Before running the following code, modify `consolidate()` function in lerobot.
> ```python
> def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False, stat_kwargs: dict = {}) -> None:
>     ...
>     if run_compute_stats:
>         self.stop_image_writer()
>         # TODO(aliberts): refactor stats in save_episodes
>         self.meta.stats = compute_stats(self, **stat_kwargs)
>     ...
> ```
> 2.for `bc_z` dataset, two source codes need to be modified.
> 
> path: `lerobot/common/datasets/video_utils.py`
> 
> method: `encode_video_frames`
> ```python
> # add the following content to line 141:
> vf: str = "pad=ceil(iw/2)*2:ceil(ih/2)*2",
> # Add the following content to line 171:
> ffmpeg_args["-vf"] = vf
> ```

Download source code:

```bash
git clone https://github.com/Tavish9/openx2lerobot.git
```

Modify path in `convert.sh`:

```bash
python openx_rlds.py \
    --raw-dir /path/to/droid/1.0.0 \
    --local-dir /path/to/LEROBOT_DATASET \
    --repo-id your_hf_id \
    --push-to-hub \
    --batch-size 16 \
    --num-workers 8 \
    --use-videos
```

Execute the script:

```bash
bash convert.sh
```

## Available OpenX_LeRobot Dataset

We have upload most of the OpenX datasets in [huggingface](https://huggingface.co/IPEC-COMMUNITY)🤗.

You can visualize the dataset in this [space](https://huggingface.co/spaces/IPEC-COMMUNITY/openx_dataset_lerobot_v2.0).

## The `LeRobotDataset` format

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}` one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](examples/1_load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of en episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ episode_data_index: contains 2 tensors with the start and end indices of each episode
  │  ├ from (1D int64 tensor): first frame index for each episode — shape (num episodes,) starts with 0
  │  └ to: (1D int64 tensor): last frame index for each episode — shape (num episodes,)
  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  ├ observation.images.cam_high: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  ...
  ├ info: a dictionary of metadata on the dataset
  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  ├ fps (float): frame per second the dataset is recorded/synchronized to
  │  ├ video (bool): indicates if frames are encoded in mp4 video files to save space or stored as png files
  │  └ encoding (dict): if video, this documents the main options that were used with ffmpeg to encode the videos
  ├ videos_dir (Path): where the mp4 videos or png images are stored/accessed
  └ camera_keys (list of string): the keys to access camera features in the item returned by the dataset (e.g. `["observation.images.cam_high", ...]`)
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:

- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can use the `local_files_only` argument and specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

## Acknowledgment

Special thanks to the [Lerobot teams](https://github.com/huggingface/lerobot) for making this great framework.
