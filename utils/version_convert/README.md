# **OpenX to LeRobot Utils **

## What's New in This Script

> [!IMPORTANT]
>
> This is not a universally applicable method, so we decided not to save it as an executable python script, but to write it in a tutorial for reference by those who need it.
>
> If you are using `libx264` encoding and want to use `decord` as the video backend to speed up the stats conversion, or want to use the process pool to speed up the conversion when converting the huge dataset like droid, you can use this script. 
>
> However, please note that the droid dataset may get stuck at episode 5545 during the conversion process.

Key improvements:

- support loading the local dataset
- support use decord as video backend (NOTICE: decord is not supported to 'libsvtav1' encode method, we test it using 'libx264', ref: https://github.com/dmlc/decord/issues/319)
- support process pool for huge dataset like droid to accelerate conversation speed



`convert_dataset_v20_to_v21.py`:

```python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_episode_stats, EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_dataset_v20_to_v21 import V20, V21, SuppressWarnings
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats, sample_episode_video_frames
from lerobot.common.datasets.compute_stats import get_feature_stats


def convert_episode_stats(dataset: LeRobotDataset, ep_idx: int, is_parallel: bool = False):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            # We sample only for videos
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            ep_ft_data = np.array(ep_data[key])

        axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
        keepdims = True if ft["dtype"] in ["image", "video"] else ep_ft_data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:  # remove batch dim
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()}

    if not is_parallel:
        dataset.meta.episodes_stats[ep_idx] = ep_stats

    return ep_stats, ep_idx


def convert_stats_by_process_pool(dataset: LeRobotDataset, num_workers: int = 0):
    """Convert stats in parallel using multiple process."""
    assert dataset.episodes is None

    total_episodes = dataset.meta.total_episodes
    futures = []

    if num_workers > 0:
        max_workers = min(cpu_count() - 1, num_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ep_idx in range(total_episodes):
                futures.append(executor.submit(convert_episode_stats, dataset, ep_idx, True))
            for future in tqdm(as_completed(futures), total=total_episodes, desc="Converting episodes stats"):
                ep_stats, ep_idx = future.result()
                dataset.meta.episodes_stats[ep_idx] = ep_stats
    else:
        for ep_idx in tqdm(range(total_episodes)):
            convert_episode_stats(dataset, ep_idx)

    for ep_idx in tqdm(range(total_episodes)):
        write_episode_stats(ep_idx, dataset.meta.episodes_stats[ep_idx], dataset.root)


def convert_dataset(
    repo_id: str,
    root: str | None = None,
    push_to_hub: bool = False,
    delete_old_stats: bool = False,
    branch: str | None = None,
    num_workers: int = 4,
    video_backend: str = "pyav",
    use_process_pool: bool = True,
):
    with SuppressWarnings():
        if root is not None:
            dataset = LeRobotDataset(repo_id, root, revision=V20, video_backend=video_backend)
        else:
            dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True, video_backend=video_backend)

    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    if use_process_pool:
        convert_stats_by_process_pool(dataset, num_workers=num_workers)
    else:
        convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = V21
    write_info(dataset.meta.info, dataset.root)

    if push_to_hub:
        dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if delete_old_stats and (dataset.root / STATS_PATH).is_file:
        (dataset.root / STATS_PATH).unlink()

    hub_api = HfApi()
    if delete_old_stats and hub_api.file_exists(
        repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset")
    if push_to_hub:
        hub_api.create_tag(repo_id, tag=V21, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the local dataset root directory. If not provided, the script will use the dataset from local.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to the hub after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--delete-old-stats",
        action="store_true",
        help="Delete the old stats.json file after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "decord"],
        help="Video backend to use. Defaults to pyav.",
    )
    parser.add_argument(
        "--use-process-pool",
        action="store_true",
        help="Use process pool for parallelizing stats compute. Defaults to False.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))

```



## Installation

Install decord: https://github.com/dmlc/decord

## Get Start

### Default usage

This equal to lerobot projects, it will use dataset from huggingface hub, delete `stats.json` and push to huggingface hub (multi-thread and `pyav` as video backend), you can:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --delete-old-stats \
    --push-to-hub \
    --num-workers=8 \
    --video-backend=pyav
```



### Using `decord` as video backend

> [!IMPORTANT]
>
> 1.We recommend use default method to convert stats and use decord and process pool if you want to convert huge dataset like droid.
>
> 2.If you want to use decord as video backend, you should modify the `video_utils.py` source code from lerobot:
>
> ```python
> def decode_video_frames(
>     video_path: Path | str,
>     timestamps: list[float],
>     tolerance_s: float,
>     backend: str | None = None,
> ) -> torch.Tensor:
>     """
>     Decodes video frames using the specified backend.
> 
>     Args:
>         video_path (Path): Path to the video file.
>         timestamps (list[float]): List of timestamps to extract frames.
>         tolerance_s (float): Allowed deviation in seconds for frame retrieval.
>         backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..
> 
>     Returns:
>         torch.Tensor: Decoded frames.
> 
>     Currently supports torchcodec on cpu and pyav.
>     """
>     if backend is None:
>         backend = get_safe_default_codec()
>     if backend == "torchcodec":
>         return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
>     elif backend in ["pyav", "video_reader"]:
>         return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
>     elif backend == "decord":
>         return decode_video_frames_decord(video_path, timestamps)
>     else:
>         raise ValueError(f"Unsupported video backend: {backend}")
>     
>     
> def decode_video_frames_decord(
>     video_path: Path | str,
>     timestamps: list[float],
> ) -> torch.Tensor:
>     video_path = str(video_path)
>     vr = decord.VideoReader(video_path)
>     num_frames = len(vr)
>     frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
>     indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
>     frames = vr.get_batch(indices)
>     
>     frames_tensor = torch.tensor(frames.asnumpy()).type(torch.float32).permute(0, 3, 1, 2) / 255
>     return frames_tensor
> ```

This will load local dataset, use `decord` as video backend and process pool, you can:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --root=/home/path/to/your/lerobot/dataset/path \
    --num-workers=8 \
    --video-backend=decord \
    --use-process-pool
    
```

### Speed Test

Table I. dataset conversation time use stats.

| dataset              | episodes | video_backend | method  | workers | video_encode | Time  |
| -------------------- | -------- | ------------- | ------- | ------- | ------------ | ----- |
| bekerley_autolab_ur5 | 896      | pyav          | thread  | 16      | libx264      | 10:56 |
| bekerley_autolab_ur5 | 896      | pyav          | process | 16      | libx264      | --    |
| bekerley_autolab_ur5 | 896      | decord        | thread  | 16      | libx264      | 11:44 |
| bekerley_autolab_ur5 | 896      | decord        | process | 16      | libx264      | 14:26 |



