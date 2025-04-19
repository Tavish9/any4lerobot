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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from huggingface_hub import HfApi
from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_episode_stats, write_info
from lerobot.common.datasets.v21.convert_dataset_v20_to_v21 import V20, V21, SuppressWarnings
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats, sample_episode_video_frames
from tqdm import tqdm


def convert_episode_stats(dataset: LeRobotDataset, ep_idx: int, is_parallel: bool = False):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            # We sample only for videos
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
            ep_ft_data = ep_ft_data[None, ...] if ep_ft_data.ndim == 3 else ep_ft_data
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