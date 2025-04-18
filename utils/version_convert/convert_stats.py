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

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_episode_stats
from lerobot.common.datasets.v21.convert_stats import sample_episode_video_frames
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
