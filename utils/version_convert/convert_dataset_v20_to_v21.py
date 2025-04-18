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

"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.0 to
2.1. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial
```

"""

import argparse
import logging

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats, convert_stats_by_process_pool

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


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

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
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
        hub_api.delete_file(
            path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )
    if push_to_hub:
        hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


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
        choices=["pyav", "decorid"],
        help="Video backend to use. Defaults to pyav.",
    )
    parser.add_argument(
        "--use-process-pool",
        action="store_true",
        help="Use process pool for parallelizing stats compute. Defaults to False.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
