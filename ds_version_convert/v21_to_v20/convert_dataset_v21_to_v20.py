import argparse

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, write_info, write_stats
from lerobot.datasets.v21.convert_dataset_v20_to_v21 import V20, V21


def convert_dataset(
    repo_id: str,
    root: str | None = None,
    push_to_hub: bool = False,
    delete_old_stats: bool = False,
    branch: str | None = None,
):
    if root is not None:
        dataset = LeRobotDataset(repo_id, root, revision=V21)
    else:
        dataset = LeRobotDataset(repo_id, revision=V21, force_cache_sync=True)

    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    write_stats(dataset.meta.stats, dataset.root)

    dataset.meta.info["codebase_version"] = V20
    write_info(dataset.meta.info, dataset.root)

    if push_to_hub:
        dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if delete_old_stats and (dataset.root / EPISODES_STATS_PATH).is_file:
        (dataset.root / EPISODES_STATS_PATH).unlink()

    hub_api = HfApi()
    if delete_old_stats and hub_api.file_exists(
        repo_id=dataset.repo_id, filename=EPISODES_STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(
            path_in_repo=EPISODES_STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )
    if push_to_hub:
        hub_api.create_tag(repo_id, tag=V20, revision=branch, repo_type="dataset")


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

    args = parser.parse_args()
    convert_dataset(**vars(args))
