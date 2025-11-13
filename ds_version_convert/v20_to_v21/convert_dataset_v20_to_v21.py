import argparse

from .convert_stats import check_aggregate_stats, convert_stats
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import LEGACY_EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
V20 = "v2.0"
V21 = "v2.1"


def convert_dataset(
    repo_id: str,
    root: str | None = None,
    push_to_hub: bool = False,
    delete_old_stats: bool = False,
    branch: str | None = None,
    num_workers: int = 4,
):
    if root is not None:
        dataset = LeRobotDataset(repo_id, root, revision=V20)
    else:
        dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True)

    if (dataset.root / LEGACY_EPISODES_STATS_PATH).is_file():
        (dataset.root / LEGACY_EPISODES_STATS_PATH).unlink()

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

    args = parser.parse_args()
    convert_dataset(**vars(args))
