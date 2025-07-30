import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """
    implemented by @Cadene
    """
    # validate same fps, robot_type, features

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}.")
        if features != meta.features:
            raise ValueError(f"Same features is expected, but got features={meta.features} instead of {features}.")

    return fps, robot_type, features
