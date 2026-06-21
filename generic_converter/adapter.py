from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .utils import ConversionTask, FeatureSpec


class BaseAdapter(ABC):
    """Dataset-specific hooks used by the generic conversion pipeline."""

    dataset_type: str
    fps: int
    robot_type: str
    features: FeatureSpec
    tags: Sequence[str] = ()

    def __init__(self, output_path: Path):
        self.output_path = output_path.expanduser().resolve()

    @property
    def temp_output_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.name}_temp")

    @abstractmethod
    def load_tasks(self) -> list[ConversionTask]:
        """Build conversion tasks from dataset-specific inputs."""

    @abstractmethod
    def load_subset(self, task: ConversionTask) -> Iterable[Any]:
        """Yield LeRobot episodes for one raw input path."""

    def create_dataset(self, task: ConversionTask):
        """Create the temporary LeRobot dataset for one conversion task."""
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset.create(
            repo_id=task.local_repo_id,
            root=task.output_path,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
        )

    def save_episode(
        self,
        dataset: Any,
        episode_data: Any,
        task: ConversionTask,
    ) -> bool:
        """Save one episode to the temporary dataset.

        Adapters can override this when a dataset needs extra per-episode
        arguments or a non-standard writer.
        """
        for frame in episode_data:
            dataset.add_frame(frame)
        dataset.save_episode()
        return True

    def get_episode_length(self, episode_data: Any) -> int:
        return len(episode_data)
