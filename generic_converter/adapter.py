from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

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

    @abstractmethod
    def load_tasks(self) -> list[ConversionTask]:
        """Build conversion tasks from dataset-specific inputs."""

    @abstractmethod
    def load_subset(
        self, task: ConversionTask
    ) -> Iterable[Sequence[dict]]:
        """Yield LeRobot episodes for one raw input path."""
