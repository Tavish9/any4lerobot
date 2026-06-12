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

    @abstractmethod
    def load_tasks(
        self,
        output_path: Path,
        *,
        src_paths: Sequence[Path] | None = None,
        tasks_file: Path | None = None,
    ) -> list[ConversionTask]:
        """Build conversion tasks from dataset-specific inputs."""

    @abstractmethod
    def load_subset(
        self, task: ConversionTask
    ) -> Iterable[Sequence[dict]]:
        """Yield LeRobot episodes for one raw input path."""
