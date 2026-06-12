import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TaskMetadata = Mapping[str, Any]
FeatureSpec = Mapping[str, dict]


@dataclass(frozen=True)
class ConversionTask:
    """One independently convertible raw input file and adapter metadata."""

    input_path: Path
    output_path: Path
    local_repo_id: str | None = None
    metadata: TaskMetadata = field(default_factory=dict)


def setup_logger():
    import sys

    from datatrove.utils.logging import logger

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    return logger


def unique_strings(values: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result
