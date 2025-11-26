"""
The converter defined in this module discovers the current dataset version, finds a
conversion route to the requested target version, and executes each hop through a
registered handler. Handlers can be bound to the existing pairwise scripts in this
folder or to custom conversion logic implemented by downstream users.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.utils import INFO_PATH


class ConversionPathError(RuntimeError):
    """Raised when a conversion path between two versions cannot be found."""


@dataclass
class DatasetContext:
    """Container storing the dataset information shared across conversion steps."""

    repo_id: str
    source_root: Path
    output_root: Path
    root: Path | None = None
    local_dir: Path | None = None
    working_dir: Path | None = None
    current_version: str | None = None
    metadata: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    current_root: Path | None = None
    step_index: int = 0

    def __post_init__(self) -> None:
        if self.current_root is None:
            self.current_root = self.source_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def get_version(self) -> str:
        """Return the current dataset version."""
        if self.current_version:
            return self.current_version

        if self.metadata and "codebase_version" in self.metadata:
            self.current_version = self.metadata["codebase_version"]
            return self.current_version

        active_root = self.current_root or self.root
        if active_root:
            info_path = active_root / INFO_PATH
            if info_path.is_file():
                with info_path.open() as f:
                    self.metadata = json.load(f)
                if "codebase_version" not in self.metadata:
                    raise ValueError(f"Missing 'codebase_version' in {info_path}")
                self.current_version = self.metadata["codebase_version"]
                return self.current_version

        raise ValueError(
            "Unable to infer current version. "
            "Set dataset version manually or provide a dataset root with meta/info.json."
        )

    def update_version(self, version: str) -> None:
        """Update the cached version after a conversion step."""
        self.current_version = version
        self.metadata = None  # Force re-read on demand.

    def get_value(self, key: str) -> Any:
        """Resolve values for handlers from the context or the extras dictionary."""
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value
        if key in self.extras:
            return self.extras[key]
        return None

    def set_extra(self, key: str, value: Any) -> None:
        """Store arbitrary metadata that may be required by downstream handlers."""
        self.extras[key] = value

    def allocate_destination(self, version: str) -> Path:
        stem = self.repo_id.replace("/", "_") or "dataset"
        dest_name = f"{stem}_step{self.step_index:02d}_{version}"
        self.step_index += 1
        dest_path = self.output_root / dest_name
        if dest_path.exists():
            shutil.rmtree(dest_path)
        dest_path.mkdir(parents=True, exist_ok=True)
        return dest_path


class BaseConversionHandler:
    """Abstract base class representing a directed conversion step."""

    source_version: str
    target_version: str

    def __init__(
        self,
        source_version: str,
        target_version: str,
        description: str | None = None,
        *,
        supports_inplace: bool = False,
    ):
        self.source_version = source_version
        self.target_version = target_version
        self.description = description or f"{source_version} -> {target_version}"
        self.supports_inplace = supports_inplace

    @property
    def edge_key(self) -> Tuple[str, str]:
        return (self.source_version, self.target_version)

    def convert(self, context: DatasetContext, **kwargs: Any) -> None:  # pragma: no cover - abstract method
        raise NotImplementedError


class FunctionConversionHandler(BaseConversionHandler):
    """
    Conversion handler that wraps an existing callable.

    Args:
        source_version: Version that the handler expects as input.
        target_version: Version produced after the handler succeeds.
        func: Callable that performs the conversion.
        context_bindings: Mapping between function parameter names and context attributes.
        static_kwargs: Keyword arguments passed to every invocation.
        required_params: Parameters that must be provided either through bindings,
            static kwargs, or the runtime kwargs supplied when executing the plan.
    """

    def __init__(
        self,
        source_version: str,
        target_version: str,
        func: Callable[..., Any],
        *,
        context_bindings: Mapping[str, str] | None = None,
        static_kwargs: Mapping[str, Any] | None = None,
        required_params: Iterable[str] | None = None,
        description: str | None = None,
        supports_inplace: bool = False,
    ):
        super().__init__(source_version, target_version, description=description, supports_inplace=supports_inplace)
        self.func = func
        self.context_bindings = dict(context_bindings or {})
        self.static_kwargs = dict(static_kwargs or {})
        self.required_params = set(required_params or [])

    def convert(self, context: DatasetContext, **runtime_kwargs: Any) -> None:
        resolved_kwargs: Dict[str, Any] = {}
        for param_name, attr_name in self.context_bindings.items():
            value = context.get_value(attr_name)
            if value is not None:
                resolved_kwargs[param_name] = value
        resolved_kwargs.update(self.static_kwargs)
        resolved_kwargs.update(runtime_kwargs)
        missing = [param for param in self.required_params if param not in resolved_kwargs]
        if missing:
            raise ValueError(
                f"Missing required parameters {missing} while executing conversion "
                f"{self.source_version}->{self.target_version}"
            )
        self.func(**resolved_kwargs)


class LeRobotDatasetVersionConverter:
    """High-level orchestrator that manages multi-step conversions between dataset versions."""

    def __init__(self):
        self._handlers: Dict[Tuple[str, str], BaseConversionHandler] = {}

    def register_handler(self, handler: BaseConversionHandler, *, overwrite: bool = False) -> None:
        """Register a handler for a source->target conversion."""
        if handler.edge_key in self._handlers and not overwrite:
            raise ValueError(f"Handler already registered for edge {handler.edge_key}")
        self._handlers[handler.edge_key] = handler

    def available_versions(self) -> Sequence[str]:
        """Return the set of versions known by the converter."""
        versions = {edge[0] for edge in self._handlers} | {edge[1] for edge in self._handlers}
        return sorted(versions)

    def list_handlers(self) -> List[BaseConversionHandler]:
        return list(self._handlers.values())

    def _find_path(self, start: str, target: str) -> List[BaseConversionHandler]:
        if start == target:
            return []

        adjacency: Dict[str, List[BaseConversionHandler]] = {}
        for handler in self._handlers.values():
            adjacency.setdefault(handler.source_version, []).append(handler)

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            for handler in adjacency.get(current, []):
                if handler.target_version in visited:
                    continue
                new_path = path + [handler]
                if handler.target_version == target:
                    return new_path
                visited.add(handler.target_version)
                queue.append((handler.target_version, new_path))

        raise ConversionPathError(f"No conversion path from {start} to {target}")

    def convert(
        self,
        *,
        context: DatasetContext,
        target_version: str,
        step_kwargs: Mapping[Tuple[str, str], Mapping[str, Any]] | None = None,
    ) -> List[BaseConversionHandler]:
        """
        Convert the dataset represented by `context` into `target_version`.

        Args:
            context: DatasetContext describing the dataset and shared parameters.
            target_version: Desired version.
            step_kwargs: Optional mapping providing extra keyword arguments per step.

        Returns:
            The ordered list of handlers that were executed.
        """
        step_kwargs = dict(step_kwargs or {})
        current_version = context.get_version()
        if current_version == target_version:
            return []

        plan = self._find_path(current_version, target_version)
        for idx, handler in enumerate(plan):
            extra_kwargs = dict(step_kwargs.get(handler.edge_key, {}))
            source_root = context.current_root
            if source_root is None:
                raise ValueError("DatasetContext.current_root must be set before running conversions.")
            inplace_step = idx > 0 and handler.supports_inplace
            if inplace_step:
                dest_root = source_root
                extra_kwargs.setdefault("inplace", True)
            else:
                dest_root = context.allocate_destination(handler.target_version)
            extra_kwargs.setdefault("source_root", source_root)
            extra_kwargs.setdefault("dest_root", dest_root)
            handler.convert(context, **extra_kwargs)
            context.update_version(handler.target_version)
            context.current_root = dest_root

        return plan


def build_default_converter() -> LeRobotDatasetVersionConverter:
    """
    Build a converter pre-loaded with handlers that wrap the scripts distributed in
    the `ds_version_convert` directory. Each handler expects users to provide the
    arguments required by the underlying conversion script through the `step_kwargs`
    argument when calling `LeRobotDatasetVersionConverter.convert`.
    """

    converter = LeRobotDatasetVersionConverter()

    from .local_pipelines import (
        convert_v16_to_v20_local,
        convert_v20_to_v21_local,
        convert_v21_to_v20_filtered_local,
        convert_v21_to_v30_local,
        convert_v30_to_v21_local,
    )

    converter.register_handler(
        FunctionConversionHandler(
            "v1.6",
            "v2.0",
            convert_v16_to_v20_local,
            required_params={"source_root", "dest_root"},
            description="Convert legacy v1.6 datasets to the v2.0 layout (local copy).",
        )
    )

    converter.register_handler(
        FunctionConversionHandler(
            "v2.0",
            "v2.1",
            convert_v20_to_v21_local,
            required_params={"source_root", "dest_root"},
            supports_inplace=True,
            description="Upgrade v2.0 datasets to v2.1 locally.",
        )
    )

    converter.register_handler(
        FunctionConversionHandler(
            "v2.1",
            "v2.0",
            convert_v21_to_v20_filtered_local,
            required_params={"source_root", "dest_root"},
            supports_inplace=True,
            description="Downgrade v2.1 datasets to v2.0 using filtered stats.",
        )
    )

    converter.register_handler(
        FunctionConversionHandler(
            "v2.1",
            "v3.0",
            convert_v21_to_v30_local,
            required_params={"source_root", "dest_root"},
            description="Upgrade datasets to the v3.0 format locally.",
        )
    )

    converter.register_handler(
        FunctionConversionHandler(
            "v3.0",
            "v2.1",
            convert_v30_to_v21_local,
            required_params={"source_root", "dest_root"},
            description="Downgrade v3.0 datasets to v2.1 locally.",
        )
    )

    return converter


__all__ = [
    "DatasetContext",
    "BaseConversionHandler",
    "FunctionConversionHandler",
    "LeRobotDatasetVersionConverter",
    "ConversionPathError",
    "build_default_converter",
]
