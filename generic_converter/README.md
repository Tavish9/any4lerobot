# Generic Converter

Shared conversion flow for turning task-based source datasets into LeRobot
datasets.

The generic package owns the execution mechanics:

- create one temporary `LeRobotDataset` per `ConversionTask`
- run tasks with a local or Ray Datatrove executor
- aggregate temporary datasets into the adapter output directory
- remove temporary task outputs by default
- optionally push the aggregated dataset to the Hub

Dataset-specific converters own the adapter logic:

- where raw inputs come from
- how tasks are discovered or loaded
- how one raw input is converted into LeRobot episodes
- how task metadata, such as language instructions, is represented

## Files

- `adapter.py`: `BaseAdapter`, the class dataset adapters inherit from.
- `pipeline.py`: the reusable conversion, executor, aggregation, cleanup, and push flow.
- `utils.py`: shared types and small helpers.

## Adapter Contract

A dataset converter should subclass `BaseAdapter` and provide dataset-level
metadata as class attributes.

Required attributes:

- `dataset_type`
- `fps`
- `robot_type`
- `features`
- `output_path`

Optional attributes:

- `tags`

Required methods:

- `load_tasks(self) -> list[ConversionTask]`
- `load_subset(self, task: ConversionTask) -> Iterable[Sequence[dict]]`

`run_converter` calls `adapter.load_tasks()` without arguments. Store paths,
task manifests, or other adapter options on the adapter instance in `__init__`.

`load_subset` receives the full `ConversionTask`, not just an input path. Use
`task.input_path` for raw data and `task.metadata` for dataset-specific values
such as language instructions. Each yielded episode must be a sequence of frame
dictionaries accepted by `LeRobotDataset.add_frame`; each frame should include
the LeRobot `task` field when language tasks are needed.

## ConversionTask

`ConversionTask` describes one independently convertible raw input:

- `input_path`: source file or directory
- `output_path`: temporary LeRobot dataset directory for this task
- `local_repo_id`: repo id used while writing the temporary dataset
- `metadata`: adapter-owned metadata

Keep dataset-specific values in `metadata`; the generic pipeline does not know
about task-file schemas or instruction formats.

## Minimal Adapter

```python
from pathlib import Path

from generic_converter import BaseAdapter, ConversionTask, run_converter


class MyAdapter(BaseAdapter):
    dataset_type = "my_dataset"
    fps = 20
    robot_type = "my_robot"
    features = MY_FEATURES
    tags = ["my_dataset"]

    def __init__(self, src_paths: list[Path], output_path: Path):
        self.src_paths = [path.expanduser().resolve() for path in src_paths]
        self.output_path = output_path.expanduser().resolve()

    def load_tasks(self) -> list[ConversionTask]:
        tasks = []
        temp_root = self.output_path.with_name(f"{self.output_path.name}_temp")
        for src_path in self.src_paths:
            for raw_file in src_path.glob("*.hdf5"):
                tasks.append(
                    ConversionTask(
                        input_path=raw_file.resolve(),
                        output_path=(temp_root / src_path.name / raw_file.stem).resolve(),
                        local_repo_id=f"{src_path.name}/{raw_file.name}",
                        metadata={"task": raw_file.stem.replace("_", " ")},
                    )
                )
        return tasks

    def load_subset(self, task: ConversionTask):
        for raw_episode in load_raw_episodes(task.input_path):
            yield [
                {
                    **frame,
                    "task": task.metadata["task"],
                }
                for frame in raw_episode
            ]


adapter = MyAdapter(src_paths=[Path("raw")], output_path=Path("output/my_dataset"))
run_converter(
    adapter=adapter,
    executor="local",
    cpus_per_task=4,
    tasks_per_job=1,
    workers=4,
)
```

## Execution Notes

`executor="local"` uses Datatrove's local executor.

`executor="ray"` uses Datatrove's Ray executor. The pipeline builds a Ray
runtime environment with `PYTHONPATH` entries for the repository root, current
working directory, the current Python path, and the existing environment
`PYTHONPATH` so Ray workers can import local packages such as
`generic_converter`.

`debug=True` forces local execution, sets `workers=1`, disables Hub upload, and
only runs the first two tasks.

`cleanup_temp=True` removes each task's temporary `output_path` after
aggregation.

## What Stays In Adapters

Task manifests are dataset-specific. If a converter wants JSON, CSV, YAML, or
another task description format, parse it inside that converter's adapter.

Adapters should also own dataset-specific naming, metadata inference, and raw
file parsing. The generic pipeline should stay limited to execution and
LeRobot dataset assembly.
