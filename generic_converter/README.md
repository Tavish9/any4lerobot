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

A dataset converter should subclass `BaseAdapter`, pass `output_path` to the
base constructor, and provide dataset-level metadata as class attributes.

Required attributes:

- `dataset_type`
- `fps`
- `robot_type`
- `features`

Optional attributes:

- `tags`

Required methods:

- `load_tasks(self) -> list[ConversionTask]`
- `load_subset(self, task: ConversionTask) -> Iterable[Sequence[dict]]`

`run_converter` reads `adapter.output_path` and calls `adapter.load_tasks()`
without arguments. Store paths, task manifests, or other adapter options on the
adapter instance in `__init__`.

Use `adapter.temp_output_path` when building task-level temporary output paths.

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

## Usage Sketch

```python
from generic_converter import BaseAdapter, ConversionTask, run_converter


class MyAdapter(BaseAdapter):
    dataset_type = "my_dataset"
    fps = 20
    robot_type = "my_robot"
    features = MY_FEATURES
    tags = ["my_dataset"]

    def __init__(self, output_path):
        super().__init__(output_path)

    def load_tasks(self) -> list[ConversionTask]:
        ...

    def load_subset(self, task: ConversionTask):
        ...


run_converter(
    adapter=adapter,
    executor="local",
    cpus_per_task=1,
    tasks_per_job=1,
    workers=-1,
)
```
