# Generic Converter

Shared building blocks for source-dataset-to-LeRobot converters.

Files:

- `adapter.py`: `BaseAdapter`, the class dataset converters inherit from.
- `pipeline.py`: the full conversion flow: write temporary LeRobot datasets, aggregate, clean up, and optionally push.
- `utils.py`: shared data structures and small framework helpers.

The intended adapter flow is:

1. Subclass `BaseAdapter`.
2. Set dataset-level fields: `dataset_type`, `fps`, `robot_type`, `features`, and optional `tags`.
3. Implement `load_tasks(...)` to discover or load `ConversionTask` objects. Use `metadata`
   for dataset-specific values such as language task strings.
4. Implement `load_subset(input_path, metadata)` to yield episodes of frame dictionaries.
5. Call `run_converter(..., adapter=adapter)`.

Minimal shape:

```python
class MyAdapter(BaseAdapter):
    dataset_type = "my_dataset"
    fps = 20
    robot_type = "robot"
    features = MY_FEATURES

    def load_tasks(self, output_path, *, src_paths=None, tasks_file=None):
        ...

    def load_subset(self, input_path, metadata):
        ...
```

Task manifests are dataset-specific. If a converter wants JSON/CSV/YAML task
files, implement that parser in its adapter rather than in this generic
package.

This generic converter is intentionally scoped to source datasets that can be
converted into normal LeRobot episodes. Datasets that require custom
LeRobotDataset writers or custom episode metadata can still reuse pieces here,
but should not force-fit into this pipeline.

Temporary task datasets are removed by default after aggregation. Pass
`cleanup_temp=False` to `run_converter` when debugging temporary outputs.
