# LeRobot Dataset v16 to v20

## Get started

1. Install v2.0 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    git checkout c574eb49845d48f5aad532d823ef56aec1c0d0f2
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python convert_dataset_v16_to_v20.py \
        --repo-id=your_id \
        --single-task=task_desc \
        --tasks-col=task_column_name \
        --tasks-path=path_to_json \
    ```