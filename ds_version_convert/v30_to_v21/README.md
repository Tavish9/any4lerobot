# LeRobot Dataset v30 to v21

## Get started

1. Install v3.0 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
        --repo-id=your_id \
        --root=./test_data
    ```

- Optional: omit `--root` to rely on the default cache (`$HF_LEROBOT_HOME`), and add `--force-conversion` if you need to discard an existing snapshot and download a fresh copy.
