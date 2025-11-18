# LeRobot Dataset v30 to v21

The script download the original dataset from HuggingFace data repo, and writes the new data to the --output-path.

## Get started

1. Install v3.0 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py \
        --repo-id=your_id \
        --output-path=./test_data
    ```

- Optional: omit `--output-path` to rely on the default cache (`$HF_LEROBOT_HOME`), and add `--force-conversion` if you need to discard an existing snapshot and download a fresh copy.
