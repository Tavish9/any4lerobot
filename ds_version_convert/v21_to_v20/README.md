# LeRobot Dataset v21 to v20

## Get started

1. Install v2.1 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    git checkout d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python convert_dataset_v21_to_v20.py \
        --repo-id=your_id \
        --root=your_local_dir \
        --delete-old-stats \
        --push-to-hub
    ```