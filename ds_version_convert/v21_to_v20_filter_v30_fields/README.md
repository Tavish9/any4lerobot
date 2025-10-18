# LeRobot Dataset v21 to v20 (filter v30 fields)

## Get started

1. Install lerobot

2. Run the converter:
    ```bash
    python ds_version_convert/v21_to_v20_filter_v30_fields/convert_dataset_v21_to_v20.py \
        --repo-id=your_id \
        --output-dir=/tmp/lerobot_datasets \
        --download-media
    ```

- Optional: omit `--download-media` to only download metadata, or add `--delete-episodes-stats` to remove episode-level files after conversion.
