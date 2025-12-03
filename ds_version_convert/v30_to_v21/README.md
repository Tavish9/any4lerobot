# LeRobot Dataset v30 to v21

## Get started

1. Downgrade datasets:

   ```bash
   pip install "datasets<4.0.0"
   ```

   > Need to downgrade datasets first since `4.0.0` introduces `List` and `Column`.

2. Install v3.0 lerobot

   ```bash
   git clone https://github.com/huggingface/lerobot.git
   pip install -e .
   ```

2. Run the converter:
   ```bash
   python convert_dataset_v30_to_v21.py \
       --repo-id=your_id \
       --root=your_local_dir
   ```
