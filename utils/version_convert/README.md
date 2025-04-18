https://github.com/huggingface/lerobot/pull/962

convert stats in parallel using multiple thread and decord video backend

This script will help you convert any LeRobot dataset using process pool and decord video backend 
from codebase version 2.0 to 2.1.

Usage:

Please install decord first: https://github.com/dmlc/decord

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --num_workers=32 \
    --video-backend="decord" \
    --use-process-pool
```