This script will help you convert any LeRobot dataset using process pool and decord video backend 
from codebase version 2.0 to 2.1.

Usage:

Please install decord first: https://github.com/dmlc/decord

the default usage, this equal to lerobot projects, this will use dataset from huggingface hub, delete stats.json and push to huggingface hub (multi-thread and pyav video backend), you can:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --delete-old-stats \
    --push-to-hub \
    --num-workers=8 \
    --video-backend=pyav
```

if you want to don't delete stats.json form lerobot dataset v2.0, use local dataset and don't push to huggingface hub, this will use decord video backend and thread pool to accelerate processing, you can:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --root=/home/path/to/your/lerobot/dataset/path \
    --num-workers=32 \
    --video-backend=decord \
    --use-process-pool
```

||||||
|--|--|--|--|--|
|pyav|thread|16|libx264|10:56|
|pyav|process|16|libx264||
|decord|thread|16|libx264|11:44|
|decord|process|16|libx264|14:26|
