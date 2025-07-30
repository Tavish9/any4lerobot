# AgiBot-World to LeRobot

AgiBot World, the first large-scale robotic learning dataset designed to advance multi-purpose robotic policies. It is accompanied by foundation models, benchmarks, and an ecosystem to democratize access to high-quality robot data for the academic community and the industry, paving the path towards the "ImageNet Moment" for Embodied AI. (Copied from [docs](https://agibot-world.com/))

## ⚠️ Dirty Tasks

| (Gripper) Task ID | (Some episodes) Reason | Fixed By |
| :---------------: | :--------------------: | -------- |
|     task_352      | action_len > state_len | skipping |
|     task_354      | action_len > state_len | skipping |
|     task_359      | action_len > state_len | skipping |
|     task_361      | action_len > state_len | skipping |
|     task_368      | action_len > state_len | skipping |
|     task_376      | action_len > state_len | skipping |
|     task_377      | action_len > state_len | skipping |
|     task_380      |     corrupted mp4      | skipping |
|     task_384      |     corrupted mp4      | skipping |
|     task_410      | action_len > state_len | skipping |
|     task_414      | action_len > state_len | skipping |
|     task_421      | action_len > state_len | skipping |
|     task_428      |     corrupted mp4      | skipping |
|     task_460      |     corrupted mp4      | skipping |
|     task_505      |     corrupted mp4      | skipping |
|     task_510      |     corrupted mp4      | skipping |
|     task_711      |     corrupted mp4      | skipping |

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **Preservation of Agibot’s Original Information** 🧠: We have preserved as much of Agibot’s original information as possible, with field names strictly adhering to the original dataset’s naming conventions to ensure compatibility and consistency.
- **State and Action as Dictionaries** 🧾: The traditional one-dimensional state and action have been transformed into dictionaries, allowing for greater flexibility in designing custom states and actions, enabling modular and scalable handling.

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v2.1", // lastest lerobot format
  "robot_type": "a2d", // specific robot type
  "fps": 30, // control frequency
  "features": {
    "observation.images.image_key": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "rgb"],
      "info": {
        "video.fps": 3.0,
        "video.height": 128,
        "video.width": 128,
        "video.channels": 3,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    // for more states key, see config.py
    "observation.states.joint.position": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "motors": [
          "left_arm_0",
          "left_arm_1",
          "left_arm_2",
          "left_arm_3",
          "left_arm_4",
          "left_arm_5",
          "left_arm_6",
          "right_arm_0",
          "right_arm_1",
          "right_arm_2",
          "right_arm_3",
          "right_arm_4",
          "right_arm_5",
          "right_arm_6"
        ]
      }
    },
    "observation.states.head.position": {
        "dtype": "float32",
        "shape": [
            2
        ],
        "names": {
            "motors": [
                "yaw",
                "patch"
            ]
        }
    },
    ...
    // for more actions key, see config.py
    "actions.head.position": {
      "dtype": "float32",
      "shape": [2],
      "names": {
        "motors": ["yaw", "patch"]
      }
    },
    "actions.waist.position": {
      "dtype": "float32",
      "shape": [2],
      "names": {
        "motors": ["pitch", "lift"]
      }
    },
    ...
  }
}
```

## Installation

1. Install LeRobot:  
   Follow instructions in [official repo](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation).

2. Install others:  
   We use ray for parallel conversion, significantly speeding up data processing tasks by distributing the workload across multiple cores or nodes (if any).
   ```bash
   pip install h5py
   pip install -U "ray[default]"
   ```

## Get started

> [!IMPORTANT]  
> 1.If you want to save depth when converting the dataset, modify `_assert_type_and_shape()` function in [lerobot.datasets.compute_stats.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py).
>
> ```python
> def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
>     for i in range(len(stats_list)):
>         for fkey in stats_list[i]:
>             for k, v in stats_list[i][fkey].items():
>                 if not isinstance(v, np.ndarray):
>                     raise ValueError(
>                         f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
>                     )
>                 if v.ndim == 0:
>                     raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
>                 if k == "count" and v.shape != (1,):
>                     raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
>                 # bypass depth check
>                 if "image" in fkey and k != "count":
>                     if "depth" not in fkey and v.shape != (3, 1, 1):
>                         raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")
>                     if "depth" in fkey and v.shape != (1, 1, 1):
>                         raise ValueError(f"Shape of '{k}' must be (1,1,1), but is {v.shape} instead.")
> ```

> [!NOTE]
> The conversion speed of this script is limited by the performance of the physical machine running it, including **CPU cores and memory**. We recommend using **3 CPU cores per task** for optimal performance. However, each task requires approximately 20 GiB of memory. To avoid running out of memory, you may need to increase the number of CPU cores per task depending on your system’s available memory.

### Download source code:

```bash
git clone https://github.com/Tavish9/any4lerobot.git
```

### Modify path in `convert.sh`:

There are three types of end-effector, `gripper`, `dexhand` and `tactile`, specify the type before converting

```bash
python convert.py \
    --src-path /path/to/AgiBotWorld-Beta \
    --output-path /path/to/local \
    --eef-type gripper \
    --num-cpus-per-task 3
```

### Execute the script:

#### For single node

```bash
bash convert.sh
```

#### For multi nodes

**Direct Access to Nodes (2 nodes in example)**

On Node 1:

```bash
ray start --head --port=6379
```

On Node 2:

```bash
ray start --address='node_1_ip:6379'
```

On either Node, check the ray cluster status, and start the script

```bash
ray status
bash convert.sh
```

**Slurm-managed System**

```bash
#!/bin/bash
#SBATCH --job-name=ray-cluster
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --partition=partition

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$port \
    --block &

sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start \
        --address "$ip_head" \
        --block &
    sleep 5
done

sleep 10

bash convert.sh
```

**Other Community Supported Cluster Managers**

See the [doc](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/index.html) for more details.
