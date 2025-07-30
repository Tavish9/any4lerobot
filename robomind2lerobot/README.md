# RoboMIND to LeRobot

RoboMIND (Multi-embodiment Intelligence Normative Data for Robot Manipulation), a dataset containing 107k demonstration trajectories across 479 diverse tasks involving 96 object classes. RoboMIND is collected through human teleoperation and encompasses comprehensive robotic-related information, including multi-view observations, proprioceptive robot state information, and linguistic task descriptions.. (Copied from [docs](https://x-humanoid-robomind.github.io/))

## ⚠️ Dirty Tasks

|              Task ID              |                      Reason                      |
| :-------------------------------: | :----------------------------------------------: |
|          3_eggplantOven           |           take - turn on, wrong order            |
|         3_eggplantoven_2          |           take - turn on, wrong order            |
|            5_eggoven_2            |                  no instruction                  |
|           10_packplate            |           no plate marker, no plate 2            |
|          10_packplate_2           |           no plate marker, no plate 2            |
|            11_brushcup            |                     two cups                     |
|            12_packcup             |                  no cup marker                   |
|            13_packbowl            |      no bowl marker, blue - greeen flipped       |
|           35_putcarrot            |                  no instruction                  |
|           36_putpepper            |                  no instruction                  |
|             37_putegg             |                  no instruction                  |
|           39_puttomato            |                  no instruction                  |
|           40_putavocado           |                  no instruction                  |
|            41_putplum             |                  no instruction                  |
|         42_putkiwifruite          |               wrong word: wifruite               |
|           43_packplate            |      last object should be "them", not "it"      |
|    44_putbluebowlongreenplate     |      last object should be "them", not "it"      |
|    45_putgreenbowlonblueplate     |     only one instruction, but two sub-tasks      |
|     46_putredbowlonwhiteplate     |     only one instruction, but two sub-tasks      |
| 48_putpotatogreenplatefromsteam_2 | last action is about "left arm", not "right arm" |
|           52_holdercup            |   wrong order: should be right - left - right    |
|            53_stackcup            |   wrong order: should be right - left - right    |
|        to be continued ...        |                                                  |

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **Preservation of RoboMIND’s Original Information** 🧠: We have preserved as much of RoboMIND’s original information as possible, with field names strictly adhering to the original dataset’s naming conventions to ensure compatibility and consistency.
- **State and Action as Dictionaries** 🧾: The traditional one-dimensional state and action have been transformed into dictionaries, allowing for greater flexibility in designing custom states and actions, enabling modular and scalable handling.

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v2.1", // lastest lerobot format
  "robot_type": "franka_3rgb", // specific robot type
  "fps": 30, // control frequency
  "features": {
    "observation.images.image_key": {
        "dtype": "video",
        "shape": [
            720,
            1280,
            3
        ],
        "names": [
            "height",
            "width",
            "rgb"
        ],
        "info": {
            "video.height": 720,
            "video.width": 1280,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": false
        }
    },
    // for more states key, see configs
    "observation.states.end_effector": {
        "dtype": "float32",
        "shape": [
            6
        ],
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "r",
                "p",
                "y"
            ]
        }
    },
    ...
    // for more actions key, see configs
    "actions.joint_position": {
        "dtype": "float32",
        "shape": [
            8
        ],
        "names": {
            "motors": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper"
            ]
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
>
> 1. If you want to save depth when converting the dataset, modify `_assert_type_and_shape()` function in [lerobot.datasets.compute_stats.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py).
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
>
> 2. The dataset needs to be organized into the following format before running the script due to differences in storage formats across platforms:
>
> ```bash
> /path/to/robomind/
> ├── benchmark1_0_release
> │   ├── h5_agilex_3rgb
> │   │   ├── 10_packplate
> │   │   ├── ...
> │   ├── h5_franka_1rgb
> │   │   ├── bread_in_basket
> │   └── ...
> ├── benchmark1_1_release
> │   ├── h5_agilex_3rgb
> │   │   ├── 20_takecorn_2
> │   │   ├── ...
> │   ├── h5_franka_3rgb
> │   │   ├── apples_placed_on_a_ceramic_plate
> │   └── ...
> ├── benchmark1_2_release
> │   ├── h5_franka_3rgb
> │   │   └── 241223_upright_cup
> │   └── h5_sim_franka_3rgb
> │       ├── 408-place_upright_mug_on_the_left_middle
> │       ├── ...
> ├── language_description_annotation_json
> │   ├── h5_agilex_3rgb.json
> │   ├── h5_franka_1rgb.json
> │   ├── h5_franka_3rgb.json
> │   ├── h5_simulation_franka.json
> │   ├── h5_tienkung_xsens.json
> │   └── h5_ur_1rgb.json
> └── RoboMIND_v1_2_instr.csv
> ```

> [!NOTE]
> The conversion speed of this script is limited by the performance of the physical machine running it, including **CPU cores and memory**. We recommend using **2 CPU cores per task** for optimal performance. However, each task requires approximately 10 GiB of memory. To avoid running out of memory, you may need to increase the number of CPU cores per task depending on your system’s available memory.

### Download source code:

```bash
git clone https://github.com/Tavish9/any4lerobot.git
```

### Modify path in `convert.sh`:

There are three benchmarks, each with several embodiments, including `agilex_3rgb`, `franka_1rgb`, `franka_3rgb`, `franka_fr3_dual`, `tienkung_gello_1rgb`, `tienkung_prod1_gello_1rgb`, `tienkung_xsens_1rgb`, `ur_1rgb`.

```bash
python robomind_h5.py \
    --src-path /path/to/robomind/ \
    --output-path /path/to/local \
    --benchmark benchmark1_1_release \
    --embodiments agilex_3rgb franka_1rgb \
    --cpus-per-task 2
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
