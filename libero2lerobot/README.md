# LIBERO to LeRobot

LIBERO consists of 4 task suites and 130 tasks for studying LLDM. Specifically, the tasks in 3 of the 4 task suites vary only in one type of knowledge, while the last task suite requires transfer of entangled knowledge. (Copied from [docs](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html))

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **OpenVLA-based LIBERO Regeneration**: Resolution enhancement, No-op action filtration, 180° RGB frame rotation, Failed trajectory filtering.
- **State Data Preservation**: Maintained native LIBERO state information (accessible via `states.ee_state`, `states.joint_state` and etc.).
- **Robust Conversion Pipeline**: Using DataTrove framework for High-speed dataset transformation and automatic failure recovery during conversion

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v2.1", // lastest lerobot format
  "robot_type": "franka", // specific robot type
  "fps": 20, // control frequency
  "features": {
    "observation.images.image": {
        "dtype": "video",
        "shape": [
            256,
            256,
            3
        ],
        "names": [
            "height",
            "width",
            "rgb"
        ],
        "info": {
            "video.height": 256,
            "video.width": 256,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": false,
            "video.fps": 20,
            "video.channels": 3,
            "has_audio": false
        }
    },
    // for more states key, see configs
    "observation.state": {
        "dtype": "float32",
        "shape": [
            8
        ],
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
                "gripper"
            ]
        }
    },
    ...
    "action": {
        "dtype": "float32",
        "shape": [
            7
        ],
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
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
   We use datatrove[ray] for parallel conversion, significantly speeding up data processing tasks by distributing the workload across multiple cores or nodes (if any).
   ```bash
   pip install h5py
   pip install -U datatrove
   pip install -U "datatrove[ray]" # if you want ray features
   ```

## Get started

> [!NOTE]
> This script supports converting from original hdf5 to lerobot. If you want to convert from rlds to lerobot, check [openx2lerobot](../openx2lerobot/README.md).

### Download source code:

```bash
git clone https://github.com/Tavish9/any4lerobot.git
```

### Regenerate LIBERO Trajectory:

1. [Install LIBERO dependency](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#installtion) 
2. Replace `libero_90` with your target libero dataset.

```bash
python libero_utils/regenerate_libero_dataset.py \
    --resolution 256 \
    --libero_task_suite libero_90 \
    --libero_raw_data_dir /path/to/libero/datasets/libero_90 \
    --libero_target_dir /path/to/libero/datasets/libero_90_no_noops
```

### Modify in `convert.sh`:

1. If you have installed `datatrove[ray]`, we recommend using `ray` executor for faster conversion.
2. Increase `workers` and `tasks-per-job` if you have sufficient computing resources.
3. To merge many datasets into one, simply specify both paths like: `--src-paths /path/libero_10 /path/libero_90`
4. To resume from a previous conversion, provide the appropriate log directory using `--resume-from-save` and `--resume-from-aggregate`
5. If you want different image resolution, regenerate the trajectory, and change the [config](./libero_utils/config.py). (DO NOT use resize)

```bash
python libero_h5.py \
    --src-paths /path/to/libero/ \
    --output-path /path/to/local \
    --executor local \
    --tasks-per-job 3 \
    --workers 10
```

### Execute the script:

#### For single node

```bash
bash convert.sh
```

#### For multi nodes (Install ray first)

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
