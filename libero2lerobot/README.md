# LIBERO to LeRobot

LIBERO consists of 4 task suites and 130 tasks for studying LLDM. Specifically, the tasks in 3 of the 4 task suites vary only in one type of knowledge, while the last task suite requires transfer of entangled knowledge. (Copied from [docs](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html))

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **OpenVLA-based LIBERO Regeneration**: Resolution enhancement, No-op action filtration, 180° RGB frame rotation, Failed trajectory filtering.
- **State Data Preservation**: Maintained native LIBERO state information (accessible via `states.ee_state`, `states.joint_state` and etc.).
- **Robust Conversion Pipeline**: Using the shared `generic_converter` pipeline with local and Ray DataTrove executors for high-speed dataset transformation and resumable conversion.

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v3.0", // latest lerobot format
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
    "observation.images.wrist_image": {
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
    // for more state keys, see LiberoAdapter.features in libero_h5.py
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
                "axis_angle1",
                "axis_angle2",
                "axis_angle3",
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
                "axis_angle1",
                "axis_angle2",
                "axis_angle3",
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
   We use DataTrove for conversion. Install the Ray extra if you want distributed execution across multiple cores or nodes.
   ```bash
   pip install h5py
   pip install -U datatrove
   pip install -U "datatrove[ray]" # optional, for --executor ray
   ```

## Get started

> [!NOTE]
> This script supports converting LIBERO-style HDF5 directories to LeRobot. If you want to convert from RLDS to LeRobot, check [openx2lerobot](../openx2lerobot/README.md).

### Download source code:

```bash
git clone https://github.com/Tavish9/any4lerobot.git
cd any4lerobot/libero2lerobot
```

### Regenerate LIBERO Trajectory:

1. [Install LIBERO dependency](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#installtion) 
2. Replace `libero_90` with your target libero dataset.
3. The converter feature schema expects `256x256x3` RGB observations. If your source HDF5 files are the original `128x128` LIBERO files, regenerate them first with `--resolution 256`, or update the image feature shapes in `libero_h5.py` to match your data.

```bash
python regenerate_libero_dataset.py \
    --resolution 256 \
    --libero_task_suite libero_90 \
    --libero_raw_data_dir /path/to/libero/datasets/libero_90 \
    --libero_target_dir /path/to/libero/datasets/libero_90_no_noops
```

### Modify in `convert.sh`:

1. `--src-paths` accepts one or more directories containing `*.hdf5` LIBERO task files. To merge many suites into one LeRobot dataset, specify all source directories, for example `--src-paths /path/libero_10 /path/libero_90`.
2. `--output-path` is the final aggregated LeRobot dataset root. Temporary per-task datasets are written next to it under `<output-name>_temp` and removed after aggregation.
3. If you have installed `datatrove[ray]`, use `--executor ray` for faster conversion. Increase `--workers`, `--tasks-per-job`, and `--cpus-per-task` if you have enough CPU and memory.
4. To resume a previous conversion, pass the existing DataTrove log directory with `--resume-dir /path/to/logs/...`.
5. Use `--debug` for a small local smoke test. It converts only the first two tasks, forces local execution, and disables Hub upload.
6. Use `--repo-id <namespace/name>` together with `--push-to-hub` to upload the aggregated dataset. Without `--push-to-hub`, `--repo-id` only controls the local aggregate repo id.

```bash
python libero_h5.py \
    --src-paths /path/to/libero/datasets/libero_90_no_noops \
    --output-path /path/to/local/libero_90_lerobot \
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
