# LeRobot to RLDS

RLDS stands for Reinforcement Learning Datasets and it is an ecosystem of tools to store, retrieve and manipulate episodic data in the context of Sequential Decision Making including Reinforcement Learning (RL), Learning for Demonstrations, Offline RL or Imitation Learning.

For more details, please check [official repo](https://github.com/google-research/rlds).

## ✨ Motivation

Some classic works like [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), etc. currently only support reading the RLDS format. To meet the community’s needs, we provide a script that converts the popular LERobot format into the RLDS format.

## 🚀 What's New in This Script

- **Complete Data Preservation**: Retains all original information from the lerobot dataset, including diverse image keys, depth maps, and associated metadata.  
- **TFDS Conversion Simplified**: Implements the first Python-based workflow to launch TensorFlow Datasets (TFDS) conversions with native support for parallel Beam processing. 
- **Customizable RLDS Metadata**: Enables flexible customization of RLDS dataset metadata fields (e.g., citations, descriptions, versioning) through a unified configuration interface.  

## Installation

1. Install LeRobot:  
    Follow instructions in [official repo](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation).

2. Install others:
    For saving tfds/rlds, we need to install `tensorflow-datasets`:
    ```bash
    pip install tensorflow
    pip install tensorflow-datasets
    ```
    If you want to enable beam processing:
    ```bash
    pip install apache-beam
    ```


## Get started

> [!WARNING]
> - Beam processing is implemented for speed improvements, but may exhibit occasional instability with Apache Beam. 
> - If your dataset is small, or you want to safely save all the data, we recommend disabling beam processing.
> - If partial episode loss is acceptable for performance gains, enable beam by adding `--enable-beam`.


### Download source code:

```bash
git clone https://github.com/Tavish9/any4lerobot.git
```

### Modify path in `convert.sh`:

```bash
python lerobot2rlds.py \
    --src-dir /path/to/lerobot/dataset \
    --output-dir /path/to/rlds_dir \
    --task-name default_task
```

### Customizing rlds:
```bash
    --encoding-format png \
    --version 1.0.0 \
    --citation "@{...}"
```

For more flags, check `python lerobot2rlds.py --help`

### Execute the script:

```bash
cd lerobot2rlds && bash convert.sh
```