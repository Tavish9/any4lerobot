# Libero2LeRobot

Tool for converting Libero RLDS format datasets to LeRobot format, following the design style of other any4lerobot projects.


## Installation

```bash
# Install tensorflow dependencies
pip install tensorflow tensorflow_datasets

# Install other dependencies
pip install numpy huggingface_hub datasets Pillow tqdm
```

## Quick Start

### Basic Conversion
```bash
python run_converter.py \
  --data-dir /path/to/libero/data \
  --repo-id username/libero_dataset \
```

### Full Conversion with Hub Push
```bash
python run_converter.py \
  --data-dir /path/to/libero/data \
  --repo-id username/libero_dataset \
  --push-to-hub \
  --use-videos \
  --verbose
```

## Command Line Arguments

Similar to ds_version_convert and openx2lerobot parameter style:

### Required Arguments
```bash
--data-dir          # Libero RLDS dataset directory path
--repo-id           # Repository ID for output dataset (username/dataset_name)
```

### Output Configuration
```bash
--output-dir        # Local output directory
--push-to-hub       # Push to Hugging Face Hub
--private           # Create private dataset
```

### Data Format
```bash
--use-videos        # Use video format for storing images (default: True)
--robot-type        # Robot type (default: panda)
--fps               # Frame rate (default: 20)
```

### Performance Configuration
```bash
--image-writer-processes    # Number of image writer processes (default: 5)
--image-writer-threads      # Number of image writer threads (default: 10)
```

### Hub Configuration
```bash
--license           # Dataset license (default: apache-2.0)
--tags              # Dataset tags
```

### Debug Options
```bash
--verbose           # Verbose logging output
--dry-run           # Dry run mode
```

## Supported Datasets

- `libero_10_no_noops`: 10 basic tasks
- `libero_goal_no_noops`: Goal-oriented tasks  
- `libero_object_no_noops`: Object manipulation tasks
- `libero_spatial_no_noops`: Spatial reasoning tasks

## Output Format

The converted dataset follows the LeRobot standard format:

```
dataset_directory/
├── meta/
│   ├── episodes.jsonl     # Episode metadata
│   ├── modality.json      # Modality information
│   ├── info.json          # Dataset information
│   └── tasks.jsonl        # Task descriptions
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        └── episode_000001.parquet
```

## Examples

### 1. Basic Usage
```bash
python run_converter.py \
  --data-dir /media/bigdisk/Isaac-GR00T/demo_data/libero \
  --repo-id pony/libero_lerobot
```

### 2. Advanced Usage
```bash
python run_converter.py \
  --data-dir /media/bigdisk/Isaac-GR00T/demo_data/libero \
  --repo-id cadene/libero_10_lerobot \
  --output-dir /media/bigdisk/Isaac-GR00T/demo_data/libero_lerobot_datasets \
  --push-to-hub \
  --use-videos \
  --fps 20 \
  --robot-type panda \
  --image-writer-processes 8 \
  --image-writer-threads 12 \
  --tags libero panda manipulation robotics lerobot \
  --license apache-2.0 \
  --verbose
```

### 3. Dry Run
```bash
python run_converter.py \
  --data-dir /path/to/libero/data \
  --repo-id test/libero_dataset \
  --dry-run
```
