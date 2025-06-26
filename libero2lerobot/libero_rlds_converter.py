#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Libero RLDS to LeRobot Dataset Converter

This module provides functionality to convert Libero RLDS format datasets to LeRobot format,
with support for custom feature configurations and output options.
"""

import argparse
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import logging

import numpy as np

# Check dependencies
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("tensorflow_datasets not installed. Run: pip install tensorflow tensorflow_datasets")


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
HAS_LEROBOT = True
HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_lerobot_dataset(
    repo_id: str,
    data_source: Union[str, Path, Any],
    robot_type: str = "panda",
    fps: int = 10,
    features: Optional[Dict[str, Dict[str, Any]]] = None,
    data_processor: Optional[Callable] = None,
    output_dir: Optional[Union[str, Path]] = None,
    push_to_hub: bool = False,
    hub_config: Optional[Dict[str, Any]] = None,
    clean_existing: bool = True,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
    run_compute_stats: bool = False,
    use_videos: bool = True,
) -> LeRobotDataset:
    """
    Convert data to LeRobot format dataset.
    
    Args:
        repo_id: Repository ID for the dataset (e.g., "username/dataset_name")
        data_source: Source data (path, dataset object, or any data structure)
        robot_type: Robot type (default: "panda")
        fps: Dataset frame rate
        features: Dictionary defining feature structure
        data_processor: Function to process raw data to LeRobot format
        output_dir: Output directory (default: LEROBOT_HOME)
        push_to_hub: Whether to push to Hugging Face Hub
        hub_config: Hub upload configuration
        clean_existing: Whether to clean existing dataset
        image_writer_threads: Number of image writer threads
        image_writer_processes: Number of image writer processes
        run_compute_stats: Whether to compute statistics after consolidation
        use_videos: Whether to use video format for storing images
        
    Returns:
        LeRobotDataset: Created dataset
    """
    
    if not HAS_LEROBOT:
        raise ImportError("lerobot package not installed. Please install lerobot first")
    
    # Set default features
    if features is None:
        features = get_default_libero_features(use_videos=use_videos)
    
    # Set default hub configuration
    if hub_config is None:
        hub_config = get_default_hub_config()
    
    # Determine output path
    if output_dir is None:
        lerobot_root = HF_LEROBOT_HOME
    else:
        lerobot_root = Path(output_dir)
        
    print("-----------------HF_LEROBOT_HOME", lerobot_root)
    # Set LEROBOT_HOME environment variable
    os.environ["LEROBOT_HOME"] = str(lerobot_root)

    # LeRobot always uses LEROBOT_HOME/repo_id as dataset root directory
    lerobot_dataset_dir = lerobot_root / repo_id

    # Clean existing dataset if requested
    if clean_existing and lerobot_dataset_dir.exists():
        logger.info(f"Cleaning existing dataset: {lerobot_dataset_dir}")
        shutil.rmtree(lerobot_dataset_dir)

    lerobot_root.mkdir(parents=True, exist_ok=True)

    # Create LeRobot dataset
    logger.info(f"Creating LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    
    # Use provided processor or default processor to process data
    if data_processor is None:
        data_processor = default_libero_processor
    
    logger.info("Processing data...")
    data_processor(dataset, data_source)
    
    # Consolidate dataset
    logger.info("Consolidating dataset...")
    dataset.consolidate(run_compute_stats=run_compute_stats)
    
    # Push to hub if requested
    if push_to_hub:
        logger.info("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(**hub_config)
    
    logger.info("Dataset conversion completed!")
    return dataset


def get_default_libero_features(use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
    """Get default feature configuration for Libero datasets."""
    image_dtype = "video" if use_videos else "image"
    
    return {
        "observation.images.front": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state_0", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6", "state_7"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action_0", "action_1", "action_2", "action_3", "action_4", "action_5", "action_6"],
        }
    }


def get_default_hub_config() -> Dict[str, Any]:
    """Get default configuration for Hugging Face Hub upload."""
    return {
        "tags": ["libero", "robotics", "lerobot", "panda"],
        "private": False,
        "push_videos": True,
        "license": "apache-2.0",
    }


def default_libero_processor(dataset: LeRobotDataset, data_source: Union[str, Path]):
    """
    Default processor for Libero RLDS datasets.
    
    Args:
        dataset: LeRobot dataset to write to
        data_source: Path to data directory containing RLDS datasets
    """
    if not HAS_TF:
        raise ImportError("tensorflow_datasets is required for Libero processing. Install with: pip install tensorflow tensorflow_datasets")
    
    # List of Libero dataset names
    raw_dataset_names = [
        "libero_10_no_noops",
        "libero_goal_no_noops", 
        "libero_object_no_noops",
        "libero_spatial_no_noops",
    ]
    
    episode_idx = 0
    
    for raw_dataset_name in raw_dataset_names:
        logger.info(f"Processing dataset: {raw_dataset_name}")
        
        try:
            # Load RLDS dataset
            raw_dataset = tfds.load(
                raw_dataset_name, 
                data_dir=data_source, 
                split="train",
                try_gcs=False  # Don't try to download from GCS
            )
            
            for episode in raw_dataset:
                logger.info(f"Processing episode {episode_idx + 1}")
                
                # Initialize task description
                task_str = f"episode_{episode_idx}"
                
                # Get first step to extract task string
                steps_list = list(episode["steps"].as_numpy_iterator())
                # Get task string
                if steps_list and "language_instruction" in steps_list[0]:
                    task_str = steps_list[0]["language_instruction"].decode()
                else:
                    task_str = f"episode_{episode_idx}"
                # Process each step in the episode
                for step_idx, step in enumerate(steps_list):
                    # Prepare frame data
                    frame_data = {
                        "observation.images.front": step["observation"]["image"],
                        "observation.images.wrist": step["observation"]["wrist_image"], 
                        "observation.state": step["observation"]["state"].astype(np.float32),
                        "action": step["action"].astype(np.float32),
                        "task": task_str,  # Each frame needs to contain task field
                    }
                    
                    # Add frame to dataset
                    dataset.add_frame(frame_data)
                dataset.save_episode()
                episode_idx += 1
                
        except Exception as e:
            logger.warning(f"Error processing dataset {raw_dataset_name}: {e}")
            continue

def convert_libero_dataset(
    data_dir: str,
    repo_id: str = "username/libero",
    push_to_hub: bool = False,
    use_videos: bool = True,
    robot_type: str = "panda",
    fps: int = 20,
    **kwargs
) -> LeRobotDataset:
    """
    Convert Libero RLDS dataset to LeRobot format.
    
    Args:
        data_dir: Directory containing Libero RLDS datasets
        repo_id: Repository ID for output dataset
        push_to_hub: Whether to push to Hugging Face Hub
        use_videos: Whether to use video format for storing images
        **kwargs: Additional arguments for convert_to_lerobot_dataset
        
    Returns:
        LeRobotDataset: Converted dataset
    """
    hub_config = {
        "tags": ["libero", "panda", "rlds", "manipulation"],
        "private": False,
        "push_videos": True,
        "license": "apache-2.0",
    }
    hub_config.update(kwargs.pop("hub_config", {}))
    
    return convert_to_lerobot_dataset(
        repo_id=repo_id,
        data_source=data_dir,
        robot_type=robot_type,
        fps=fps,  # Libero default frequency
        features=get_default_libero_features(use_videos=use_videos),
        data_processor=default_libero_processor,
        push_to_hub=push_to_hub,
        hub_config=hub_config,
        use_videos=use_videos,
        **kwargs
    )


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Convert Libero RLDS dataset to LeRobot format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python libero_rlds_converter.py \\
    --data-dir /path/to/libero/rlds/data \\
    --repo-id username/libero_dataset \\
    --output-dir /path/to/output \\
    --push-to-hub \\
    --use-videos

Supported Libero datasets:
  - libero_10_no_noops
  - libero_goal_no_noops  
  - libero_object_no_noops
  - libero_spatial_no_noops

For more information, see README.md documentation.
        """
    )
    
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory path containing Libero RLDS datasets"
    )
    required_group.add_argument(
        "--repo-id", 
        type=str,
        required=True,
        help="Repository ID for output dataset (format: username/dataset_name, e.g., cadene/libero_10)"
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local output directory (default: LEROBOT_HOME environment variable or ~/.cache/huggingface/lerobot)"
    )
    output_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to Hugging Face Hub after conversion (requires huggingface_hub login)"
    )
    output_group.add_argument(
        "--private",
        action="store_true",
        help="Create private dataset on Hub (default: public)"
    )
    output_group.add_argument(
        "--clean-existing",
        action="store_true",
        default=True,
        help="Clean existing dataset directory (default: True)"
    )
    
    # Data format configuration
    format_group = parser.add_argument_group('Data Format Configuration')
    format_group.add_argument(
        "--use-videos",
        action="store_true", 
        default=True,
        help="Use video format for storing image data (default: True, reduces storage by 60x)"
    )
    format_group.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "opencv"],
        help="Video encoding backend (default: pyav)"
    )
    format_group.add_argument(
        "--robot-type",
        type=str,
        default="panda",
        help="Robot type identifier (default: panda)"
    )
    format_group.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Dataset frame rate (default: 20 Hz)"
    )
    
    # Performance configuration
    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="Number of image writer processes (default: 5)"
    )
    perf_group.add_argument(
        "--image-writer-threads", 
        type=int,
        default=10,
        help="Number of image writer threads per process (default: 10)"
    )
    perf_group.add_argument(
        "--run-compute-stats",
        action="store_true",
        help="Compute dataset statistics after conversion (may take a long time)"
    )
    
    # Hub configuration
    hub_group = parser.add_argument_group('Hub Configuration')
    hub_group.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="Dataset license (default: apache-2.0)"
    )
    hub_group.add_argument(
        "--tags",
        nargs="+",
        default=["libero", "panda", "robotics", "lerobot"],
        help="Dataset tags (default: libero panda robotics lerobot)"
    )
    hub_group.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch name for Hub push (default: main)"
    )
    
    # Debug and logging
    debug_group = parser.add_argument_group('Debug and Logging')
    debug_group.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging output"
    )
    debug_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode, only check parameters and data source without executing conversion"
    )
    debug_group.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit maximum number of episodes to process (for testing, default: process all)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Validate arguments
    data_dir_path = Path(args.data_dir)
    if not data_dir_path.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    # Dry run mode
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info(f"Data source: {args.data_dir}")
        logger.info(f"Repository ID: {args.repo_id}")
        logger.info(f"Output directory: {args.output_dir or 'LEROBOT_HOME'}")
        logger.info(f"Push to Hub: {args.push_to_hub}")
        logger.info(f"Use videos: {args.use_videos}")
        logger.info(f"Robot type: {args.robot_type}")
        logger.info(f"Frame rate: {args.fps}")
        logger.info("Parameter validation passed, dry run completed")
        return
    
    # Build hub configuration
    hub_config = {
        "tags": args.tags,
        "private": args.private,
        "push_videos": args.use_videos,
        "license": args.license,
    }
    if args.branch:
        hub_config["branch"] = args.branch
    
    try:
        dataset = convert_libero_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            clean_existing=args.clean_existing,
            robot_type=args.robot_type,
            fps=args.fps,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            run_compute_stats=args.run_compute_stats,
        )
        
        logger.info(f"✅ Conversion completed! Dataset saved as: {args.repo_id}")
        logger.info(f"📁 Dataset path: {getattr(dataset, 'root', 'Unknown')}")
        
        # Display dataset statistics
        meta = getattr(dataset, 'meta', None)
        if meta:
            logger.info(f"📊 Dataset statistics:")
            logger.info(f"   - Total episodes: {getattr(meta, 'total_episodes', 'Unknown')}")
            logger.info(f"   - Total frames: {getattr(meta, 'total_frames', 'Unknown')}")
            logger.info(f"   - Robot type: {getattr(meta, 'robot_type', 'Unknown')}")
            logger.info(f"   - Frame rate: {getattr(meta, 'fps', 'Unknown')} Hz")
        
        if args.push_to_hub:
            logger.info(f"🚀 Dataset pushed to Hub: https://huggingface.co/datasets/{args.repo_id}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ User interrupted conversion process")
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Detailed error information:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
