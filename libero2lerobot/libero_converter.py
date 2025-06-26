# -*- coding: utf-8 -*-
"""
LIBERO Unified Data Converter

Supports automatic detection of RLDS and HDF5 formats, auto-parses to LeRobotDataSet format with multi-threading support
"""

import argparse
import ast
import json
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import concurrent.futures
from functools import partial

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# 检查依赖
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("tensorflow_datasets not installed, RLDS support will be disabled")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetFormatDetector:
    """Dataset format detector"""
    
    @staticmethod
    def detect_format(data_path: Union[str, Path]) -> str:
        """
        Automatically detect dataset format
        
        Args:
            data_path: Dataset path
            
        Returns:
            str: 'rlds' or 'hdf5'
        """
        data_path = Path(data_path)
        
        # Check HDF5 format: look for .hdf5 or .h5 files
        hdf5_files = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        # Check RLDS format: look for tfrecord files or dataset_info.json
        rlds_indicators = (
            list(data_path.rglob("*.tfrecord*")) + 
            list(data_path.rglob("dataset_info.json")) +
            list(data_path.rglob("features.json"))
        )
        
        if hdf5_files and not rlds_indicators:
            logger.info(f"Detected HDF5 format, found {len(hdf5_files)} HDF5 files")
            return "hdf5"
        elif rlds_indicators and not hdf5_files:
            logger.info(f"Detected RLDS format, found related files: {[f.name for f in rlds_indicators[:3]]}")
            return "rlds"
        elif hdf5_files and rlds_indicators:
            logger.warning("Found both HDF5 and RLDS files, prioritizing HDF5 format")
            return "hdf5"
        else:
            raise ValueError(f"Unable to detect data format: {data_path}")


class HDF5Processor:
    """HDF5 data processor"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), use_videos: bool = False):
        """
        Initialize HDF5 processor
        
        Args:
            image_size: Image size (height, width) - matches numpy array format
            use_videos: Whether to use video format
        """
        self.image_size = image_size  # (height, width)
        self.use_videos = use_videos
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get default feature configuration for LIBERO dataset"""
        image_dtype = "video" if use_videos else "image"
        
        return {
            "observation.images.front": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"state_{i}" for i in range(7)],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"action_{i}" for i in range(7)],
            }
        }
    
    def process_episode(self, episode_path: Path, dataset: LeRobotDataset, task_name: str) -> bool:
        """
        Process single episode data
        
        Args:
            episode_path: Episode file path
            dataset: LeRobot dataset
            task_name: Task name
            
        Returns:
            bool: Whether processing was successful
        """
        try:
            with h5py.File(episode_path, "r") as file:
                logger.debug(f"HDF5 file keys: {list(file.keys())}")
                
                # Detect HDF5 file format, support multiple LIBERO formats
                if "data" in file:
                    # New LIBERO format: data/demo_N/...
                    return self._process_libero_demo_format(file, dataset, task_name)
                else:
                    logger.warning(f"Unrecognized HDF5 format: {episode_path}")
                    return False

        except (FileNotFoundError, OSError, KeyError) as e:
            logger.error(f"Skipping {episode_path}: {str(e)}")
            return False
            

    def _process_libero_demo_format(self, file: h5py.File, dataset: LeRobotDataset, task_name: str) -> bool:
        """Process new LIBERO demo format: data/demo_N/..."""
        data_group = file["data"]
        
        # Get all demos
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # Sort by number
        
        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            logger.info(f"Processing {demo_key}")
            
            # Read action data
            actions = np.array(demo_group["actions"])
            
            # Read observation data
            obs_group = demo_group["obs"]
            
            # Read joint states - as observation.state
            joint_states = np.array(obs_group["joint_states"])
            
            # Read image data
            agentview_rgb = np.array(obs_group["agentview_rgb"])  # Front camera
            eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # Wrist camera
            
            # Ensure all arrays have consistent length
            num_frames = min(len(actions), len(joint_states), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # Process each frame
            for i in tqdm(range(num_frames), desc=f"Processing {demo_key}", leave=False):
                # Process images: resize to target dimensions
                front_img = cv2.resize(agentview_rgb[i], (self.image_size[1], self.image_size[0]))
                wrist_img = cv2.resize(eye_in_hand_rgb[i], (self.image_size[1], self.image_size[0]))
                
                # Prepare frame data - reference RLDS format
                frame_data = {
                    "task": task_name,
                    "action": actions[i].astype(np.float32),
                    "observation.state": joint_states[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                }
                
                # Add frame to dataset
                dataset.add_frame(frame_data)
            
            # Save each demo as one episode
            dataset.save_episode()
        
        return True

    def _extract_libero_demo_frames(self, file: h5py.File, task_name: str) -> List[List[Dict]]:
        """Extract LIBERO demo format frame data (for multi-threading) - returns data grouped by demo"""
        demos_frames = []
        data_group = file["data"]
        
        # Get all demos
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # Sort by number
        
        for demo_key in demo_keys:
            demo_frames = []
            demo_group = data_group[demo_key]
            
            # Read action data
            actions = np.array(demo_group["actions"])
            
            # Read observation data
            obs_group = demo_group["obs"]
            
            # Read joint states - as observation.state
            joint_states = np.array(obs_group["joint_states"])
            
            # Read image data
            agentview_rgb = np.array(obs_group["agentview_rgb"])  # Front camera
            eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # Wrist camera
            
            # Ensure all arrays have consistent length
            num_frames = min(len(actions), len(joint_states), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # Process each frame
            for i in range(num_frames):
                # Process images: resize to target dimensions
                front_img = cv2.resize(agentview_rgb[i], (self.image_size[1], self.image_size[0]))
                wrist_img = cv2.resize(eye_in_hand_rgb[i], (self.image_size[1], self.image_size[0]))
                
                # Prepare frame data - reference RLDS format
                frame_data = {
                    "task": task_name,
                    "action": actions[i].astype(np.float32),
                    "observation.state": joint_states[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                }
                demo_frames.append(frame_data)
            
            demos_frames.append(demo_frames)
        
        return demos_frames

    def _extract_direct_episode_frames(self, file: h5py.File, task_name: str) -> List[List[Dict]]:
        """Extract direct episode format frame data (fallback option for multi-threading) - returns data grouped by episode"""
        # Fallback: treat entire file as one episode
        episode_frames = []
        
        # Try to find possible state data
        state_keys = ["joint_states", "states", "robot_states", "state"]
        state_data = None
        
        for key in state_keys:
            if key in file:
                state_data = np.array(file[key])
                logger.info(f"Found state data: {key}, shape: {state_data.shape}")
                break
        
        if state_data is None:
            raise KeyError("No state data found")
        
        # Try to find action data
        action_keys = ["actions", "action"]
        action_data = None
        
        for key in action_keys:
            if key in file:
                action_data = np.array(file[key])
                break
        
        if action_data is None:
            logger.warning("No action data found, using state data as actions")
            action_data = state_data
        
        # Try to find image data
        image_keys = ["agentview_rgb", "images", "rgb"]
        image_data = None
        
        for key in image_keys:
            if key in file:
                image_data = np.array(file[key])
                break
        
        if image_data is None:
            logger.warning("No image data found, will create empty images")
            image_data = np.zeros((len(state_data), *self.image_size, 3), dtype=np.uint8)
        
        # Ensure all arrays have consistent length
        num_frames = min(len(state_data), len(action_data), len(image_data))
        
        # Process each frame
        for i in range(num_frames):
            # Process images
            if image_data.ndim == 4:  # Time dimension
                img = cv2.resize(image_data[i], (self.image_size[1], self.image_size[0]))
            else:  # No time dimension, use first image
                img = cv2.resize(image_data[0] if len(image_data) > 0 else np.zeros((*self.image_size, 3), dtype=np.uint8), (self.image_size[1], self.image_size[0]))
            
            frame_data = {
                "task": task_name,
                "action": action_data[i].astype(np.float32),
                "observation.state": state_data[i].astype(np.float32),
                "observation.images.front": img,
                "observation.images.wrist": img,  # Use same image as wrist view
            }
            episode_frames.append(frame_data)
        
        return [episode_frames]  # Return list of single episode


class RLDSProcessor:
    """RLDS data processor"""
    
    def __init__(self):
        if not HAS_TF:
            raise ImportError("tensorflow_datasets is RLDS processing required, please run: pip install tensorflow tensorflow_datasets")
        # Add compatible attributes with HDF5Processor
        self.image_size = (256, 256)
        self.use_videos = False
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get default feature configuration for LIBERO dataset"""
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
                "names": [f"state_{i}" for i in range(8)],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"action_{i}" for i in range(7)],
            }
        }
    
    def process_dataset(self, dataset: LeRobotDataset, data_source: Union[str, Path]):
        """Process RLDS dataset"""
        # List of LIBERO dataset names. Modify according to your needs
        raw_dataset_names = [
            "libero_10_no_noops",
            "libero_goal_no_noops", 
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ]
        
        episode_idx = 0
        
        for raw_dataset_name in raw_dataset_names:
            logger.info(f"Processing RLDS dataset: {raw_dataset_name}")
            
            try:
                # Load RLDS dataset
                raw_dataset = tfds.load(
                    raw_dataset_name, 
                    data_dir=data_source, 
                    split="train",
                    try_gcs=False
                )
                
                for episode in raw_dataset:
                    logger.info(f"Processing episode {episode_idx + 1}")
                    
                    # Get task description
                    steps_list = list(episode["steps"].as_numpy_iterator())
                    task_str = f"episode_{episode_idx}"
                    
                    if steps_list and "language_instruction" in steps_list[0]:
                        task_str = steps_list[0]["language_instruction"].decode()
                    
                    # Process each step in episode
                    for step_idx, step in enumerate(steps_list):
                        frame_data = {
                            "observation.images.front": step["observation"]["image"],
                            "observation.images.wrist": step["observation"]["wrist_image"], 
                            "observation.state": step["observation"]["state"].astype(np.float32),
                            "action": step["action"].astype(np.float32),
                            "task": task_str,
                        }
                        dataset.add_frame(frame_data)
                    
                    dataset.save_episode()
                    episode_idx += 1
                    
            except Exception as e:
                logger.warning(f"Error processing dataset {raw_dataset_name}: {e}")
                continue


class UnifiedConverter:
    """Unified converter class"""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize unified converter
        
        Args:
            num_workers: Number of parallel processing worker threads
        """
        self.num_workers = num_workers
        self.detector = DatasetFormatDetector()
    
    def convert_dataset(
        self,
        data_dir: Union[str, Path],
        repo_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        use_videos: bool = True,
        robot_type: str = "panda",
        fps: int = 20,
        task_name: str = "default_task",
        hub_config: Optional[Dict[str, Any]] = None,
        clean_existing: bool = True,
        image_writer_threads: int = 10,
        image_writer_processes: int = 5,
        run_compute_stats: bool = False,
        **kwargs
    ) -> LeRobotDataset:
        """
        Unified conversion interface
        
        Args:
            data_dir: Data directory path
            repo_id: Dataset repository ID
            output_dir: Output directory
            push_to_hub: Whether to push to Hub
            use_videos: Whether to use video format
            robot_type: Robot type
            fps: Frame rate
            task_name: Task name (used for HDF5 format)
            hub_config: Hub configuration
            clean_existing: Whether to clean existing dataset
            image_writer_threads: Number of image writing threads
            image_writer_processes: Number of image writing processes
            run_compute_stats: Whether to compute statistics
            
        Returns:
            LeRobotDataset: Converted dataset
        """
        data_path = Path(data_dir)
        
        # Automatically detect format
        format_type = self.detector.detect_format(data_path)
        logger.info(f"Detected data format: {format_type}")
        
        # Select processor and features based on format
        if format_type == "hdf5":
            processor = HDF5Processor()
            features = processor.get_default_features(use_videos)
        else:  # rlds
            processor = RLDSProcessor()
            features = processor.get_default_features(use_videos)
        
        # Set output path
        if output_dir is None:
            lerobot_root = HF_LEROBOT_HOME
        else:
            lerobot_root = Path(output_dir)
        
        os.environ["LEROBOT_HOME"] = str(lerobot_root)
        lerobot_dataset_dir = lerobot_root / repo_id
        
        # Clean existing dataset
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
        
        # Process data
        if format_type == "hdf5":
            self._process_hdf5_data(processor, dataset, data_path, task_name)
        else:  # rlds
            processor.process_dataset(dataset, data_path)
        
        # Consolidate dataset
        logger.info("Consolidating dataset...")
        dataset.consolidate(run_compute_stats=run_compute_stats)
        
        # Push to Hub
        if push_to_hub:
            if hub_config is None:
                hub_config = self._get_default_hub_config()
            logger.info("Pushing to Hugging Face Hub...")
            dataset.push_to_hub(**hub_config)
        
        logger.info("✅ Dataset conversion completed!")
        return dataset
    
    def _process_hdf5_data(self, processor: Union[HDF5Processor, RLDSProcessor], dataset: LeRobotDataset, data_path: Path, task_name: str):
        """Use multi-threading to process HDF5 data"""
        # Ensure HDF5Processor
        if not isinstance(processor, HDF5Processor):
            raise TypeError("processor must be HDF5Processor instance")
            
        # Find all episodes
        episodes = []
        for ep_dir in data_path.iterdir():
            if ep_dir.is_dir():
                ep_path = ep_dir / "data" / "trajectory.hdf5"
                if ep_path.exists():
                    episodes.append(ep_path)
        
        if not episodes:
            # Directly find HDF5 files
            episodes = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        logger.info(f"Found {len(episodes)} episode files")
        
        if self.num_workers == 1:
            # Single-thread processing
            for ep_path in tqdm(episodes, desc="Processing Episodes"):
                processor.process_episode(ep_path, dataset, task_name)
                logger.info(f"Processing completed: {ep_path.name}")
        else:
            # Multi-thread processing
            self._process_episodes_parallel(processor, dataset, episodes, task_name)
    
    def _process_episodes_parallel(self, processor: HDF5Processor, dataset: LeRobotDataset, episodes: List[Path], task_name: str):
        """Parallel processing of episodes, using unified method of HDF5Processor"""
        # Create processing function
        def process_single_episode(ep_path: Path) -> Tuple[Path, bool, List[List[Dict]]]:
            """Process single episode and return frame data"""
            demos_frames = []
            try:
                with h5py.File(ep_path, "r") as file:
                    logger.debug(f"Multi-thread processing HDF5 file keys: {list(file.keys())}")
                    
                    # Use same detection and processing logic as HDF5Processor
                    if "data" in file:
                        # New LIBERO format: data/demo_N/...
                        demos_frames = processor._extract_libero_demo_frames(file, task_name)
                    else:
                        # Try fallback processing for other formats
                        demos_frames = processor._extract_direct_episode_frames(file, task_name)
                    
                return ep_path, True, demos_frames
            except Exception as e:
                logger.error(f"Processing {ep_path} failed: {e}")
                return ep_path, False, []
        
        # Parallel processing
        logger.info(f"Using {self.num_workers} worker threads to process episodes")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_episode = {executor.submit(process_single_episode, ep): ep for ep in episodes}
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_episode), total=len(episodes), desc="Processing Episodes"):
                ep_path, success, demos_frames_list = future.result()
                
                if success and demos_frames_list:
                    # Each demo saved as independent episode
                    for demo_idx, demo_frames in enumerate(demos_frames_list):
                        for frame_data in demo_frames:
                            dataset.add_frame(frame_data)
                        dataset.save_episode()
                        logger.info(f"Saved episode: {ep_path.name}_demo_{demo_idx}")
    
    def _get_default_hub_config(self) -> Dict[str, Any]:
        """Get default Hub configuration"""
        return {
            "tags": ["libero", "robotics", "lerobot", "unified"],
            "private": False,
            "push_videos": True,
            "license": "apache-2.0",
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Libero unified data converter - Supports automatic detection of RLDS and HDF5 formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Automatically detect format and convert
  python unified_converter.py \\
    --data-dir /path/to/data \\
    --repo-id username/dataset_name \\
    --push-to-hub \\
    --use-videos \\
    --num-workers 4

  # HDF5 format, specify config file
  python unified_converter.py \\
    --data-dir /path/to/hdf5/data \\
    --repo-id username/hdf5_dataset \\
    --config config.json \\
    --task-name "pick_and_place" \\
    --num-workers 8
        """
    )
    
    # Required arguments
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repository ID")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hub")
    parser.add_argument("--private", action="store_true", help="Create private dataset")
    
    # Data format
    parser.add_argument("--use-videos", action="store_true", default=True, help="Use video format")
    parser.add_argument("--robot-type", type=str, default="panda", help="Robot type")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate")
    
    # HDF5 specific parameters
    parser.add_argument("--task-name", type=str, default="default_task", help="Task name")
    
    # Performance parameters
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel worker threads")
    parser.add_argument("--image-writer-processes", type=int, default=5, help="Number of image writing processes")
    parser.add_argument("--image-writer-threads", type=int, default=1, help="Number of image writing threads")
    
    # Hub configuration
    parser.add_argument("--license", type=str, default="apache-2.0", help="Dataset license")
    parser.add_argument("--tags", nargs="+", default=["libero", "robotics", "lerobot"], help="Dataset tags")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate parameters
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1
    
    if "/" not in args.repo_id:
        logger.error(f"repo_id format error: {args.repo_id}")
        return 1
    
    logger.info("📋 Conversion configuration:")
    logger.info(f"   Data source: {args.data_dir}")
    logger.info(f"   Repository ID: {args.repo_id}")
    logger.info(f"   Number of parallel threads: {args.num_workers}")
    logger.info(f"   Use videos: {args.use_videos}")
    logger.info(f"   Push to Hub: {args.push_to_hub}")
    
    if args.dry_run:
        logger.info("✅ Dry run completed, parameter validation passed")
        return 0
    
    # Execute conversion
    try:
        converter = UnifiedConverter(num_workers=args.num_workers)
        
        hub_config = {
            "tags": args.tags,
            "private": args.private,
            "license": args.license,
        }

        
        dataset = converter.convert_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            task_name=args.task_name,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        
        logger.info("✅ Conversion completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 