#!/usr/bin/env python3
"""
Libero RLDS to LeRobot Converter Runner Script

Provides complete command line interface, similar to other any4lerobot projects.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

def create_argument_parser():
    """Create command line argument parser, similar to other any4lerobot projects"""
    parser = argparse.ArgumentParser(
        description="Convert Libero RLDS dataset to LeRobot format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_converter.py \\
    --data-dir /path/to/libero/data \\
    --repo-id username/libero_dataset \\
    --push-to-hub \\
    --use-videos

Supported datasets: libero_10_no_noops, libero_goal_no_noops, 
                   libero_object_no_noops, libero_spatial_no_noops
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory path containing Libero RLDS datasets"
    )
    parser.add_argument(
        "--repo-id", 
        type=str,
        required=True,
        help="Repository ID for output dataset (format: username/dataset_name)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local output directory (default: LEROBOT_HOME)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to Hugging Face Hub after conversion"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private dataset"
    )
    
    # Data format
    parser.add_argument(
        "--use-videos",
        action="store_true",
        default=True,
        help="Use video format for storing images (default: True)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="panda",
        help="Robot type (default: panda)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate (default: 20)"
    )
    
    # Performance parameters
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="Number of image writer processes (default: 5)"
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of image writer threads (default: 10)"
    )
    
    # Hub configuration
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="Dataset license (default: apache-2.0)"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["libero", "panda", "robotics", "lerobot"],
        help="Dataset tags"
    )
    
    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode"
    )
    
    return parser


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1
    
    if "/" not in args.repo_id:
        logger.error(f"Invalid repo_id format: {args.repo_id}")
        return 1
    
    logger.info("📋 Conversion configuration:")
    logger.info(f"  Data source: {args.data_dir}")
    logger.info(f"  Repository ID: {args.repo_id}")
    logger.info(f"  Output directory: {args.output_dir or 'LEROBOT_HOME'}")
    logger.info(f"  Use videos: {args.use_videos}")
    logger.info(f"  Push to Hub: {args.push_to_hub}")
    
    if args.dry_run:
        logger.info("✅ Dry run completed, parameter validation passed")
        return 0
    
    # Call converter
    try:
        from libero_rlds_converter import convert_libero_dataset
        
        hub_config = {
            "tags": args.tags,
            "private": args.private,
            "license": args.license,
        }
        
        dataset = convert_libero_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        
        logger.info("✅ Conversion completed!")
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())