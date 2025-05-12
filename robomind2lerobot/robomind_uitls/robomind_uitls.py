from pathlib import Path

import cv2
import h5py
import numpy as np


def decode_images(camera_key, input_images):
    if "depth" not in camera_key:
        rgb_images = []
        camera_rgb_images = input_images
        for camera_rgb_image in camera_rgb_images:
            camera_rgb_image = np.array(camera_rgb_image)
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if rgb is None:
                rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8)
                if rgb.size == 2764800:
                    rgb = rgb.reshape(720, 1280, 3)
                elif rgb.size == 921600:
                    rgb = rgb.reshape(480, 640, 3)
            rgb_images.append(rgb)
        rgb_images = np.asarray(rgb_images)
        return rgb_images
    else:
        depth_images = []
        camera_depth_images = input_images
        for camera_depth_image in camera_depth_images:
            if isinstance(camera_depth_image, np.ndarray):
                depth_array = camera_depth_image
            else:
                depth_array = np.frombuffer(camera_depth_image, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            if depth is None:
                if depth_array.size == 921600:
                    depth = depth_array.reshape(720, 1280)
                elif depth_array.size == 307200:
                    depth = depth_array.reshape(480, 640)
            depth_images.append(depth)
        depth_images = np.asarray(depth_images)[..., None]
        return depth_images


def load_local_dataset(episode_path: Path, config: dict, save_depth: bool):
    try:
        images = {}
        states = {}
        actions = {}
        with h5py.File(episode_path, "r") as file:
            for key in config["images"]:
                if save_depth and "depth" in key:
                    image_key = f"observations/depth_images/{key[:-6]}"
                elif "depth" not in key:
                    image_key = f"observations/rgb_images/{key}"
                else:
                    continue
                images[f"observation.images.{key}"] = decode_images(image_key, file[image_key])
            for key in config["states"]:
                states[f"observation.states.{key}"] = np.array(file[f"puppet/{key}"], dtype=np.float32)
            for key in config["actions"]:
                actions[f"actions.{key}"] = np.array(file[f"master/{key}"], dtype=np.float32)

        num_frames = len(next(iter(states.values())))
        frames = [
            {
                **{key: value[i] for key, value in images.items() if save_depth or "depth" not in key},
                **{key: value[i] for key, value in states.items()},
                **{key: value[i] for key, value in actions.items()},
            }
            for i in range(num_frames)
        ]
        return True, frames, ""

    except (FileNotFoundError, OSError, KeyError) as e:
        return False, [], e
