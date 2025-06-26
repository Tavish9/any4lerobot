LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]},
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
    },
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}
