UR_1RGB_Config = {
    "images": {
        "camera_top": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_top_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
    },
    "states": {
        "end_effector": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "r", "p", "y"]},
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
    },
    "actions": {
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
    },
}
