Franka_3RGB_Config = {
    "images": {
        "camera_top": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_left": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_right": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_top_depth": {
            "dtype": "image",
            "shape": (720, 1280, 1),
            "names": ["height", "width", "channel"],
        },
        "camera_left_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "camera_right_depth": {
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
            "shape": (8,),
            "names": {
                "motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
            },
        },
    },
    "actions": {
        "joint_position": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
            },
        },
    },
}
