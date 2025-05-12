AgileX_3RGB_Config = {
    "images": {
        "camera_front": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_left_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_right_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "camera_front_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "camera_left_wrist_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "camera_right_wrist_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
    },
    "states": {
        "end_effector_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw"]},
        },
        "end_effector_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw"]},
        },
        "joint_effort_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_effort_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_position_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_position_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_velocity_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_velocity_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
    },
    "actions": {
        "end_effector_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw"]},
        },
        "end_effector_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw"]},
        },
        "joint_effort_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_effort_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_position_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_position_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_velocity_left": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
        "joint_velocity_right": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
        },
    },
}
