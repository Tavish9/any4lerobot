AgiBotWorld_BETA_GRIPPER_CONFIG = {
    "images": {
        "head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_center_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "head_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_left": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_right": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
    },
    "states": {
        "effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["left_gripper", "right_gripper"]},
        },
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": {"motors": ["left_xyzw", "right_xyzw"]}},
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": {"motors": ["left_xyz", "right_xyz"]}},
        "head.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["yaw", "patch"]}},
        "joint.current_value": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "joint.position": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "robot.orientation": {"dtype": "float32", "shape": (4,), "names": {"motors": ["x", "y", "z", "w"]}},
        "robot.position": {"dtype": "float32", "shape": (3,), "names": {"motors": ["x", "y", "z"]}},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["pitch", "lift"]}},
    },
    "actions": {
        "effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["left_gripper", "right_gripper"]},
        },
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": {"motors": ["left_xyzw", "right_xyzw"]}},
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": {"motors": ["left_xyz", "right_xyz"]}},
        "head.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["yaw", "patch"]}},
        "joint.position": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "robot.velocity": {"dtype": "float32", "shape": (2,), "names": {"motors": ["x_vel", "yaw_vel"]}},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["pitch", "lift"]}},
    },
}

AgiBotWorld_BETA_DEXHAND_CONFIG = {
    "images": {
        "head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_center_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "head_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
    },
    "states": {
        **AgiBotWorld_BETA_GRIPPER_CONFIG["states"],
        "effector.position": {
            "dtype": "float32",
            "shape": (12,),
            "names": {
                "motors": [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                ]
            },
        },
    },
    "actions": {
        **AgiBotWorld_BETA_GRIPPER_CONFIG["actions"],
        "effector.position": {
            "dtype": "float32",
            "shape": (12,),
            "names": {
                "motors": [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                ]
            },
        },
    },
}

AgiBotWorld_BETA_TACTILE_CONFIG = {
    **AgiBotWorld_BETA_GRIPPER_CONFIG,
    "images": {
        **AgiBotWorld_BETA_GRIPPER_CONFIG["images"],
        "left_sensor_1": {
            "dtype": "video",
            "shape": (700, 400, 3),
            "names": ["height", "width", "rgb"],
        },
        "left_sensor_2": {
            "dtype": "video",
            "shape": (700, 400, 3),
            "names": ["height", "width", "rgb"],
        },
        "right_sensor_1": {
            "dtype": "video",
            "shape": (700, 400, 3),
            "names": ["height", "width", "rgb"],
        },
        "right_sensor_2": {
            "dtype": "video",
            "shape": (700, 400, 3),
            "names": ["height", "width", "rgb"],
        },
    },
}

# Task statistics coming from https://docs.google.com/spreadsheets/d/1GWMFHYo3UJADS7kkScoJ5ObbQfAFasPuaeC7TJUr1Cc/edit?gid=0#gid=0
AgiBotWorld_TASK_TYPE = {
    "gripper": {
        "task_config": AgiBotWorld_BETA_GRIPPER_CONFIG,
        "task_ids": [],  # The remaining are all gripper
    },
    "dexhand": {
        "task_config": AgiBotWorld_BETA_DEXHAND_CONFIG,
        "task_ids": [
            "task_475",
            "task_536",
            "task_547",
            "task_548",
            "task_549",
            "task_554",
            "task_577",
            "task_578",
            "task_591",
            "task_595",
            "task_608",
            "task_620",
            "task_622",
            "task_660",
            "task_679",
            "task_705",
            "task_710",
            "task_727",
            "task_730",
            "task_731",
            "task_749",
            "task_753",
        ],
    },
    "tactile": {
        "task_config": AgiBotWorld_BETA_TACTILE_CONFIG,
        "task_ids": [
            "task_666",
            "task_675",
            "task_676",
            "task_677",
            "task_694",
            "task_737",
            "task_774",
        ],
    },
}
