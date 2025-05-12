Franka_Fr3_Dual_Arm_Config = {
    "images": {
        "camera_front": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "rgb"],
        },
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
        "camera_front_depth": {
            "dtype": "image",
            "shape": (720, 1280, 1),
            "names": ["height", "width", "channel"],
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
            "shape": (12,),
            "names": {"motors": ["left_xyzrpy", "right_xyzrpy"]},
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "motors": [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "left_joint_6",
                    "left_gripper",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                    "right_joint_6",
                    "right_gripper",
                ]
            },
        },
    },
    "actions": {
        "joint_position": {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "motors": [
                    "left_joint_0",
                    "left_joint_1",
                    "left_joint_2",
                    "left_joint_3",
                    "left_joint_4",
                    "left_joint_5",
                    "left_joint_6",
                    "left_gripper",
                    "right_joint_0",
                    "right_joint_1",
                    "right_joint_2",
                    "right_joint_3",
                    "right_joint_4",
                    "right_joint_5",
                    "right_joint_6",
                    "right_gripper",
                ]
            },
        },
    },
}
