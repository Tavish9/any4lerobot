Tien_Kung_Xsens_1RGB_Config = {
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
            "shape": (12,),
            "names": {
                "motors": [
                    "left_little_finger",
                    "left_ring_finger",
                    "left_middle_finger",
                    "left_index_finger",
                    "left_thumb0_for_bending",
                    "left_thumb1_for_rotation",
                    "right_little_finger",
                    "right_ring_finger",
                    "right_middle_finger",
                    "right_index_finger",
                    "right_thumb0_for_bending",
                    "right_thumb1_for_rotation",
                ]
            },
        },
        "joint_position": {
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
    },
    "actions": {
        "end_effector": {
            "dtype": "float32",
            "shape": (12,),
            "names": {
                "motors": [
                    "left_little_finger",
                    "left_ring_finger",
                    "left_middle_finger",
                    "left_index_finger",
                    "left_thumb0_for_bending",
                    "left_thumb1_for_rotation",
                    "right_little_finger",
                    "right_ring_finger",
                    "right_middle_finger",
                    "right_index_finger",
                    "right_thumb0_for_bending",
                    "right_thumb1_for_rotation",
                ]
            },
        },
        "joint_position": {
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
    },
}
