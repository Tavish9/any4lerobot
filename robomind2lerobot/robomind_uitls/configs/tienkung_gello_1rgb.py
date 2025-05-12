Tien_Kung_Gello_1RGB_Config = {
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
        "joint_position": {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "left hand_closure",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                    "right hand closure",
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
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "left hand_closure",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                    "right hand closure",
                ]
            },
        },
    },
}
