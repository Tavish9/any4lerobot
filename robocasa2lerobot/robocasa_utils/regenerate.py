import json
import os

import h5py
import numpy as np
import robosuite
from robocasa.scripts.playback_dataset import reset_to
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_extrinsic_matrix_rel,
    get_camera_intrinsic_matrix,
)
from tqdm import tqdm

ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]


def get_camera_info(sim, camera_name, camera_height, camera_width):
    camera_intrinsics = get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
    camera_extrinsics = get_camera_extrinsic_matrix(sim, camera_name)

    return camera_intrinsics, camera_extrinsics


def creat_env_from_hdf5(f):
    env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta["env_kwargs"]["camera_depths"] = True
    env_meta["env_kwargs"]["camera_heights"] = 256
    env_meta["env_kwargs"]["camera_widths"] = 256
    env_meta["env_kwargs"]["camera_segmentations"] = "element"  # element' #'instance'
    # f.close()

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True  # write_video
    env_kwargs["use_camera_obs"] = True
    env_kwargs["ignore_done"] = False

    env = robosuite.make(**env_kwargs)

    return env, env_meta


def reset_each_demo(env, demo):
    # demo = f["data"]["demo_<idx>"]
    model_xml = demo.attrs["model_file"]
    init_state = demo["states"][()][0]
    ep_meta = demo.attrs["ep_meta"]

    state = {"states": init_state, "model": model_xml, "ep_meta": ep_meta}
    reset_to(env, state)


def process_1_demo(env, f, demo_id, grp):
    demo = f["data"][demo_id]
    reset_each_demo(env, demo)

    ep_meta = env.get_ep_meta()
    model_file = env.model.get_xml()

    for _ in range(10):
        obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)

    obs_keys = list(obs.keys())
    obs_keys += [
        "robot0_agentview_left_intrinsics",
        "robot0_agentview_right_intrinsics",
        "robot0_eye_in_hand_intrinsics",
    ]
    obs_keys += [
        "robot0_agentview_left_extrinsics",
        "robot0_agentview_right_extrinsics",
        "robot0_eye_in_hand_extrinsics",
    ]
    obs_keys += [
        "robot0_agentview_left_extrinsicsR",
        "robot0_agentview_right_extrinsicsR",
        "robot0_eye_in_hand_extrinsicsR",
    ]
    obs_keys += ["robot0_agentview_left_depthW", "robot0_agentview_right_depthW", "robot0_eye_in_hand_depthW"]

    obs_dict = {key: [] for key in obs_keys}
    # action_dict = {key: [] for key in act_keys}
    actions = []
    actions_abs = []
    rewards = []
    dones = []
    states = []  # env state, not robot. The state for robot is included in obs

    # for key in obs_keys:
    #     obs_dict[key] = obs[key]
    orig_actions = demo["actions"][()]
    orig_actions_abs = demo["actions_abs"][()]
    # orig_action_dict = demo['action_dict']

    for i, action in enumerate(orig_actions):
        # for i, action in enumerate(orig_actions_abs):
        extent = env.sim.model.stat.extent
        far = env.sim.model.vis.map.zfar * extent
        near = env.sim.model.vis.map.znear * extent
        left_depth = obs["robot0_agentview_left_depth"].copy()
        right_depth = obs["robot0_agentview_right_depth"].copy()
        wrist_depth = obs["robot0_eye_in_hand_depth"].copy()
        left_depth = near / (1.0 - left_depth * (1.0 - near / far))[::-1]
        right_depth = near / (1.0 - right_depth * (1.0 - near / far))[::-1]
        wrist_depth = near / (1.0 - wrist_depth * (1.0 - near / far))[::-1]

        obs["robot0_agentview_left_depthW"] = left_depth
        obs["robot0_agentview_right_depthW"] = right_depth
        obs["robot0_eye_in_hand_depthW"] = wrist_depth

        left_intrinsics, left_extrinsics = get_camera_info(env.sim, "robot0_agentview_left", 256, 256)
        right_intrinsics, right_extrinsics = get_camera_info(env.sim, "robot0_agentview_right", 256, 256)
        wrist_intrinsics, wrist_extrinsics = get_camera_info(env.sim, "robot0_eye_in_hand", 256, 256)

        obs["robot0_agentview_left_intrinsics"] = left_intrinsics
        obs["robot0_agentview_right_intrinsics"] = right_intrinsics
        obs["robot0_eye_in_hand_intrinsics"] = wrist_intrinsics
        obs["robot0_agentview_left_extrinsics"] = left_extrinsics
        obs["robot0_agentview_right_extrinsics"] = right_extrinsics
        obs["robot0_eye_in_hand_extrinsics"] = wrist_extrinsics

        left_intrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_agentview_left")
        right_intrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_agentview_right")
        wrist_intrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_eye_in_hand")

        obs["robot0_agentview_left_extrinsicsR"] = left_intrinsics_rel
        obs["robot0_agentview_right_extrinsicsR"] = right_intrinsics_rel
        obs["robot0_eye_in_hand_extrinsicsR"] = wrist_intrinsics_rel

        # append all keys
        for key in obs_keys:
            if (
                ("eye_in_hand" in key or "agentview" in key)
                and "depthW" not in key
                and "intrinsics" not in key
                and "extrinsics" not in key
            ):
                obs_dict[key].append(obs[key][::-1, :, :])
            else:
                obs_dict[key].append(obs[key])

        # for key in act_keys:
        #     action_dict[key].append(orig_action_dict[key][i])

        actions.append(action)
        actions_abs.append(orig_actions_abs[i])

        rewards.append(reward)
        dones.append(done)

        current_state = env.sim.get_state().flatten()
        states.append(current_state)

        # step env
        obs, reward, done, info = env.step(action.tolist())

        done = done or env._check_success()
        # if done:
        # print(f" Step {i} done: {done}")
        # print(f" Step {i} info: {info}")
        # print(f" Step {i} is_success: {env._check_success()}" )

    # save successful episode only
    if done:
        print(f"Demo {demo_id} is done after {i} actions! -> SAVE!!!")

        # save to new hdf5 file here
        ep_data = grp.create_group(demo_id)
        # set attribute for ep_data here ...
        ep_data.attrs["model_file"] = model_file
        ep_data.attrs["ep_meta"] = json.dumps(ep_meta, indent=4)

        # obs group
        obs_grp = ep_data.create_group("obs")
        for key in obs_keys:
            obs_grp.create_dataset(key, data=np.stack(obs_dict[key], axis=0))

        # actions dataset
        ep_data.create_dataset("actions", data=np.stack(actions, axis=0))

        ep_data.create_dataset("actions_abs", data=np.stack(actions_abs, axis=0))

        ep_data.create_dataset("dones", data=np.stack(dones, axis=0))

        ep_data.create_dataset("rewards", data=np.stack(rewards, axis=0))

        # state dataset
        ep_data.create_dataset("states", data=np.stack(states, axis=0))

    elif not done:
        print(f"Demo {demo_id} not done after all actions executed! -> does not SAVE!")


def regenerate_hdf5_dataset(input_path, output_path, debug=False):
    f = h5py.File(input_path, "r")
    env, env_meta = creat_env_from_hdf5(f)

    out_f = h5py.File(output_path, "w")
    out_f.attrs["env_args"] = json.dumps(env_meta)

    grp = out_f.create_group("data")

    all_demo_ids = list(f["data"].keys())
    if debug:
        all_demo_ids = all_demo_ids[: min(2, len(all_demo_ids))]
    for demo_id in tqdm(all_demo_ids):
        print(f"Processing {demo_id} ...")
        process_1_demo(env, f, demo_id, grp)

    f.close()
    if len(out_f["data"].keys()) == 0:
        print("No demos were processed successfully. Deleting output file.")
        out_f.close()
        os.remove(output_path)
    else:
        print(f"Processed data saved {len(out_f['data'].keys())} demos to {output_path}")
        out_f.close()


def process_task_wrapper(args):
    """Wrapper function for multiprocessing to process a single task."""
    task, origin_dir, regenerate_dir, debug = args
    input_path = os.path.join(origin_dir, f"{task}.hdf5")
    output_path = os.path.join(regenerate_dir, f"{task}.hdf5")

    print(f"Regenerating dataset for task {task} ...")
    regenerate_hdf5_dataset(input_path, output_path, debug=debug)
    print(f"Completed task {task}")


if __name__ == "__main__":
    n_demo = 100  # 100
    origin_dir = f"<directory/contain/original/hdf5/files/>"
    regenerate_dir = f"<directory/contain/regenerated/hdf5/files/>"
    os.makedirs(regenerate_dir, exist_ok=True)

    task_list = [
        "PnPCabToCounter",
        "PnPCounterToCab",
        "CoffeeSetupMug",
        "TurnOffStove",
        "TurnOnMicrowave",
        # ... add other tasks as needed
        # "CoffeePressButton",
        # "CoffeeServeMug",
        # "TurnOffMicrowave",
        # "TurnOffSinkFaucet",
        # "TurnOnSinkFaucet",
        # "TurnOnStove",
        # "TurnSinkSpout"
        # "CloseDoubleDoor",
        # "CloseDrawer",
        # "CloseSingleDoor",
        # "OpenDoubleDoor",
        # "OpenDrawer",
        # "OpenSingleDoor"
        # "PnPCounterToMicrowave",
        # "PnPCounterToSink",
        # "PnPCounterToStove",
        # "PnPMicrowaveToCounter",
        # "PnPSinkToCounter",
        # "PnPStoveToCounter"
    ]  # 24 tasks in robocasa kitchen dataset

    debug = False
    if debug:
        task_list = task_list[:2]

    for task in task_list:
        input_path = os.path.join(origin_dir, f"{task}.hdf5")
        output_path = os.path.join(regenerate_dir, f"{task}.hdf5")

        print(f"Regenerating dataset for task {task} ...")
        regenerate_hdf5_dataset(input_path, output_path, debug=False)
