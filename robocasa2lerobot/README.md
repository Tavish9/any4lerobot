# ROBOCASA TO LEROBOT

## ROBOCASA installation

- Clone this repo: https://github.com/robocasa/robocasa
- Follow README.md to install packages and download assets

## Data Preparation

- Check files: `robocasa/scripts/download_datasets.py`, `robocasa/utils/dataset_registry.py`
- Download original datasets by python scripts or wget/curl (recommended)

## Example:

```bash
wget https://utexas.box.com/shared/static/7y9csrcx6uhhq4p3yctmm2df3rjqpw6g.hdf5 -O PnPCounterToCab.hdf5
```

- Extract subset data: Original hdf5 files contain about 3000 episodes. However, it contains a key "masks", which contain list of subset demo_ids. For example: 30_demos : `[demo123, demo234, demo345, etc.]`.Run the code in the notebook to extract only chosen subset demos, which is much smaller and easier for later processes.

## Code Modification

- Add functions in `camera_utils.py` to your `robosuite/robosuite/utils/camera_utils.py` for camera parameters extraction (May be useful for experiments which requires multiview rendering)

- Change args to render depth and segmentation masks for new regenerated dataset. Change in `robocasa/environments/kitchen/kitchen.py`

```python

class Kitchen(ManipulationEnv, metaclass=KitchenEnvMeta):
    ...
    EXCLUDE_LAYOUTS = []
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False, # -> True
        renderer="mjviewer",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        scene_split=None,  # unsued, for backwards compatibility
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
        camera_segmentations="instance", # add camera segmentation here: semantic/instance/element
    ):
        ...

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations, # add camera segmentation here
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )
```

## Regenerate

- Check file: `regenerate.py`
- Original dataset contain image in 128x128 resolution and does not contain segmentation mask, depth, etc. We need to rerender it in 256x256 and save segmentation mask, and depth
- Overall re-render flow:
  - (1) load hdf5 file and create env
  - (2) reset env to first state in the dataset
  - (3) Execute action in action label of original dataset, at each step, we collect observation data, camera parameters, state, etc. from simulation.
  - (4) Save only successful episode to new hdf5 file (original data contain unsuccessful episode or wrong action)
- Change `origin_dir` and `regenerate_dir` to your dir in `regenerate.py` then run `python regenerate.py` to regenerate

## Get started

1. Download source code:

   ```bash
   git clone https://github.com/Tavish9/any4lerobot.git
   ```

2. Modify path in `convert.sh`:

   ```bash
   python robocasa_h5.py \
       --raw-dir /path/to/your/hdf5/files \
       --repo-id your_hf_id \
       --local-dir /path/to/your/output/dataset
   ```

3. Execute the script:

   ```bash
   bash convert.sh
   ```

## Example output datasets:

- ROBOCASA 100 demos: https://huggingface.co/datasets/binhng/robocasa_merged_24_tasks_100demos_v1
- ROBOCASA 30 demos: https://huggingface.co/datasets/binhng/robocasa_merged_24_tasks_30demos_v3
