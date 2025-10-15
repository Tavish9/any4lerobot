<h1 align="center">
    <p>Any4LeRobot: A tool collection for LeRobot</p>
</h1>

<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Tavish9/any4lerobot)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![LeRobot Dataset](https://img.shields.io/badge/dynamic/json?url=https://api.github.com/repos/tavish9/any4lerobot/commits?per_page=1&query=$[0].commit.committer.date&label=LeRobot&color=blue)](https://github.com/huggingface/lerobot)
[![LeRobot Dataset](https://img.shields.io/badge/LeRobot%20Dataset-v3.0-ff69b4.svg)](https://github.com/huggingface/lerobot/pull/1412)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

> [!IMPORTANT]
>
> **Star and Contribute**, let's make community of robotics better and better! 🔥

A curated collection of utilities for [LeRobot Projects](https://github.com/huggingface/lerobot), including data conversion scripts, preprocessing tools, training workflow helpers and etc..

## 📣 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2025.10.04\]** We have collected and updated all Dataset Version Conversion Scripts for LeRobot! 🔥🔥🔥
- **\[2025.09.28\]** We have upgraded LeRobotDataset from v2.1 to v3.0! 🔥🔥🔥
- **\[2025.06.27\]** We have supported Data Conversion from LIBERO to LeRobot! 🔥🔥🔥
- **\[2025.05.16\]** We have supported Data Conversion from LeRobot to RLDS! 🔥🔥🔥
- **\[2025.05.12\]** We have supported Data Conversion from RoboMIND to LeRobot! 🔥🔥🔥
<details>
<summary>More News</summary>

- **\[2025.04.15\]** We add Dataset Merging Tool for merging multi-source lerobot datasets! 🔥🔥🔥
- **\[2025.04.14\]** We have supported Data Conversion from AgiBotWorld to LeRobot! 🔥🔥🔥
- **\[2025.04.11\]** We change the repo from `openx2lerobot` to `any4lerobot`, making a ​​universal toolbox for LeRobot​​! 🔥🔥🔥
- **\[2025.02.19\]** We have supported Data Conversion from Open X-Embodiment to LeRobot! 🔥🔥🔥
</details>

## ✨ Features

- ​**​Data Conversion​**​:

  - [x] [Open X-Embodiment to LeRobot](./openx2lerobot/README.md)
  - [x] [AgiBot-World to LeRobot](./agibot2lerobot/README.md)
  - [x] [RoboMIND to LeRobot](./robomind2lerobot/README.md)
  - [x] [LeRobot to RLDS](./lerobot2rlds/README.md)
  - [x] [LIBERO to LeRobot](./libero2lerobot/README.md)

- **Training**:

  - [ ] MultiLeRobotDataset

- **Dataset Preprocess**:

  - [x] [Dataset Merging](./dataset_merging/README.md)
  - [ ] Dataset Filtering
  - [ ] Dataset Sampling

- ​[**Version Conversion​**​](./ds_version_convert/README.md):

  - [x] [LeRobotv1.6 to LeRobotv2.0](./ds_version_convert/v16_to_v20/README.md)
  - [x] [LeRobotv2.0 to LeRobotv2.1](./ds_version_convert/v20_to_v21/README.md)
  - [x] [LeRobotv2.1 to LeRobotv2.0](./ds_version_convert/v21_to_v20/README.md)
  - [x] [LeRobotv2.1 to LeRobotv3.0](./ds_version_convert/v21_to_v30/README.md)
  - [ ] LeRobotv3.0 to LeRobotv2.1

- [**Want more features?**](https://github.com/Tavish9/any4lerobot/issues/new?template=feature-request.yml)

## 📚 Awesome LeRobot

### Model

- [EO1](https://eo-robotics.ai/eo-1): An Open Unified Embodied Foundation Model for General Robot Control Trained on Interleaved Vision-Text-Action Data [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/EO-Robotics/EO1">](https://github.com/EO-Robotics/EO1)
- [Hume](https://hume-vla.github.io): A Dual-System VLA with System2 Thinking [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/hume-vla/hume">](https://github.com/hume-vla/hume)
- [OneTwoVLA](https://one-two-vla.github.io/): A Unified Vision-Language-Action Model with Adaptive Reasoning [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Fanqi-Lin/OneTwoVLA">](https://github.com/Fanqi-Lin/OneTwoVLA)
- [SmolVLA](https://huggingface.co/blog/smolvla): Efficient Vision-Language-Action Model trained on Lerobot Community Data [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/lerobot">](https://github.com/huggingface/lerobot)
- [SpatialVLA](https://spatialvla.github.io/): a spatial-enhanced vision-language-action model that is trained on 1.1 Million real robot episodes [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SpatialVLA/SpatialVLA">](https://github.com/SpatialVLA/SpatialVLA)
- [openpi](https://www.physicalintelligence.company/blog/pi0): the official implemenation of $π_0$: A Vision-Language-Action Flow Model for General Robot Control [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Physical-Intelligence/openpi">](https://github.com/Physical-Intelligence/openpi)
- [Isaac-GR00T](https://developer.nvidia.com/isaac/gr00t): NVIDIA Isaac GR00T N1 is the world's first open foundation model for generalized humanoid robot reasoning and skills [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/NVIDIA/Isaac-GR00T">](https://github.com/NVIDIA/Isaac-GR00T)

### Dataset

- [Official](https://huggingface.co/lerobot): State-of-the-art Machine Learning for real-world robotics.
- [IPEC-COMMUNITY/OpenX](https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot-67c29b2ee5911f17dbea635e): Open X-Embodiment datasets in LeRobot format with standard transfomation
- [IPEC-COMMUNITY/LIBERO](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset-684837af28d465aa8b043950): LIBERO datasets in LeRobot format with standard transfomation and filtering
- [weijian-sun/agibotworld-lerobot](https://huggingface.co/datasets/weijian-sun/agibotworld-lerobot): AgibotWorld-LeRobot v2.0
- [GR00T-Dateset](https://huggingface.co/GR00T-Dateset): Isaac-GR00T training dataset
- [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim): Isaac-GR00T training dataset

### Embodiment Extensions

- [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot): a training framework enabling the training and testing of data collected using Unitree's G1 robot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/unitreerobotics/unitree_IL_lerobot">](https://github.com/unitreerobotics/unitree_IL_lerobot)
- [Dora-LeRobot](https://github.com/dora-rs/dora-lerobot): Lerobot boosted with dora [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/dora-rs/dora-lerobot">](https://github.com/dora-rs/dora-lerobot)
- [Fourier-Lerobot](https://github.com/FFTAI/fourier-lerobot): A training pipeline with Fourier dataset [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/FFTAI/fourier-lerobot">](https://github.com/FFTAI/fourier-lerobot)
- [Adora-LeRobot](https://github.com/Ryu-Yang/adora-lerobot): a modified version of lerobot, specifically adapted for the Adora robot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Ryu-Yang/adora-lerobot">](https://github.com/Ryu-Yang/adora-lerobot)
- [BiLerobot](https://github.com/LiZhYun/BiLerobot): A bimanual robotics platform combining LeRobot and ManiSkill for advanced dual-arm manipulation tasks using the SO100 robot digital twin [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/LiZhYun/BiLerobot">](https://github.com/LiZhYun/BiLerobot)
- [lerobot-piper](https://github.com/lykycy123/lerobot-piper): About Use Lerobot to collect piper robot arm data, and perform training and reasoning [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/lykycy123/lerobot-piper">](https://github.com/lykycy123/lerobot-piper)
- [Lerobot-koch](https://github.com/LilyHuang-HZ/Lerobot-koch): LeRobot Training Notes for Koch Arm [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/LilyHuang-HZ/Lerobot-koch">](https://github.com/LilyHuang-HZ/Lerobot-koch)
- [LeFranX](https://github.com/wengmister/LeFranX): Franka and XHand Extension for LeRobot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wengmister/LeFranX">](https://github.com/wengmister/LeFranX)
- [U-Arm](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm): Lerobot-Everything-Cross-Embodiment-Teleoperation [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/MINT-SJTU/LeRobot-Anything-U-Arm">](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm)
- [lerobot-robot-xarm](https://github.com/SpesRobotics/lerobot-robot-xarm): xArm integration for LeRobot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SpesRobotics/lerobot-robot-xarm">](https://github.com/SpesRobotics/lerobot-robot-xarm)

### Hardware

- [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi): Low-Cost Mobile Manipulator [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SIGRobotics-UIUC/LeKiwi">](https://github.com/SIGRobotics-UIUC/LeKiwi)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot): Fully Autonomous Household Dual-Arm Mobile Robot for $998 [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Vector-Wangel/XLeRobot">](https://github.com/Vector-Wangel/XLeRobot)
- [LeRobot-Kinematics](https://github.com/box2ai-robotics/lerobot-kinematics): Simple and Accurate Forward and Inverse Kinematics Examples for the Lerobot SO100 ARM [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/box2ai-robotics/lerobot-kinematics">](https://github.com/box2ai-robotics/lerobot-kinematics)
- [lerobotdepot](https://github.com/maximilienroberti/lerobotdepot): a reoi for hardware, components, and 3D-printable projects compatible with the LeRobot library [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/maximilienroberti/lerobotdepot">](https://github.com/maximilienroberti/lerobotdepot)
- [PingTi-Arm](https://github.com/nomorewzx/PingTi-Arm): A human-scale robotic arm compatible with Lerobot, based on SO100 [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/nomorewzx/PingTi-Arm">](https://github.com/nomorewzx/PingTi-Arm)

### Tutorial / Utils

- [Official Docs](https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot): This tutorial will explain how to train a neural network to control a real robot autonomously.
- [YouTube: LeRobot Tutorials](https://www.youtube.com/playlist?list=PLo2EIpI_JMQu5zrDHe4NchRyumF2ynaUN)
- [LeRobot Tutorial with MuJoCo](https://github.com/jeongeun980906/lerobot-mujoco-tutorial): Examples for collecting data and training with MuJoCo [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/jeongeun980906/lerobot-mujoco-tutorial">](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)
- [LeRobot Sim2Real](https://github.com/StoneT2000/lerobot-sim2real): Train in fast simulation and deploy visual policies zero shot to the real world [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/StoneT2000/lerobot-sim2real">](https://github.com/StoneT2000/lerobot-sim2real)
- [lerobot-hilserl-guide](https://github.com/michel-aractingi/lerobot-hilserl-guide): Guide and tutorial to run the HILSerl implementation of LeRobot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/michel-aractingi/lerobot-hilserl-guide">](https://github.com/michel-aractingi/lerobot-hilserl-guide)
- [LeRobotTutorial-CN](https://github.com/CSCSX/LeRobotTutorial-CN): a tutorial for LeRobot in Chinese [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/CSCSX/LeRobotTutorial-CN">](https://github.com/CSCSX/LeRobotTutorial-CN)
- [PathOn.AI](https://learn-robotics.pathon.ai/): Learn Robotics at PathOn.AI is a platform for learning robotics and AI
- [NVIDIA Jetson Tutorials](https://www.jetson-ai-lab.com/lerobot.html)
- [lerobot-on-ascend](https://github.com/hexchip/lerobot-on-ascend): Tutorial of Deploying ACT on Huawei Ascend 310B [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/hexchip/lerobot-on-ascend">](https://github.com/hexchip/lerobot-on-ascend)
- [lerobot_ws](https://github.com/Pavankv92/lerobot_ws): ROS 2 Package for LeRobot SO-ARM101 [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Pavankv92/lerobot_ws">](https://github.com/Pavankv92/lerobot_ws)
- [lerobot-ros](https://github.com/astroyat/lerobot-ros): Running LeRobot and ROS 2 on custom LIDAR [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/astroyat/lerobot-ros">](https://github.com/astroyat/lerobot-ros)
- [Physical AI Tools](https://github.com/ROBOTIS-GIT/physical_ai_tools): Physical AI Development Interface with LeRobot and ROS 2 [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/ROBOTIS-GIT/physical_ai_tools">](https://github.com/ROBOTIS-GIT/physical_ai_tools)
- [LeRobot.js](https://github.com/TimPietrusky/lerobot.js): interact with your robot in JS, inspired by LeRobot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/TimPietrusky/lerobot.js">](https://github.com/TimPietrusky/lerobot.js)
- [LeLab](https://github.com/nicolas-rabault/leLab): A web UI interface on top of lerobot [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/nicolas-rabault/leLab">](https://github.com/nicolas-rabault/leLab)
- [LeRobot Episode Scoring Toolkit](https://github.com/RoboticsData/score_lerobot_episodes): One-click tool to score, filter, and export higher-quality LeRobot datasets [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/RoboticsData/score_lerobot_episodes">](https://github.com/RoboticsData/score_lerobot_episodes)
- [LERO](https://github.com/masato-ka/lero): LeRobot dataset operations toolkit [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/masato-ka/lero">](https://github.com/masato-ka/lero)
- [LeRobot Dataset Visualizer](https://github.com/huggingface/lerobot-dataset-visualizer): Web application for visualizing robotics datasets in LeRobot format [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/lerobot-dataset-visualizer">](https://github.com/huggingface/lerobot-dataset-visualizer)
- [lerobot_so101_teleop](https://github.com/liorbenhorin/lerobot_so101_teleop): Sample Environment for the LeRobot SO-101 Robot in Isaac Lab to collect demonstrations in a simulation [<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/liorbenhorin/lerobot_so101_teleop">](https://github.com/liorbenhorin/lerobot_so101_teleop)

## 👷‍♂️ Contributing

We appreciate all contributions to improving Any4LeRobot.

<a href="https://github.com/Tavish9/any4lerobot/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=tavish9/any4lerobot"><br><br>
      </th>
    </tr>
  </table>
</a>

## 🤝 Acknowledgements

Special thanks to the [LeRobot teams](https://github.com/huggingface/lerobot) for making this great framework.

Thanks to everyone for supporting this project.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/Tavish9/any4lerobot" />
  <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/Tavish9/any4lerobot" />
  <img alt="github-stargazers" src="https://github.com/Tavish9/any4lerobot/stargazers" />
</picture>

<p align="right"><a href="#top">🔝Back to top</a></p>
