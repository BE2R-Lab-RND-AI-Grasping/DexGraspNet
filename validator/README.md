# Pose Validator

A comprehensive pose validation package for DexGraspNet that validates robotic hand grasping poses using MuJoCo physics simulation.

## Overview

The Pose Validator package simulates robotic hand grasping scenarios to validate whether a given hand pose can maintain a stable grasp on an object. It uses MuJoCo physics engine to create realistic simulations with contact forces, disturbance forces, and stability analysis.



## Installation

### Prerequisites

- Python 3.9+
- MuJoCo 3.2.7
- Conda/Miniconda

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/BE2R-Lab-RND-AI-Grasping/pose_validator.git
cd pose_validator
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pose_validator
```

3. Install the package:
```bash
pip install -e .
```

## Project Structure

```
pose_validator/
├── app/                      # Contains validator usage
├── pose_validator/
│   ├── __init__.py
│   ├── validator.py          # Main validation class
│   ├── grasp_env_utils.py    # Utility functions for grasp environment
│   └── load_complex_obj.py   # Object loading utilities
├── data/
│   ├── final_positions/      # Stored grasp poses for different hands
│   │   ├── barret/
│   │   ├── shadow_hand/
│   │   └── shadow_dexee/
│   └── mjcf/
│       └── models/         # Contains hands .xml
|       └── objs/           # Contains object mesh

├── setup.py
├── environment.yml
└── README.md
```

## Usage


### Hand Configuration Format

Each hand requires a JSON configuration file with the following structure:

```json
{
    "joint_finger_names": [
        "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
        "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
        "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
        "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
        "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
    ],
    "wrist_translation_names": ["WRJTx", "WRJTy", "WRJTz"],
    "wrist_rotational_joint_names": ["WRJRx", "WRJRy", "WRJRz"],
    "hand_base_joint_name": "robot0:slide"
}
```

## Datasets 

1. [Validated dataset for ShadowDexEE](https://drive.google.com/file/d/12fKwEnFPYV-CQfMsGCYw6ckcFmHD4zVi/view?usp=drive_link)
2. [Validated dataset for DIP-Flex_opened_kinematics](https://drive.google.com/file/d/10qHUfePBs_2LKfrtCdchxfpx5vDo9x6g/view?usp=drive_link)