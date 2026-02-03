import json
import os

from matplotlib import pyplot as plt
import numpy as np
from pose_validator.validator import Validator, ValidatorConfig, get_all_paths
from PIL import Image
import numpy as np


target_hand = "DIP-Flex_opened_kinematics"
pose_folder = "data/results/DIP-Flex_opened_kinematics"


filenames = list(os.walk(pose_folder))[0][2]
pose_names = [f.split(".")[0] for f in filenames]
 
for pose_i in pose_names:
    image_save_path = "data/results/processed/pictures/" + target_hand + "/" + pose_i + "/"

    paths = get_all_paths(
        hands_folder="data/mjcf/models/hand_models",
        final_positions_folder="data/results/processed",
        object_folder="data/mjcf/models/objs",
        target_hand=target_hand,
        object_name=pose_i,
    )
    hand_model_path = paths["hand_model"]
    hand_config = json.load(open(paths["hand_config"], "r"))
    pose_dict = np.load(paths["final_positions"], allow_pickle=True)
    object_folder = paths["object"]

    validator = Validator(
        hand_model=hand_model_path,
        hand_config=hand_config,
        validator_config=ValidatorConfig(),
    )
    success_count = 0
    success_poses = []
    success_poses_num = []

    for number, pose_i in enumerate(pose_dict):
        qpos_history, object_height, object_contact, control_error_fingers, penetration = validator.experiment(
            position_dict=pose_i, object_folder=object_folder, visualize=False
        )
        frame = validator.get_pose_reneder(pose_i, object_folder)
        if frame.dtype != np.uint8:

            # Convert to uint8 if necessary
            frame = (frame * 255).astype(np.uint8)

        image = Image.fromarray(frame)
        os.makedirs(image_save_path, exist_ok=True)
        image.save(f'{image_save_path}image_{number}.png')
