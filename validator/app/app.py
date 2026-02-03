import json

from matplotlib import pyplot as plt
import numpy as np
from pose_validator.validator import Validator, ValidatorConfig, get_all_paths


target_hand = "shadow_dexee"
target_object = "core-bowl-a593e8863200fdb0664b3b9b23ddfcbc"

paths = get_all_paths(
    hands_folder="data/mjcf/models/hand_models",
    final_positions_folder="data/final_positions",
    object_folder="data/mjcf/models/objs",
    target_hand=target_hand,
    object_name=target_object,
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
    if pose_i["energy"] > 20:
        continue

    print(f"pose_i['energy']: {pose_i['energy']}")
    qpos_history, object_height, object_contact, control_error_fingers, penetration = validator.experiment(
        position_dict=pose_i, object_folder=object_folder, visualize=True
    )
    if len(penetration) > 0:
        is_chating = np.abs(penetration).max() > 0.009
    else:
        is_chating = False
    
    if validator.is_stable_grasp(object_contact) and not is_chating:
        success_count += 1
        success_poses.append(pose_i)
        success_poses_num.append(number)
        validator.experiment(position_dict=pose_i, object_folder=object_folder, visualize=True)
    else:
        # plt.plot(np.rad2deg(control_error_fingers))
        # plt.show()
        pass

print(f"Success rate: {success_count}/{len(pose_dict)}")
print(f"Success poses: {success_poses_num}")
