import json
import pickle

from matplotlib import pyplot as plt
import numpy as np
from pose_validator.validator import Validator, ValidatorConfig, get_all_paths
import os
from tqdm import tqdm


def get_file_names_without_extension(folder_path):
    """
    Gets all file names in the specified folder and removes the .npy extension.

    Args:
        folder_path (str): Path to the folder containing the files

    Returns:
        list: List of file names without the .npy extension
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return []

    file_names = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_names.append(os.path.splitext(file)[0])

    return file_names


def fetch_paths_for_final_positions(target_hand, final_positions_folder, object_folder, hands_folder, result_dir):
    final_positions_folder_hand = os.path.join(final_positions_folder, target_hand)
    file_names = get_file_names_without_extension(final_positions_folder_hand)

    path_for_one_folder = []
    for file_i in file_names:
        paths = get_all_paths(
            hands_folder=hands_folder,
            final_positions_folder=final_positions_folder,
            object_folder=object_folder,
            target_hand=target_hand,
            object_name=file_i,
        )
        result_dir_full = os.path.join(result_dir, target_hand)
        paths["result_dir"] = result_dir_full
        paths["object_name"] = file_i
        path_for_one_folder.append(paths)
    return path_for_one_folder


def run_one_position_file(current_experiment_setup: dict, validator_config: ValidatorConfig = ValidatorConfig()):
    hand_model_path = current_experiment_setup["hand_model"]
    hand_config = json.load(open(current_experiment_setup["hand_config"], "r"))
    pose_dict = np.load(current_experiment_setup["final_positions"], allow_pickle=True)
    object_folder = current_experiment_setup["object"]
    validator = Validator(
        hand_model=hand_model_path,
        hand_config=hand_config,
        validator_config=validator_config,
    )

    pose_validation_results = []

    for number, pose_i in tqdm(enumerate(pose_dict), total=len(pose_dict), desc="Processing poses"):
        qpos_history, object_height, object_contact, control_error_fingers, penetration = validator.experiment(
            position_dict=pose_i, object_folder=object_folder, visualize=False
        )
        is_valid = False
        if len(penetration) > 0:
            is_chating = np.abs(penetration).max() > 0.009
        else:
            is_chating = False

        if validator.is_stable_grasp(object_contact) and not is_chating:
            is_valid = True

        pose_validation_results.append(
            {
                "qpos_history": qpos_history,
                "object_height": object_height,
                "object_contact": object_contact,
                "control_error_fingers": control_error_fingers,
                "penetration": penetration,
                "is_valid": is_valid,
                "dexgrasppose_data" : pose_i,
                "qpos": pose_i["qpos"],
                "scale": pose_i["scale"]
            }
        )

    object_name = current_experiment_setup["object_name"]
    result_dir = os.path.join(current_experiment_setup["result_dir"], f"{object_name}.npy")
    if not os.path.exists(os.path.dirname(result_dir)):
        os.makedirs(os.path.dirname(result_dir))

    # penetration_list_dumped = pickle.dumps(penetration_list)

    np.save(
        result_dir,
        pose_validation_results,
    )
    return pose_validation_results


def main():

    target_hand = "shadow_dexee"
    final_positions_folder = "data/final_positions"
    object_folder = "data/mjcf/models/objs"
    hands_folder = "data/mjcf/models/hand_models"
    result_dir = "data/results"

    path_for_one_folder = fetch_paths_for_final_positions(
        target_hand, final_positions_folder, object_folder, hands_folder, result_dir
    )
    validator_config = ValidatorConfig()
    for path in path_for_one_folder:
        valid_counter = 0
        print(f"Processing path: {path['object_name']}")
        print("###############################################")
        pose_validation_results = run_one_position_file(path, validator_config=validator_config)
        for res in pose_validation_results:
            if res["is_valid"]:
                valid_counter += 1
        print(f"Found {valid_counter} valid poses out of {len(pose_validation_results)}")
        print("###############################################")



if __name__ == "__main__":
    main()
