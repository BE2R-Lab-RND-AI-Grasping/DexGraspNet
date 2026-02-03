import numpy as np
import matplotlib.pyplot as plt
import os


def extract_valid_pose_data(raw_dataset_dir, filtered_dataset_dir):
    pose_files = []
    for root, dirs, files in os.walk(raw_dataset_dir):
        for file in files:
            if file.endswith(".npy"):  # Assuming you're looking for .npy files
                pose_files.append(os.path.join(root, file))

    for file_i in pose_files:
        full_data = np.load(file_i, allow_pickle=True)
        is_valid_array = []
        for pose_i in full_data:
            if pose_i["is_valid"]:
                is_valid_array.append(pose_i["dexgrasppose_data"])

        file_path = os.path.join(filtered_dataset_dir, os.path.basename(file_i))

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "wb") as f:
            np.save(f, is_valid_array)


def dataset_dexgrasppose_report(dataset_dir, report_path):
    pose_files = []
    pose_names = []
    pose_data = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".npy") or file.endswith(".npz"):  # Assuming you're looking for .npy files
                pose_files.append(os.path.join(root, file))
                pose_names.append(file)
                pose_data_i = np.load(os.path.join(root, file), allow_pickle=True)
                pose_data.append(pose_data_i)
    report = []
    for num, pose_data_i in enumerate(pose_data):
        pose_data_name = pose_names[num]
        # Calculate average energy for this pose data file
        energies = [pose_j["energy"] for pose_j in pose_data_i]
        avg_energy = np.mean(energies) if energies else 0
        # Instead of printing, write this information to the report file
        report.append(f"{pose_data_name}: Average energy = {avg_energy:.4f}, Count = {len(energies)}\n\n")

    with open(report_path, "w") as f:
        f.writelines(report)


raw_dataset_dir = "data/results/shadow_dexee"
filtered_dataset_dir = "data/results/processed/shadow_dexee"


extract_valid_pose_data(raw_dataset_dir, filtered_dataset_dir)
dataset_dexgrasppose_report(filtered_dataset_dir, "data/results/processed/report.txt")
