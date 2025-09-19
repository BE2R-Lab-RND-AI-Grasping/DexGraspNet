# %%
import os
import random
from utils.hand_model_lite import HandModelMJCFLite 
from utils.hand_model import HandModel_Mujoco 
import numpy as np
import transforms3d
import torch
import trimesh
import json
import plotly.graph_objects as go


# %% [markdown]
# Here you need to choose your `hand_name`

# %%
mesh_path = "../data/meshdata"
# hand_name ="shadow_dexee"
hand_name ="DIP-Flex_opened_kinematics_simpl"
# hand_name ="DIP-Flex_opened_kinematics"
# hand_name = "robotiq_2"
# hand_name = "panda"

use_visual_mesh = True

if hand_name =="shadow_dexee":
    '''For shadow dexee'''
    data_path = "../data/dataset/shadow_dexee/"
    hand_file = "mjcf/shadow_dexee.xml"
    joint_names = [
                    "F0_J0", "F0_J1", "F0_J2", "F0_J3", "F1_J0", "F1_J1", "F1_J2", "F1_J3", "F2_J0", "F2_J1", "F2_J2", "F2_J3"
    ]

elif hand_name =="barret":
    ''' For BarretHand'''
    data_path = "../data/dataset/barret/"
    hand_file = "mjcf/barret.xml"
    joint_names = [
                    "wam_bhand_finger_1_prox_joint", "wam_bhand_finger_1_med_joint", "wam_bhand_finger_1_dist_joint", 
                    "wam_bhand_finger_2_prox_joint", "wam_bhand_finger_2_med_joint", "wam_bhand_finger_2_dist_joint",
                    "wam_bhand_finger_3_med_joint", "wam_bhand_finger_3_dist_joint"
    ]

elif hand_name =="DIP-Flex_opened_kinematics":
    ''' For Egorhand'''
    data_path = "../data/dataset/DIP-Flex_opened_kinematics"
    hand_file = "mjcf/DIP-Flex_opened_kinematics.xml"
    joint_names = [
                    "Joint_pinkie_abduction", "Joint_pinkie_PPflexion", "Joint_pinkie_DPflexion",
                    "Joint_index_abduction", "Joint_index_PPflexion", "Joint_index_DPflexion",
                    "Joint_thumb_rotation", "Joint_thumb_abduction", "Joint_thumb_PPflexion", "Joint_thumb_DPflexion"
    ]

elif hand_name == "robotiq_2":
    '''For robotiq'''
    data_path = "../data/dataset/robotiq_2/"
    hand_file = "mjcf/robotiq_2 simpl.xml"
    joint_names = [
                    "left_spring_link_joint", "left_follower",
                    "right_spring_link_joint", "right_follower_joint"
    ]

elif hand_name == "panda":
    '''For panda'''
    data_path = "../data/dataset/panda/"
    hand_file = "mjcf/panda.xml"
    joint_names = [
                    "finger_joint1", "finger_joint2"
    ]

elif hand_name == "shadow_dex_ee_simpl":
    data_path = "../data/dataset/shadow_dexee/"
    hand_file = "mjcf/shadow_dexee simpl.xml"
    joint_names = [
                    "F0_J0", "F0_J1", "F0_J2", "F0_J3", "F1_J0", "F1_J1", "F1_J2", "F1_J3", "F2_J0", "F2_J1", "F2_J2", "F2_J3"
    ]
elif hand_name =="barret_simpl":
    ''' For BarretHand'''
    data_path = "../data/dataset/barret/"
    hand_file = "mjcf/barret_simpl.xml"
    joint_names = [
                    "wam_bhand_finger_1_prox_joint", "wam_bhand_finger_1_med_joint", "wam_bhand_finger_1_dist_joint", 
                    "wam_bhand_finger_2_prox_joint", "wam_bhand_finger_2_med_joint", "wam_bhand_finger_2_dist_joint",
                    "wam_bhand_finger_3_med_joint", "wam_bhand_finger_3_dist_joint"
    ]

elif hand_name =="DIP-Flex_opened_kinematics_simpl":
    ''' For Egorhand'''
    data_path = "../data/dataset/DIP-Flex_opened_kinematics"
    hand_file = "mjcf/DIP-Flex_opened_kinematics simpl.xml"
    joint_names = [
                    "Joint_pinkie_abduction", "Joint_pinkie_PPflexion", "Joint_pinkie_DPflexion",
                    "Joint_index_abduction", "Joint_index_PPflexion", "Joint_index_DPflexion",
                    "Joint_thumb_rotation", "Joint_thumb_abduction", "Joint_thumb_PPflexion", "Joint_thumb_DPflexion"
    ]
 
    

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']


# %%
 

# %%
hand_config = json.load(open("mjcf/DIP-Flex_opened_kinematics/DIP-Flex_opened_kinematics.json", 'r'))
device = "cpu"

# %%


hand_model = HandModel_Mujoco(
    hand_config=hand_config,
    mjcf_path="mjcf/DIP-Flex_opened_kinematics/DIP-Flex_opened_kinematics_prim.xml",
    mesh_path='mjcf/assets/DIP-Flex_opened_kinematics',
    contact_points_path='mjcf/DIP-Flex_opened_kinematics/contact_points_DIP-Flex_opened_kinematics.json',
    penetration_points_path='mjcf/DIP-Flex_opened_kinematics/penetration_points_DIP-Flex_opened_kinematics.json',
    n_surface_points=200,
    device=device
)

# %%
grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])

print(grasp_code_list)

# %%
grasp_code = random.choice(grasp_code_list)
grasp_data = np.load(
    os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code, "coacd/decomposed.obj"))
print(grasp_code)

print(grasp_data)
print(len(grasp_data))

# %%
index = random.randint(0, len(grasp_data) - 1)
# index = 3

qpos = grasp_data[index]['qpos']
print(index)
rot = np.array(transforms3d.euler.euler2mat(
    *[qpos[name] for name in rot_names]))
rot = rot[:, :2].T.ravel().tolist()
hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                         for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
hand_model.set_parameters(hand_pose)
# hand_mesh = hand_model.get_plotly_data(0)
# object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])

# # Задаем цвета (RGB в формате [R, G, B, A], где значения от 0.0 до 1.0)
# hand_color = [0.7, 0.7, 0.7, 1.0]  # Красноватый цвет для руки
# object_color = [0.2, 0.5, 0.8, 1.0]  # Голубоватый цвет для объекта

# # Применяем цвета к мешам
# hand_mesh[0].visual.face_colors = hand_color
# object_mesh.visual.face_colors = object_color


# %%


hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue')
 
fig = go.Figure( hand_en_plotly  )
fig.show()
print("dddd")

 



