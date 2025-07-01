"""
Last modified date: 2025.04.17
Author: Olga Borisova
Description: energy functions with grasp region support

Этот Python-скрипт реализует энергетическую функцию (cal_energy) для оценки взаимодействия руки с объектом, используя PyTorch. 
Функция вычисляет различные компоненты энергии, связанные с контактом, расстоянием, проникновением и ограничениями суставов, 
чтобы проверить, насколько физически правдоподобно взаимодействие.

"""

import torch
import open3d as o3d
import numpy as np


# def load_grasp_region_from_ply(ply_path: str, color=[1.0, 0.0, 0.0]) -> torch.Tensor:
#     """
#     Загружает grasp-точки из .ply, отбирая точки по цвету (по умолчанию — красные).
#     Возвращает: (N, 3) тензор grasp-точек
#     """
#     point_cloud = o3d.io.read_point_cloud(ply_path)
#     if not point_cloud.has_colors():
#         raise ValueError("Point cloud does not have color information.")

#     points_coordinates = np.asarray(point_cloud.points)
#     points_colors = np.asarray(point_cloud.colors)

#     # выбираем точки, у которых цвет близок к заданному
#     mask = np.all(np.isclose(points_colors, np.array(color), atol=0.05), axis=1)
#     selected_points = points_coordinates[mask]

#     if selected_points.shape[0] == 0:
#         raise ValueError("No grasp region points found with the specified color.")

#     return torch.tensor(selected_points, dtype=torch.float32)
    

def cal_energy(hand_model, object_model, w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0, w_grasp=50.0, grasp_points:torch.Tensor = None, verbose=False):
    
    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape # возвращает размерности тензора. contact points из contact_points.json. n_contact — количество контактных точек руки. 
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points) # cal_distance из файла object_model.py
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)], 
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_joints
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
        torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)

    # E_pen
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_spen
    E_spen = hand_model.self_penetration()


    '''Energy grasp region. If hand is close to labed region of object - energy term is minimising.'''

    # E_grasp = torch.tensor(0.0, device=device)

    # grasp_points = load_grasp_region_from_ply("grasp_region.ply")

    # if grasp_points is not None:
    #     grasp_points = grasp_points.to(device)
    #     contact_points = hand_model.contact_points  # (B, N, 3)
    #     B, N, _ = contact_points.shape
    #     M = grasp_points.shape[0]

    #     expanded_grasp = grasp_points.unsqueeze(0).expand(B, M, 3)  # (B, M, 3)
    #     dists = torch.cdist(contact_points, expanded_grasp)  # (B, N, M)
    #     min_dists, _ = torch.min(dists, dim=-1)  # (B, N)
    #     E_grasp = torch.mean(min_dists, dim=-1)  # (B,)

    if verbose:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_grasp * E_grasp, E_fc, E_dis, E_pen, E_spen, E_joints
    else:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_grasp * E_grasp
