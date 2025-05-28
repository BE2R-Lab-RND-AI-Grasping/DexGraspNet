"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

import open3d as o3d
from scipy.spatial import cKDTree

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=500, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/graspdata", type=str)
    parser.add_argument('--result_path', default="../data/dataset", type=str)
    parser.add_argument('--object_code',
                        default="sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    args = parser.parse_args()

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]

    os.environ.pop("CUDA_VISIBLE_DEVICES")
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        # device = torch.device(
            # f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')

        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            rot = np.array(transforms3d.euler.euler2mat(
                *[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
                qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)
        # print(scale_tensor.dtype)
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            n_surface_points=2000,
            device=device
        )
        hand_model.set_parameters(hand_state)
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor

        '''COLOR VALIDATION'''
        # Загружаем цветное облако точек объекта
        # Файл должен быть с цветами! (например, object_colored.ply)
        colored_pcd_path = os.path.join(args.mesh_path, args.object_code, "object_colored.ply")
        pcd = o3d.io.read_point_cloud(colored_pcd_path)

        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)

        # KD-дерево по точкам
        kdtree = cKDTree(pcd_points)

        def is_in_red_area(contact_point, radius=0.005):
            indices = kdtree.query_ball_point(contact_point, r=radius)
            for idx in indices:
                r, g, b = pcd_colors[idx]
                if r > 0.8 and g < 0.2 and b < 0.2:  # Красный цвет
                    return True
            return False

        '''...'''



        # calculate contact points and contact normals
        contact_points_hand = torch.zeros((batch_size, 19, 3)).to(device)
        contact_normals = torch.zeros((batch_size, 19, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(
                surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(
                surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i:i+1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() -
                contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    sim = IsaacValidator(gpu=args.gpu)
    if (args.index is not None):
        sim = IsaacValidator(gpu=args.gpu, mode="gui")

    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name]
                            for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                           os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
            for index in range(offset, offset_):
                sim.add_env(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        
        
        '''COLOR VALIDATION'''
        
        color_valid = np.zeros(batch_size, dtype=np.bool8)

        for i in range(batch_size):
            points = contact_points_hand[i].cpu().numpy()  # shape: (19, 3)
            hits = [is_in_red_area(pt) for pt in points]
            # Захват валиден, если хотя бы 2 пальца (или все) в красной зоне
            color_valid[i] = np.sum(hits) >= 2  # можешь поменять логику

        '''...'''
        
        valid = simulated * estimated * color_valid # add * color_valid
        
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'color valid: {color_valid.sum().item()}/{batch_size}, ' # new print about color validation
            f'valid: {valid.sum().item()}/{batch_size}')
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
        np.save(os.path.join(args.result_path, args.object_code +
                '.npy'), result_list, allow_pickle=True)
    sim.destroy()
