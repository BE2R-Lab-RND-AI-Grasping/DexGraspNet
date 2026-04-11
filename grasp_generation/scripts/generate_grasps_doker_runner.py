"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
import json
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method
 


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list
    hand_config = json.load(open('mjcf/' + args.hand_name + "/" + args.hand_name +  '.json', 'r'))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    # prepare models

    n_objects = len(object_code_list)

    # worker = multiprocessing.current_process()._identity[0]
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')
    print(args.hand_name)
    
    hand_model = HandModel(
        hand_config=hand_config,
        mjcf_path='mjcf/' + args.hand_name + "/" + args.hand_name + '_simpl.xml',
        mesh_path='mjcf/assets/' + args.hand_name,
        contact_points_path='mjcf/' + args.hand_name + "/" + "contact_points_" + args.hand_name +  '.json',
        penetration_points_path='mjcf/' + args.hand_name  + "/" + "penetration_points_" + args.hand_name +  '.json',
        n_surface_points=200,
        device=device
    )

    # Load object scales from JSON file
    object_scales_path = os.path.join(args.data_root_path, 'object_scales.json')
    with open(object_scales_path, 'r') as f:
        object_scale_dict = json.load(f)

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000, 
        device=device,
        scale_dict=object_scale_dict
    )


    object_model.initialize(object_code_list)

    initialize_convex_hull(hand_config['init_pos'], hand_model, object_model, args)
    
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    # optimize
    
    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
    )
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)

    for step in tqdm(range(1, args.n_iter + 1)):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward()

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]
 

    # save results
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = hand_config['joint_names']
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos.update(dict(zip(rot_names, euler)))
            qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos_st.update(dict(zip(rot_names, euler)))
            qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
                E_fc=E_fc[idx].item(),
                E_dis=E_dis[idx].item(),
                E_pen=E_pen[idx].item(),
                E_spen=E_spen[idx].item(),
                E_joints=E_joints[idx].item(),
            ))
        np_energy = energy.cpu().detach().numpy().mean()
        print(f"Mean energy for {object_code}")
        print(f" Value  {np_energy}")
        print()
        np.save(os.path.join(args.result_path + '/' + args.hand_name, object_code + '.npy'), data_list, allow_pickle=True)
        
    print("")

    return 

 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--hand_name', default='shadow_dexee')
    parser.add_argument('--result_path', default="../data/grasp_data_BIG_BOWLS_CLASS_1000", type=str)
    parser.add_argument('--data_root_path', default="../data/BIG_BOWLS_CLASS", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size_each', default=500, type=int) # Number of generated poses
    parser.add_argument('--max_total_batch_size', default=1500, type=int)
    parser.add_argument('--n_iter', default=1000, type=int)
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.004, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=200.0, type=float) # default=100
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=10.0, type=float) # default=1.0
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.3, type=float)
    parser.add_argument('--distance_upper', default=0.4, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)

    args = parser.parse_args()

    gpu_list = [0]
    print(f'gpu_list: {gpu_list}')

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.result_path + '/' + args.hand_name):
        os.makedirs(args.result_path + '/' + args.hand_name)
    
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = [f for f in os.listdir(args.data_root_path) 
                                if not os.path.isfile(os.path.join(args.data_root_path, f))]
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path + '/' + args.hand_name, object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    
    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))
 
    # with multiprocessing.Pool(len(gpu_list)) as p:
    #     it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating', maxinterval=1000)
    for number, grasp_args_i in enumerate(process_args):
        print(f"Object group {number} / {len(process_args)}")
        p = multiprocessing.Process(target=generate, args=(grasp_args_i,))
        p.start()
        p.join()
 
 