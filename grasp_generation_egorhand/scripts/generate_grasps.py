"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
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

try:
    set_start_method('spawn')
except RuntimeError:
    pass


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Отключается дублирование библиотек KMP_DUPLICATE_LIB_OK = 'True'.
np.seterr(all='raise') # настраивает numpy так, чтобы все ошибки вычислений вызывали исключения (raise) вместо предупреждений (warn) или игнорирования (ignore).


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list # Эта строка распаковывает переданный список аргументов (args_list) в четыре переменные

    np.random.seed(args.seed)    # Фиксирует seed для NumPy.
    torch.manual_seed(args.seed) # Фиксирует seed для Pytorch.

    # prepare models

    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')

    hand_model = HandModel(                                     # Загружает модель руки
        mjcf_path='mjcf/DP-Flex_opened_kinematics.xml',            # XML-файл с физической моделью руки. Определяет суставы, ограничения и динамику (используется в симуляторах вроде MuJoCo).
        mesh_path='mjcf/assets',                                # Папка с 3D-мешами руки (визуальное представление модели).
        contact_points_path='mjcf/contact_points.json',         # JSON-файл с контактными точками (места, где пальцы могут касаться объекта).
        penetration_points_path='mjcf/penetration_points.json', # JSON-файл с точками проникновения (области, где пальцы НЕ могут касаться объекта).
        device=device
    )

    object_model = ObjectModel(                 # Загружает модель объектов
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000, 
        device=device
    )

    object_model.initialize(object_code_list) # Этот вызов инициализирует модель объектов, загружая их геометрию и подготавливая их для симуляции.

    initialize_convex_hull(hand_model, object_model, args) # Эта функция инициализирует выпуклую оболочку (convex hull) для руки и объекта.
    
    hand_pose_st = hand_model.hand_pose.detach() #  это копия текущей позы руки, но без градиентов. Она полезна, если нужно работать с этими данными вне контекста обучения.

    optim_config = {                                        # Этот фрагмент кода создаёт конфигурацию для оптимизатора Annealing, который отвечает за поиск оптимального захвата (grasp) руки на объекте.
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

    for step in range(1, args.n_iter + 1):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

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
    joint_names = [
        'Joint_left_abduction', 'Joint_left_flexion', 'Joint_left_finray_proxy',
        'Joint_left_dynamixel_crank', 'Joint_left_crank_pusher',
        'Joint_right_abduction', 'Joint_right_flexion', 'Joint_right_finray_proxy',
        'Joint_right_dynamixel_crank', 'Joint_right_crank_pusher',
        'Joint_thumb_rotation', 'Joint_thumb_abduction', 'Joint_thumb_flexion', 'Joint_thumb_finray_proxy',
        'Joint_thumb_dynamixel_crank', 'Joint_thumb_crank_pusher'
    ]
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
        np.save(os.path.join(args.result_path, object_code + '.npy'), data_list, allow_pickle=True)


if __name__ == '__main__':              # Эта строка проверяет, запущен ли скрипт напрямую, а не импортирован как модуль.
    parser = argparse.ArgumentParser()  # Эта строка создает парсер аргументов командной строки с помощью модуля argparse. Это позволяет передавать параметры в скрипт при его запуске.
    # experiment settings
    parser.add_argument('--result_path', default="../data/graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size_each', default=5, type=int)       # число объектов, обрабатываемых за один проход.
    parser.add_argument('--max_total_batch_size', default=10, type=int) # максимальное число объектов за один запуск.
    parser.add_argument('--n_iter', default=6000, type=int)
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)  # управляет вероятностью переключения состояния в алгоритме оптимизации
    parser.add_argument('--mu', default=0.98, type=float)                 # представляет собой коэффициент затухания или скорости изменения
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.2, type=float)
    parser.add_argument('--distance_upper', default=0.3, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)

    args = parser.parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",") # Определяет, какие GPU доступны для использования.
    print(f'gpu_list: {gpu_list}')

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)    # Фиксирует seed для NumPy. Любые вызовы np.random.rand(), np.random.randint() и т. д. теперь будут воспроизводимыми.
    torch.manual_seed(args.seed) # Фиксирует seed для PyTorch, чтобы все случайные операции (например, инициализация нейросетей) давали одинаковые результаты.
    random.seed(args.seed)       # Фиксирует seed для стандартного модуля random в Python.

    if not os.path.exists(args.result_path): # Если папки для сгенерированных захватов нет, то создает.
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.data_root_path): # Если нет папки с мэшэм объектов, то останавливает программу и вызывает ошибку.
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if (args.object_code_list is not None) + args.all != 1: # Этот код проверяет, что либо args.object_code_list, либо args.all переданы, но не оба сразу.
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified') # Если переданы оба или ни один, программа остановится с ошибкой ValueError.
    
    if args.todo: # Если передано todo = True, то открыть текстовый файл todo.txt и построчно читать. И записывать каждую строку в список object_code_list_all.
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path) # Если False в список object_code_list_all записывать имена файлов из папки меша.
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)): # Проверяем, что переданные объекты существуют в data_root_path.
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path') # Если нет - вызываем ValueError, т.к. пользователь указал несуществующие объекты.
    else:
        object_code_list = object_code_list_all # Если --object_code_list не передан, берем все объекты
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    
    # generate
    # Этот блок кода перемешивает список объектов и разбивает его на группы, чтобы выполнять обработку параллельно.
    random.seed(args.seed) # Устанавливаем случайное зерно (seed) для воспроизводимости. Устанавливает фиксированное зерно (args.seed) для random, чтобы перемешивание всегда происходило одинаково при каждом запуске.
    random.shuffle(object_code_list) # Перемешиваем список объектов
    objects_each = args.max_total_batch_size // args.batch_size_each # Определяем, сколько объектов будет в каждой группе
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)] # Разбиваем список объектов на группы. Разбивает object_code_list на части размером objects_each. Это делается, чтобы разделить нагрузку между процессами.

    # Этот блок распределяет объекты по процессам и запускает параллельную обработку с multiprocessing.Pool.
    process_args = []
    for id, object_code_group in enumerate(object_code_groups): # Формируем список аргументов для каждого процесса.
        process_args.append((args, object_code_group, id + 1, gpu_list)) # функция generate() получит эти параметры.

    with multiprocessing.Pool(len(gpu_list)) as p: # Запускаем параллельную обработку
        it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating', maxinterval=1000) # Передаем задачи в generate() и отслеживаем прогресс
        list(it) # запускает итерацию по объекту it
