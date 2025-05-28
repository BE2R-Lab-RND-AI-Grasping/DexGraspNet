'''
Last modified: 06.05.2025
Author: Olga Borisova
Description: Create a Hand Model from XML. Load hand visual and contact capsules
'''

import os
import numpy as np
import torch
import trimesh
from xml.dom import minidom
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Transform3d
import xml.etree.ElementTree as ET # for working with XML files (their tree structure)


class HandModel:
    def __init__(self, xml_path, stl_dir, mesh_color, device='cpu', num_points_per_capsule=10):
        '''
        Initialization hand model
        xml_path - path to XML file of hand
        stl_dir - directory with STL files
        num_points_per_capsule - how much points has to generte per capsule
        '''
        self.xml_path = xml_path
        self.stl_dir = stl_dir
        self.mesh_color = mesh_color
        self.device = device
        self.num_points_per_capsule = num_points_per_capsule

        # Load hand meshes (from STL files)
        self.meshes = self.load_meshes()

        # Load capsules from XML file
        self.contact_capsules = self.load_capsules(self.xml_path)

        # Sampling contact points from capsules
        self.contact_points = self.sample_points_from_capsules(self.contact_capsules, num_points_per_capsule=self.num_points_per_capsule)



    def load_capsules(self, xml_path):
            '''
            Читает MJCF XML-файл и извлекает параметры всех капсул (тип geom="capsule").
            Возвращает список словарей с полями pos, fromto и size.
            '''
            tree = ET.parse(xml_path)
            root = tree.getroot()

            capsules = []
            for geom in root.findall(".//geom[@type='capsule']"):
                fromto = list(map(float, geom.attrib['fromto'].split()))
                radius = float(geom.attrib['size'])
                name = geom.attrib.get('name', 'noname')

                p1 = np.array(fromto[:3]) # начальная точка оси
                p2 = np.array(fromto[3:]) # конечная точка оси

                capsules.append({
                    'name': name,
                    'p1': torch.tensor(p1, dtype=torch.float32),
                    'p2': torch.tensor(p2, dtype=torch.float32),
                    'radius': radius,
                })
            
            return capsules

    def sample_points_from_capsules(self, capsules, num_points_per_capsule=10):
        """
        Генерирует точки на поверхности каждой капсулы:
        - Случайно выбирает точку вдоль оси капсулы.
        - Случайным образом смещает её на расстояние радиуса в произвольном направлении.
        Возвращает Pointclouds из PyTorch3D.
        """
        all_points = []

        for capsule in capsules:
            p1 = capsule['p1']
            p2 = capsule['p2']
            radius = capsule['radius']

            for i in range(num_points_per_capsule):
                t = np.random.rand()                    # случайная позиция вдоль оси
                center = (1 - t) * p1 + t * p2          # интерполяция между p1 и p2

                direction = np.random.randn(3)          # случайное направление
                direction /= np.linalg.norm(direction)  # нормализация
                point = center + radius * direction     # точка на поверхности
                
                all_points.append(point)
        
        all_points = np.array(all_points)
        pointcloud = Pointclouds(points=[torch.tensor(all_points, dtype=torch.float32, device=self.device)])
        
        return pointcloud

    def load_meshes(self):
    """
    Загружает все .STL файлы из указанной директории и применяет к ним цвет.
    """
    mesh_files = sorted(os.listdir(self.stl_dir))
    meshes = []

    for filename in mesh_files:
        if filename.endswith('.STL'):
            mesh_path = os.path.join(self.stl_dir, filename)
            mesh = trimesh.load(mesh_path, process=False)  # process=False — не менять геометрию
            mesh.visual.vertex_colors = self.mesh_color    # применяем цвет к вершинам
            meshes.append(mesh)

    return meshes