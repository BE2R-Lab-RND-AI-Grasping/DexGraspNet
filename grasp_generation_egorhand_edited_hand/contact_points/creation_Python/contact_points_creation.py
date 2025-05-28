'''
WORK CODE
'''


import os
import trimesh
import numpy as np
import json
from tqdm import tqdm

# Настройки
meshes_dir = "assets/"  # Папка с STL-файлами
output_json = "contact_points.json"  # Выходной файл
num_points_per_part = 20  # Количество точек на деталь

# Загрузка всех мешей
meshes = {}
for mesh_file in os.listdir(meshes_dir):
    if mesh_file.endswith('.STL'):
        part_name = os.path.splitext(mesh_file)[0]
        mesh_path = os.path.join(meshes_dir, mesh_file)
        meshes[part_name] = trimesh.load(mesh_path)

# Генерация точек на поверхности мешей
contact_points = {} # Словарь для контактных точек

for part_name, mesh in tqdm(meshes.items()):
    points = []

    # Способ 1: Случайные точки на всей поверхности (по умолчанию)
    # points, _ = trimesh.sample.sample_surface(mesh, count=num_points_per_part)
    # contact_points[f"{part_name}"] = points.tolist()

    # Способ 2: Точки на конкретной грани AABB (для пальцев)
    aabb = mesh.bounding_box
    for _ in range(num_points_per_part):
        # Настраиваем диапазон координат для нужной грани:
        # Например, для верхней грани по Z:
        point = [

            # Задняя грань
            np.random.uniform(aabb.bounds[0][0], aabb.bounds[1][0]),
            aabb.bounds[0][1] + 0.001,
            np.random.uniform(aabb.bounds[0][2], aabb.bounds[1][2])

        ]
        points.append(point)

    contact_points[f"{part_name}"] = points



# Сохранение в JSON
with open(output_json, 'w') as f:
    json.dump(contact_points, f, indent=2)

print(f"Создан файл контактных точек: {output_json}")


