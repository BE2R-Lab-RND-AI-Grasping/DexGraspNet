'''
WORK CODE
There is vizualization contact points on STL mesh
'''


import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import trimesh
import json
import os
from scipy.spatial.transform import Rotation as R

def load_mesh_trimesh_to_open3d(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces)
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def create_sphere(center, radius=0.0025, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def parse_chain_transformations(xml_path, target_body):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    chain = []
    def find_chain_recursive(body, path):
        if body.attrib.get("name") == target_body:
            chain.append(body)
            return True
        for child in body.findall("body"):
            if find_chain_recursive(child, path):
                chain.append(body)
                return True
        return False

    worldbody = root.find("worldbody")
    for body in worldbody.findall("body"):
        if find_chain_recursive(body, []):
            break

    chain.reverse()  # from root to target
    T = np.eye(4)
    for body in chain:
        pos = np.fromstring(body.attrib.get("pos", "0 0 0"), sep=' ')
        quat = np.fromstring(body.attrib.get("quat", "1 0 0 0"), sep=' ')  # w x y z
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        Ti = np.eye(4)
        Ti[:3, :3] = rot
        Ti[:3, 3] = pos
        T = T @ Ti  # accumulate
    return T

def transform_mesh(mesh, transform):
    mesh.transform(transform)
    return mesh

def visualize_contact_points(xml_path, mesh_path, contact_json_path, body_name):
    # Загрузить и трансформировать меш
    T = parse_chain_transformations(xml_path, body_name)
    mesh = load_mesh_trimesh_to_open3d(mesh_path)
    mesh.paint_uniform_color([0.7, 0.7, 0.8])
    # mesh = transform_mesh(mesh, T)

    # Загрузить точки
    with open(contact_json_path, 'r') as f:
        contact_data = json.load(f)

    geometries = [mesh]
    for pt in contact_data.get(body_name, []):
        sphere = create_sphere(np.array(pt))
        geometries.append(sphere)

    print("Центр меша (после трансформации):", mesh.get_center())
    print("Первая контактная точка:", contact_data[body_name][0])

    o3d.visualization.draw_geometries(geometries)

# Путь к файлам
visualize_contact_points(
    xml_path="DIP-Flex_opened_kinematics.xml",                     
    # mesh_path="assets/Link_pinkie_DPflexion.STL",         
    # mesh_path="assets/Link_index_DPflexion.STL",
    # mesh_path="assets/Link_thumb_DPflexion.STL",
    # mesh_path="assets/Link_index_PPflexion.STL",
    # mesh_path="assets/Link_pinkie_PPflexion.STL",
    mesh_path="assets/Link_thumb_PPflexion.STL",
    
    contact_json_path="contact_points.json",    
    
    # body_name="Link_pinkie_DPflexion"
    # body_name="Link_index_DPflexion"
    # body_name="Link_thumb_DPflexion"
    # body_name="Link_index_PPflexion"
    # body_name="Link_pinkie_PPflexion"
    body_name="Link_thumb_PPflexion"
)

