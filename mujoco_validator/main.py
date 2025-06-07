import time

import mujoco
import mujoco.viewer

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import argparse
import json
import os
import xml.etree.ElementTree as ET

class Validator():

  def __init__(self, hand_name):
    self.translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    self.rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    hand_config = json.load(open(hand_name + '/' + hand_name + '.json', 'r'))
    self.joint_names = hand_config['joint_names']    
    self.hand_name = hand_name


  def corr_model(self, object_name):
    self.object_name = object_name
    self.scale = self.data_dict['scale']
    self.qpos = self.data_dict['qpos']
    self.rot = np.array([self.qpos[name] for name in self.rot_names])
    self.translate = np.array([self.qpos[name] for name in self.translation_names])
    self.joints_control = np.array([self.qpos[name] for name in self.joint_names])


    tree = ET.parse('meshh/' + self.object_name + '/coacd/decomposed/decomposed.xml')
    root = tree.getroot()
    for mesh in root.findall('.//mesh'):
        if 'scale' in mesh.attrib: 
            mesh.set('scale', f'{self.scale} {self.scale} {self.scale}') 
    tree.write('meshh/' + self.object_name + '/coacd/decomposed/decomposed.xml')
    
    tree = ET.parse(self.hand_name + '/scene.xml')
    root = tree.getroot()
    for include in root.findall('.//include'):
        if 'decomposed' in include.attrib['file']: 
          include.set('file', '../meshh/' + self.object_name + '/coacd/decomposed/decomposed.xml') 
    tree.write(self.hand_name + '/scene.xml')


  def sim(self, object_name):
    success_list = []
    data_dict = np.load(os.path.join('result/' + self.hand_name, object_name + '.npy'), allow_pickle=True)
    for i in range(len(data_dict)):
      self.data_dict = data_dict[i]
      self.corr_model(object_name)
      m = mujoco.MjModel.from_xml_path('shadow_dexee/scene.xml')
      d = mujoco.MjData(m)
      m.opt.viscosity = 0
      qpos = self.data_dict['qpos']
      rot = np.array([qpos[name] for name in self.rot_names])
      translate = np.array([qpos[name] for name in self.translation_names])
      joints_control = np.array([qpos[name] for name in self.joint_names])
      scene_option = mujoco.MjvOption()
      scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

      success = False
      q1_list = []
      
      # setting preset positions
      d.qpos[:3]=translate
      d.qpos[7:len(self.joint_names)+7] = joints_control
      r = Rotation.from_euler('xyz', rot, degrees=False)
      r = r.as_quat()
      r = np.roll(r, 1)
      d.qpos[3:7]=r
      for i in range(1000):
        mujoco.mj_step(m, d)

      # determining contact with object
      for i in range(d.ncon):
        geom1_id = d.contact[i].geom1
        geom2_id = d.contact[i].geom2

        body1_id = m.geom_bodyid[geom1_id]
        body2_id = m.geom_bodyid[geom2_id]

        body1_name = m.body(body1_id).name
        body2_name = m.body(body2_id).name

        if body1_name == 'decomposed' or body2_name == 'decomposed':
          success = True
      success_list.append(success)
    success_rate = sum(success_list) / len(success_list)
    return success_rate

parser = argparse.ArgumentParser()
parser.add_argument('--hand_name', default='shadow_dexee')
parser.add_argument('--object_code_list', default=
                    [
                      'sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5',
                      'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03',
                      'ddg-gd_banana_poisson_002',
                      'mujoco-Ecoforms_Plant_Plate_S11Turquoise',
                      'sem-Bottle-437678d4bc6be981c8724d5673a063a6'
                    ])
args = parser.parse_args()

valid = Validator(args.hand_name)
for object in args.object_code_list:
  print(object, '\n', valid.sim(object), '\n')



