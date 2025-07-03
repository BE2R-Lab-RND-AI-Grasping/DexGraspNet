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

parser = argparse.ArgumentParser()
parser.add_argument('--hand_name', default='shadow_dexee')
parser.add_argument('--object_code', default='sem-Bottle-437678d4bc6be981c8724d5673a063a6')
parser.add_argument('--num', default=0)
args = parser.parse_args()

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
hand_config = json.load(open(args.hand_name + '/' + args.hand_name + '.json', 'r'))
joint_names = hand_config['joint_names']

#load npy file
data_dict_all = np.load(os.path.join('result/' + args.hand_name, args.object_code + '.npy'), allow_pickle=True)

for data_dict in data_dict_all:
  scale = data_dict['scale']
  qpos = data_dict['qpos']
  rot = np.array([qpos[name] for name in rot_names])
  translate = np.array([qpos[name] for name in translation_names])
  joints_control = np.array([qpos[name] for name in joint_names])

  # load XML file
  tree = ET.parse('meshh/' + args.object_code + '/coacd/decomposed/decomposed.xml')
  root = tree.getroot()
  for mesh in root.findall('.//mesh'):
      if 'scale' in mesh.attrib: 
          mesh.set('scale', f'{scale} {scale} {scale}') 
  tree.write('meshh/' + args.object_code + '/coacd/decomposed/decomposed.xml')

  tree = ET.parse(args.hand_name + '/scene.xml')
  root = tree.getroot()
  for include in root.findall('.//include'):
      if 'decomposed' in include.attrib['file']: 
        include.set('file', '../meshh/' + args.object_code + '/coacd/decomposed/decomposed.xml') 
  tree.write(args.hand_name + '/scene.xml')

  success = False
  q1_list = []

  m = mujoco.MjModel.from_xml_path(args.hand_name + '/scene.xml')
  d = mujoco.MjData(m)
  d.qpos[7:len(joint_names)+7] = hand_config['init_pos']
  d.qpos[:3]=translate
  r = Rotation.from_euler('xyz', rot, degrees=False)
  r = r.as_quat()
  r = np.roll(r, 1)
  d.qpos[3:7]=r
  m.opt.viscosity = 0
  d.qpos[7:len(joint_names)+7] = joints_control + np.pi/30

  with mujoco.viewer.launch_passive(m, d) as viewer:
    while d.time < 2:
      step_start = time.time()
      
      # setting preset positions
      d.qpos[:3]=translate
      d.ctrl[0:len(joint_names)] = joints_control
      # d.qpos[7:len(joint_names)+7] = joints_control
      r = Rotation.from_euler('xyz', rot, degrees=False)
      r = r.as_quat()
      r = np.roll(r, 1)
      d.qpos[3:7]=r

      if d.time < 0.8:
        d.qpos[len(joint_names)+7:len(joint_names)+10] = [0,0,0] 
        d.qpos[len(joint_names)+10:len(joint_names)+14] = [1,0,0,0]
        m.opt.gravity = [0, 0, 0]
      else:
        m.opt.gravity = [0, 0, -9.81]

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
      mujoco.mj_step(m, d)
      viewer.sync()