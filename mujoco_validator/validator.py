import mujoco
import mujoco.viewer

import numpy as np
from scipy.spatial.transform import Rotation
import json
import os
import xml.etree.ElementTree as ET
import warnings

class Validator():

  def __init__(self, hand_name, angle='', path=''):
    self.angle = angle
    self.path = path
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
    data_dict = np.load(os.path.join('result/' + self.path + self.hand_name, object_name + self.angle + '.npy'), allow_pickle=True)
    for i in range(len(data_dict)):
      self.data_dict = data_dict[i]
      self.corr_model(object_name)
      m = mujoco.MjModel.from_xml_path(self.hand_name + '/scene.xml')
      d = mujoco.MjData(m)
      m.opt.viscosity = 0
      qpos = self.data_dict['qpos']
      rot = np.array([qpos[name] for name in self.rot_names])
      translate = np.array([qpos[name] for name in self.translation_names])
      joints_control = np.array([qpos[name] for name in self.joint_names])

      success = False
      
      # Установка заданных положений
      d.qpos[:3]=translate
      d.qpos[7:len(self.joint_names)+7] = joints_control
      r = Rotation.from_euler('xyz', rot, degrees=False)
      r = r.as_quat()
      r = np.roll(r, 1)
      d.qpos[3:7]=r
      m.opt.viscosity = 0
      d.qpos[7:len(self.joint_names)+7] = joints_control - np.pi/15
      # with mujoco.viewer.launch_passive(m, d) as viewer:
      flag = False
    # Сохранить последнее стабильное состояние
      while not(d.time > 2 or flag):
        #########################################################
        ## Закон управления
        # Установка заданных положений
        d.qpos[:3]=translate
        d.ctrl[0:len(self.joint_names)] = joints_control
        # d.qpos[7:len(self.joint_names)+7] = joints_control
        r = Rotation.from_euler('xyz', rot, degrees=False)
        r = r.as_quat()
        r = np.roll(r, 1)
        d.qpos[3:7]=r
        if d.time < 0.8:
          d.qpos[len(self.joint_names)+7:len(self.joint_names)+10] = [0,0,0] 
          d.qpos[len(self.joint_names)+10:len(self.joint_names)+14] = [1,0,0,0] 
          m.opt.gravity = [0, 0, 0]
        else:
          m.opt.gravity = [0, 0, -9.81]
        #########################################################
        mujoco.mj_step(m, d)
        if d.warning.number.any():
          flag = True
        # viewer.sync()

      # Определение наличия контакта с объектом
      if not flag:
        for i in range(d.ncon):
          geom1_id = d.contact[i].geom1
          geom2_id = d.contact[i].geom2

          body1_id = m.geom_bodyid[geom1_id]
          body2_id = m.geom_bodyid[geom2_id]

          body1_name = m.body(body1_id).name
          body2_name = m.body(body2_id).name

          if (body1_name == 'decomposed' and body2_name != 'world') or \
          (body2_name == 'decomposed' and body1_name != 'world'):
            success = True
      else:
        success = False
      success_list.append(success)
    success_rate = sum(success_list) / len(success_list)
    return success_rate