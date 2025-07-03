import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import xml.etree.ElementTree as ET

from validator import Validator

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



