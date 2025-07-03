import numpy as np
from validator import Validator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hand_name', default='shadow_dexee')
parser.add_argument('--object_code_list', default=
                    [
                      'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03',
                      'sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5',
                      'ddg-gd_banana_poisson_002',
                      'mujoco-Ecoforms_Plant_Plate_S11Turquoise',
                      'sem-Bottle-437678d4bc6be981c8724d5673a063a6'
                    ])
args = parser.parse_args()
Matrix = []

for object in args.object_code_list:
    angles_list = np.linspace(-np.pi/6, np.pi/6, 9)
    row = []
    print(object)
    for angle in angles_list:
        print(angle)
        valid = Validator(args.hand_name, angle='_angle' + str(round(angle, 2)), path='diff_angles/')
        SR = valid.sim(object)
        row.append(SR)
    Matrix.append(row)
print(np.array(Matrix))