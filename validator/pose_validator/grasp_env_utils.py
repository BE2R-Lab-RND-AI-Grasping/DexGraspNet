from copy import deepcopy
import time
import mujoco
import mujoco.viewer
import numpy as np
import transforms3d.euler as euler
from transforms3d import affines
from numpy.typing import NDArray

# from pose_validator.load_complex_obj import add_graspable_body, add_meshes_from_folder
from load_complex_obj import add_graspable_body, add_meshes_from_folder


shadow_hand_mapping = {
    "WRJTx": "WRJTx",
    "WRJTy": "WRJTy",
    "WRJTz": "WRJTz",
    "WRJRx": "WRJRx",
    "WRJRy": "WRJRy",
    "WRJRz": "WRJRz",
    "robot0:FFJ3": "robot0:A_FFJ3",
    "robot0:FFJ2": "robot0:A_FFJ2",
    "robot0:FFJ1": "robot0:A_FFJ1",
    "robot0:FFJ0": "robot0:A_FFJ0",
    "robot0:MFJ3": "robot0:A_MFJ3",
    "robot0:MFJ2": "robot0:A_MFJ2",
    "robot0:MFJ1": "robot0:A_MFJ1",
    "robot0:MFJ0": "robot0:A_MFJ0",
    "robot0:RFJ3": "robot0:A_RFJ3",
    "robot0:RFJ2": "robot0:A_RFJ2",
    "robot0:RFJ1": "robot0:A_RFJ1",
    "robot0:RFJ0": "robot0:A_RFJ0",
    "robot0:LFJ4": "robot0:A_LFJ4",
    "robot0:LFJ3": "robot0:A_LFJ3",
    "robot0:LFJ2": "robot0:A_LFJ2",
    "robot0:LFJ1": "robot0:A_LFJ1",
    "robot0:LFJ0": "robot0:A_LFJ0",
    "robot0:THJ4": "robot0:A_THJ4",
    "robot0:THJ3": "robot0:A_THJ3",
    "robot0:THJ2": "robot0:A_THJ2",
    "robot0:THJ1": "robot0:A_THJ1",
    "robot0:THJ0": "robot0:A_THJ0",
}


def get_key_bodies_pose(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> dict[str, NDArray]:
    body_names = get_key_bodies_shadow_names(mj_model)

    body_pos_dict = {}
    for b_name in body_names:
        body_id = mj_data.model.body(name=b_name).id
        body_centr_pose = mj_data.xipos[body_id]
        body_pos_dict[b_name] = body_centr_pose
    return body_pos_dict


def set_position(mj_data: mujoco.MjData, qpos: dict[str, float], maping: dict[str, str] = None):
    if maping is None:
        for key, value in qpos.items():
            mj_data.actuator(key).ctrl = value
    else:
        for key, value in qpos.items():
            mj_data.actuator(maping[key]).ctrl = value


def set_position_kinematics(mj_data: mujoco.MjData, qpos: dict[str, float], maping: dict[str, str] = None):
    if maping is None:
        for key, value in qpos.items():
            qpos_id = mj_data.model.joint(name=key).qposadr
            mj_data.qpos[qpos_id] = value

    else:
        for key, value in qpos.items():
            qpos_id = mj_data.model.joint(name=maping[key]).qposadr
            mj_data.qpos[qpos_id] = value


def get_key_bodies_shadow_names(composite_model):

    bodies_names = []
    for i in range(composite_model.nbody):
        bodies_names.append(composite_model.body(i).name)
    bodies_names = [name for name in bodies_names if "distal" in name or "palm" in name]
    return bodies_names


def transform_wirst_pos_to_obj(
    pos_obj: NDArray, quat_obj: NDArray, pos_hand: NDArray, quat_hand: NDArray
) -> tuple[NDArray, NDArray]:

    rotation_matrix_obj = euler.quat2mat(quat_obj)

    homogeneous_matrix_obj = affines.compose(T=pos_obj, R=rotation_matrix_obj, Z=np.ones(3))

    rotation_matrix_hand = euler.quat2mat(quat_hand)
    # Create a homogeneous transformation matrix
    homogeneous_matrix_hand = affines.compose(T=pos_hand, R=rotation_matrix_hand, Z=np.ones(3))

    transformed_pos = homogeneous_matrix_obj.dot(homogeneous_matrix_hand)
    T, R, _, _ = affines.decompose(transformed_pos)
    return T, R


def add_body_key_points(spec_mujoco, key_pose_dict):
    for pose_name, pose in key_pose_dict.items():
        spec_mujoco.worldbody.add_geom(
            name=pose_name + "ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[1, 1, 0, 0.25],
            size=[0.005, 0.005, 0.1],
            pos=pose,
        )


def get_final_bodies_pose(final_position: dict[str, float], hand_model_path: str):

    model_for_pose = mujoco.MjModel.from_xml_path(hand_model_path)
    data_for_pose = mujoco.MjData(model_for_pose)

    set_position_kinematics(data_for_pose, final_position)
    mujoco.mj_kinematics(model_for_pose, data_for_pose)

    key_bodies_pose = get_key_bodies_pose(model_for_pose, data_for_pose)
    return key_bodies_pose


def main():
    pass


if __name__ == "__main__":
    main()
