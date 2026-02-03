from dataclasses import dataclass, field
from time import sleep, time
from typing import Union
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer
import logging

import numpy as np
from scipy.spatial.transform import Rotation
import json
import os
import xml.etree.ElementTree as ET
import warnings

# from pose_validator.grasp_env_utils import set_position, set_position_kinematics
from grasp_env_utils import set_position, set_position_kinematics
# from pose_validator.load_complex_obj import add_graspable_body, add_meshes_from_folder
from load_complex_obj import add_graspable_body, add_meshes_from_folder

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ValidatorConfig:
    testing_time: float = 3
    simulation_timestep: float = 0.005
    gravity: list = field(default_factory=lambda: [0, 0, -9.81])
    object_mass: float = 0.2
    disturbance_force_magnitude: float = 2  # Magnitude of the disturbance force applied to the object


class Validator:
    def __init__(
        self,
        hand_model: Union[str, mujoco.MjSpec],
        hand_config: dict,
        validator_config: ValidatorConfig = ValidatorConfig(),
    ):
        if isinstance(hand_model, str):
            hand_model_spec = mujoco.MjSpec.from_file(hand_model)
        elif isinstance(hand_model, mujoco.MjSpec):
            hand_model_spec = hand_model
        else:
            raise ValueError("hand_model must be a string or mujoco.MjSpec")

        self.hand_model_spec = hand_model_spec  # type: ignore

        self.joint_finger_names = hand_config["joint_finger_names"]
        self.wrist_t_joint_names_xyz = hand_config["wrist_translation_names"]
        self.wrist_r_joint_names_xyz = hand_config["wrist_rotational_joint_names"]
        self.hand_base_joint_name = hand_config["hand_base_joint_name"]
        self.validator_config = validator_config
        
        # Set up logger for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def setup_simulation(self, mj_spec: mujoco.MjSpec):
        mj_spec.option.timestep = self.validator_config.simulation_timestep
        self.hand_model_spec.option.gravity = np.array(self.validator_config.gravity)
        # self.hand_model_spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT

        mj_spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_spec.option.iterations = 100  # More solver iterations
        mj_spec.option.tolerance = 1e-8
        mj_spec.option.viscosity = 0.001  # Small damping helps stability
        mj_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        mj_spec.option.noslip_iterations = 50
        mj_spec.option.ccd_iterations = 100
        mj_spec.option.impratio = 100
        mj_spec.option.o_margin = 0.001
        mj_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        mj_spec.option.ccd_tolerance = 1e-7
        mj_spec.option.o_friction = np.array([0.6, 0.7, 5.0e-03, 1.0e-03, 1.0e-03])
        mj_spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_LIMIT

    def get_joint_to_actuator_map(self):
        """
        Creates a mapping from joint names to actuator names.

        Returns:
            dict: A dictionary mapping joint names to their corresponding actuator names.
        """
        joint_to_actuator = {}

        # Iterate through all actuators in the model
        for actuator_i in self.hand_model_spec.actuators:
            joint_to_actuator[actuator_i.target] = actuator_i.name

        return joint_to_actuator

    def check_contact_for_obj(self, mj_model, mj_data):
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]

            geom1_body = mj_model.geom_bodyid[contact.geom1]
            geom2_body = mj_model.geom_bodyid[contact.geom2]

            body1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
            body2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)

            if body1_name == "graspable_object" or body2_name == "graspable_object":
                return True
        return False

    def get_contact_depth(self, mj_model, mj_data):
        contact_depths = []
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]

            geom1_body = mj_model.geom_bodyid[contact.geom1]
            geom2_body = mj_model.geom_bodyid[contact.geom2]

            body1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
            body2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)

            if body1_name == "graspable_object" or body2_name == "graspable_object":
                contact_depths.append(contact.dist)
        return contact_depths

    def experiment(self, position_dict: dict, object_folder: str, visualize: bool = True):
        mj_model, mj_data = self.create_experiment_model(position_dict, object_folder)
        is_simulation_crash = False
        timesteps_number = int(self.validator_config.testing_time / self.validator_config.simulation_timestep)

        object_height = np.zeros(timesteps_number)
        object_contact = np.zeros(timesteps_number, dtype=bool)
        qpos_history = np.zeros((timesteps_number, mj_data.qpos.shape[0]))
        control_error_fingers = np.zeros((timesteps_number, len(self.joint_finger_names)))

        desiried_qpos = position_dict["qpos"]
        mujoco.mj_forward(mj_model, mj_data)
        penetration = self.get_contact_depth(mj_model, mj_data)

        if visualize:
            viewer = self.launch_viewer(mj_model, mj_data)
        for step in range(timesteps_number):
            # Check for collisions between the object and hand

            contact = self.check_contact_for_obj(mj_model, mj_data)
            object_contact[step] = contact

            object_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "free_object")
            object_height[step] = mj_data.xpos[object_body_id][2]  # z-coordinate
            qpos_history[step] = mj_data.qpos.copy()
            self.apply_disturbance_force(mj_model, mj_data, step, timesteps_number)

            # Get position error for finger joints
            finger_joint_ids = [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) + 6 for name in self.joint_finger_names
            ]
            desired_positions = np.array([desiried_qpos[name] for name in self.joint_finger_names])
            current_positions = np.array([mj_data.qpos[joint_id] for joint_id in finger_joint_ids])
            position_error = desired_positions - current_positions
            control_error_fingers[step] = position_error

            # Check if desired positions are within actuator range
            
            mujoco.mj_step(mj_model, mj_data)
            if visualize:
                viewer.sync()
                sleep(0.007)  # To make the visualization slower

        if visualize:
            viewer.close()


        return qpos_history, object_height, object_contact, control_error_fingers, penetration
    
    def get_pose_reneder(self, position_dict: dict, object_folder: str):
        mj_model, mj_data = self.create_experiment_model(position_dict, object_folder)
        is_simulation_crash = False
        timesteps_number = int(self.validator_config.testing_time / self.validator_config.simulation_timestep)

        object_height = np.zeros(timesteps_number)
        object_contact = np.zeros(timesteps_number, dtype=bool)
        qpos_history = np.zeros((timesteps_number, mj_data.qpos.shape[0]))
        control_error_fingers = np.zeros((timesteps_number, len(self.joint_finger_names)))

        desiried_qpos = position_dict["qpos"]
        mujoco.mj_forward(mj_model, mj_data)
        penetration = self.get_contact_depth(mj_model, mj_data)
        mujoco.mj_step(mj_model, mj_data)

        viewer = self.launch_viewer(mj_model, mj_data)
        
        one_shot_render = mujoco.renderer.Renderer(mj_model)
        viewer.sync()
        one_shot_render.update_scene(mj_data)
        frame = one_shot_render.render()
        viewer.close()

        return frame



    def range_warning(self, mj_model, desired_positions):
        for i, joint_name in enumerate(self.joint_finger_names):
            actuator_name = self.get_joint_to_actuator_map().get(joint_name)
            if actuator_name:
                actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0:
                    ctrlrange = mj_model.actuator_ctrlrange[actuator_id]
                    desired_pos = desired_positions[i]
                    if desired_pos < ctrlrange[0] or desired_pos > ctrlrange[1]:
                        self.logger.debug(
                            f"{joint_name} desired position {desired_pos:.3f} outside actuator range "
                            f"[{ctrlrange[0]:.3f}, {ctrlrange[1]:.3f}]"
                        )
                    # Also check joint limits directly
                    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        joint_range = mj_model.jnt_range[joint_id]
                        if len(joint_range) == 2 and (desired_pos < joint_range[0] or desired_pos > joint_range[1]):
                            self.logger.debug(
                                f"{joint_name} desired position {desired_pos:.3f} outside joint range "
                                f"[{joint_range[0]:.3f}, {joint_range[1]:.3f}]"
                            )

    def is_stable_grasp(self, object_contact: np.ndarray):
        if np.mean(object_contact[-10:]) < 0.8:
            return False

        return True

    def launch_viewer(self, mj_model, mj_data):
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        return viewer

    def create_experiment_model(self, position_dict: dict, object_folder: str):
        scale = position_dict["scale"]
        qpos = position_dict["qpos"]

        wirst_rot = np.array([qpos[name] for name in self.wrist_r_joint_names_xyz])
        wirst_translate = np.array([qpos[name] for name in self.wrist_t_joint_names_xyz])
        fingers_joints_control = np.array([qpos[name] for name in self.joint_finger_names])

        mj_spec = self.hand_model_spec.copy()
        combined_mesh, mesh_names = add_meshes_from_folder(
            mj_spec, object_folder, prefix="obj_", scale=[scale, scale, scale]
        )
        graspable_body = add_graspable_body(
            mj_spec,
            combined_mesh,
            mesh_names,
            init_pos=[0.0, 0.0, 0.0],
            mass=self.validator_config.object_mass,
        )
        graspable_body.add_joint(name="free_object", type=mujoco.mjtJoint.mjJNT_FREE)
        self.fix_base_joint_over_damping(mj_spec)
        self.setup_simulation(mj_spec)

        mj_model = mj_spec.compile()
        mj_data = mujoco.MjData(mj_model)

        # Filter qpos dictionary to only include finger joint entries
        desired_finger_joint_pos_dict = {
            joint_name: qpos[joint_name] for joint_name in self.joint_finger_names if joint_name in qpos
        }

        base_body_pos = mj_model.body_pos[1]
        self.move_hand_base(wirst_rot, wirst_translate, mj_data, base_body_pos)

        set_position(mj_data=mj_data, qpos=desired_finger_joint_pos_dict, maping=self.get_joint_to_actuator_map())
        set_position_kinematics(mj_data, desired_finger_joint_pos_dict)

        mujoco.mj_kinematics(mj_model, mj_data)

        desired_positions = np.array([qpos[name] for name in self.joint_finger_names])
        self.range_warning(mj_model, desired_positions)
        return mj_model, mj_data

    def fix_base_joint_over_damping(self, mj_spec: mujoco.MjSpec):
        for joint_i in mj_spec.joints:
            if joint_i.name == self.hand_base_joint_name:
                joint_i.damping = 1000
                return
        raise NotImplementedError(
            f"Didn't find the base joint in the model." f"Model should contain: {self.hand_base_joint_name} joint"
        )

    def move_hand_base(self, wirst_rot, wirst_translate, mj_data, base_body_pose):
        
        base_rotation = Rotation.from_euler("xyz", wirst_rot, degrees=False)
        r = base_rotation.as_quat()
        r = np.roll(r, 1)
        
        base_pose = np.array(base_body_pose) @ base_rotation.as_matrix().T + wirst_translate

        mj_data.qpos[3:7] = r
        mj_data.qpos[:3] = base_pose

    def apply_disturbance_force(self, mj_model, mj_data, step, timesteps_number):
        """
        Apply a disturbance force to the graspable object that changes direction periodically.

        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
            step: Current simulation step number
            timesteps_number: Total number of timesteps
        """
        # Get object body ID
        object_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")

        # Parameters for disturbance force

        direction_change_period = int(timesteps_number / 6)  # Change direction 6 times during simulation

        # Calculate force direction based on current step
        cycle = (step // direction_change_period) % 6

        if cycle == 0:
            force_direction = np.array([0.0, 0.0, 0.0])  # without force
        elif cycle == 1:
            force_direction = np.array([0.0, 1.0, 0.0])  # +Y direction
        elif cycle == 2:
            force_direction = np.array([0.0, -1.0, 0.0])  # -Y direction
        elif cycle == 3:
            force_direction = np.array([1.0, 0.0, 0.0])  # +X direction
        elif cycle == 4:
            force_direction = np.array([-1.0, 0.0, 0.0])  # -X direction
        elif cycle == 5:
            force_direction = np.array([0.0, 0.0, 2.0])  # +Z direction

        # Apply force to the object
        force = self.validator_config.disturbance_force_magnitude * force_direction
        mj_data.xfrc_applied[object_body_id][:3] = force


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to write logs to
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('pose_validator').setLevel(level)


def get_path_for_hand_model(hands_folder: str, target_hand: str) -> str:
    return f"{hands_folder}/{target_hand}/{target_hand}.xml"


def get_path_for_config(hands_folder: str, target_hand: str) -> str:
    return f"{hands_folder}/{target_hand}/{target_hand}.json"


def get_path_for_final_positions(final_positions_folder: str, target_hand: str, object_name: str) -> str:
    return f"{final_positions_folder}/{target_hand}/{object_name}.npy"


def get_path_for_object(object_folder: str, object_name: str) -> str:
    return f"{object_folder}/{object_name}/coacd"


def get_all_paths(
    hands_folder: str, final_positions_folder: str, object_folder: str, target_hand: str, object_name: str
):
    return {
        "hand_model": get_path_for_hand_model(hands_folder, target_hand),
        "hand_config": get_path_for_config(hands_folder, target_hand),
        "final_positions": get_path_for_final_positions(final_positions_folder, target_hand, object_name),
        "object": get_path_for_object(object_folder, object_name),
    }


def main():
    # Set up logging
    setup_logging(level=logging.INFO, log_file='pose_validation.log')
    
    # target_hand = "barret"
    # target_object = "sem-Bottle-437678d4bc6be981c8724d5673a063a6"

    # object_folder = "data/mjcf/models/objs/core-camera-e9f2c58d90e723f7cc57882dfaef8a57/coacd"
    # if target_hand == "shadow_hand":
    #     pose_dict = np.load(
    #         "data/final_positions/shadow_hand/core-camera-e9f2c58d90e723f7cc57882dfaef8a57.npy", allow_pickle=True
    #     )
    #     hand_config = json.load(open("data/mjcf/models/hand_models/shadow_hand/shadow_hand.json", "r"))
    #     hand_model_path = "data/mjcf/models/hand_models/shadow_hand/shadow_hand.xml"
    # elif target_hand == "shadow_dexee":
    #     pose_dict = np.load(
    #         "data/final_positions/shadow_dexee/sem-Bottle-437678d4bc6be981c8724d5673a063a6_angle-0.13.npy",
    #         allow_pickle=True,
    #     )
    #     hand_config = json.load(open("data/mjcf/models/hand_models/shadow_dexee/shadow_dexee.json", "r"))
    #     hand_model_path = "data/mjcf/models/hand_models/shadow_dexee/shadow_dexee copy.xml"
    #     object_folder = "data/mjcf/models/objs/sem-Bottle-437678d4bc6be981c8724d5673a063a6/coacd"
    # elif target_hand == "barret":
    #     pose_dict = np.load(
    #         "data/final_positions/barret/sem-Bottle-437678d4bc6be981c8724d5673a063a6.npy",
    #         allow_pickle=True,
    #     )
    #     hand_config = json.load(open("data/mjcf/models/hand_models/barret/barret.json", "r"))
    #     hand_model_path = "data/mjcf/models/hand_models/barret/barret.xml"
    #     object_folder = "data/mjcf/models/objs/sem-Bottle-437678d4bc6be981c8724d5673a063a6/coacd"
    # else:
    #     raise ValueError("Unknown hand model")

    # target_hand = "egor_hand"
    target_object = "sem-Bottle-437678d4bc6be981c8724d5673a063a6"

    target_hand = "hand_reorient"
    # target_object = "hummer_0"

    logger.info(f"Starting pose validation for hand: {target_hand}, object: {target_object}")

    paths = get_all_paths(
        hands_folder="data/mjcf/models/hand_models",
        final_positions_folder="data/final_positions",
        object_folder="data/mjcf/models/objs",
        target_hand=target_hand,
        object_name=target_object,
    )
    hand_model_path = paths["hand_model"]
    hand_config = json.load(open(paths["hand_config"], "r"))
    pose_dict = np.load(paths["final_positions"], allow_pickle=True)
    object_folder = paths["object"]

    logger.info(f"Loaded {len(pose_dict)} poses from {paths['final_positions']}")

    validator = Validator(
        hand_model=hand_model_path,
        hand_config=hand_config,
        validator_config=ValidatorConfig(),
    )

    successful_experiments = 0
    for i, pose_i in enumerate(pose_dict):
        if pose_i["energy"] > 20:
            logger.debug(f"Skipping pose {i}: energy {pose_i['energy']} > 20")
            continue
            
        logger.info(f"Processing pose {i+1}/{len(pose_dict)} (energy: {pose_i['energy']:.2f})")
        
        try:
            qpos_history, object_height, object_contact, control_error_fingers, penetration = validator.experiment(
                position_dict=pose_i, object_folder=object_folder
            )
            
            is_stable = validator.is_stable_grasp(object_contact)
            logger.info(f"Pose {i+1} result: stable_grasp = {is_stable}")
            successful_experiments += 1
            
            plt.plot(np.rad2deg(control_error_fingers))
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to process pose {i+1}: {str(e)}")
    
    logger.info(f"Completed validation: {successful_experiments} successful experiments")


if __name__ == "__main__":
    main()
