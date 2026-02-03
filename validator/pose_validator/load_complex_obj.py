import time
import mujoco
import mujoco.viewer
import numpy as np
import transforms3d.euler as euler
import os
import trimesh


def add_meshes_from_folder(mj_spec: mujoco.MjSpec, folder_path, prefix="_mesh", scale=[1, 1, 1]):
    """
    Add all meshes from a folder to the MJCF model.

    Args:
        model: The MJCF model to add meshes to.
        folder_path: The path to the folder containing the .obj files.
        prefix: Prefix to add to all names to avoid conflicts.
    """

    mesh_dir = mj_spec.meshdir
    modelfile_dirdir = mj_spec.modelfiledir  + "/" + mesh_dir
    # Get relative path from modelfiledir to folder_path
    rel_path = os.path.relpath(folder_path, modelfile_dirdir)
    # Set meshdir to this relative path
    # mj_spec.meshdir = rel_path
    mj_meshes_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj") and filename != "decomposed.obj":
            mesh_name = prefix + filename
            mesh_path = os.path.join(folder_path, filename)
            mesh_path2 = os.path.join(rel_path, filename)
            mj_mesh = mj_spec.add_mesh(file=mesh_path2, name=mesh_name, scale=scale)
            mj_meshes_names.append(mesh_name)
    convex_parts = [trimesh.load(os.path.join(folder_path, p)) for p in os.listdir(folder_path) if p.endswith(".obj")]
    combined_mesh = trimesh.util.concatenate(convex_parts)
    combined_mesh.apply_scale(scale)
    return combined_mesh, mj_meshes_names


def add_graspable_body(
    mj_spec: mujoco.MjSpec,
    combined_mesh: trimesh.Trimesh,
    mj_meshes_names,
    obj_name: str = "graspable_object",
    mass=0.5,
    init_pos=[0.0, 0.0, 0.0],
):

    init_quat = euler.euler2quat(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    density = 2*mass/(combined_mesh.volume)
    body = mj_spec.worldbody.add_body(pos=init_pos, quat=init_quat, name=obj_name, mass=0)
    for name in mj_meshes_names:
        body.add_geom(
            name=name + "_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            rgba=[0.7, 0, 0.3, 0.5],
            meshname=name,
            condim=4,
            margin=0.001,
            friction = [0.7, 0.002, 0.002],
            density=density,
            solref=np.array([-7000, -167]),
            solimp=np.array([0.9, 0.95, 0.001, 0.5, 5])
        )
    return body


def main():
    mesh_dir = "data/mjcf/models/objs/core-bowl-a593e8863200fdb0664b3b9b23ddfcbc/coacd"
    spec_hand = mujoco.MjSpec.from_file(
        "data/mjcf/models/hand_models/shadow_hand/shadow_hand_wrist_free_special_path.xml"
    )
    spec_hand.option.timestep = 0.002
    spec_hand.option.gravity = [0, 0, -10]
    combined_mesh, mesh_names = add_meshes_from_folder(spec_hand, mesh_dir, prefix="obj_", scale=[0.1, 0.1, 0.1])
    graspable_body = add_graspable_body(spec_hand, combined_mesh, mesh_names, init_pos=[0, 0, 0.4], mass=0.2)
    graspable_body.add_joint(name="free_object", type=mujoco.mjtJoint.mjJNT_FREE)
    composite_model = spec_hand.compile()
    composite_data = mujoco.MjData(composite_model)

    with mujoco.viewer.launch(composite_model, composite_data) as viewer:
        while True:
            step_start = time.time()
            mujoco.mj_step(composite_model, composite_data)
            viewer.sync()
            time_until_next_step = composite_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
