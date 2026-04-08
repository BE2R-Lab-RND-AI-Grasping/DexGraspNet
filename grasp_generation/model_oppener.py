import mujoco
import mujoco.viewer

mj_model = mujoco.MjModel.from_xml_path("/home/karich/prj/DexGraspNet/grasp_generation/mjcf/shadow_dexee/shadow_dexee_simpl.xml")
mj_data = mujoco.MjData(mj_model)

with mujoco.viewer.launch(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
