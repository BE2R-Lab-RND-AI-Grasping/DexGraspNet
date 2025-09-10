import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

m = mujoco.MjModel.from_xml_path('shadow_dexee/scene.xml')
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    d.qpos[0]=1*np.cos(step_start)
    d.qpos[1]=1*np.sin(step_start)
    d.qpos[2]=1*np.cos(step_start)
    # quat=np.zeros((4,1), dtype=np.float64)
    # euler=np.zeros((3,1), dtype=np.float64)
    # mujoco.mju_euler2Quat(quat=quat, euler=euler)
    angle = 1*np.cos(step_start)
    r = Rotation.from_euler('xyz', [0, angle, 0], degrees=False)
    r = r.as_quat()
    d.qpos[3:7]=r#mujoco.euler2quat([1*np.cos(step_start), 0, 0])

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)