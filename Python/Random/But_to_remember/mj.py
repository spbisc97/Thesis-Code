import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
import time

XML_MODEL = """
<? xml version="1.0" encoding="utf-8"?>
<mujoco>
    <option gravity="0 0 0" integrator="RK4" timestep="0.1" />
    <worldbody>
        <camera name="top" euler="0 0 0" pos="6e6 0 1"  fovy="120" />
        <!-- Define chaser body -->
        <body name="chaser" pos="6e6 0 1">
            <!-- Define satellite geometry -->
            <geom type="box" size="0.2 0.2 0.2" rgba="0.5 0.5 0.5 1"/>
            <!-- Define satellite joint -->
            <joint type="free"/>
        </body>
        <body name="target" pos="6e6 1 0">
            <!-- Define target geometry -->
            <geom type="box" size="0.2 0.2 0.2" rgba="0 0 1 1"/>
            <!-- Define target joint -->
            <joint type="free"/>
        </body>
    </worldbody>
</mujoco>   
"""

# Load the model
model = mujoco.MjModel.from_xml_string(XML_MODEL)
model.ngeom

data = mujoco.MjData(model)

import glfw

glfw.init()
window = glfw.create_window(640, 480, "Hello World", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)


camera = mujoco.MjvCamera()
option = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, maxgeom=100000)
perturb = mujoco.MjvPerturb()


print('raw access:\n', data.geom_xpos)

# MjData also supports named access:
print('\nnamed access:\n', data.body('chaser').xpos)


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 60:
        # data.qfrc_applied[:3] = -data.qpos[:3] - data.qvel[:3]

        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                data.time % 2
            )
        viewer.cam.trackbodyid = 0

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    print(data.time)
