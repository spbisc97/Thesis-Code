import mujoco
import mujoco.viewer
import mujoco_viewer
import mediapy as media
import numpy as np
import time

XML_MODEL = """
<? xml version="1.0" encoding="utf-8"?>
<mujoco>
    <option gravity="0 0 0" integrator="RK4" timestep="0.1" />
    <asset>
        <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800" mark="random" markrgb="1 1 1" />
    </asset>
    <default>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0"/>
    </default>
    <worldbody>
        <camera name="top" euler="0 0 0" pos="0 0 1"  fovy="120" />
        <!-- Define chaser body -->
        <body name="chaser" pos="1 0 1">
            <!-- Define satellite geometry -->
            <geom type="box" size="0.2 0.2 0.2" rgba="0.5 0.5 0.5 1"/>
            <!-- Define satellite joint -->
            <joint type="free"/>
        </body>
        <body name="target" pos="0 0 0">
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
scene = mujoco.MjvScene(model, maxgeom=100000)
perturb = mujoco.MjvPerturb()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

camera.type = mujoco.mjtCamera.mjCAMERA_FREE
mj_viewer = mujoco_viewer.MujocoViewer(model, data)
mj_viewer._run_speed = 0.1

print('raw access:\n', data.geom_xpos)

# MjData also supports named access:
print('\nnamed access:\n', data.body('chaser').xpos)


for _ in range(3600 * 10 * 10):
    mujoco.mj_step(model, data)
    data.qfrc_applied[:3] = -data.qpos[:3] - data.qvel[:3]
    mj_viewer.render()
