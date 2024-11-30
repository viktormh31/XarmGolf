import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces



p.connect(p.GUI)
p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0.3,0.1,-0.1])

p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0.0,0.0,-9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
fullpath = os.path.join(os.path.dirname(__file__), "urdf/xarm7.urdf")
joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 
                                0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 
                                1.1524205092196147, -0.0004991403332530034] + [0]*9

xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)
for i in range(17):
    p.resetJointState(xarm,i,joint_init_pos[i])

fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_sphere.urdf')
sphere = p.loadURDF(fullpath,[0.5,0,0.02],useFixedBase=True)






time.sleep(20)
