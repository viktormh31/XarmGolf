import time
import os
import gym
import numpy as np
import XarmEnv

config = {

    'GUI' : True,
    'reward_type' : "sparse",
}


env = XarmEnv.XarmRobotEnv(config)
action = np.array([0.5,0.0,0.3])
env.make_goal(action)
time.sleep(2)
for i in range(50000):
    env.move_joint(16)
   
    time.sleep(.06)
    #print(action)
time.sleep(122)