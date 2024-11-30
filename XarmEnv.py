import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces

class XarmRobotEnv():

    def __init__(self, config):
        self.time_step = 1./240


        # robot parameters
        self.distance_threshold = 0.05
        self.num_joints = 17
        self.gripper_driver_index = 10
        self.gripper_base_index = 9
        self.arm_eef_index = 8
        self.reward_type = config['reward_type']
        self.start_pos = [0,0,0]
        self.start_orientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 
                                0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 
                                1.1524205092196147, -0.0004991403332530034] + [0]*9
        self.max_vel = 1
        self.dt = self.time_step * 20
        self.pos_space = spaces.Box(low=np.array([0.2,-0.4,0.02]), high=np.array([0.8,0.4,0.6]))

        # connect bullet
        #if self.num_client == 1 and config['GUI']:
        p.connect(p.GUI)


        # bullet setup
        ###self.seed()
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0,0.0,-9.81)
        p.setRealTimeSimulation(0)


        # load plane
        p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        # load robot
        fullpath = os.path.join(os.path.dirname(__file__), "urdf/xarm7.urdf")
        self.xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)
        for i in range(17):
            p.resetJointState(self.xarm,i,self.joint_init_pos[i])
        
        # env setup
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        #self.goal
        #self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        # self.observation_space = spaces.Dict(dict(
        #     observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        #     achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        # ))

    #  Basic methods
    # ---------------

    def step(self,action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p.stepSimulation()
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'],self.goal)
        done = (reward == 0)

        return obs, reward, done, #info
        
    def reset():
        raise NotImplementedError

    def compute_reward(self, achieved_goal, goal):
        distance = np.linalg.norm(achieved_goal, goal, axis=-1)

        return (distance < self.distance_threshold).astype(np.float32)

    #  RobotEnv methods
    # ------------------

    def _set_action(self,action):

        assert action.shape == (3,), 'action shape error'
        current_position = np.array(p.getLinkState(self.xarm,self.arm_eef_index)[0])
        new_position = current_position + action * self.max_vel * self.dt
        new_position = np.clip(new_position, self.pos_space.low,self.pos_space.high)
        joint_poses = p.calculateInverseKinematics(self.xarm,self.arm_eef_index,new_position,[1,0,0,0])
        for i in range(1,self.arm_eef_index):
            p.setJointMotorControl2(self.xarm,i,p.POSITION_CONTROL, joint_poses[i-1], force=5*240.)
        
    def move_joint(self,joint_index):
        p.setJointMotorControl2(self.xarm,joint_index,p.POSITION_CONTROL,1)
        p.stepSimulation()


    def _get_obs(self):
        # robot state
        robot_state = p.getJointStates(self.xarm, np.arange(0,self.num_joints))
        ee_position = np.array([robot_state[self.gripper_driver_index][0]])



        pass
    def _reset_sim(self):
        raise NotImplementedError
    

    #  Goal methods
    # --------------

    def make_goal(self,point_position):
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] + 0.05], lineColorRGB=[1, 0, 0], lineWidth=5)