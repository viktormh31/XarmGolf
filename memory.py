import numpy as np


class HerBuffer():
    def __init__(self,batch_size,batch_ratio,max_episode_timesteps, obs_dims, n_actions,goal_dim,max_buffer_size = 1e6) -> None:
        self.batch_size = batch_size
        self.batch_ratio = batch_ratio
        self.her_batch_size = int(self.batch_ratio*self.batch_size)
        self.real_batch_size = int((1-self.batch_ratio)*self.batch_size)
        self.max_episode_timesteps = max_episode_timesteps
        self.max_buffer_size = max_buffer_size
        self.goal_dim = goal_dim # default 3

        self.real_buffer = BaseBuffer(self.max_buffer_size,self.real_batch_size, obs_dims,n_actions,goal_dim)
        self.her_buffer = BaseBuffer(self.max_buffer_size,self.her_batch_size, obs_dims,n_actions,goal_dim)
        self.episode_buffer = BaseBuffer(self.max_episode_timesteps,self.max_episode_timesteps,obs_dims,n_actions,goal_dim)

    def sample(self):
        real_batch = self.real_buffer.sample()
        her_batch = self.her_buffer.sample()

        combined_batch = {key: np.concatenate([real_batch[key], her_batch[key]],axis=0) for key in real_batch}
        
        return combined_batch

class BaseBuffer():
    def __init__(self,max_size,batch_size, input_dims, n_actions,goal_dim) -> None:

        self.max_size = max_size
        self.batch_size = batch_size
        self.counter = 0
        self.rng = np.random.default_rng()

        self.observations = np.zeros((max_size,input_dims), dtype= np.float32)
        self.achieved_goals = np.zeros((max_size,goal_dim), dtype= np.float32)
        self.desired_goals = np.zeros((max_size,goal_dim), dtype= np.float32)
        self.actions = np.zeros((max_size,n_actions), dtype= np.float32)
        self.rewards = np.zeros((max_size), dtype= np.float32)
        #self.rewards -= 1
        self.dones = np.zeros((max_size),dtype=np.int32)
        self.next_observations = np.zeros((max_size,input_dims), dtype= np.float32)
        self.next_achieved_goals = np.zeros((max_size,goal_dim), dtype= np.float32)
        self.next_desired_goals = np.zeros((max_size,goal_dim), dtype= np.float32)
    
    def append(self,obs,action,reward,done,next_obs,size_of_append):
        
        observation = obs["observation"]
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]

        next_observation = next_obs["observation"]
        next_achieved_goal = next_obs["achieved_goal"]
        next_desired_goal = next_obs["desired_goal"]

        index = self.counter % self.max_size

        if size_of_append == 1:
            self.observations[index] = observation
            self.achieved_goals[index] = achieved_goal
            self.desired_goals[index] = desired_goal
            self.actions[index] = action
            self.rewards[index] = reward
            self.dones[index] = done
            self.next_observations[index] = next_observation
            self.next_achieved_goals[index] = next_achieved_goal
            self.next_desired_goals[index] = next_desired_goal
            
        else:
            indices = np.arange(index, index + size_of_append) % self.max_size
            print(size_of_append)
            self.observations[indices] = observation[0:size_of_append]
            self.achieved_goals[indices] = achieved_goal[0:size_of_append]
            self.desired_goals[indices] = desired_goal[0:size_of_append]
            self.actions[indices] = action[0:size_of_append]
            self.rewards[indices] = reward[0:size_of_append]
            self.dones[indices] = done[0:size_of_append]
            self.next_observations[indices] = next_observation[0:size_of_append]
            self.next_achieved_goals[indices] = next_achieved_goal[0:size_of_append]
            self.next_desired_goals[indices] = next_desired_goal[0:size_of_append]

        self.counter+=size_of_append

    def sample(self):

        #if self.counter < self.batch_size: # episode buffer ne koristi ovu funkciju, ali bi inace uvek radio prazan return
        #    return
        max_index = min(self.counter, self.max_size)
        indices = np.random.choice(max_index, self.batch_size, replace=False)
      
        return {
            'observations': self.observations[indices],
            'achieved_goals': self.achieved_goals[indices],
            'desired_goals': self.desired_goals[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'next_observations': self.next_observations[indices],
            'next_achieved_goals': self.next_achieved_goals[indices],
            'next_desired_goals': self.next_desired_goals[indices],
        }
        
    def reset_buffer(self):
        self.counter = 0
        self.observations.fill(0)
        self.achieved_goals.fill(0)
        self.desired_goals.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.next_observations.fill(0)
        self.next_achieved_goals.fill(0)
        self.next_desired_goals.fill(0)












