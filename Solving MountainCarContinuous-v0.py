#!/usr/bin/env python
# coding: utf-8

# In[1]:

import ddpg_agent, os
from ddpg_agent.agent import DDPG
from ddpg_agent.visualizations import create_animation

print("Using ddpg_agent:%s at %s."%(
    ddpg_agent.__version__,
    os.path.dirname(ddpg_agent.__file__)))


# In[2]:


import gym
import numpy as np
import warnings

warnings.simplefilter('ignore')

env = gym.make('MountainCarContinuous-v0')
print('Continuous action space: (%.3f to %.3f)'%(env.action_space.low, env.action_space.high))
print('Reward range: %s'%(str(env.reward_range)))
for i in range(len(env.observation_space.low)):
    print('Observation range, dimension %i: (%.3f to %.3f)'%
          (i,env.observation_space.low[i], env.observation_space.high[i]))


# In[3]:


from ddpg_agent.agent import DDPG
from ddpg_agent.visualizations import create_animation


# In[4]:


agent = DDPG(env, train_during_episode=True, ou_mu=0, ou_theta=.05, ou_sigma=.25,
             discount_factor=.999, replay_buffer_size=10000, replay_batch_size=1024,
             tau_actor=.3, tau_critic=.1,
             relu_alpha_actor=.01, relu_alpha_critic=.01,
             lr_actor=.0001, lr_critic=.005, activation_fn_actor='tanh',
             l2_reg_actor=.01, l2_reg_critic=.01,
             bn_momentum_actor=0, bn_momentum_critic=.7,
             hidden_layer_sizes_actor=[16,32,16], hidden_layer_sizes_critic=[[16,32],[16,32]], )
agent.print_summary()


# In[5]:


agent.train_n_episodes(50, eps=1, eps_decay=1/50, action_repeat=5,
                       run_tests=True, gen_q_a_frames_every_n_steps=10, )


# In[7]:


create_animation(agent, display_mode='video_file', every_n_steps=10, fps=15)


# In[ ]:
