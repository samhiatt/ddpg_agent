# ddpg_agent
Reinforcement Learning agent using Deep Deterministic Policy Gradients (DDPG).

This reinforcement lerning model is a modified version of [Udacity's DDPG model](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) which is based on the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). This project was developed as part of the [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) quadcopter project and the model is based on code provided in the project assignment.

Solving OpenAI Gym's [MountainCarContinuous-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) continuous control problem with this model provides a particularly good learning example as its 2-dimensional continuous state space (position and velocity) and 1-dimensional continuous action space (forward, backward) are easy to visualize in two dimensions, lending to an intuitive understanding of hyperparameter tuning.

Project development began as a kaggle kernel. Initial code in this repo is based on [DDPG_OpenAI-MountainCarContinuous-V0 Version 74](https://www.kaggle.com/samhiatt/mountaincarcontinuous-v0-ddpg?scriptVersionId=16052313).

## Demo

See [Solving MountainCarContinuous-v0.ipynb](https://nbviewer.jupyter.org/github/samhiatt/ddpg_agent/blob/master/Solving%20MountainCarContinuous-v0.ipynb) for a demo training session of the Mountain Car continuous control problem (with training animation video).

Or for a live demo, launch a jupyter labs instance and run the notebook on [Binder](https://mybinder.org) by pressing here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samhiatt/ddpg_agent/master?filepath=Solving%20MountainCarContinuous-v0.ipynb)

## Usage

First make a new OpenAi Gym environment.
``` python
import gym
env = gym.make('MountainCarContinuous-v0')
```

Instantiate a new DDPG agent with the OpenAi environment.
``` python
from ddpg_agent.agent import DDPG
agent = DDPG(env, ou_mu=0, ou_theta=.05, ou_sigma=.25,
             discount_factor=.999, replay_buffer_size=10000, replay_batch_size=1024,
             tau_actor=.3, tau_critic=.1,
             relu_alpha_actor=.01, relu_alpha_critic=.01,
             lr_actor=.0001, lr_critic=.005, activation_fn_actor='tanh',
             l2_reg_actor=.01, l2_reg_critic=.01,
             bn_momentum_actor=0, bn_momentum_critic=.7,
             hidden_layer_sizes_actor=[16,32,16], hidden_layer_sizes_critic=[[16,32],[16,32]], )
agent.print_summary()
```
```
Actor model summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
states (InputLayer)          (None, 2)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                48        
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                544       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 32)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                528       
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 16)                0         
_________________________________________________________________
actions (Dense)              (None, 1)                 17        
=================================================================
Total params: 1,137
Trainable params: 1,137
Non-trainable params: 0
_________________________________________________________________
Critic model summary:
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
states (InputLayer)             (None, 2)            0                                            
__________________________________________________________________________________________________
actions (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 16)           48          states[0][0]                     
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 16)           32          actions[0][0]                    
__________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)       (None, 16)           0           dense_7[0][0]                    
__________________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)       (None, 16)           0           dense_9[0][0]                    
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 32)           544         leaky_re_lu_7[0][0]              
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 32)           544         leaky_re_lu_9[0][0]              
__________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)       (None, 32)           0           dense_8[0][0]                    
__________________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)      (None, 32)           0           dense_10[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32)           0           leaky_re_lu_8[0][0]              
                                                                 leaky_re_lu_10[0][0]             
__________________________________________________________________________________________________
leaky_re_lu_11 (LeakyReLU)      (None, 32)           0           add_1[0][0]                      
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32)           128         leaky_re_lu_11[0][0]             
__________________________________________________________________________________________________
q_values (Dense)                (None, 1)            33          batch_normalization_1[0][0]      
==================================================================================================
Total params: 1,329
Trainable params: 1,265
Non-trainable params: 64
__________________________________________________________________________________________________
Hyperparameters:
{'train_during_episode': True, 'discount_factor': 0.999, 'tau_actor': 0.3, 'tau_critic': 0.1, 'lr_actor': 0.0001, 'lr_critic': 0.005, 'bn_momentum_actor': 0, 'bn_momentum_critic': 0.7, 'ou_mu': 0, 'ou_theta': 0.05, 'ou_sigma': 1, 'activation_fn_actor': 'tanh', 'replay_buffer_size': 10000, 'replay_batch_size': 1024, 'l2_reg_actor': 0.01, 'l2_reg_critic': 0.01, 'relu_alpha_actor': 0.01, 'relu_alpha_critic': 0.01, 'dropout_actor': 0, 'dropout_critic': 0, 'hidden_layer_sizes_actor': [16, 32, 16], 'hidden_layer_sizes_critic': [[16, 32], [16, 32]]}
```
Train it, generating animation frames for Q and action visualizations every nth step, with epsilon starting at `eps` and decaying by `eps -= eps_decay` after every episode.
``` python
agent.train_n_episodes(50, eps=1, eps_decay=1/50, action_repeat=5,
                       run_tests=True, gen_q_a_frames_every_n_steps=10, )
```
```
Episode 1 - epsilon: 0.98, memory size: 57, training score: 83.69, test score: -0.01
Episode 2 - epsilon: 0.96, memory size: 171, training score: 80.90, test score: -0.00
...
```

Visualize it, every nth step per frame.
``` python
from ddpg_agent.visualizations import create_animation

create_animation(agent, display_mode='video_file', every_n_steps=10, fps=15)
```
```
Using ffmpeg at '/Users/sam/mlenv/lib/python3.6/site-packages/imageio_ffmpeg/binaries/ffmpeg-osx64-v4.1'.
Video saved to training_animation_1561324217.mp4.
```

## Credits
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* Andre Muta's [DDPG-MountainCarContinuous-v0](https://github.com/amuta/DDPG-MountainCarContinuous-v0) repo was helpful in suggesting some good visualizations as well as giving some good hyperparameters to start with. It looks like he uses the same code from the nanodegree quadcopter project and uses it to solve the MountainCarContinuous problem as well. His [plot_Q method in MountainCar.py](https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/MountainCar.py) was particularly helpful by showing how to plot Q_max, Q_std, Action at Q_max, and Policy. Adding a visualization of the policy gradients and animating the training process ended up helping me better understand the problem and the effects of various hypterparemeters.
* Thanks to [Eli Bendersky](https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/) for help with matplotlib animations.
* Thanks to [Joseph Long](https://joseph-long.com/writing/colorbars/) for help with matplotlib colorbar axes placement.
