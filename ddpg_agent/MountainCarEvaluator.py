import ddpg_agent, os
from ddpg_agent.agent import DDPG
import gym
import numpy as np

def evaluate_hyperparams(params):

    agent = DDPG(env, train_during_episode=True, ou_mu=0, ou_theta=.05, ou_sigma=.25, 
                 discount_factor=.999, replay_buffer_size=10000, replay_batch_size=1024,
                 tau_actor=.3, tau_critic=.1, 
                 relu_alpha_actor=.01, relu_alpha_critic=.01,
                 lr_actor=.0001, lr_critic=.005, activation_fn_actor='tanh',
                 l2_reg_actor=.01, l2_reg_critic=.01, 
                 bn_momentum_actor=0, bn_momentum_critic=.7,
                 hidden_layer_sizes_actor=[16,32,16], hidden_layer_sizes_critic=[[16,32],[16,32]], )

    agent.train_n_episodes(30, eps=1, eps_decay=1/50, action_repeat=5, run_tests=True, )
    
    # Return a negative score (since hyperopt will minimize this function)
    return -agent.history.max_test_score