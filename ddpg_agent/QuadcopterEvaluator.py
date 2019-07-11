import warnings; warnings.simplefilter('ignore')

from ddpg_agent.agent import DDPG#, Q_a_frames_spec
from ddpg_agent.visualizations import plot_quadcopter_episode
from ddpg_agent.quadcopter_environment import Task
import numpy as np
from collections import namedtuple

task = Task(init_pose=np.array([0., 0., 10, 0., 0., 0.]),
            #init_pose=np.array([0., 0., 10, math.pi/2., math.pi/2., 0.]),
            init_velocities=np.array([0., 0., 0.]),
            init_angle_velocities=np.array([0., 0., 0.]),
            runtime=10.,
            vert_dist_thresh=1, horiz_dist_thresh=1,
            target_steps_within_goal=25,
            target_pos=np.array([0., 0., 20.]),
           )

def noise_evaluator(params):
    """ Evaluator to test different noise parameters looking to maximize average training_score.
        No model learning is done during this step.
    """
    params = namedtuple('NoiseParams',['ou_mu','ou_theta','ou_sigma','n_episodes','eps'])(*params)
    print(params)

    agent = DDPG(task,
                 ou_mu=params.ou_mu, ou_theta=params.ou_theta, ou_sigma=params.ou_sigma,
                 replay_buffer_size=0,
                 replay_batch_size=params.n_episodes, # suppress model training
                )
    # agent.print_summary()
    agent.train_n_episodes(params.n_episodes, eps_decay=0, run_tests=False,
                           # Notice we're acting randomly for all n_episodes
                           act_random_first_n_episodes=params.n_episodes,
                           primary_exploration_eps=params.eps, )

    # Return a negative score (since hyperopt will minimize this function)
    return -np.mean([ep.score for ep in agent.history.training_episodes])

def evaluator(params):
    """ Evaluator to test learning parameters, using the noise parameters learned using
        the evaluator above.
    """
    params = namedtuple('LearningParams',[
            'ou_mu','ou_theta','ou_sigma','n_episodes','eps',
            'eps_decay','primary_exploration_eps','act_random_first_n_episodes',
            'discount_factor','replay_buffer_size','replay_batch_size',
            'tau_actor','tau_critic','lr_actor','lr_critic',
        ])(*params)
    print(params)
    agent = DDPG(task, discount_factor=params.discount_factor,
                 ou_mu=params.ou_mu, ou_theta=params.ou_theta, ou_sigma=params.ou_sigma,
                 replay_buffer_size=params.replay_buffer_size,
                 replay_batch_size=params.replay_batch_size,
                 tau_actor=params.tau_actor, tau_critic=params.tau_critic,
                 lr_actor=params.lr_actor, lr_critic=params.lr_critic, #activation_fn_actor='tanh',
    #              relu_alpha_actor=.01, relu_alpha_critic=.01,
    #              l2_reg_actor=.01, l2_reg_critic=.01,
    #              bn_momentum_actor=0, bn_momentum_critic=.7,
    #              q_a_frames_spec=q_a_frames_spec,
    #              do_preprocessing=False,
    #              input_bn_momentum_actor=.7,
    #              input_bn_momentum_critic=.7,
    #              activity_l2_reg=50,
    #              output_action_regularizer=10,
                )
    # agent.print_summary()

    agent.train_n_episodes(params.n_episodes, eps=params.eps, eps_decay=params.eps_decay,
                           act_random_first_n_episodes=params.act_random_first_n_episodes,
                           primary_exploration_eps=params.primary_exploration_eps,
                           )

    # Return a negative score (since hyperopt will minimize this function)
    # Mean test score/variance, last 100 episodes
    last_n_scores = [e.score for e in agent.history.test_episodes[-100:]]
    return -np.mean(last_n_scores)/np.std(last_n_scores)




# labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
#           'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
#           'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4
# [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)

# def episode_callback(episode_num):
#     last_training_episode = agent.history.training_episodes[-1]
#     last_test_episode = agent.history.test_episodes[-1]
#     num_steps = last_training_episode.last_step - last_training_episode.first_step+1
#     message = "Episode %i - epsilon: %7.4g, memory: %i, step: %i, training score: %6.2f, test score: %6.2f"\
#                 %(episode_num,
#                   last_training_episode.epsilon,
#                   len(agent.memory),
#                   num_steps,
#                   last_training_episode.score,
#                   last_test_episode.score
#                  )
#     print(message)
#
#     if episode_num%10==0:
#         fig = plot_quadcopter_episode(last_training_episode)
#         display(fig)

# agent.set_episode_callback(episode_callback)

# def max_training_score_callback(episode):
#     last_training_episode = agent.history.training_episodes[-1]
#     print("New best training score.")
#     fig = plot_quadcopter_episode(last_training_episode)
#     display(fig)
#
# # agent.set_max_training_score_callback(max_training_score_callback)
#
# def max_test_score_callback(episode):
#     last_test_episode = agent.history.test_episodes[-1]
#     print("New best test score.")
#     fig = plot_quadcopter_episode(last_test_episode)
#     display(fig)

# agent.set_max_test_score_callback(max_test_score_callback)
