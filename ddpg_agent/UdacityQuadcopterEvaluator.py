import warnings; warnings.simplefilter('ignore')

from ddpg_agent.udacity_models import DDPG
from ddpg_agent.contrib.physics_sim import PhysicsSim
# from ddpg_agent.quadcopter_environment import Task
import numpy as np
from collections import namedtuple
from hyperopt import STATUS_OK
import json
import sys
import warnings; warnings.simplefilter('ignore')

class Task():
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        reward = 0
        #Reward for horizontal distance to goal
        horiz_dist, vert_dist = self.get_horiz_vert_distance_from_goal()
        if vert_dist<10 and horiz_dist<10:
            reward += 10-vert_dist
            reward += .1*(10-horiz_dist)
        return reward

    def get_horiz_vert_distance_from_goal(self):
        horiz_dist = np.sqrt((self.sim.pose[0]-self.target_pos[0])**2 +(self.sim.pose[1]-self.target_pos[1])**2)
        vert_dist = np.abs(self.target_pos[2]-self.sim.pose[2])
        return horiz_dist, vert_dist

    def step(self, rotor_speeds):
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

task = Task(init_pose=np.array([0., 0., 10, 0., 0., 0.]),
            #init_pose=np.array([0., 0., 10, math.pi/2., math.pi/2., 0.]),
            init_velocities=np.array([0., 0., 0.]),
            init_angle_velocities=np.array([0., 0., 0.]),
            runtime=10.,
            # vert_dist_thresh=1, horiz_dist_thresh=1,
            # target_steps_within_goal=25,
            target_pos=np.array([0., 0., 20.]),
           )
UdacityDDPGLearningParams = namedtuple('LearningParams',[
        'n_training_courses','n_episodes',
        'exploration_mu','exploration_theta','exploration_sigma',
        'buffer_size','batch_size',
        'gamma','tau',
    ])

def evaluator(params):
    params = UdacityDDPGLearningParams(*params)
    print(params)
    losses = []
    learning_curves = []
    for i_training_course in range(params.n_training_courses):
        agent = DDPG(task, dict(gamma=params.gamma, tau=params.tau,
                     exploration_mu=params.exploration_mu,
                     exploration_theta=params.exploration_theta,
                     exploration_sigma=params.exploration_sigma,
                     buffer_size=params.buffer_size,
                     batch_size=params.batch_size,
                    ))
        # agent.print_summary()
        scores = []
        for i_episode in range(1, params.n_episodes+1):
            state = agent.reset_episode() # start a new episode
            total_reward=0
            while True:
                action = agent.act(state)
                next_state, reward, done = task.step(action)
                total_reward += reward
                agent.step(action, reward, next_state, done)
                state = next_state

                if done:
                    break

            print("\rEpisode {:4d}:{:4d}, total_reward = {}".format(
                i_training_course, i_episode, total_reward), end="")
            scores.append(total_reward)
            sys.stdout.flush()

        # Return a negative score (since hyperopt will minimize this function)
        loss = -np.mean(scores[-10:])
        losses.append(loss)
        print("\rCourse %i learning curves:\n"%i_training_course,learning_curves)
        print("\rCourse %i: loss=%7.4f"%(i_training_course, loss))
        learning_curves.append(scores)
        #return -np.mean(last_n_scores)/np.std(last_n_scores)

    return {'status':STATUS_OK,
        'loss': np.mean(losses),
        'loss_variance': np.var(losses),
        'max_score': np.max(learning_curves),
        # 'scores': scores,
        'params': dict(params),
        # 'attachments':{
        #     },
        'learning_curves':learning_curves,
        }
