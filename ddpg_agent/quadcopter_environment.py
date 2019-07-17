import math
import numpy as np
from ddpg_agent.contrib.physics_sim import PhysicsSim
from collections import namedtuple
import copy

QuadcopterState = namedtuple("QuadcopterState",[
                     'x', 'y', 'z', 'phi', 'theta', 'psi',
                     'x_velocity', 'y_velocity', 'z_velocity',
                     'phi_velocity', 'theta_velocity', 'psi_velocity',
                     'x_linear_accel','y_linear_accel','z_linear_accel',
                     'phi_angular_accel','theta_angular_accel','psi_angular_accel',
                    ])

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=10., target_pos=None,
        vert_dist_thresh=1, horiz_dist_thresh=1,
        target_steps_within_goal=1,):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        # TODO: Make action_repeat align with agent.action_repeat
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12
        self.observation_space = Space(
            list(np.hstack(( self.sim.lower_bounds, [ -math.pi ]*3, [float('-inf')]*6 )))*self.action_repeat,
            list(np.hstack(( self.sim.upper_bounds, [ -math.pi ]*3, [float('inf') ]*6 )))*self.action_repeat,
        )
        # self.observation_space = Space( list(list(self.sim.lower_bounds) + \
        #                                      [ -math.pi ]*3)*self.action_repeat + [float('-inf')]*6,
        #                                list(list(self.sim.upper_bounds) + \
        #                                     [ math.pi ]*3)*self.action_repeat + [float('inf')]*6 )
        self.action_space = Space([0,0,0,0], [900,900,900,900])
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_steps_within_goal = target_steps_within_goal
        self.steps_within_goal = 0
        self.horiz_dist_thresh = horiz_dist_thresh
        self.vert_dist_thresh = vert_dist_thresh

        # History
        self.step_history = []

    def reached_goal(self):
        horiz_distance_from_goal = np.sqrt((self.sim.pose[0]-self.target_pos[0])**2
                                           +(self.sim.pose[1]-self.target_pos[1])**2)
        vert_distance_from_goal = np.abs(self.sim.pose[2]-self.target_pos[2])
        return horiz_distance_from_goal < self.horiz_dist_thresh and \
                vert_distance_from_goal <= self.vert_dist_thresh

    def get_full_state(self):
        return QuadcopterState( *self.sim.pose, *self.sim.v, *self.sim.angular_v,
                                *self.sim.linear_accel, *self.sim.angular_accels )

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 0
        # Reward for staying at target altitude
#         target_alt=self.target_pos[2]
#         reward = .1*(target_alt - np.abs(self.sim.pose[2] - target_alt))/target_alt
#         distance_from_goal = np.sqrt((self.sim.pose[0]-self.target_pos[0])**2
#                  +(self.sim.pose[1]-self.target_pos[1])**2
#                  +(self.sim.pose[2]-self.target_pos[2])**2)
#         reward += 1-distance_from_goal/10.
        #Intermediate reward for flying at altitude
#         if np.abs(self.sim.pose[2] - self.target_pos[2]) < 1:
#             reward += 1
        vert_dist = np.abs(self.target_pos[2]-self.sim.pose[2])
        if vert_dist<10:
            reward += 10-vert_dist

        # Punish for high angular velocity
        reward -= (self.sim.angular_v[0]/30.)**2 + (self.sim.angular_v[1]/30.)**2

        # Punishment for crashing (altitude < 1 m)
        if self.sim.pose[2]<=0: reward -= 100
#         if self.sim.pose[2]<2: reward -= 1
        # Reward for being within goal radius
#         horiz_distance_from_goal = np.sqrt((self.sim.pose[0]-self.target_pos[0])**2
#                                            +(self.sim.pose[1]-self.target_pos[1])**2)
        # Reward for going up, up to 10m above the goal height
        if self.sim.v[2]>0 and self.sim.pose[2]<(self.target_pos[2]+10):
            reward += 1
        # Penalty for falling
        if self.sim.v[2]<0:
            reward -= .01*(self.sim.v[2]**2)

#         if self.reached_goal():
#             self.steps_within_goal += 1
#             reward += 1
# #             if self.steps_within_goal / self.action_repeat >= self.target_steps_within_goal:
# #                 reward += 1000
#         else:
#             self.steps_within_goal = 0
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
#             pose_all.append(self.sim.pose)
            pose_all.append(np.hstack(( self.sim.pose, self.sim.v, self.sim.angular_v )))
        # next_state = list(np.concatenate(pose_all))+list(self.sim.v)+list(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        #import pdb; pdb.set_trace()
        # Punish and end episode for crashing
        if self.sim.pose[2]<=0:
#             reward -= 100
            done = True

#         # Punish for excessive rotor speeds
#         reward -= np.sum(((rotor_speeds-self.action_space.low)/(self.action_space.high-self.action_space.low))**2)

        # end episode if at goal state
#         if self.steps_within_goal / self.action_repeat >= self.target_steps_within_goal:
        # if self.sim.pose[2] > self.target_pos[2]:
        #     reward += 1000
        #     done = True
        # Scale reward.
        # TODO: How can the agent detect need for reward scaling automatically?
#         reward = np.log1p(reward) if reward>0 else np.log1p(-reward)

        return next_state, reward, done, None

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # state = list(np.concatenate([self.sim.pose] * self.action_repeat)) + \
        #         list(self.sim.v) + list(self.sim.angular_v)
        state = list(np.hstack((self.sim.pose, self.sim.v, self.sim.angular_v)))*self.action_repeat
        self.steps_within_goal = 0
        return state

class Space():
    def __init__(self, low, high):
        low = np.array(low)
        high = np.array(high)
        assert low.shape == high.shape,\
            "Expected bounds to be of same shape."
        self.low = low
        self.high = high
        self.shape = low.shape
