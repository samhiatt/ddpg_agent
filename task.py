import numpy as np
from ddpg_agent.contrib.physics_sim import PhysicsSim
from ddpg_agent.quadcopter_environment import QuadcopterState

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

#         self.state_size = 18
#         self.observation_space = Space(
#             np.hstack(( self.sim.lower_bounds, [-np.pi]*3, [float('-inf')]*6, [float('-inf')]*6)),
#             np.hstack(( self.sim.upper_bounds, [np.pi]*3, [float('inf')]*6, [float('inf')]*6)) )
        self.state_size = self.action_repeat*6
        self.observation_space = Space(
            list( list(self.sim.lower_bounds) + [-np.pi]*3 )*self.action_repeat,
            list( list(self.sim.upper_bounds) + [np.pi]*3 )*self.action_repeat )

        self.action_space = Space([0,0,0,0], [900,900,900,900])
        self.action_size = 4

    def get_reward(self):
        reward = 0
        #Reward for horizontal distance to goal
        horiz_dist, vert_dist = self.get_horiz_vert_distance_from_goal()
        if vert_dist<10 and horiz_dist<10:
            reward += 10-vert_dist
            reward += .1*(10-horiz_dist)
        return reward

    def get_full_state(self):
        return QuadcopterState( *self.sim.pose, *self.sim.v, *self.sim.angular_v,
                                *self.sim.linear_accel, *self.sim.angular_accels )

    def get_horiz_vert_distance_from_goal(self):
        horiz_dist = np.sqrt((self.sim.pose[0]-self.target_pos[0])**2 +(self.sim.pose[1]-self.target_pos[1])**2)
        vert_dist = np.abs(self.target_pos[2]-self.sim.pose[2])
        return horiz_dist, vert_dist

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, None

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
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
