import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import copy
import random
import sys
from collections import namedtuple, deque

from models import Actor
from models import Critic

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, train_during_episode=True,
                 discount_factor=.999,
                 tau_actor=.2, tau_critic=.2,
                 lr_actor=.0001, lr_critic=.005,
                 bn_momentum_actor=.9, bn_momentum_critic=.9,
                 ou_mu=0, ou_theta=.1, ou_sigma=1,
                 activation_fn_actor='sigmoid',
                 replay_buffer_size=10000, replay_batch_size=64,
                 l2_reg_actor=.01, l2_reg_critic=.01,
                 relu_alpha_actor=.01, relu_alpha_critic=.01,
                 dropout_actor=0, dropout_critic=0,
                 hidden_layer_sizes_actor=[32,64,32],
                 hidden_layer_sizes_critic=[[32,64],[32,64]], ):

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.train_during_episode = train_during_episode

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low,
                self.action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor, )
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low,
                self.action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor, )

        # Critic (Q-Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, bn_momentum=bn_momentum_critic, relu_alpha=relu_alpha_critic,
                hidden_layer_sizes=hidden_layer_sizes_critic, dropout=dropout_critic, )
        self.critic_target = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, bn_momentum=bn_momentum_critic, relu_alpha=relu_alpha_critic,
                hidden_layer_sizes=hidden_layer_sizes_critic, dropout=dropout_critic, )

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = ou_mu
        self.exploration_theta = ou_theta
        self.exploration_sigma = ou_sigma
        self.noise = OUNoise(self.action_size,
                             self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = replay_buffer_size
        self.batch_size = replay_batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = discount_factor  # discount factor

        self.tau_actor = tau_actor
        self.tau_critic = tau_critic

        # Training history
        self.training_scores = []
        self.test_scores = []
        self.training_history = TrainingHistory(env)
        self.q_a_frames_spec = Q_a_frames_spec(env)

        # Track training steps and episodes
        self.steps = 0
        self.episodes = 0

        self.reset_episode()

    def print_summary(self):
        print("Actor model summary:")
        self.actor_local.model.summary()
        print("Critic model summary:")
        self.critic_local.model.summary()

    def preprocess_state(self, state):
        obs_space = self.env.observation_space
        return np.array([
            (state[i]-obs_space.low[i])/(obs_space.high[i]-obs_space.low[i])*2 - 1
            for i in range(len(obs_space.low))])

    def reset_episode(self):
        self.noise.reset()
        state = self.preprocess_state(self.env.reset())
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        next_state = self.preprocess_state(next_state)
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and (self.train_during_episode or done):
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        self.steps += 1

    def act(self, state=None, eps=0, verbose=False):
        """Returns actions for given state(s) as per current policy."""
        if state is None:
            state = self.last_state
        else:
            state = self.preprocess_state(state)
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        noise_sample = self.noise.sample() * max(0,eps)
        res = list(np.clip(action + noise_sample, self.action_low, self.action_high))
        if verbose:
            print("State: (%6.3f, %6.3f), Eps: %6.3f, Action: %6.3f + %6.3f = %6.3f"%
                  (state[0][0], state[0][1], eps, action, noise_sample, res[0]))
        return res  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]
                        ).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(
            self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model, self.tau_critic)
        self.soft_update(self.actor_local.model, self.actor_target.model, self.tau_actor)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), \
            "Local and target model parameters must have the same size"

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)

    def train_n_episodes(self, n_episodes, eps=1, eps_decay=None, action_repeat=1,
                         run_tests=True, gen_q_a_frames_every_n_steps=0, draw_plots=False ):
        if eps_decay is None: eps_decay = 1/n_episodes
        n_training_episodes = len(self.training_scores)
        for i_episode in range(n_training_episodes+1, n_training_episodes+n_episodes+1):
            eps -= eps_decay
            eps = max(eps,0)
            episode_start_step = self.steps
            self.run_episode(train=True, action_repeat=action_repeat, eps=eps,
                             gen_q_a_frames_every_n_steps=gen_q_a_frames_every_n_steps )
            if run_tests is True:
                self.run_episode(train=False, eps=0, action_repeat=action_repeat)
            message = "Episode %i - epsilon: %.2f, memory size: %i, training score: %.2f"\
                        %(self.episodes, eps, len(self.memory), self.training_history.training_episodes[-1].score)
            if run_tests: message += ", test score: %.2f"%self.training_history.test_episodes[-1].score
            print(message)
            sys.stdout.flush()

    def run_episode(self, action_repeat=1, eps=0, train=False, gen_q_a_frames_every_n_steps=0 ):
        next_state = self.reset_episode()
        if train: episode_history = self.training_history.new_training_episode(self.episodes+1,eps)
        else: episode_history = self.training_history.new_test_episode(self.episodes,eps)
        q_a_frame = None
        while True:
            action = self.act(next_state, eps=eps)
            sum_rewards=0
            # Repeat action `action_repeat` times, summing up rewards
            for i in range(action_repeat):
                next_state, reward, done, info = self.env.step(action)
                sum_rewards += reward
                if done:
                    break
            #sum_rewards = np.log1p(sum_rewards)
            episode_history.append(self.steps, next_state, action, sum_rewards)
            if train:
                self.step(action, sum_rewards, next_state, done)
                if gen_q_a_frames_every_n_steps > 0 and self.steps%gen_q_a_frames_every_n_steps==0:
                    self.training_history.add_q_a_frame(self.get_q_a_frames())
            if done:
                if train:
                    self.episodes += 1
                break

    def get_q_a_frames(self):
        """ TODO: Figure out how to work with added dimensions.
                - Use x_dim, y_dim, and a_dim to know which dimensions of state and action to vary.
                    Maybe fill in the unvaried dimensions of states and actions with agent's current state
                    and anticipated action (according to policy).
        """
        xs = self.q_a_frames_spec.xs
        nx = self.q_a_frames_spec.nx
        ys = self.q_a_frames_spec.ys
        ny = self.q_a_frames_spec.ny
        action_space = self.q_a_frames_spec.action_space
        na = self.q_a_frames_spec.na
        x_dim = self.q_a_frames_spec.x_dim
        y_dim = self.q_a_frames_spec.y_dim
        a_dim = self.q_a_frames_spec.a_dim

        def get_state(x,y):
            s=copy.copy(self.last_state)
            s[x_dim]=x
            s[y_dim]=y
            return s
        raw_states = np.array([[ get_state(x,y) for x in xs ] for y in ys ]).reshape(nx*ny, self.state_size)

        def get_action(action):
            a=self.act() if self.action_size>1 else [0]
            a[a_dim]=action
            return a
        actions = np.array([get_action(a) for a in action_space]*nx*ny)

        preprocessed_states = np.array([ self.preprocess_state(s) for s in raw_states])
        Q = self.critic_local.model.predict_on_batch(
            [np.repeat(preprocessed_states,na,axis=0),actions]).reshape((ny,nx,na))
        Q_max = np.max(Q,axis=2)
        Q_std = np.std(Q,axis=2)
        max_action = np.array([action_space[a] for a in np.argmax(Q,axis=2).flatten()]).reshape((ny,nx))
        actor_policy = np.array([ self.act(s)[0] for s in raw_states]).reshape(ny,nx)
        action_gradients = self.critic_local.get_action_gradients(
            [preprocessed_states,actor_policy.reshape(nx*ny,-1),0])[0].reshape(ny,nx)

        return namedtuple( 'q_a_frames',[
                'step_idx', 'episode_idx', 'Q_max', 'Q_std', 'max_action', 'action_gradients', 'actor_policy'
            ])(self.steps, self.episodes, Q_max, Q_std, max_action, action_gradients, actor_policy)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=min(self.batch_size, len(self)))

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck noise process."""
    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, sigma=None):
        """Update internal state and return it as a noise sample."""
        if sigma is None:
            sigma = self.sigma
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Q_a_frames_spec():
    """
    Tracks training history, including a snapshot of rasterized Q values and actions
    for use in visualizations.
    """
    def __init__(self, env, nx=16, ny=16, na=11, x_dim=0, y_dim=1, a_dim=0):
        """
        Initialize Q_a_frame_set object with Q grid shape.
        Params
        ======
             env (obj): OpenAi Gym environment
             nx (int): Width of Q grid (default: 16)
             ny (int): Height of Q grid (default: 16)
             na (int): Depth of Q grid (default: 11)
             x_dim (int): Observation dimension to use as x-axis (default: 0)
             y_dim (int): Observation dimension to use as y-axis (default: 1)
             a_dim (int): Action dimension to use as x-axis (default: 0)
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.a_dim = a_dim

        self.xmin = env.observation_space.low[x_dim]
        self.xmax = env.observation_space.high[x_dim]
        self.xs = np.arange(self.xmin, self.xmax, (self.xmax-self.xmin)/nx)[:nx]
        self.nx = len(self.xs)

        self.ymin = env.observation_space.low[y_dim]
        self.ymax = env.observation_space.high[y_dim]
        self.ys = np.arange(self.ymin, self.ymax, (self.ymax-self.ymin)/ny)[:ny]
        self.ny = len(self.ys)

        self.amin = env.action_space.low[a_dim]
        self.amax = env.action_space.high[a_dim]
        self.action_space = np.linspace(self.amin,self.amax,na)
        self.na = len(self.action_space)

class TrainingHistory:
    """
    Tracks training history, including a snapshot of rasterized Q values and actions
    for use in visualizations.
    """
    def __init__(self, env, nx=16, ny=16, na=11, x_dim=0, y_dim=1, a_dim=0):
        """
        Initialize TrainingHistory object with Q grid shape.
        Params
        ======
        """
        self.training_episodes = []
        self.test_episodes = []
        self.q_a_frames = []
        self.last_step = 0
        self.q_a_frames_spec = Q_a_frames_spec(env, nx=nx, ny=ny, na=na,
                                         x_dim=x_dim, y_dim=y_dim, a_dim=a_dim)

    def __repr__(self):
        return "TrainingHistory ( %i training_episodes, %i test_episodes, %i qa_grids, last_step: %i )"%\
                (len(self.training_episodes), len(self.test_episodes), len(self.q_a_frames), self.last_step)

    def add_q_a_frame(self, q_a_frame):
        self.q_a_frames.append(q_a_frame)

    def new_training_episode(self, idx, epsilon=None):
        episode = EpisodeHistory(idx, epsilon)
        self.training_episodes.append(episode)
        return episode

    def new_test_episode(self, idx, epsilon=None):
        episode = EpisodeHistory(idx, epsilon)
        self.test_episodes.append(episode)
        return episode

    def get_training_episode_for_step(self, step_idx):
        for ep in self.training_episodes:
            if ep.last_step>=step_idx:
                return ep
    def get_test_episode_for_step(self, step_idx):
        for ep in self.test_episodes:
            if (ep.last_step+1)>=step_idx:
                return ep
    def get_q_a_frames_for_step(self, step_idx):
        for g in self.q_a_frames:
            if g.step_idx>=step_idx:
                return g
        return g
    def append_training_step(self, step, state, action, reward):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            step (int): Step index
            state (list|array): State, array-like of shape env.observation_space.shape
            action (list|array): Action, array-like of shape env.action_space.shape
            reward (float): Reward, scalar value
        """
        if len(self.training_episodes)==0:
            raise "No training episodes exist yet. "
        self.training_episodes[-1].append(step, state, action, reward)
        self.last_step = step
        return self
    def append_test_step(self, step, state, action, reward):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            step (int): Step index
            state (list|array): State, array-like of shape env.observation_space.shape
            action (list|array): Action, array-like of shape env.action_space.shape
            reward (float): Reward, scalar value
        """
        if len(self.test_episodes)==0:
            raise "No test episodes exist yet. "
        self.test_episodes[-1].append(step, state, action, reward)
        self.last_step = step
        return self

class EpisodeHistory:
    """ Tracks the history for a single episode, including the states, actions, and rewards.
    """
    def __init__(self, episode_idx=None, epsilon=None):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            epsilon (float): Exploration factor
        """
        self.episode_idx = episode_idx
        self.epsilon = epsilon
        self.steps = []
        self.first_step = None
        self.last_step = None
        self.states = []
        self.actions = []
        self.rewards = []

        self.score = self._get_score()

    def _get_score(self):
        return sum(self.rewards)

    def append(self, step, state, action, reward):
        self.steps.append(step)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.first_step is None: self.first_step = step
        self.last_step = step
        self.score += reward

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return "EpisodeHistory ( idx: %i, len: %i, first_step: %i, last_step: %i, epsilon: %.3f, score: %.3f )"%\
            (self.episode_idx,len(self.steps),self.first_step, self.last_step, self.epsilon, self.score)
