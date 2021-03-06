import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import copy
import random
import sys
import json
from collections import namedtuple, deque, defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ddpg_agent.models import Actor
from ddpg_agent.models import Critic

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, train_during_episode=True,
                 discount_factor=.999,
                 tau_actor=.5, tau_critic=.5,
                 lr_actor=.001, lr_critic=.001,
                 bn_momentum_actor=0, bn_momentum_critic=0,
                 ou_mu=0, ou_theta=.1, ou_sigma=1,
                 activation_fn_actor='sigmoid',
                 replay_buffer_size=10000, replay_batch_size=64,
                 l2_reg_actor=0, l2_reg_critic=0,
                 relu_alpha_actor=0, relu_alpha_critic=0,
                 dropout_actor=0, dropout_critic=0,
                 hidden_layer_sizes_actor=[32,64,32],
                 hidden_layer_sizes_critic=[[32,64],[32,64]],
                 q_a_frames_spec=None,
                 do_preprocessing=True,
                 input_bn_momentum_actor=0,
                 input_bn_momentum_critic=0,
                 activity_l1_reg=0,
                 activity_l2_reg=0,
                 output_action_regularizer=0,
                 output_action_variance_regularizer=0,
                 normalize_rewards=True,
                ):

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.observation_scalers = defaultdict(lambda: StandardScaler().fit([[0]]))
        for i in range(env.observation_space.shape[0]):
            if env.observation_space.low[i] != float('-inf') and env.observation_space.high[i] != float('inf'):
                self.observation_scalers[i] = MinMaxScaler((-1,1)).fit(
                    [[env.observation_space.low[i]], [env.observation_space.high[i]]])

#         self.action_scalers = defaultdict(lambda: MinMaxScaler((-1,1)).fit([[],[]]))
#         for i in range(env.action_space.shape[0]):
#             if env.action_space.low[i] != float('-inf') and env.action_space.high[i] != float('inf'):
#                 self.action_scalers = MinMaxScaler((-1,1)).fit(
#                     [[env.action_space.low[i]], [env.action_space.high[i]]])
        self.action_scaler = MinMaxScaler((-1,1)).fit([env.action_space.low, env.action_space.high])

        self.train_during_episode = train_during_episode

        action_low, action_high = (-1,1) if do_preprocessing else \
                                  (self.env.action_space.low, self.env.action_space.high)

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, action_low,
                action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor,
                input_bn_momentum=input_bn_momentum_actor,
                activity_l1_reg=activity_l1_reg, activity_l2_reg=activity_l2_reg,
                output_action_regularizer=output_action_regularizer,
                output_action_variance_regularizer=output_action_variance_regularizer)
        self.actor_target = Actor(self.state_size, self.action_size, action_low,
                action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor,
                input_bn_momentum=input_bn_momentum_actor,
                activity_l1_reg=activity_l1_reg, activity_l2_reg=activity_l2_reg,
                output_action_regularizer=output_action_regularizer,
                output_action_variance_regularizer=output_action_variance_regularizer)

        # Critic (Q-Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, relu_alpha=relu_alpha_critic,
                input_bn_momentum=input_bn_momentum_critic, bn_momentum=bn_momentum_critic,
                hidden_layer_sizes=hidden_layer_sizes_critic, dropout=dropout_critic, )
        self.critic_target = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, relu_alpha=relu_alpha_critic,
                input_bn_momentum=input_bn_momentum_critic, bn_momentum=bn_momentum_critic,
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
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.dropout_actor = dropout_actor
        self.dropout_critic = dropout_critic
        self.input_bn_momentum_actor = input_bn_momentum_actor
        self.input_bn_momentum_critic = input_bn_momentum_critic
        self.bn_momentum_actor = bn_momentum_actor
        self.bn_momentum_critic = bn_momentum_critic
        self.activation_fn_actor = activation_fn_actor
        self.ou_mu=ou_mu
        self.ou_theta=ou_theta
        self.ou_sigma=ou_sigma
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.l2_reg_actor = l2_reg_actor
        self.l2_reg_critic = l2_reg_critic
        self.relu_alpha_actor = relu_alpha_actor
        self.relu_alpha_critic = relu_alpha_critic
        self.hidden_layer_sizes_actor = hidden_layer_sizes_actor
        self.hidden_layer_sizes_critic = hidden_layer_sizes_critic
        self.activity_l1_reg = activity_l1_reg
        self.activity_l2_reg = activity_l2_reg
        self.output_action_regularizer = output_action_regularizer
        self.output_action_variance_regularizer = output_action_variance_regularizer
        self.normalize_rewards = normalize_rewards

        self.tau_actor = tau_actor
        self.tau_critic = tau_critic

        self.step_callback = None

        # Training history
        self.history = TrainingHistory(env)
        self.q_a_frames_spec = Q_a_frames_spec(env) if q_a_frames_spec is None else q_a_frames_spec

        # Track training steps and episodes
        self.steps = 0
        self.episodes = 0

        self.do_preprocessing = do_preprocessing

        self.reset_episode()

        self.max_training_score_callback = lambda x: None
        self.max_test_score_callback = lambda x: None
        self.episode_callback = lambda x: None
        self.step_callback = lambda x: None

    def print_summary(self):
        print("Actor model summary:")
        self.actor_local.model.summary()
        print("Critic model summary:")
        self.critic_local.model.summary()
        print("Hyperparameters:")
        print(str(dict(
            train_during_episode=self.train_during_episode,
            discount_factor=self.gamma,
            tau_actor=self.tau_actor, tau_critic=self.tau_critic,
            lr_actor=self.lr_actor, lr_critic=self.lr_critic,
            bn_momentum_actor=self.bn_momentum_actor,
            bn_momentum_critic=self.bn_momentum_critic,
            ou_mu=self.ou_mu, ou_theta=self.ou_theta, ou_sigma=1,
            activation_fn_actor=self.activation_fn_actor,
            replay_buffer_size=self.replay_buffer_size,
            replay_batch_size=self.replay_batch_size,
            l2_reg_actor=self.l2_reg_actor, l2_reg_critic=self.l2_reg_critic,
            relu_alpha_actor=self.relu_alpha_actor,
            relu_alpha_critic=self.relu_alpha_critic,
            dropout_actor=self.dropout_actor, dropout_critic=self.dropout_critic,
            hidden_layer_sizes_actor=self.hidden_layer_sizes_actor,
            hidden_layer_sizes_critic=self.hidden_layer_sizes_critic,
            input_bn_momentum_actor=self.input_bn_momentum_actor,
            input_bn_momentum_critic=self.input_bn_momentum_critic,
            activity_l1_reg=self.activity_l1_reg,
            activity_l2_reg=self.activity_l2_reg,
            output_action_regularizer=self.output_action_regularizer,
            output_action_variance_regularizer=self.output_action_variance_regularizer,
            normalize_rewards=self.normalize_rewards,
        )))

    def set_episode_callback(self, episode_callback=None):
        if not callable(episode_callback):
            self.episode_callback = lambda x: None
        else: self.episode_callback=episode_callback

    def set_step_callback(self, step_callback=None):
        if not callable(step_callback):
            self.step_callback = lambda x: None
        else: self.step_callback=step_callback

    def set_max_training_score_callback(self, max_training_score_callback=None):
        if not callable(max_training_score_callback):
            self.max_training_score_callback = lambda x: None
        else: self.max_training_score_callback=max_training_score_callback

    def set_max_test_score_callback(self, max_test_score_callback=None):
        if not callable(max_test_score_callback):
            self.max_test_score_callback = lambda x: None
        else: self.max_test_score_callback=max_test_score_callback

    def preprocess_state(self, state):
        res = np.array(state, copy=True)
        for i in range(self.env.observation_space.shape[0]):
            res[i] = self.observation_scalers[i].transform(np.array(state[i]).reshape(-1,1))[0]
        return res

    def transform_state(self, preprocessed_state):
        res = np.array(preprocessed_state, copy=True)
        for i in range(self.env.observation_space.shape[0]):
             res[i] = self.observation_scalers[i].inverse_transform(
                np.array(preprocessed_state[i]).reshape(-1,1))[0]
        return res

    def preprocess_action(self, action):
        return self.action_scaler.transform(np.array(action, copy=True).reshape(-1,self.action_size))[0]

    def transform_action(self, preprocessed_action):
        return self.action_scaler.inverse_transform(
            np.array(preprocessed_action, copy=True).reshape(-1,self.action_size))[0]

    def reset_episode(self):
        self.noise.reset()
        if self.do_preprocessing: state = self.preprocess_state(self.env.reset())
        else: state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done, learn=False):
         # Save experience / reward
        if self.do_preprocessing:
            next_state = self.preprocess_state(next_state)
            action = self.preprocess_action(action)

        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if learn or (len(self.memory) > self.batch_size and (self.train_during_episode or done)):
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        self.steps += 1

    def act(self, state=None):
        """Returns actions for given state(s) as per current policy."""
        if state is None:
            state = self.last_state
        else:
            if self.do_preprocessing: state = self.preprocess_state(state)

        action = self.actor_local.model.predict(np.reshape(state, [-1, self.state_size]))[0]
        if self.do_preprocessing:
            action = self.transform_action(action)
        return action

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

        # Normalize rewards
        # https://datascience.stackexchange.com/questions/20098/
        # why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
        if self.normalize_rewards:
            rewards -= self.memory.reward_mean
            rewards /= np.sqrt(self.memory.reward_var)

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

    def train_n_episodes(self, n_episodes, eps=1, eps_decay=0, action_repeat=1,
                         run_tests=True, gen_q_a_frames_every_n_steps=0,
                         learn_from_test=False, draw_plots=False,
                         act_random_first_n_episodes=0,
                         primary_exploration_eps=None, ):
        if primary_exploration_eps is None: primary_exploration_eps = eps
        n_training_episodes = len(self.history.training_episodes)
        for i_episode in range(n_training_episodes+1, n_training_episodes+n_episodes+1):
            eps -= eps_decay
            eps = max(eps,0)
            episode_start_step = self.steps
            training_score = self.run_episode(train=True, action_repeat=action_repeat,
                         eps=primary_exploration_eps if (i_episode<act_random_first_n_episodes) else eps,
                         gen_q_a_frames_every_n_steps=gen_q_a_frames_every_n_steps,
                         act_random=(i_episode<act_random_first_n_episodes), )
            if run_tests is True:
                test_score = self.test_agent(action_repeat=action_repeat,
                                gen_q_a_frames_every_n_steps=gen_q_a_frames_every_n_steps,
                                learn_from_test=learn_from_test, )
            self.episode_callback(self.episodes)

            last_training_episode = self.history.training_episodes[-1]
            num_steps = last_training_episode.last_step - last_training_episode.first_step+1
            message = "Episode %i - epsilon: %8.4g, memory size: %i, num steps: %i, training score: %6.2f"\
                        %(i_episode, eps, len(self.memory), num_steps, last_training_episode.score)
            if run_tests: message += ", test score: %.2f"%self.history.test_episodes[-1].score
            print(message)
            sys.stdout.flush()

    def test_agent(self, action_repeat=1, gen_q_a_frames_every_n_steps=0, learn_from_test=False, ):
        return self.run_episode(train=False, eps=0,
                    action_repeat=action_repeat,
                    gen_q_a_frames_every_n_steps=gen_q_a_frames_every_n_steps,
                    learn_from_test=learn_from_test, )

    def act_with_noise(self, eps, next_state):
        action_range = self.env.action_space.high - self.env.action_space.low
        raw_action = self.act(next_state)
        noise_sample = action_range * self.noise.sample() * max(0,eps) # some noise for exploration
        action = list(np.clip(raw_action + noise_sample, self.env.action_space.low, self.env.action_space.high))
        return action, raw_action

    def run_episode(self, action_repeat=1, eps=0, train=False, gen_q_a_frames_every_n_steps=0,
                    learn_from_test=False, act_random=False, ):
        next_state = self.reset_episode()
        if train: episode_history = self.history.new_training_episode(self.episodes+1,eps)
        else: episode_history = self.history.new_test_episode(self.episodes,eps)
        q_a_frame = None
        while True:
            if act_random:
                action, raw_action = np.random.uniform(self.action_low,self.action_high,self.action_size),\
                                     np.zeros(self.action_size)
                # action_range = (self.action_high-self.action_low)
                # action_mid = self.action_low+action_range/2
                # action, raw_action = np.random.normal(action_mid, action_range/6, self.action_size),\
                #                      np.zeros(self.action_size)
            else:
                action, raw_action = self.act_with_noise(eps, next_state)
            sum_rewards=0
            # Repeat action `action_repeat` times, summing up rewards
            for i in range(action_repeat):
                next_state, reward, done, info = self.env.step(action)
                sum_rewards += reward
                if done:
                    break
            # TODO: Snapshot agent's weights after each episode, and/or every gen_q_a_frames_every_n_steps
            env_state = None
            if callable(self.step_callback): self.step_callback(self.steps)
            if hasattr(self.env,'get_full_state') and callable(self.env.get_full_state):
                # Get some extra environment state data to store in episode history for visualization
                env_state = self.env.get_full_state()
            episode_history.append(self.steps, next_state, raw_action, action, sum_rewards, env_state)
            if train is True:
                self.step(action, sum_rewards, next_state, done)
                # TODO: Replace this with a callback function that will enable
                #       visualization snapshots to be generated externally
                if gen_q_a_frames_every_n_steps > 0 and self.steps%gen_q_a_frames_every_n_steps==0:
                    self.history.add_q_a_frame(self.get_q_a_frames())
            elif learn_from_test is True:
                self.step(action, sum_rewards, next_state, done)
            if done:
                if train:
                    self.episodes += 1
                    if self.history.max_training_score is None or \
                    episode_history.score > self.history.max_training_score:
                        self.history.max_training_score = episode_history.score
                        self.max_training_score_callback(episode_history)
                elif self.history.max_test_score is None or \
                episode_history.score > self.history.max_test_score:
                    self.history.max_test_score = episode_history.score
                    self.max_test_score_callback(episode_history)
                break
        return episode_history.score

    def get_q_a_frames(self, q_a_frames_spec=None):
        """ TODO: Figure out how to work with added dimensions.
                - Use x_dim, y_dim, and a_dim to know which dimensions of state and action to vary.
                    Maybe fill in the unvaried dimensions of states and actions with agent's current state
                    and anticipated action (according to policy).
        """
        if q_a_frames_spec is None:
            q_a_frames_spec = self.q_a_frames_spec
        xs = q_a_frames_spec.xs
        nx = q_a_frames_spec.nx
        ys = q_a_frames_spec.ys
        ny = q_a_frames_spec.ny
        action_space = q_a_frames_spec.action_space
        na = q_a_frames_spec.na
        x_dim = q_a_frames_spec.x_dim
        y_dim = q_a_frames_spec.y_dim
        a_dim = q_a_frames_spec.a_dim

        def get_state(x,y):
            # TODO: Re-consider how we're copying the rest of the state. In the case of the quadcopter we have
            #       position for the last 3 (action-repeat) time steps. These should be adjusted to match with the
            #       variation we are adding to any position dimensions we are visualizing.
            #       This function will have to know somehow that state dimensions [0,1,2,6,7,8,12,13,14] need to
            #       apply this spatial translation.

            s=copy.copy(self.last_state)
            # Special case for quadcopter with action_repeat=3
#             def adjust_state_position(dim, val):
#                 # if we're visualizing a spatial dimension
#                 if len(s)>14:
#                     orig_val = s[x_dim]
#                     diff = val - orig_val
#                     s[dim+6] += diff
#                     s[dim+12] += diff
#                 s[dim]=val
# #             s[x_dim]=x
# #             s[y_dim]=y
#             adjust_state_position(x_dim,x)
#             adjust_state_position(y_dim,y)
            s[x_dim]=x
            s[y_dim]=y
            return s
        raw_states = np.array([[ get_state(x,y) for x in xs ] for y in ys ]).reshape(nx*ny, self.state_size)

        def get_action(action):
            a=self.act() if self.action_size>1 else [0]
            a[a_dim]=action
            return a
        actions = np.array([get_action(a) for a in action_space]*nx*ny)

        preprocessed_states = np.array([ self.preprocess_state(s) for s in raw_states]
                                      ) if self.do_preprocessing else raw_states
        preprocessed_actions = np.array([ self.preprocess_action(a) for a in actions ]
                                      ) if self.do_preprocessing else actions
        Q = self.critic_local.model.predict_on_batch(
                [np.repeat(preprocessed_states,na,axis=0),preprocessed_actions]).reshape((ny,nx,na))
        Q_max = np.max(Q,axis=2)
        Q_std = np.std(Q,axis=2)
        max_action = np.array([action_space[a] for a in np.argmax(Q,axis=2).flatten()]).reshape((ny,nx))
        actor_policy = np.array([
                self.act(s)
                #self.preprocess_action(self.act(s))
                if self.do_preprocessing else self.act(s)
                for s in raw_states]).reshape(ny,nx,self.action_size)
        preprocessed_policy = np.array([self.preprocess_action(a)
                                for a in actor_policy.reshape(nx*ny, self.action_size)])
        action_gradients = self.critic_local.get_action_gradients(
            [preprocessed_states,preprocessed_policy,0])[0].reshape(ny,nx,self.action_size)

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
        self._field_names = field_names=["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", self._field_names)
        self.action_means = []
        self.action_vars = []
        self.action_M2s = []
        self.state_means = []
        self.state_vars = []
        self.state_M2s = []
        self.reward_mean = None
        self.reward_var = None
        self.reward_M2 = None

    def __len__(self):
        return len(self.memory)

    def sample(self):
        return random.sample(self.memory, k=min(self.batch_size, len(self)))

#     def normalize_state(self, state, eps=.0000001):
#         return [ (a-self.state_means[i])/(np.sqrt(self.state_vars[i])+eps) for i, a in enumerate(state) ]

#     def normalize_action(self, action, eps=.0000001):
#         return [ (a-self.action_means[i])/(np.sqrt(self.action_vars[i])+eps) for i, a in enumerate(action) ]

#     def transform_action(self, normalized_action):
#         return [ np.sqrt(self.action_vars[i])*a+self.action_means[i] for i, a in enumerate(normalized_action) ]

#     def normalize_reward(self, reward, eps=.0000001):
#         if None in [self.reward_mean, self.reward_var]:
#             return 0
#         return (reward-self.reward_mean)/(np.sqrt(self.reward_var)+eps)

#     def normalize_sample(self, sample):
#         """ normalizes given sample with the mean and variance of each experience in memory. """
#         return namedtuple("normalizedExperience", self._field_names)(
#                             self.normalize_state(sample.state),
#                             self.normalize_action(sample.action),
#                             self.normalize_reward(sample.reward),
#                             self.normalize_state(sample.next_state),
#                             sample.done )

#     def sample_normalized(self):
#         """ Return states, actions, and rewards normalized
#             by the mean and variance of each experience in memory. """
#         return [ self.normalize_sample(exp) for exp in self.sample() ]

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        # Update mean and std of actions, rewards and each state dimension before adding to memory.
        """ Using Welford's online algorithm for calculating rolling variance
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        """
        # for a new value newValue, compute the new count, new mean, the new M2.
        # mean accumulates the mean of the entire dataset
        # M2 aggregates the squared distance from the mean
        # count aggregates the number of samples seen so far
        def _update(existingAggregate, newValue):
            (count, mean, M2) = existingAggregate
            count += 1
            delta = newValue - mean
            mean += delta / count
            delta2 = newValue - mean
            M2 += delta * delta2

            return (count, mean, M2)

        # retrieve the mean, variance and sample variance from an aggregate
        def _finalize(existingAggregate, newValue):
            (count, mean, M2) = _update(existingAggregate, newValue)
            (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
            if count < 2:
                return float('nan')
            else:
                return (mean, variance, sampleVariance)
        """
        End copy from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        """
        if len(self)==0:
            self.state_means = copy.copy(state)
            self.state_M2s = np.zeros(len(state))
            self.state_vars = np.zeros(len(state))
            self.action_means = copy.copy(action)
            self.action_M2s = np.zeros(len(action))
            self.action_vars = np.zeros(len(action))
            self.reward_mean = copy.copy(reward)
            self.reward_M2 = 0
            self.reward_var = 0
        else:
            for i, s in enumerate(state):
                existingAggregate = (len(self), self.state_means[i], self.state_M2s[i])
                (count, mean, self.state_M2s[i]) = _update(existingAggregate, s)
                (self.state_means[i], self.state_vars[i], _) = _finalize(existingAggregate, s)
            for i, a in enumerate(action):
                existingAggregate = (len(self), self.action_means[i], self.action_M2s[i])
                (count, mean, self.action_M2s[i]) = _update(existingAggregate, a)
                (self.action_means[i], self.action_vars[i], _) = _finalize(existingAggregate, a)
            existingAggregate = (len(self), self.reward_mean, self.reward_M2)
            (count, mean, self.reward_M2) = _update(existingAggregate, reward)
            (self.reward_mean, self.reward_var, _) = _finalize(existingAggregate, reward)

        self.memory.append(e)

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
        self.max_training_score = None
        self.max_test_score = None

    def __repr__(self):
        return "TrainingHistory ( %i training_episodes, %i test_episodes, %i qa_grids, last_step: %i )"%\
                (len(self.training_episodes), len(self.test_episodes), len(self.q_a_frames), self.last_step)

    def __iter__(self):
        for key in ['test_episodes','training_episodes','max_test_score','max_training_score']:
            if key in ['test_episodes','training_episodes']:
                yield (key, [ dict(ep) for ep in getattr(self,key)])
            else:
                yield (key, getattr(self,key))

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
        self.raw_actions = []
        self.actions = []
        self.rewards = []
        self.env_state = []

        self.score = self._get_score()

    def _fromDict(obj):
        qs = namedtuple("QuadcopterState",[
                     'x', 'y', 'z', 'phi', 'theta', 'psi',
                     'x_velocity', 'y_velocity', 'z_velocity',
                     'phi_velocity', 'theta_velocity', 'psi_velocity',
                     'x_linear_accel','y_linear_accel','z_linear_accel',
                     'phi_angular_accel','theta_angular_accel','psi_angular_accel',
                    ])
        eh = EpisodeHistory(obj['episode_idx'],obj['epsilon'])
        eh.steps = obj['steps']
        eh.first_step = obj['first_step']
        eh.last_step = obj['last_step']
        eh.states = obj['states']
        eh.raw_actions = obj['raw_actions']
        eh.actions = obj['actions']
        eh.rewards = obj['rewards']
        eh.score = np.sum(obj['rewards'])
        eh.env_state = [qs(**dict(ep)) for ep in obj['env_state']]
        return eh

    def __iter__(self):
        for key in ['episode_idx','epsilon','steps','first_step','last_step',\
                    'states','raw_actions','actions','rewards','env_state']:
            attr = getattr(self,key)
            if type(attr)==list:
                yield (key, [a.tolist() \
                                if hasattr(a,'tolist') else \
                                dict(a._asdict()) if hasattr(a,'_asdict') else \
                                a for a in attr])
            else:
                yield(key, attr)

    def _get_score(self):
        return sum(self.rewards)

    def append(self, step, state, raw_action, action, reward, env_state):
        self.steps.append(step)
        self.states.append(state)
        self.raw_actions.append(raw_action)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.env_state is not None: self.env_state.append(env_state)
        if self.first_step is None: self.first_step = step
        self.last_step = step
        self.score += reward

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return "EpisodeHistory ( idx: %i, len: %i, first_step: %i, last_step: %i, epsilon: %.3f, score: %.3f )"%\
            (self.episode_idx,len(self.steps),self.first_step, self.last_step, self.epsilon, self.score)
