from keras import backend as K
from keras import layers, models, optimizers, regularizers

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, learn_rate,
                 activation_fn, input_bn_momentum, bn_momentum, relu_alpha, l2_reg,
                 dropout, hidden_layer_sizes, activity_l2_reg, output_action_regularizer):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            action_low (array): Min value of each action dimension.
            action_high (array): Max value of each action dimension.
            learn_rate (float): Learning rate.
            activation_fn (string): Activation function, either 'sigmoid' or 'tanh'.
            bn_momentum (float): Batch Normalization momentum .
            relu_alpha (float): LeakyReLU alpha, allowing small gradient when the unit is not active.
            l2_reg (float): L2 regularization factor for each dense layer.
            dropout (float): Dropout rate
            hidden_layer_sizes (list): List of hidden layer sizes.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learn_rate = learn_rate
        self.activation = activation_fn
        self.input_bn_momentum = input_bn_momentum
        self.bn_momentum = bn_momentum
        self.relu_alpha = relu_alpha
        self.l2_reg = l2_reg
        self.activity_l2_reg = activity_l2_reg
        self.output_action_regularizer=output_action_regularizer
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        states = layers.Input(shape=(self.state_size,), name='states')
        net = states

        # Batch Norm instead of input preprocessing (Since we don't know up front the range of state values.)
        if self.input_bn_momentum>0: net = layers.BatchNormalization(momentum=self.input_bn_momentum)(states)

        # Add a hidden layer for each element of hidden_layer_sizes
        for size in self.hidden_layer_sizes:
            net = layers.Dense(units=size, kernel_regularizer=regularizers.l2(l=self.l2_reg))(net)
#             if self.input_bn_momentum>0: net = layers.BatchNormalization(momentum=self.input_bn_momentum)(net)
            if self.relu_alpha>0: net = layers.LeakyReLU(alpha=self.relu_alpha)(net)
            else: net = layers.Activation('relu')(net)

        if self.dropout>0: net = layers.Dropout(.2)(net)

        if self.bn_momentum>0: net = layers.BatchNormalization(momentum=self.bn_momentum)(net)

        if self.activation=='tanh':
            # Add final output layer with tanh activation with [-1, 1] output
            # raw_actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions',
            #                            activity_regularizer=regularizers.l2(self.activity_l2_reg))(net)
            # actions = layers.Lambda(lambda x: ((x+1)/2. * self.action_range) + self.action_low,
            #     name='actions')(raw_actions)
            actions = layers.Dense(units=self.action_size, activation='tanh', name='actions',
                                       activity_regularizer=regularizers.l2(self.activity_l2_reg))(net)
        elif self.activation=='sigmoid':
            # Add final output layer with sigmoid activation
            raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions',
                                      activity_regularizer=regularizers.l2(self.activity_l2_reg))(net)
            # Scale [0, 1] output for each action dimension to proper range
            # actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            #     name='actions')(raw_actions)
            # Scale to -1 to 0 (assume all preprocessing done already by agent)
            actions = layers.Lambda(lambda x: (x * 2) -1, name='actions')(raw_actions)
        else:
            raise "Expected 'activation' to be one of: 'tanh', or 'sigmoid'."

        self.model = models.Model(inputs=states, outputs=actions)
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # These next two lines thanks to https://gist.github.com/kkweon/a82980f3d60ffce1d69ad6da8af0e124
        for l2_regularizer_loss in self.model.losses:
            loss += l2_regularizer_loss

        # Incorporate any additional losses here (e.g. from regularizers)

        # These next two lines thanks to https://gist.github.com/kkweon/a82980f3d60ffce1d69ad6da8af0e124
        for l2_regularizer_loss in self.model.losses:
            loss += l2_regularizer_loss

        if self.output_action_regularizer:
            # mid_action = (self.action_high-self.action_low)/2.
            # loss += self.output_action_regularizer*K.mean(K.square(actions-mid_action))
            loss += self.output_action_regularizer*K.mean(K.square(actions))
        # if self.output_action_regularizer:
        #     loss += self.output_action_regularizer*K.sum(K.square(net))
        # if self.output_action_regularizer:
        #     loss += self.output_action_regularizer*K.var(net)
        # if self.output_action_regularizer:
        #     loss += self.output_action_regularizer*K.mean(K.square(raw_actions))

        optimizer = optimizers.Adam(lr=self.learn_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, learn_rate, input_bn_momentum, bn_momentum,
                 relu_alpha, l2_reg, dropout, hidden_layer_sizes,
                ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            learn_rate (float): Learning rate.
            bn_momentum (float): Batch Normalization momentum.
            relu_alpha (float): LeakyReLU alpha, allowing small gradient when the unit is not active.
            l2_reg (float): L2 regularization factor for each dense layer.
            dropout (float): Dropout rate
            hidden_layer_sizes (list[list]): List of two lists with hidden layer sizes for state and action pathways.
        """
        self.state_size = state_size
        self.action_size = action_size

#         assert len(hidden_layer_sizes)==2 \
#             and len(hidden_layer_sizes[0])==len(hidden_layer_sizes[1]),\
#             "Expected Critic's hidden_layer_sizes to be a list of two arrays of equal length."
        assert len(hidden_layer_sizes)==2 \
            and hidden_layer_sizes[0][-1]==hidden_layer_sizes[1][-1], \
            "Critic's hidden_layer_sizes should be a list of two arrays where the last element "+\
            "of each array is equal to each other."

        # Initialize any other variables here
        self.learn_rate = learn_rate
        self.input_bn_momentum = input_bn_momentum
        self.bn_momentum = bn_momentum
        self.relu_alpha = relu_alpha
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        net_states = states
        net_actions = actions

        if self.input_bn_momentum>0:
            net_states = layers.BatchNormalization(momentum=self.input_bn_momentum)(net_states)
            net_actions = layers.BatchNormalization(momentum=self.input_bn_momentum)(net_actions)

        # Add hidden layer(s) for state pathway
        for size in self.hidden_layer_sizes[0]:
            net_states = layers.Dense(units=size,
                                      kernel_regularizer=regularizers.l2(l=self.l2_reg))(net_states)
            #net_states = layers.BatchNormalization(momentum=self.bn_momentum)(net_states)
            if self.relu_alpha>0: net_states = layers.LeakyReLU(alpha=self.relu_alpha)(net_states)
            else: net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        for size in self.hidden_layer_sizes[1]:
            net_actions = layers.Dense(units=size,
                                       kernel_regularizer=regularizers.l2(l=self.l2_reg))(net_actions)
            #net_actions = layers.BatchNormalization(momentum=self.bn_momentum)(net_actions)
            if self.relu_alpha>0: net_actions = layers.LeakyReLU(alpha=self.relu_alpha)(net_actions)
            else: net_actions = layers.Activation('relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        if self.relu_alpha>0: net = layers.LeakyReLU(alpha=self.relu_alpha)(net)
        else: net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        if self.dropout>0: net = layers.Dropout(self.dropout)(net)

        # Normalize the final activations
        if self.bn_momentum>0: net = layers.BatchNormalization(momentum=self.bn_momentum)(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learn_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
