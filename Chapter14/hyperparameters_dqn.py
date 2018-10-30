"""This module contains hyperparameters for the DQN model."""

# Discount in Bellman Equation
gamma = 0.95
# Epsilon
epsilon = 1.0
# Minimum Epsilon
epsilon_min = 0.01
# Decay multiplier for epsilon
epsilon_decay = 0.99
# Size of deque container
deque_len = 20000
# Average score needed over 100 epochs
target_score = 200
# Number of games
episodes = 2000
# Data points per episode used to train the agent
batch_size = 64
# Optimizer for training the agent
optimizer = 'adam'
# Loss for training the agent
loss = 'mse'
