"""This module implements training and testing of SARSA agent."""
import gym
import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

# load the environment
env = gym.make('CartPole-v1')

# set seed
seed_val = 456
env.seed(seed_val)
np.random.seed(seed_val)

states = env.observation_space.shape[0]
actions = env.action_space.n


def agent(states, actions):
    """Agent/Deep Neural Network."""
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))
    return model


model = agent(states, actions)

# Define the policy
policy = EpsGreedyQPolicy()
# Define SARSA agent by feeding it the policy and the model
sarsa = SARSAAgent(model=model, nb_actions=actions, nb_steps_warmup=10,
                   policy=policy)
# compile sarsa with mean squared error loss
sarsa.compile('adam', metrics=['mse'])
# train the agent for 50000 steps
sarsa.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Evaluate the agent on 100 new episodes.
scores = sarsa.test(env, nb_episodes=100, visualize=False)
print('Average score over 100 test games: {}'
      .format(np.mean(scores.history['episode_reward'])))
