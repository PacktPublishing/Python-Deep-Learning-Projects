"""This module is used to train and test the DQN agent."""
import random
import numpy as np
from agent_replay_dqn import agent, agent_action, replay, performance_plot
from hyperparameters_dqn import *
from test_dqn import test
from keras import backend as k
from collections import deque
import gym

env = gym.make('CartPole-v1')

# Set seed for reproducibility
seed_val = 456
np.random.seed(seed_val)
env.seed(seed_val)
random.seed(seed_val)

states = env.observation_space.shape[0]
actions = env.action_space.n
training_data = deque(maxlen=deque_len)


def memory(state, new_state, reward, done, action):
    """Function to store data points in the deque container."""
    training_data.append((state, new_state, reward, done, action))


def train(target_score, batch_size, episodes,
          optimizer, loss, epsilon,
          gamma, epsilon_min, epsilon_decay, actions, render=False):
    """Training the agent on games."""
    print('----Training----')
    k.clear_session()

    # define empty list to store the score at the end of each episode
    scores = []
    # load the agent
    model = agent(states, actions)
    # compile the agent with mean squared error loss
    model.compile(loss=loss, optimizer=optimizer)

    for episode in range(1, (episodes+1)):
        # reset environment at the end of each episode
        state = env.reset()
        # reshape state to shape 1*4
        state = state.reshape(1, states)
        # set done value to False
        done = False
        # counter to keep track of actions taken in each episode
        time_step = 0
        # play the game until done value changes to True
        while not done:
            if render:
                env.render()

            # call the agent_action function to decide on an action
            action = agent_action(model, epsilon, state, actions)
            # take the action
            new_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            # reshape new_state to shape 1*4
            new_state = new_state.reshape(1, states)
            # call memory function to store info in the deque container
            memory(state, new_state, reward, done, action)
            # set state to new state
            state = new_state
            # increment timestep
            time_step += 1

        # call the replay function to train the agent
        epsilon = replay(epsilon, gamma, epsilon_min, epsilon_decay, model,
                         training_data)
        # save score after the game/episode ends
        scores.append(time_step)

        if episode % 100 == 0:
            print('episode {}, score {}, epsilon {:.4}'.format(episode,
                                                               time_step,
                                                               epsilon))
            print('Avg Score over last 100 epochs', sum(scores[-100:])/100)
            if sum(scores[-100:])/100 > target_score:
                print('------ Goal Achieved After {} Episodes ------'
                      .format(episode))
                # plot the scores over time
                performance_plot(scores, target_score)
                break
            # Uncomment below line to plot score progress every 100 episodes
            # performance_plot(scores, target_score)
    return model


model = train(target_score=target_score, batch_size=batch_size,
              episodes=episodes, optimizer=optimizer, loss=loss,
              epsilon=epsilon, gamma=gamma, epsilon_min=epsilon_min,
              epsilon_decay=epsilon_decay, actions=actions, render=False)

test(env, model, states, render=False)
