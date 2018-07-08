"""This module contains."""
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam


def agent(states, actions):
    """Simple Deep Neural Network."""
    model = Sequential()
    model.add(Dense(16, input_dim=states))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))
    return model


def agent_action(model, epsilon, state, actions):
    """Define action to be taken."""
    if np.random.rand() <= epsilon:
        act = random.randrange(actions)
    else:
        act = np.argmax(model.predict(state)[0])
    return act


def performance_plot(scores, target_score):
    """Plot the game progress."""
    scores_arr = np.array(scores)
    scores_arr[np.where(scores_arr > target_score)] = target_score
    plt.figure(figsize=(20, 5))
    plt.title('Plot of Score v/s Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.plot(scores_arr)
    plt.show()


def replay(epsilon, gamma, epsilon_min, epsilon_decay, model,
           training_data, batch_size=64):
    """Train the agent on a batch of data."""
    idx = random.sample(range(len(training_data)),
                        min(len(training_data), batch_size))
    train_batch = [training_data[j] for j in idx]
    for state, new_state, reward, done, action in train_batch:
        target = reward
        if not done:
            target = reward+gamma*np.amax(model.predict(new_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    return epsilon
