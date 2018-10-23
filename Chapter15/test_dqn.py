"""This module contains function to test the performance of the DQN model."""
import numpy as np


def test(env, model, states, episodes=100, render=False):
    """Test the performance of the DQN agent."""
    scores_test = []
    for episode in range(1, (episodes+1)):
        state = env.reset()
        state = state.reshape(1, states)

        done = False
        time_step = 0

        while not done:
            if render:
                env.render()
            action = np.argmax(model.predict(state)[0])
            new_state, reward, done, info = env.step(action)
            new_state = new_state.reshape(1, states)
            state = new_state
            time_step += 1
        scores_test.append(time_step)
        if episode % 10 == 0:
            print('episode {}, score {} '.format(episode, time_step))
    print('Average score over 100 test games: {}'.format(np.mean(scores_test)))
