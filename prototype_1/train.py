from typing import Dict
from collections import defaultdict
from custom_types import MTDTechnique, Behavior

from collections import defaultdict
from custom_types import MTDTechnique, Behavior
from data_manager import DataManager
from prototype_1.environment import SensorEnvironment
from prototype_1.agent import Agent, DeepQNetwork

import numpy as np
import random

# Hyperparams
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 10
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 0.001



if __name__ == '__main__':
    # read in all data for a simulated, supervised environment to sample from
    env = SensorEnvironment(DataManager.parse_all_behavior_data())

    #
    # env.reset()
    # print(f"state features num: {env.observation_space}")
    # print(f"actions num: {env.action_space}")
    # print(f"initial behavior: {env.current_state.iloc[0]['attack']}")
    # action = MTDTechnique.RANSOMWARE_DIRTRAP
    # new_state, reward, isTerminalState = env.step(action)
    # print(f"new state: {new_state.iloc[0]['attack']}")
    # print(f"is terminal: {isTerminalState}")

    agent = Agent(input_dims=env.observation_space_size, n_actions=len(env.actions),
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)
    scores, eps_history = [], []
    n_episodes = 500

    # initialize memory replay buffer
    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = random.choice(list(MTDTechnique))

        new_obs, reward, done = env.step(action)
        transition = (obs, action, reward, new_obs, done)
        agent.replay_buffer.append(transition)

        obs = new_obs
        if done:
            obs = env.reset()

    # # training
    # for i in range(n_episodes):
    #     score = 0



    # for i in range(n_games):
    #     score = 0
    #     done = False
    #     observation = env.reset()
    #     while not done:
    #         action = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         score += reward
    #         agent.store_transition(observation, action, reward,
    #                                observation_, done)
    #         agent.learn()
    #         observation = observation_
    #     scores.append(score)
    #     eps_history.append(agent.epsilon)
    #
    #     avg_score = np.mean(scores[-100:])
    #
    #     print('episode ', i, 'score %.2f' % score,
    #           'average score %.2f' % avg_score,
    #           'epsilon %.2f' % agent.epsilon)
    # x = [i + 1 for i in range(n_games)]
    # filename = 'lunar_lander.png'
    # #plotLearning(x, scores, eps_history, filename)
