from typing import Dict
from collections import defaultdict
from custom_types import MTDTechnique, Behavior


from collections import defaultdict
from custom_types import MTDTechnique, Behavior
from data_manager import DataManager
from prototype_1.environment import SensorEnvironment
from prototype_1.agent import Agent, DeepQNetwork

import numpy as np



if __name__ == '__main__':
    # read in all data for a simulated, supervised environment to sample from
    env = SensorEnvironment(DataManager.parse_all_behavior_data())
    env.reset()
    print(f"state features num: {env.observation_space}")
    print(f"actions num: {env.action_space}")
    print(f"initial behavior: {env.current_state.iloc[0]['attack']}")

    action = MTDTechnique.RANSOMWARE_DIRTRAP
    new_state, reward, isTerminalState = env.step(action)
    print(f"new state: {new_state.iloc[0]['attack']}")
    print(f"is terminal: {isTerminalState}")

    input_dims = env.observation_space
    # agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
    #               input_dims=[8], lr=0.001)
    # scores, eps_history = [], []
    # n_games = 500

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
