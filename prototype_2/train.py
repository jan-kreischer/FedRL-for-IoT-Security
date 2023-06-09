from data_manager import DataManager
from prototype_2.environment import SensorEnvironment, supervisor_map
from prototype_2.agent import Agent, DeepQNetwork
from custom_types import Behavior
from autoencoder import AutoEncoderInterpreter
from utils.utils import plot_learning, seed_random
from time import time
import torch
import numpy as np
import random

# Hyperparams
GAMMA = 0.99
BATCH_SIZE = 100
BUFFER_SIZE = 500
MIN_REPLAY_SIZE = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
TARGET_UPDATE_FREQ = 100
LEARNING_RATE = 1e-5
N_EPISODES = 5000
LOG_FREQ = 100
DIMS = 15

if __name__ == '__main__':
    seed_random()
    start = time()

    # TODO: autoencoder training

    # read in all preprocessed data for a simulated, supervised environment to sample from
    #train_data, test_data, scaler = DataManager.get_scaled_train_test_split()
    train_data, test_data = DataManager.get_reduced_dimensions_with_pca(DIMS)



    env = SensorEnvironment(train_data, test_data)

    agent = Agent(input_dims=env.observation_space_size, n_actions=len(env.actions), buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)
    episode_returns, eps_history = [], []

    # initialize memory replay buffer (randomly)
    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = random.choice(env.actions)

        new_obs, reward, done = env.step(action)
        transition = (obs[:, :-1], action, reward, new_obs[:, :-1], done)
        agent.replay_buffer.append(transition)

        obs = new_obs
        if done:
            obs = env.reset()

    # main training
    step = 0
    for i in range(N_EPISODES):
        episode_return = 0
        episode_steps = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs[:, :-1])

            new_obs, reward, done = env.step(action)
            episode_return += reward
            agent.replay_buffer.append((obs[:, :-1], action, reward,
                                        new_obs[:, :-1], done))
            agent.reward_buffer.append(reward)

            agent.learn()
            obs = new_obs

            episode_steps += 1
            # update target network
            step += 1
            if step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # if step % LOG_FREQ == 0:
            # print("Episode: ", i, "Step: ", step, ", Avg Reward: ", np.mean(agent.reward_buffer), "epsilon: ", agent.epsilon)

        episode_returns.append(episode_return / episode_steps)
        avg_episode_return = np.mean(episode_returns[-10:])
        eps_history.append(agent.epsilon)

        print('episode ', i, '| episode_return %.2f' % episode_returns[-1],
              '| average episode_return %.2f' % avg_episode_return,
              '| epsilon %.2f' % agent.epsilon)

    end = time()
    print("Total training time: ", end - start)

    agent.save_dqns(0)

    x = [i + 1 for i in range(N_EPISODES)]
    filename = 'mtd_agent.pdf'
    plot_learning(x, episode_returns, eps_history, filename)


    # check predictions with learnt dqn
    agent.online_net.eval()
    results = {}
    with torch.no_grad():
        for b, d in test_data.items():
            if b != Behavior.NORMAL:
                cnt_corr = 0
                cnt = 0
                for state in d:
                    action = agent.take_greedy_action(state[:-1])
                    if b in supervisor_map[action]:
                        cnt_corr += 1
                    cnt += 1
                results[b] = (cnt_corr, cnt)

    print(results)






