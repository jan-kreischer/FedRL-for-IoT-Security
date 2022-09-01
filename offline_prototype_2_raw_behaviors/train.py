from data_provider import DataProvider
from offline_prototype_2_raw_behaviors.environment import SensorEnvironment, supervisor_map
from agent import Agent
from custom_types import Behavior
from autoencoder import AutoEncoder, AutoEncoderInterpreter
from utils.evaluation_utils import plot_learning, seed_random, calculate_metrics, get_pretrained_agent, evaluate_agent
from utils.autoencoder_utils import evaluate_ae_on_no_mtd_behavior, pretrain_ae_model, get_pretrained_ae
from tabulate import tabulate
from time import time
import torch
import numpy as np
import random
import os

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
DIMS = 30  # TODO check
PI = 3
SAMPLES = 1

if __name__ == '__main__':
    os.chdir("..")
    seed_random()
    start = time()

    # read in all preprocessed data for a simulated, supervised environment to sample from
    # train_data, test_data, scaler = DataProvider.get_scaled_train_test_split()
    train_data, test_data = DataProvider.get_reduced_dimensions_with_pca(DIMS, pi=PI)
    # get splits for RL & AD of normal data
    n = 100
    s = 0.8
    b = Behavior.NORMAL
    normal_data = train_data[b]
    train_data[b] = normal_data[:n]  # use fixed number of samples for Reinforcement Agent training
    # COMMENT/UNCOMMENT BELOW for pretraining of autoencoder
    ae_path = "offline_prototype_2_raw_behaviors/trained_models/ae_model_pi3.pth"
    ae_data = normal_data[n:]  # use remaining samples for autoencoder
    train_ae_x, valid_ae_x = pretrain_ae_model(ae_data=ae_data, path=ae_path)


    # AE evaluation of pretrained model
    ae_interpreter = get_pretrained_ae(path=ae_path, dims=DIMS)
    # AE can directly be tested on the data that will be used for RL: pass train_data to testing
    evaluate_ae_on_no_mtd_behavior(ae_interpreter=ae_interpreter, test_data=train_data)

    # Reinforcement Learning
    env = SensorEnvironment(train_data, test_data, interpreter=ae_interpreter, state_samples=SAMPLES)

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
        if i >= N_EPISODES - 6:
            print(episode_returns[-10:])

    end = time()
    print("Total training time: ", end - start)

    num = 0
    agent.save_agent_state(num, "offline_prototype_2_raw_behaviors")

    x = [i + 1 for i in range(N_EPISODES)]
    filename = f'offline_prototype_2_raw_behaviors/mtd_agent_p2_{SAMPLES}_sample.pdf'
    plot_learning(x, episode_returns, eps_history, filename)

    # check predictions with dqn from trained and stored agent
    pretrained_agent = get_pretrained_agent(path=f"offline_prototype_2_raw_behaviors/trained_models/agent_{num}.pth",
                                            input_dims=env.observation_space_size, n_actions=len(env.actions),
                                            buffer_size=BUFFER_SIZE)

    evaluate_agent(agent, test_data=test_data)
