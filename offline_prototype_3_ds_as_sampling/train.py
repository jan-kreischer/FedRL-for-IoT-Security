from data_provider import DataProvider
from offline_prototype_3_ds_as_sampling.environment import SensorEnvironment, supervisor_map
from agent import Agent
from custom_types import Behavior, MTDTechnique
from autoencoder import AutoEncoder, AutoEncoderInterpreter
from utils.evaluation_utils import plot_learning, seed_random, calculate_metrics, evaluate_agent, \
    evaluate_agent_on_afterstates
from utils.autoencoder_utils import pretrain_ae_model, get_pretrained_ae, evaluate_ae_on_no_mtd_behavior, \
    evaluate_ae_on_afterstates, pretrain_all_afterstate_ae_models, split_as_data_for_ae_and_rl, \
    split_ds_data_for_ae_and_rl, evaluate_all_as_ae_models, pretrain_all_ds_as_ae_models
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
N_EPISODES = 10000
LOG_FREQ = 100
DIMS = 15
SAMPLES = 20

if __name__ == '__main__':
    os.chdir("..")
    seed_random()
    start = time()

    # read in all preprocessed data for a simulated, supervised environment to sample from
    # train_data, test_data, scaler = DataProvider.get_scaled_train_test_split()
    # train_data, test_data = DataProvider.get_reduced_dimensions_with_pca(DIMS)
    # dtrain, dtest, atrain, atest, scaler = DataProvider.get_scaled_scaled_train_test_split_with_afterstates()
    dtrain, dtest, atrain, atest = DataProvider.get_reduced_dimensions_with_pca_ds_as(DIMS,
                                                                                      dir="offline_prototype_3_ds_as_sampling/")
    # get splits for RL & AD of normal data
    dir = "offline_prototype_3_ds_as_sampling/trained_models/"
    model_name = "ae_model_ds.pth"
    path = dir + model_name
    ae_ds_train, dtrain_rl = split_ds_data_for_ae_and_rl(dtrain)
    # pretrain_ae_model(ae_ds_train, path=path)

    # AE evaluation of pretrained model
    # ae_interpreter = get_pretrained_ae(path=path, dims=DIMS)
    # AE can directly be tested on the data that will be used for RL: pass train_data to testing
    # print("---AE trained on decision state normal data---")
    # print("---Evaluation on decision behaviors train---")
    # evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain_rl)
    # print("---Evaluation on afterstate behaviors train---")
    # evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)

    ae_train_dict, atrain_rl = split_as_data_for_ae_and_rl(atrain)
    # SHOWN that the DIRTRAP model has the least FP on the testset
    # pretrain_all_afterstate_ae_models(ae_train_dict, dir=dir)
    # evaluate_all_as_ae_models(dtrain_rl, atrain_rl, dims=DIMS, dir=dir)

    # MODEL trained on all ds and as normal data assumes the least -> MOST REALISTIC
    # pretrain_all_ds_as_ae_models(ae_ds_train, ae_train_dict)
    # print("Evaluating AE trained on all decision and afterstates normal")
    path = dir + "ae_model_all_ds_as.pth"
    ae_interpreter = get_pretrained_ae(path=path, dims=DIMS)
    # print("---Evaluation on decision behaviors train---")
    # evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
    # print("---Evaluation on afterstate behaviors train---")
    # evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)

    # Reinforcement Learning
    env = SensorEnvironment(decision_train_data=dtrain_rl, decision_test_data=dtest,
                            after_train_data=atrain, after_test_data=atest, interpreter=ae_interpreter,
                            state_samples=SAMPLES)

    agent = Agent(input_dims=env.observation_space_size, n_actions=len(env.actions), buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)
    episode_returns, eps_history = [], []

    # initialize memory replay buffer (randomly)
    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = random.choice(env.actions)

        new_obs, reward, done = env.step(action)
        idx1 = -1 if obs[0, -1] in Behavior else -2
        idx2 = -1 if new_obs[0, -1] in Behavior else -2
        transition = (obs[:, :idx1], action, reward, new_obs[:, :idx2], done)
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
            idx1 = -1 if obs[0, -1] in Behavior else -2
            action = agent.choose_action(obs[:, :idx1])

            new_obs, reward, done = env.step(action)
            idx2 = -1 if new_obs[0, -1] in Behavior else -2
            episode_return += reward
            agent.replay_buffer.append((obs[:, :idx1], action, reward,
                                        new_obs[:, :idx2], done))
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
    agent.save_agent_state(num, "offline_prototype_3_ds_as_sampling")

    x = [i + 1 for i in range(N_EPISODES)]
    filename = f'offline_prototype_3_ds_as_sampling/mtd_agent_p3_{SAMPLES}_samples.pdf'
    plot_learning(x, episode_returns, eps_history, filename)

    # check predictions with dqn from trained and stored agent
    pretrained_state = torch.load(f"offline_prototype_3_ds_as_sampling/trained_models/agent_{num}.pth")
    pretrained_agent = Agent(input_dims=DIMS, n_actions=4, buffer_size=BUFFER_SIZE,
                             batch_size=pretrained_state['batch_size'], lr=pretrained_state['lr'],
                             gamma=pretrained_state['gamma'], epsilon=pretrained_state['eps'],
                             eps_end=pretrained_state['eps_min'], eps_dec=pretrained_state['eps_dec'])
    pretrained_agent.online_net.load_state_dict(pretrained_state['online_net_state_dict'])
    pretrained_agent.target_net.load_state_dict(pretrained_state['target_net_state_dict'])
    pretrained_agent.replay_buffer = pretrained_state['replay_buffer']

    evaluate_agent(agent=pretrained_agent, test_data=dtest)
    evaluate_agent_on_afterstates(agent=pretrained_agent, test_data=atest)
