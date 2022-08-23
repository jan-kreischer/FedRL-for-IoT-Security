from data_provider import DataProvider
from offline_prototype_3_ds_as_sampling.environment import SensorEnvironment, supervisor_map
from agent import Agent
from custom_types import Behavior, MTDTechnique
from autoencoder import AutoEncoder, AutoEncoderInterpreter
from utils.utils import plot_learning, seed_random, calculate_metrics
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
DIMS = 15

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
    n = 300
    s = 0.8
    b = Behavior.NORMAL
    normal_data = dtrain[b]
    dtrain[b] = normal_data[:n]  # use fixed number of samples for Reinforcement Agent training
    # COMMENT/UNCOMMENT BELOW for retraining of autoencoder
    # ae_data = normal_data[n:]  # use remaining samples for autoencoder
    # idx = int(len(ae_data) * s)
    # # TODO: clean up placeholder
    # train_ae_x, train_ae_y = ae_data[:idx, :-1].astype(np.float32), np.arange(
    #     idx)  # just a placeholder for the torch dataloader
    # valid_ae_x, valid_ae_y = ae_data[idx:, :-1].astype(np.float32), np.arange(len(ae_data) - idx)
    # print(f"size train: {train_ae_x.shape}, size valid: {valid_ae_x.shape}")
    # # AD training
    # ae = AutoEncoder(train_x=train_ae_x, train_y=train_ae_y, valid_x=valid_ae_x,
    #                  valid_y=valid_ae_y)
    # ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=0.0001, momentum=0.8), num_epochs=1000)
    # ae.determine_threshold()
    # print(f"ae threshold: {ae.threshold}")
    # ae.save_model(dir="offline_prototype_3_ds_as_sampling/")

    # AE evaluation of pretrained model
    pretrained_model = torch.load("offline_prototype_3_ds_as_sampling/trained_models/autoencoder_model.pth")
    ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
                                            pretrained_model['threshold'], in_features=DIMS)
    print(f"ae_interpreter threshold: {ae_interpreter.threshold}")

    # AE can directly be tested on the data that will be used for RL: pass train_data to testing
    # print("---AE trained on decision state normal data---")
    # print("---Evaluation on decision behaviors train---")
    # res_dict = {}
    # for b, d in dtrain.items():
    #     y_test = np.array([0 if b == Behavior.NORMAL else 1] * len(d))
    #     y_predicted = ae_interpreter.predict(d[:, :-1].astype(np.float32))
    #
    #     acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
    #     res_dict[b] = f'{(100 * acc):.2f}%'
    #
    # labels = ["Behavior"] + ["Accuracy"]
    # results = []
    # for b, a in res_dict.items():
    #     results.append([b.value, res_dict[b]])
    # print(tabulate(results, headers=labels, tablefmt="pretty"))
    # print("---Evaluation on afterstate behaviors train---")
    # res_dict = {}
    # for t in atrain:
    #     y_test = np.array([0 if t[0] == Behavior.NORMAL else 1] * len(atrain[t]))
    #     y_predicted = ae_interpreter.predict(atrain[t][:, :-2].astype(np.float32))
    #
    #     acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
    #     res_dict[t] = f'{(100 * acc):.2f}%'
    # labels = ["Behavior", "MTD", "Accuracy"]
    # results = []
    # for t, a in res_dict.items():
    #     results.append([t[0].value, t[1].value, a])
    # print(tabulate(results, headers=labels, tablefmt="pretty"))

    # TODO: train here another autoencoder on normal-mtd afterstate data to be used by env.step
    #  -> possibly multiple AEs (one per mtd if needed...)

    normal_mtd_train = atrain[(Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER)]
    atrain[(Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER)] = normal_mtd_train[:n]
    ae_data = normal_mtd_train[n:]
    idx = int(len(ae_data) * s)
    all_train_ae_x, all_train_ae_y = ae_data[:idx, :-2].astype(np.float32), np.arange(
        idx)  # just a placeholder for the torch dataloader
    all_valid_ae_x, all_valid_ae_y = ae_data[idx:, :-2].astype(np.float32), np.arange(len(ae_data) - idx)

    # Train AD on each normal-mtd afterstate combination
    for i, mtd in enumerate(MTDTechnique):
        print("training:" + str(i) + " " + mtd.value)
        if mtd == MTDTechnique.ROOTKIT_SANITIZER:
            ae = AutoEncoder(train_x=train_ae_x, train_y=train_ae_y, valid_x=valid_ae_x,
                             valid_y=valid_ae_y)
            ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=0.0001, momentum=0.8), num_epochs=1000)
            ae.determine_threshold()
            print(f"ae threshold: {ae.threshold}")
            ae.save_model(dir=f"offline_prototype_3_ds_as_sampling/", num=i)

        normal_mtd_train = atrain[(Behavior.NORMAL, mtd)]
        atrain[(Behavior.NORMAL, mtd)] = normal_mtd_train[:n]
        ae_data = normal_mtd_train[n:]
        idx = int(len(ae_data) * s)
        train_ae_x, train_ae_y = ae_data[:idx, :-2].astype(np.float32), np.arange(
            idx)  # just a placeholder for the torch dataloader
        valid_ae_x, valid_ae_y = ae_data[idx:, :-2].astype(np.float32), np.arange(len(ae_data) - idx)
        all_train_ae_x, all_train_ae_y = np.vstack((all_train_ae_x, train_ae_x)), np.vstack((all_train_ae_y, train_ae_y))
        all_valid_ae_x, all_valid_ae_y = np.vstack((all_valid_ae_x, valid_ae_x)), np.vstack((all_valid_ae_y, valid_ae_y))

        print(f"size train: {train_ae_x.shape}, size valid: {valid_ae_x.shape}")
        print(f"size alltrain: {all_train_ae_x.shape}, size valid: {all_valid_ae_x.shape}")
        # AD training
        ae = AutoEncoder(train_x=train_ae_x, train_y=train_ae_y, valid_x=valid_ae_x,
                         valid_y=valid_ae_y)
        ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=0.0001, momentum=0.9), num_epochs=500)
        ae.determine_threshold()
        print(f"ae threshold: {ae.threshold}")
        ae.save_model(dir=f"offline_prototype_3_ds_as_sampling/", num=i)

    # train all after data AE
    ae = AutoEncoder(train_x=all_train_ae_x, train_y=all_train_ae_y, valid_x=all_valid_ae_x,
                     valid_y=all_valid_ae_y)
    ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=0.0001, momentum=0.9), num_epochs=500)
    ae.determine_threshold()
    print(f"ae threshold: {ae.threshold}")
    ae.save_model(dir="offline_prototype_3_ds_as_sampling/", num=i+1)

    # LOOP over all test data
    # for t in atrain:
    #     y_test = np.array([0 if t[0] == Behavior.NORMAL else 1] * len(atrain[t]))
    #         y_predicted = ae_interpreter.predict(atrain[t][:, :-2].astype(np.float32))
    #
    #         acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
    #         res_dict[t] = f'{(100 * acc):.2f}%'
    #     labels = ["Behavior", "MTD", "Accuracy"]
    #     results = []
    #     for t, a in res_dict.items():
    #         results.append([t[0].value, t[1].value, a])
    #     print(tabulate(results, headers=labels, tablefmt="pretty"))





    # Reinforcement Learning
    env = SensorEnvironment(decision_train_data=dtrain, decision_test_data=dtest,
                            after_train_data=atrain, after_test_data=atest, interpreter=ae_interpreter)

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

    agent.save_agent_state(0, "offline_prototype_2_raw_behaviors")

    x = [i + 1 for i in range(N_EPISODES)]
    filename = 'mtd_agent.pdf'
    plot_learning(x, episode_returns, eps_history, filename)

    # check predictions with dqn from trained and stored agent
    pretrained_state = torch.load("offline_prototype_2_raw_behaviors/trained_models/agent_0.pth")
    pretrained_agent = Agent(input_dims=15, n_actions=4, buffer_size=BUFFER_SIZE,
                             batch_size=pretrained_state['batch_size'], lr=pretrained_state['lr'],
                             gamma=pretrained_state['gamma'], epsilon=pretrained_state['eps'],
                             eps_end=pretrained_state['eps_min'], eps_dec=pretrained_state['eps_dec'])
    pretrained_agent.online_net.load_state_dict(pretrained_state['online_net_state_dict'])
    pretrained_agent.target_net.load_state_dict(pretrained_state['target_net_state_dict'])
    pretrained_agent.replay_buffer = pretrained_state['replay_buffer']

    pretrained_agent.online_net.eval()
    results = {}
    with torch.no_grad():
        for b, d in test_data.items():
            if b != Behavior.NORMAL:
                cnt_corr = 0
                cnt = 0
                for state in d:
                    action = pretrained_agent.take_greedy_action(state[:-1])
                    if b in supervisor_map[action]:
                        cnt_corr += 1
                    cnt += 1
                results[b] = (cnt_corr, cnt)

    print(results)
