from data_provider import DataProvider
from offline_prototype_3_ds_as_sampling.environment import SensorEnvironment
from agent import Agent
from custom_types import Behavior
from simulation_engine import SimulationEngine
from utils.evaluation_utils import plot_learning, seed_random, evaluate_agent, \
    evaluate_agent_on_afterstates, get_pretrained_agent
from utils.autoencoder_utils import get_pretrained_ae, split_as_data_for_ae_and_rl, \
    split_ds_data_for_ae_and_rl
from time import time
import numpy as np
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
    env = SensorEnvironment(decision_train_data=dtrain_rl,
                            after_train_data=atrain, interpreter=ae_interpreter,
                            state_samples=SAMPLES)

    agent = Agent(input_dims=env.observation_space_size, n_actions=len(env.actions), buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)

    # initialize memory replay buffer (randomly)
    SimulationEngine.init_replay_memory(agent=agent, env=env, min_size=MIN_REPLAY_SIZE)

    # main training
    episode_returns, eps_history = SimulationEngine.learn_agent_offline(agent=agent, env=env, num_episodes=N_EPISODES,
                                                                        t_update_freq=TARGET_UPDATE_FREQ)

    end = time()
    print("(Adapt!) Total training time: ", end - start)

    # save pretrained agent for later (online) use
    num = 0
    agent.save_agent_state(num, "offline_prototype_3_ds_as_sampling")

    x = [i + 1 for i in range(N_EPISODES)]
    filename = f'offline_prototype_3_ds_as_sampling/mtd_agent_p3_{SAMPLES}_samples.pdf'
    plot_learning(x, episode_returns, eps_history, filename)

    # check predictions with dqn from trained and stored agent
    path = f"offline_prototype_3_ds_as_sampling/trained_models/agent_{num}.pth"
    pretrained_agent = get_pretrained_agent(path=path, input_dims=env.observation_space_size,
                                            n_actions=len(env.actions), buffer_size=BUFFER_SIZE)
    evaluate_agent(agent=pretrained_agent, test_data=dtest)
    evaluate_agent_on_afterstates(agent=pretrained_agent, test_data=atest)
