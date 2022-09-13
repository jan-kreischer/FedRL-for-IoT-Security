from data_provider import DataProvider
from offline_prototype_2_raw_behaviors.environment import SensorEnvironment
from agent import Agent
from custom_types import Behavior
from simulation_engine import SimulationEngine
from utils.evaluation_utils import plot_learning, seed_random, get_pretrained_agent, evaluate_agent, \
    evaluate_agent_on_afterstates
from utils.autoencoder_utils import evaluate_ae_on_no_mtd_behavior, get_pretrained_ae, pretrain_ae_model
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
N_EPISODES = 5000
LOG_FREQ = 100
DIMS = 20  # TODO check
PI = 3
SAMPLES = 10

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
    train_ae_x, valid_ae_x = pretrain_ae_model(ae_data=ae_data, path=ae_path, split=0.8, lr=1e-4, momentum=0.8,
                                              num_epochs=50)

    # AE evaluation of pretrained model
    ae_interpreter = get_pretrained_ae(path=ae_path, dims=DIMS)
    # AE can directly be tested on the data that will be used for RL: pass train_data to testing
    evaluate_ae_on_no_mtd_behavior(ae_interpreter=ae_interpreter, test_data=train_data)

    # Reinforcement Learning
    env = SensorEnvironment(train_data, interpreter=ae_interpreter, state_samples=SAMPLES)

    agent = Agent(input_dims=env.observation_space_size, n_actions=len(env.actions), buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)

    # initialize memory replay buffer (randomly)
    SimulationEngine.init_replay_memory(agent=agent, env=env, min_size=MIN_REPLAY_SIZE)

    # main training
    episode_returns, eps_history = SimulationEngine.learn_agent_offline(agent=agent, env=env, num_episodes=N_EPISODES,
                                                                        t_update_freq=TARGET_UPDATE_FREQ)

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

    evaluate_agent(pretrained_agent, test_data=test_data)

    print("evaluate p2 agent on 'real' decision and afterstate data:")
    dtrain, dtest, atrain, atest = DataProvider.get_reduced_dimensions_with_pca_ds_as(DIMS,
                                                                                      dir="offline_prototype_2_raw_behaviors/")
    evaluate_agent(agent=pretrained_agent, test_data=dtest)
    evaluate_agent_on_afterstates(agent=pretrained_agent, test_data=atest)
