import os

from data_provider import DataProvider
from offline_prototype_1_raw_behaviors.environment import SensorEnvironment
from agent import Agent
from simulation_engine import SimulationEngine
from utils.evaluation_utils import plot_learning, seed_random, evaluate_agent, get_pretrained_agent, evaluate_agent_on_afterstates
from time import time
import numpy as np

# Hyperparams
GAMMA = 0.99
BATCH_SIZE = 100
BUFFER_SIZE = 500
MIN_REPLAY_SIZE = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
TARGET_UPDATE_FREQ = 100
LEARNING_RATE = 1e-5
N_EPISODES = 6500
LOG_FREQ = 100
DIMS = 20
PI = 3

if __name__ == '__main__':
    os.chdir("..")
    seed_random()
    start = time()

    # read in all preprocessed data for a simulated, supervised environment to sample from
    #train_data, test_data, scaler = DataProvider.get_scaled_train_test_split()
    train_data, test_data = DataProvider.get_reduced_dimensions_with_pca(DIMS, pi=PI, normal_only=False)
    env = SensorEnvironment(train_data)

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
    agent.save_agent_state(0, "offline_prototype_1_raw_behaviors")

    x = [i + 1 for i in range(N_EPISODES)]
    filename = 'offline_prototype_1_raw_behaviors/mtd_agent_p1.pdf'
    plot_learning(x, episode_returns, eps_history, filename)

    # check predictions with dqn from trained and stored agent
    pretrained_agent = get_pretrained_agent(path=f"offline_prototype_1_raw_behaviors/trained_models/agent_{num}.pth",
                                            input_dims=env.observation_space_size, n_actions=len(env.actions),
                                            buffer_size=BUFFER_SIZE)
    # check predictions with learnt dqn
    evaluate_agent(pretrained_agent, test_data=test_data)

    # TODO check scaling/how it can be evaluated better
    print("evaluate p1 agent on 'real' decision and afterstate data:")
    dtrain, dtest, atrain, atest = DataProvider.get_reduced_dimensions_with_pca_ds_as(DIMS,
                                                                                      dir="offline_prototype_1_raw_behaviors/")
    evaluate_agent(agent=pretrained_agent, test_data=dtest)
    evaluate_agent_on_afterstates(agent=pretrained_agent, test_data=atest)






