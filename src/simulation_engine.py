import random
import numpy as np
from src.custom_types import Behavior
from src.agent import Agent


class SimulationEngine:
    # memory buffer is influenced by env.step -> resetting to previous action, which may result in unbalanced training
    @staticmethod
    def init_replay_memory(agent: Agent, env, min_size):
        obs = env.reset()
        episode_action_memory = []
        i = 0
        while i < min_size:
            try:
                action = np.random.choice(list({0,1,2,3}.difference(episode_action_memory)))
                episode_action_memory.append(action)
            except ValueError:
                obs = env.reset()
                episode_action_memory = []
                # results in slightly less entries than min_size
                print("exhausted all mtd techniques")
                continue
            i += 1

            new_obs, reward, done = env.step(action)
            idx1 = -1 if obs[0, -1] in Behavior else -2
            idx2 = -1 if new_obs[0, -1] in Behavior else -2
            transition = (obs[:, :idx1], action, reward, new_obs[:, :idx2], done)
            agent.replay_buffer.append(transition)

            obs = new_obs
            if done:
                obs = env.reset()
                episode_action_memory = []

    @staticmethod
    def learn_agent_offline(agent: Agent, env, num_episodes, t_update_freq):
        episode_returns, eps_history = [], []
        step = 0
        for i in range(num_episodes):
            episode_return = 0
            episode_steps = 0
            done = False
            obs = env.reset()
            while not done:
                idx1 = -1 if obs[0, -1] in Behavior else -2
                action = agent.choose_action(obs[:, :idx1])
                if action == -1:
                    print("Agent exhausted all MTD techniques upon behavior: ", obs[0, -1])
                    agent.episode_action_memory = set()
                    done = True
                    continue

                new_obs, reward, done = env.step(action)
                idx2 = -1 if new_obs[0, -1] in Behavior else -2
                episode_return += reward
                agent.replay_buffer.append((obs[:, :idx1], action, reward,
                                            new_obs[:, :idx2], done))
                agent.reward_buffer.append(reward)
                if done:
                    agent.episode_action_memory = set()

                agent.learn()
                obs = new_obs

                episode_steps += 1
                # update target network
                step += 1
                if step % t_update_freq == 0:
                    agent.update_target_network()

                # if step % LOG_FREQ == 0:
                # print("Episode: ", i, "Step: ", step, ", Avg Reward: ", np.mean(agent.reward_buffer), "epsilon: ", agent.epsilon)

            episode_returns.append(episode_return / episode_steps)
            avg_episode_return = np.mean(episode_returns[-10:])
            eps_history.append(agent.epsilon)

            print('episode ', i, '| episode_return %.2f' % episode_returns[-1],
                  '| average episode_return %.2f' % avg_episode_return,
                  '| epsilon %.2f' % agent.epsilon)
            if i >= num_episodes - 6:
                print(episode_returns[-10:])

        return episode_returns, eps_history
