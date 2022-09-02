import random
from custom_types import Behavior


def init_replay_memory(agent, env, min_size):
    obs = env.reset()
    for _ in range(min_size):
        action = random.choice(env.actions)

        new_obs, reward, done = env.step(action)
        idx1 = -1 if obs[0, -1] in Behavior else -2
        idx2 = -1 if new_obs[0, -1] in Behavior else -2
        transition = (obs[:, :idx1], action, reward, new_obs[:, :idx2], done)
        agent.replay_buffer.append(transition)

        obs = new_obs
        if done:
            obs = env.reset()