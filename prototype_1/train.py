

from collections import defaultdict
from custom_types import MTDTechnique, Behavior


from collections import defaultdict
from custom_types import MTDTechnique, Behavior



import gym
from prototype_1 import Agent
from utils import plotLearning
import numpy as np


# define MTD - Attack Mapping
# TODO: Multiple attacks to same MTD, same attack to multiple MTD, i.e. Ransomware?
supervisor_map: Dict[Behavior, MTDTechnique] = defaultdict(lambda: MTDTechnique.NO_MTD, {
    Behavior.NORMAL: MTDTechnique.NO_MTD,
    Behavior.CNC_BACKDOOR_JAKORITAR: MTDTechnique.CNC_IP_SHUFFLE,
    Behavior.CNC_THETICK: MTDTechnique.CNC_IP_SHUFFLE,
    Behavior.ROOTKIT_BDVL: MTDTechnique.ROOTKIT_SANITIZER,
    Behavior.ROOTKIT_BEURK: MTDTechnique.ROOTKIT_SANITIZER,
    Behavior.RANSOMWARE_POC: MTDTechnique.RANSOMWARE_DIRTRAP
})



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    #plotLearning(x, scores, eps_history, filename)
