from typing import Tuple, Any
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from agent import Agent
from custom_types import Behavior
from offline_prototype_1_raw_behaviors.environment import supervisor_map


def plot_learning(x, returns, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(returns)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(returns[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1", s=2**2)
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)


def seed_random():
    random.seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)


def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Any]:
    correct = np.count_nonzero(y_test == y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    cm_fed = confusion_matrix(y_test, y_pred)  # could also extract via tn, fp, fn, tp = confusion_matrix().ravel()
    return correct / len(y_pred), f1, cm_fed


def get_pretrained_agent(path, input_dims, n_actions, buffer_size):
    pretrained_state = torch.load(path)
    pretrained_agent = Agent(input_dims=input_dims, n_actions=n_actions, buffer_size=buffer_size,
                             batch_size=pretrained_state['batch_size'], lr=pretrained_state['lr'],
                             gamma=pretrained_state['gamma'], epsilon=pretrained_state['eps'],
                             eps_end=pretrained_state['eps_min'], eps_dec=pretrained_state['eps_dec'])
    pretrained_agent.online_net.load_state_dict(pretrained_state['online_net_state_dict'])
    pretrained_agent.target_net.load_state_dict(pretrained_state['target_net_state_dict'])
    pretrained_agent.replay_buffer = pretrained_state['replay_buffer']
    return pretrained_agent

def evaluate_agent(agent: Agent, test_data):
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