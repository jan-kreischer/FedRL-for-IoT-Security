from typing import Tuple, Any
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tabulate import tabulate
from math import ceil
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from agent import Agent
from custom_types import Behavior, MTDTechnique, actions, supervisor_map, normal_afterstates


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

    ax2.scatter(x, running_avg, color="C1", s=2 ** 2)
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
    res_dict = {}
    objective_dict = {}
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
                res_dict[b] = (cnt_corr, cnt)

            for i in range(len(actions)):
                if b in supervisor_map[i]:
                    objective_dict[b] = actions[i]

    print(res_dict)
    labels = ["Behavior", "Accuracy", "Objective"]
    results = []
    for b, t in res_dict.items():
        results.append([b.value, f'{(100 * t[0] / t[1]):.2f}%', objective_dict[b].value])
    print(tabulate(results, headers=labels, tablefmt="latex"))


def evaluate_agent_on_afterstates(agent: Agent, test_data):
    agent.online_net.eval()
    res_dict = {}
    objectives_dict = {}
    with torch.no_grad():
        for t, d in test_data.items():
            if t[0] != Behavior.NORMAL and t not in normal_afterstates:
                # every MTD technique is both correct and incorrect for normal afterstates
                # so it doesnt make sense to evaluate the agent on these states as they correspond to false positives
                # flagged by the autoencoder.
                cnt_corr = 0
                cnt = 0
                for state in d:
                    action = agent.take_greedy_action(state[:-2])
                    if t[0] in supervisor_map[action]:
                        cnt_corr += 1
                    cnt += 1
                res_dict[t] = (cnt_corr, cnt)
            for i in range(len(actions)):
                if t[0] in supervisor_map[i]:
                    objectives_dict[t] = actions[i]

    labels = ["Behavior", "MTD", "Accuracy", "Objective"]
    results = []
    for t, cs in res_dict.items():
        results.append([t[0].value, t[1].value, f'{(100 * cs[0] / cs[1]):.2f}%', objectives_dict[t].value])
    print(tabulate(results, headers=labels, tablefmt="latex"))


def evaluate_anomaly_detector_ds(dtrain, clf):
    res_dict = {}
    for b, d in dtrain.items():
        y_test = np.array([1 if b == Behavior.NORMAL else -1] * len(d))
        y_predicted = clf.predict(d[:, :-1].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten())
        res_dict[b] = f'{(100 * acc):.2f}%'
    labels = ["Behavior", "Accuracy"]
    results = []
    for b, a in res_dict.items():
        results.append([b.value, a])
    print(tabulate(results, headers=labels, tablefmt="pretty"))


def evaluate_anomaly_detector_as(atrain, clf):
    res_dict = {}
    for t in atrain:
        isAnomaly = check_anomalous(t[0], t[1])
        y_test = np.array([-1 if isAnomaly else 1] * len(atrain[t]))
        y_predicted = clf.predict(atrain[t][:, :-2].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten())
        res_dict[t] = f'{(100 * acc):.2f}%, {"anomaly" if isAnomaly else "normal"}'
    labels = ["Behavior", "MTD", "Accuracy", "Objective"]
    results = []
    for t, a in res_dict.items():
        res = a.split(",")
        results.append([t[0].value, t[1].value, res[0], res[1]])
    print(tabulate(results, headers=labels, tablefmt="pretty"))


def check_anomalous(b: Behavior, m: MTDTechnique):
    if b == Behavior.NORMAL:
        return 0
    if (b == Behavior.ROOTKIT_BDVL or b == Behavior.ROOTKIT_BEURK) and m == MTDTechnique.ROOTKIT_SANITIZER:
        return 0
    if b == Behavior.RANSOMWARE_POC and (
            m == MTDTechnique.RANSOMWARE_DIRTRAP or m == MTDTechnique.RANSOMWARE_FILE_EXT_HIDE):
        return 0
    if (b == Behavior.CNC_BACKDOOR_JAKORITAR or b == Behavior.CNC_THETICK or
        b == Behavior.CNC_OPT1 or b == Behavior.CNC_OPT2) and m == MTDTechnique.CNC_IP_SHUFFLE:
        return 0
    return 1


def plot_state_samples_upper_binom_cdf():
    ns = [i + 1 for i in range(2, 100, 2)]
    ks_half = [ceil(n / 2) - 1 for n in ns]
    for p, c in zip([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], ["c", "m", "y", "b", "g", "r", "k"]):
        upper_bcdf = [1 - binom.cdf(k=ks_half[i], n=ns[i], p=p) for i in range(len(ns))]
        plt.plot(ns, upper_bcdf, f"-{c}", label=f"Accuracy p={p}")
    plt.ylabel("Probability for more than 50% correct predictions")
    plt.xlabel("Nr of trials (n)")
    plt.title("Number of Samples and Anomaly Detection Success")#fontsize='xx-large')
    plt.legend()
    plt.savefig("upper_binom_cdf_tests.pdf")
