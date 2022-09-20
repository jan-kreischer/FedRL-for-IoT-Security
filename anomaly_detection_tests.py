from data_provider import DataProvider
from offline_prototype_3_ds_as_sampling.environment import SensorEnvironment
from agent import Agent
from custom_types import Behavior
from simulation_engine import SimulationEngine
from utils.evaluation_utils import plot_learning, seed_random, evaluate_agent, \
    evaluate_agent_on_afterstates, get_pretrained_agent
from utils.autoencoder_utils import get_pretrained_ae, split_as_data_for_ae_and_rl, \
    split_ds_data_for_ae_and_rl, evaluate_ae_on_afterstates, evaluate_ae_on_no_mtd_behavior, pretrain_ae_model, \
    pretrain_all_afterstate_ae_models, evaluate_all_as_ae_models, pretrain_all_ds_as_ae_models
from time import time
import numpy as np
import os
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils.evaluation_utils import calculate_metrics
from utils.autoencoder_utils import check_anomalous
from tabulate import tabulate

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
SAMPLES = 5

if __name__ == '__main__':
    # os.chdir("")
    seed_random()
    start = time()

    # read in all preprocessed data for a simulated, supervised environment to sample from
    # train_data, test_data, scaler = DataProvider.get_scaled_train_test_split()
    # train_data, test_data = DataProvider.get_reduced_dimensions_with_pca(DIMS)
    # dtrain, dtest, atrain, atest, scaler = DataProvider.get_scaled_scaled_train_test_split_with_afterstates()
    # dtrain, dtest, atrain, atest = DataProvider.get_reduced_dimensions_with_pca_ds_as(DIMS,
    #                                                                                   dir="offline_prototype_3_ds_as_sampling/")
    dtrain, dtest, atrain, atest, scaler = DataProvider.get_scaled_scaled_train_test_split_with_afterstates(scaling_minmax=True)

    # get splits for RL & AD of normal data
    dir = "offline_prototype_3_ds_as_sampling/trained_models/"
    model_name = "ae_model_ds.pth"
    path = dir + model_name
    ae_ds_train, dtrain_rl = split_ds_data_for_ae_and_rl(dtrain)
    ae_train_dict, atrain_rl = split_as_data_for_ae_and_rl(atrain)

    # fit diverse classifiers and test them

    # LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=40, novelty=True, contamination=0.1)
    clf.fit(ae_ds_train[:, :-1])

    # IsolationForest
    # clf = IsolationForest(n_estimators=15, random_state=0)
    # clf.fit(ae_ds_train[:, :-1])  # fit 15 trees
    #y_pred = clf.predict(ae_ds_train[:, :-1])
    #print(y_pred[y_pred == -1].size)

    # One-Class SVM for novelty detection
    #clf = svm.OneClassSVM(nu=0.35, kernel="rbf", gamma="scale")
    #clf.fit(ae_ds_train[:, :-1])
    #
    # # Evaluate on all behaviors:
    # print("Evaluate OneClassSVM trained on ds normal")
    # # n_error_train = y_pred_train[y_pred_train == -1].size
    res_dict = {}
    for b, d in dtrain_rl.items():

        y_test = np.array([1 if b == Behavior.NORMAL else -1] * len(d))
        y_predicted = clf.predict(d[:, :-1].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten())
        res_dict[b] = f'{(100 * acc):.2f}%'

    labels = ["Behavior"] + ["Accuracy"]
    results = []
    for b, a in res_dict.items():
        results.append([b.value, res_dict[b]])
    print(tabulate(results, headers=labels, tablefmt="pretty"))

    res_dict = {}
    for t in atrain_rl:
        isAnomaly = check_anomalous(t[0], t[1])
        y_test = np.array([-1 if isAnomaly else 1] * len(atrain_rl[t]))
        y_predicted = clf.predict(atrain_rl[t][:, :-2].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten())
        res_dict[t] = f'{(100 * acc):.2f}%'
    labels = ["Behavior", "MTD", "Accuracy"]
    results = []
    for t, a in res_dict.items():
        results.append([t[0].value, t[1].value, a])
    print(tabulate(results, headers=labels, tablefmt="pretty"))

