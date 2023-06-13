import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.experiment import Experiment
from src.agent import Agent
from src.deep_q_network import DeepQNetwork
from src.server import Server
from src.client import Client
from src.client import Environment
from src.autoencoder import AutoEncoder
from src.custom_types import Behavior, Execution, Evaluation, MTDTechnique, actions, mitigated_by
import torch 
import torch.nn as nn 

'''
    This function computes the weighted cosine similarity 
    for a numpy 2d matrix
    For each client one row is added to the matrix
    Each row contains the sample frequency for each label as columns
'''
def weighted_cosine_similarity(lis):
    L = np.sum(lis, axis=0)
    L_one_norm = norm(L, 1)
    L_two_norm = norm(L, 2)
    #n = lis.shape[0]

    cosine_similarity = 0
    for li in lis:
        li_one_norm = norm(li, 1)
        li_two_norm = norm(li, 2)
        cosine_similarity +=  (li_one_norm/li_two_norm) * np.dot(L,li) 

    weighted_cosine_similarity = 1/(L_one_norm * L_two_norm)*cosine_similarity
    return round(weighted_cosine_similarity, 4)

'''
    This function computes the mean cosine similarity 
    for a numpy 2d matrix
    For each client one row is added to the matrix
    Each row contains the sample frequency for each label as columns
'''
def mean_cosine_similarity(lis):
    L = np.sum(lis, axis=0)
    L_two_norm = norm(L, 2)
    n = lis.shape[0]

    mean_cosine_similarity = 0
    for li in lis:
        mean_cosine_similarity += (1 / n) * (np.dot(L,li) / (L_two_norm * norm(li, 2)))

    return mean_cosine_similarity

'''
'''
def multiclass_imbalance_degree(M):
    N = np.sum(M)
    n_cs = np.sum(M, axis=0)

    C = len(n_cs)
    MID = 0
    for n_c in n_cs:
        relative_label_frequency =  n_c / N
        MID+=relative_label_frequency*np.emath.logn(C, C*relative_label_frequency)   
    return round(MID, 4)

'''
'''
def calculate_balance_metrics(sampling_probability_1, sampling_probabiliy_2, N=1000):
    NR_SAMPLES_1 = np.array(list(sampling_probability_1.values()))*N
    NR_SAMPLES_2 = np.array(list(sampling_probabiliy_2.values()))*N

    sample_matrix = np.vstack([NR_SAMPLES_1, NR_SAMPLES_2])
    MID = multiclass_imbalance_degree(sample_matrix)
    WCS = weighted_cosine_similarity(sample_matrix)
    print(f"Dataset Balance Metrics: MID={MID} & WCS={WCS}")
    return MID, WCS

def split_training_data(training_data, n_strides):
    strides = []
    for i in range(n_strides):
        strides.append(dict())

    for key, value in training_data.items():
        #print(f"{key} => {len(value)}")
        array_split = np.array_split(value, n_strides)
        for i in range(n_strides):
            strides[i][key] = array_split[i]
        
    return strides

def split_data(data, split=0.8):
    row = int(len(data) * split)
    X_train = data[:row, :-1].astype(np.float32)
    X_valid = data[row:, :-1].astype(np.float32)
    return X_train, X_valid


def plot_test_accuracies(final_training_accuracies, final_mean_class_accuracies, title, xlabel, loc):
    plt.figure(figsize=(9,6))
    xlabel = "Multiclass Imbalance Degree (MID)"
    plt.xlabel(xlabel)
    ylabel = "Test-Accuracy [%]"
    plt.ylabel(ylabel)

    plt.ylim([0, 101])
    plt.yticks(np.arange(0, 101, 10))
    
    plt.plot(list(final_training_accuracies.keys()), np.array(list(final_training_accuracies.values()))*100, marker='o', linestyle='solid', color='red', label='Micro Accuracy')
    plt.plot(list(final_mean_class_accuracies.keys()), np.array(list(final_mean_class_accuracies.values()))*100, marker='o', linestyle='dashed', color='blue', label='Macro Accuracy')
    
    plt.legend(loc=loc)
    plt.title(f"Final Test Accuracy for different Globally Imbalanced but Locally Balanced Data-Splits")
    
    plt.show()
   

def plot_mid_sweep_test_accuracies(final_training_accuracies, final_mean_class_accuracies):
    plot_test_accuracies(final_training_accuracies, final_mean_class_accuracies, "Final Test Accuracy for different Globally Imbalanced but Locally Balanced Data-Splits", "Multiclass Imbalance Degree (MID)", "lower left")
    
    
def plot_wcs_sweep_test_accuracies(final_training_accuracies, final_mean_class_accuracies):
    plot_test_accuracies(final_training_accuracies, final_mean_class_accuracies, "Final Test Accuracy for different Locally Imbalanced but Globally Balanced Data-Splits", "Weighted Cosine Similarity (WCS)", "lower right")

def convert_grid_search_result(grid_search_result, display_latex=True):
    df = pd.concat([pd.DataFrame(grid_search_result.cv_results_["mean_test_score"], columns=["mean_validation_accuracy"]), pd.DataFrame(grid_search_result.cv_results_["params"])],axis=1).sort_values(by=['mean_validation_accuracy'], ascending=False)
    df.index = np.arange(1, len(df) + 1)
    return df

def display_grid_search_result(df, n_items=10):
    print(df.head(n_items).to_latex(escape=True, bold_rows=True))
    
def evaluate(model, data_dict, tablefmt='latex_raw'):
        results = []
        labels= [-1,1]
        pos_label = 1
        
        y_true_total = np.empty([0])
        y_pred_total = np.empty([0])
        for behavior, data in data_dict.items():
            y_true = data[:,-1].astype(int)
            y_true_total = np.concatenate((y_true_total, y_true))

            y_pred = model.predict(data[:, :-1].astype(np.float32))
            y_pred_total = np.concatenate((y_pred_total, y_pred))

            accuracy = accuracy_score(y_true, y_pred)

            n_samples = len(y_true)
            results.append([behavior.name.replace("_", "\_"), f'{(100 * accuracy):.2f}\%', '\\notCalculated', '\\notCalculated', '\\notCalculated', str(n_samples)])

        accuracy = accuracy_score(y_true_total, y_pred_total)
        precision = precision_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        recall = recall_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        f1 = f1_score(y_true_total, y_pred_total, average='binary', labels=labels, pos_label=pos_label, zero_division=1)
        n_samples = len(y_true_total)
        results.append(["GLOBAL", f'{(100 * accuracy):.2f}\%', f'{(100 * precision):.2f}\%', f'{(100 * recall):.2f}\%', f'{(100 * f1):.2f}\%', n_samples])
        print(tabulate(results, headers=["Behavior", "Accuracy", "Precision", "Recall", "F1-Score", "\\#Samples"], tablefmt=tablefmt)) 
        
        
def get_training_datasets(training_data_dict, normal_label, abnormal_label):
    training_data_50 = np.empty([0,47])
    training_data_80 = np.empty([0,47])
    for behavior, behavior_data in training_data_dict.items():

        if behavior == Behavior.NORMAL:
            behavior_data[:, -1] =  normal_label # SVM uses 1 for normal
            training_data_50 = np.concatenate([training_data_50, behavior_data[:7000,:]], axis=0)
            training_data_80 = np.concatenate([training_data_80, behavior_data[:8000,:]], axis=0)
        else:
            behavior_data[:, -1] =  abnormal_label # SVM uses -1 for outlier
            training_data_50 = np.concatenate([training_data_50, behavior_data[:1000,:]], axis=0)
            training_data_80 = np.concatenate([training_data_80, behavior_data[:286,:]], axis=0)

    return training_data_50, training_data_80

import numpy as np

def get_test_datasets(test_data, normal_label, abnormal_label):
    test_data_dict = {}
    test_data_flat = np.zeros([0, 47])

    for behavior, behavior_data in test_data.items():
        if behavior == Behavior.NORMAL:
            behavior_data = behavior_data[:2800]
            behavior_data[:, -1] =  1
        else:
            behavior_data = behavior_data[:400]
            behavior_data[:, -1] =  -1

        test_data_dict[behavior] = behavior_data
        test_data_flat = np.vstack([test_data_flat, behavior_data])

    return test_data_dict, test_data_flat


def run_sample_distribution_experiment(training_data_strides, 
                                       test_data, 
                                       sampling_probabilities_01, 
                                       sampling_probabilities_02,
                                       state_interpreter: AutoEncoder=None,
                                       nr_rounds=30,
                                       nr_episodes_per_round=100,
                                       evaluate_local_clients=False):
    experiment = Experiment()

    # HYPERPARAMETER for State Anomaly Detector
    N_STD = 2.5

    # HYPERPARAMETER for Federated Learning
    NR_CLIENTS = len(training_data_strides)
    NR_ROUNDS = nr_rounds
    NR_EPISODES_PER_ROUND = nr_episodes_per_round
    NR_EPISODES_PER_CLIENT = NR_ROUNDS * NR_EPISODES_PER_ROUND

    # DQN Architecture Hyperparameter
    N_ACTIONS = len(MTDTechnique)
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 64
    N_HIDDEN_3 = 32
    N_FEATURES = training_data_strides[0][Behavior.NORMAL].shape[1] - 1

    # DQN Training Hyperparameter
    GAMMA = 0.5 #0.99 # discount factor for future rewards
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = NR_EPISODES_PER_ROUND - 1
    BUFFER_SIZE = 500
    MIN_REPLAY_SIZE = 100

    #SCHEDULER = 
    LOSS=nn.MSELoss()
    L2 = 0.01


    # HYPERPARAMETER for Epsilon Delta Strategy
    EPSILON_START = 1.0
    EPSILON_DEC = 1/(NR_EPISODES_PER_CLIENT*0.8)
    EPSILON_END = 0.01


    calculate_balance_metrics(sampling_probabilities_01, sampling_probabilities_02)

    deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
    global_agent = Agent(0, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=None, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
    server = Server(global_agent, test_data, state_interpreter)

    clients = []
    for i in range(NR_CLIENTS):
        entity_id = i + 1
        if entity_id % 2 == 0:
            SAMPLING_PROBABILITIES = sampling_probabilities_01
        else:
            SAMPLING_PROBABILITIES = sampling_probabilities_02
            
        environment = Environment(entity_id, training_data_strides[i], state_interpreter=state_interpreter, sampling_probabilities=SAMPLING_PROBABILITIES , verbose=False)
        deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
        optimizer = torch.optim.Adam(deep_q_network.parameters(), lr=LR,  weight_decay=L2, amsgrad=True)
        agent = Agent(entity_id, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
        client = Client(entity_id, agent, environment, MIN_REPLAY_SIZE)
        server.add_client(client)

    evaluations = [Evaluation.GLOBAL_PERFORMANCE_EVALUATION]
    if evaluate_local_clients:
        print("Evaluating local clients as well")
        evaluations.append(Evaluation.LOCAL_PERFORMANCE_EVALUATION)
    experiment.execute(server, nr_rounds=NR_ROUNDS, nr_episodes_per_round=NR_EPISODES_PER_ROUND, evaluations=evaluations, evaluation_frequency=1)
    return experiment

'''
# Ex 1.3
def get_mid_sweep_sampling_probabilities(current_step, n_steps):
    lower_sampling_threshold = 0.0001
    unit_sampling_probability = 1/7
    
    fraction = (unit_sampling_probability - lower_sampling_threshold)/n_steps
    offset = current_step * fraction
    
    minor_sampling_probability = unit_sampling_probability - offset
    major_sampling_probability = unit_sampling_probability + 6 * offset

    sampling_probabilities = {
        Behavior.RANSOMWARE_POC: major_sampling_probability,
        Behavior.ROOTKIT_BDVL: minor_sampling_probability,
        Behavior.ROOTKIT_BEURK: minor_sampling_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: minor_sampling_probability ,
        Behavior.CNC_THETICK: minor_sampling_probability,
        Behavior.CNC_OPT1: minor_sampling_probability,
        Behavior.CNC_OPT2: minor_sampling_probability,
    }
    return sampling_probabilities

# Ex 1.3
def run_mid_sweep_experiment(experiment_id, experiment_version, training_data_strides, n_steps):
    experiment = Experiment(base_path=original_working_directory_path, experiment_id=experiment_id, experiment_version=experiment_version)
    experiment_path = experiment.get_experiment_path()

    # HYPERPARAMETER for Federated Learning
    NR_CLIENTS = len(training_data_strides)
    NR_ROUNDS = 30
    NR_EPISODES_PER_ROUND = 100
    NR_EPISODES_PER_CLIENT = NR_ROUNDS * NR_EPISODES_PER_ROUND

    # DQN Architecture Hyperparameter
    N_ACTIONS = len(MTDTechnique)
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 64
    N_HIDDEN_3 = 32
    N_FEATURES = rl_training_data_dict[Behavior.NORMAL].shape[1] -1

    # DQN Training Hyperparameter
    GAMMA = 0.5 #0.99 # discount factor for future rewards
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = NR_EPISODES_PER_ROUND - 1
    BUFFER_SIZE = 500
    MIN_REPLAY_SIZE = 100

    #SCHEDULER = 
    LOSS=nn.MSELoss()
    L2 = 0.01

    # HYPERPARAMETER for Epsilon Delta Strategy
    EPSILON_START = 1.0
    EPSILON_DEC = 1/(NR_EPISODES_PER_CLIENT*0.8)
    EPSILON_END = 0.01

    final_micro_accuracies = {}
    final_macro_accuracies = {}
    
    for current_step in range(n_steps+1):
        print(f"=== Running MID Sweep Split {current_step}/{n_steps} ===")
        sampling_probabilities = get_mid_sweep_sampling_probabilities(current_step, n_steps)
        MID, WCS = calculate_balance_metrics(sampling_probabilities, sampling_probabilities, NR_EPISODES_PER_CLIENT)
        assert round(WCS, 4) == 1, "WCS has to be 1"
        
        deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
        global_agent = Agent(0, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=None, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
        server = Server(global_agent, rl_test_data_dict, experiment_id, experiment_path)

        clients = []
        for i in range(NR_CLIENTS):
            entity_id = i + 1
            environment = Environment(entity_id, training_data_strides[i], sampling_probabilities=sampling_probabilities, verbose=False)
            deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
            optimizer = torch.optim.Adam(deep_q_network.parameters(), lr=LR,  weight_decay=L2, amsgrad=True)
            agent = Agent(entity_id, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
            client = Client(entity_id, agent, environment, experiment_path, MIN_REPLAY_SIZE)
            server.add_client(client)


        evaluations = [Evaluation.FINAL_PERFORMANCE_EVALUATION]
        #server.run_federation(evaluations=evaluations, evaluation_frequency=1)
        experiment.execute(server, nr_rounds=NR_ROUNDS, nr_episodes_per_round=NR_EPISODES_PER_ROUND, evaluations=evaluations, evaluation_frequency=1, verbose=False)
        micro_accuracy, macro_accuracy = server.final_training_accuracy()
        final_micro_accuracies[MID] = micro_accuracy
        final_macro_accuracies[MID] = macro_accuracy
    return experiment, final_micro_accuracies, final_macro_accuracies


# Ex 1.3
def get_wcs_sweep_sampling_probabilities(current_step, n_steps):
    baseline_probability  = 1/7
    lower_sampling_threshold = 0.0001

    fraction = (baseline_probability - lower_sampling_threshold)/n_steps
    baseline_deviation = current_step * fraction
    
    inc_probability = baseline_probability + baseline_deviation
    dec_probability = baseline_probability - baseline_deviation


    sampling_probabilities_1 = {
        #Behavior.NORMAL: 0,
        Behavior.RANSOMWARE_POC: inc_probability,
        Behavior.ROOTKIT_BDVL: inc_probability,
        Behavior.ROOTKIT_BEURK: inc_probability,
        Behavior.CNC_THETICK: baseline_probability, 
        Behavior.CNC_OPT1: dec_probability,
        Behavior.CNC_OPT2: dec_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: dec_probability,
    }

    sampling_probabilities_2 = {
        #Behavior.NORMAL: 0,
        Behavior.RANSOMWARE_POC: dec_probability, 
        Behavior.ROOTKIT_BDVL: dec_probability,
        Behavior.ROOTKIT_BEURK: dec_probability,
        Behavior.CNC_THETICK: baseline_probability, 
        Behavior.CNC_OPT1: inc_probability,
        Behavior.CNC_OPT2: inc_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: inc_probability,
    }
    
    return sampling_probabilities_1, sampling_probabilities_2


# Ex 1.3
def run_wcs_sweep_experiment(experiment_id, experiment_version, training_data_strides, n_steps):
    experiment = Experiment(base_path=original_working_directory_path, experiment_id=experiment_id, experiment_version=experiment_version)
    experiment_path = experiment.get_experiment_path()

    # HYPERPARAMETER for Federated Learning
    NR_CLIENTS = len(training_data_strides)
    NR_ROUNDS = 30
    NR_EPISODES_PER_ROUND = 100
    NR_EPISODES_PER_CLIENT = NR_ROUNDS * NR_EPISODES_PER_ROUND

    # DQN Architecture Hyperparameter
    N_ACTIONS = len(MTDTechnique)
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 64
    N_HIDDEN_3 = 32
    N_FEATURES = rl_training_data_dict[Behavior.NORMAL].shape[1] -1

    # DQN Training Hyperparameter
    GAMMA = 0.5 #0.99 # discount factor for future rewards
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = NR_EPISODES_PER_ROUND - 1
    BUFFER_SIZE = 500
    MIN_REPLAY_SIZE = 100

    #SCHEDULER = 
    LOSS=nn.MSELoss()
    L2 = 0.01

    # HYPERPARAMETER for Epsilon Delta Strategy
    EPSILON_START = 1.0
    EPSILON_DEC = 1/(NR_EPISODES_PER_CLIENT*0.8)
    EPSILON_END = 0.01

    final_micro_accuracies = {}
    final_macro_accuracies = {}
    
    for current_step in range(n_steps+1):
        print(f"=== Running MID Sweep Split {current_step}/{n_steps} ===")
        sampling_probabilities_01, sampling_probabilities_02 = get_wcs_sweep_sampling_probabilities(current_step, n_steps)
        MID, WCS = calculate_balance_metrics(sampling_probabilities_01, sampling_probabilities_02)
        assert round(MID, 4) == 0, "MID has to be 1"
        
        deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
        global_agent = Agent(0, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=None, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
        server = Server(global_agent, rl_test_data_dict, experiment_id, experiment_path)

        clients = []
        for i in range(NR_CLIENTS):
            entity_id = i + 1
            if entity_id % 2 == 0:
                sampling_probabilities = sampling_probabilities_01
            else:
                sampling_probabilities = sampling_probabilities_02
            environment = Environment(entity_id, training_data_strides[i], sampling_probabilities=sampling_probabilities, verbose=False)
            deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
            optimizer = torch.optim.Adam(deep_q_network.parameters(), lr=LR,  weight_decay=L2, amsgrad=True)
            agent = Agent(entity_id, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
            client = Client(entity_id, agent, environment, experiment_path, MIN_REPLAY_SIZE)
            server.add_client(client)


        evaluations = [Evaluation.FINAL_PERFORMANCE_EVALUATION]
        #server.run_federation(evaluations=evaluations, evaluation_frequency=1)
        experiment.execute(server, nr_rounds=NR_ROUNDS, nr_episodes_per_round=NR_EPISODES_PER_ROUND, evaluations=evaluations, evaluation_frequency=1, verbose=False)
        micro_accuracy, macro_accuracy = server.final_training_accuracy()
        final_micro_accuracies[WCS] = micro_accuracy
        final_macro_accuracies[WCS] = macro_accuracy
    return experiment, final_micro_accuracies, final_macro_accuracies

# Ex 2.2
def get_mid_sweep_sampling_probabilities(current_step, n_steps):
    lower_sampling_threshold = 0.0001
    unit_sampling_probability = 1/7
    
    fraction = (unit_sampling_probability - lower_sampling_threshold)/n_steps
    offset = current_step * fraction
    
    minor_sampling_probability = unit_sampling_probability - offset
    major_sampling_probability = unit_sampling_probability + 6 * offset

    sampling_probabilities = {
        Behavior.RANSOMWARE_POC: major_sampling_probability,
        Behavior.ROOTKIT_BDVL: minor_sampling_probability,
        Behavior.ROOTKIT_BEURK: minor_sampling_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: minor_sampling_probability ,
        Behavior.CNC_THETICK: minor_sampling_probability,
        Behavior.CNC_OPT1: minor_sampling_probability,
        Behavior.CNC_OPT2: minor_sampling_probability,
    }
    return sampling_probabilities

# Ex 2.2
def run_mid_sweep_experiment(experiment_id, experiment_version, training_data_strides, n_steps):
    experiment = Experiment(base_path=original_working_directory_path, experiment_id=experiment_id, experiment_version=experiment_version)
    experiment_path = experiment.get_experiment_path()

    # HYPERPARAMETER for Federated Learning
    NR_CLIENTS = len(training_data_strides)
    NR_ROUNDS = 30
    NR_EPISODES_PER_ROUND = 100
    NR_EPISODES_PER_CLIENT = NR_ROUNDS * NR_EPISODES_PER_ROUND

    # DQN Architecture Hyperparameter
    N_ACTIONS = len(MTDTechnique)
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 64
    N_HIDDEN_3 = 32
    N_FEATURES = training_data_strides[0][Behavior.NORMAL].shape[1] -1

    # DQN Training Hyperparameter
    GAMMA = 0.5 #0.99 # discount factor for future rewards
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = NR_EPISODES_PER_ROUND - 1
    BUFFER_SIZE = 500
    MIN_REPLAY_SIZE = 100

    #SCHEDULER = 
    LOSS=nn.MSELoss()
    L2 = 0.01

    # HYPERPARAMETER for Epsilon Delta Strategy
    EPSILON_START = 1.0
    EPSILON_DEC = 1/(NR_EPISODES_PER_CLIENT*0.8)
    EPSILON_END = 0.01

    final_micro_accuracies = {}
    final_macro_accuracies = {}
    
    for current_step in range(n_steps+1):
        print(f"=== Running MID Sweep Split {current_step}/{n_steps} ===")
        sampling_probabilities = get_mid_sweep_sampling_probabilities(current_step, n_steps)
        MID, WCS = calculate_balance_metrics(sampling_probabilities, sampling_probabilities, NR_EPISODES_PER_CLIENT)
        assert round(WCS, 4) == 1, "WCS has to be 1"
        
        deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
        global_agent = Agent(0, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=None, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
        server = Server(global_agent, rl_test_data, autoencoder, experiment_id, experiment_path)

        clients = []
        for i in range(NR_CLIENTS):
            entity_id = i + 1
            environment = Environment(entity_id, training_data_strides[i], state_interpreter=autoencoder, sampling_probabilities=sampling_probabilities, verbose=False)
            deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
            optimizer = torch.optim.Adam(deep_q_network.parameters(), lr=LR,  weight_decay=L2, amsgrad=True)
            agent = Agent(entity_id, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
            client = Client(entity_id, agent, environment, experiment_path, MIN_REPLAY_SIZE)
            server.add_client(client)

        evaluations = [Evaluation.FINAL_PERFORMANCE_EVALUATION]
        experiment.execute(server, nr_rounds=NR_ROUNDS, nr_episodes_per_round=NR_EPISODES_PER_ROUND, evaluations=evaluations, evaluation_frequency=1, verbose=False)
        micro_accuracy, macro_accuracy = server.final_training_accuracy()
        final_micro_accuracies[MID] = micro_accuracy
        final_macro_accuracies[MID] = macro_accuracy
    return experiment, final_micro_accuracies, final_macro_accuracies

# Ex 2.2
def get_wcs_sweep_sampling_probabilities(current_step, n_steps):
    baseline_probability  = 1/7
    lower_sampling_threshold = 0.0001

    fraction = (baseline_probability - lower_sampling_threshold)/n_steps
    baseline_deviation = current_step * fraction
    
    inc_probability = baseline_probability + baseline_deviation
    dec_probability = baseline_probability - baseline_deviation


    sampling_probabilities_1 = {
        #Behavior.NORMAL: 0,
        Behavior.RANSOMWARE_POC: inc_probability,
        Behavior.ROOTKIT_BDVL: inc_probability,
        Behavior.ROOTKIT_BEURK: inc_probability,
        Behavior.CNC_THETICK: baseline_probability, 
        Behavior.CNC_OPT1: dec_probability,
        Behavior.CNC_OPT2: dec_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: dec_probability,
    }

    sampling_probabilities_2 = {
        #Behavior.NORMAL: 0,
        Behavior.RANSOMWARE_POC: dec_probability, 
        Behavior.ROOTKIT_BDVL: dec_probability,
        Behavior.ROOTKIT_BEURK: dec_probability,
        Behavior.CNC_THETICK: baseline_probability, 
        Behavior.CNC_OPT1: inc_probability,
        Behavior.CNC_OPT2: inc_probability,
        Behavior.CNC_BACKDOOR_JAKORITAR: inc_probability,
    }
    
    return sampling_probabilities_1, sampling_probabilities_2

# Ex 2.2
def run_wcs_sweep_experiment(experiment_id, experiment_version, training_data_strides, n_steps):
    experiment = Experiment(base_path=original_working_directory_path, experiment_id=experiment_id, experiment_version=experiment_version)
    experiment_path = experiment.get_experiment_path()

    # HYPERPARAMETER for Federated Learning
    NR_CLIENTS = len(training_data_strides)
    NR_ROUNDS = 30
    NR_EPISODES_PER_ROUND = 100
    NR_EPISODES_PER_CLIENT = NR_ROUNDS * NR_EPISODES_PER_ROUND

    # DQN Architecture Hyperparameter
    N_ACTIONS = len(MTDTechnique)
    N_HIDDEN_1 = 128
    N_HIDDEN_2 = 64
    N_HIDDEN_3 = 32
    N_FEATURES = rl_training_data[Behavior.NORMAL].shape[1] -1

    # DQN Training Hyperparameter
    GAMMA = 0.5 #0.99 # discount factor for future rewards
    LR = 1e-3
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = NR_EPISODES_PER_ROUND - 1
    BUFFER_SIZE = 500
    MIN_REPLAY_SIZE = 100

    #SCHEDULER = 
    LOSS=nn.MSELoss()
    L2 = 0.01

    # HYPERPARAMETER for Epsilon Delta Strategy
    EPSILON_START = 1.0
    EPSILON_DEC = 1/(NR_EPISODES_PER_CLIENT*0.8)
    EPSILON_END = 0.01

    final_micro_accuracies = {}
    final_macro_accuracies = {}
    
    for current_step in range(n_steps+1):
        print(f"=== Running MID Sweep Split {current_step}/{n_steps} ===")
        sampling_probabilities_01, sampling_probabilities_02 = get_wcs_sweep_sampling_probabilities(current_step, n_steps)
        MID, WCS = calculate_balance_metrics(sampling_probabilities_01, sampling_probabilities_02)
        assert round(MID, 4) == 0, f"MID has to be 0, but right now it is {MID}"
        
        deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
        global_agent = Agent(0, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=None, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
        server = Server(global_agent, rl_test_data, autoencoder, experiment_id, experiment_path)

        clients = []
        for i in range(NR_CLIENTS):
            entity_id = i + 1
            if entity_id % 2 == 0:
                sampling_probabilities = sampling_probabilities_01
            else:
                sampling_probabilities = sampling_probabilities_02
            environment = Environment(entity_id, training_data_strides[i], state_interpreter=autoencoder, sampling_probabilities=sampling_probabilities, verbose=False)
            deep_q_network = DeepQNetwork(n_features=N_FEATURES, n_hidden_1=N_HIDDEN_1, n_hidden_2=N_HIDDEN_2, n_hidden_3=N_HIDDEN_3, n_actions=N_ACTIONS, loss=LOSS)
            optimizer = torch.optim.Adam(deep_q_network.parameters(), lr=LR,  weight_decay=L2, amsgrad=True)
            agent = Agent(entity_id, deep_q_network, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, eps=EPSILON_START, eps_min=EPSILON_END, eps_dec=EPSILON_DEC)
            client = Client(entity_id, agent, environment, experiment_path, MIN_REPLAY_SIZE)
            server.add_client(client)

        evaluations = [Evaluation.FINAL_PERFORMANCE_EVALUATION]
        experiment.execute(server, nr_rounds=NR_ROUNDS, nr_episodes_per_round=NR_EPISODES_PER_ROUND, evaluations=evaluations, evaluation_frequency=1, verbose=False)
        micro_accuracy, macro_accuracy = server.final_training_accuracy()
        final_micro_accuracies[WCS] = micro_accuracy
        final_macro_accuracies[WCS] = macro_accuracy
    return experiment, final_micro_accuracies, final_macro_accuracies
'''