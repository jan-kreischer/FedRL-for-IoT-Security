import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

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