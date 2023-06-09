from data_provider import DataProvider
from custom_types import Behavior
from utils.evaluation_utils import seed_random, calculate_metrics, evaluate_anomaly_detector_ds, \
    evaluate_anomaly_detector_as, check_anomalous
from time import time
import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


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
    ae_ds_train, dtrain_rl = DataProvider.split_ds_data_for_ae_and_rl(dtrain)
    ae_train_dict, atrain_rl = DataProvider.split_as_data_for_ae_and_rl(atrain)

    # read in all data
    ntrain, test_ddata, test_adata, scaler = DataProvider.get_scaled_train_test_split_anomaly_detection_afterstates()

    # create train/test split & scale
    # pretrain



    # fit diverse classifiers and test them

    # LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=40, novelty=True, contamination=0.1)
    clf.fit(ntrain[:, :-1])
    #clf.fit(ae_ds_train[:, :-1])

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

    print("Evaluate on decision state data")
    #evaluate_anomaly_detector_ds(dtrain_rl, clf)
    evaluate_anomaly_detector_ds(test_ddata, clf)
    print("Evaluate on afterstate data")
    #evaluate_anomaly_detector_as(atrain_rl, clf)
    evaluate_anomaly_detector_as(test_adata, clf)