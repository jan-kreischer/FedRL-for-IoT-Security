import os
import sys
import numpy as np
from src.autoencoder import AutoEncoder, AutoEncoderInterpreter
import torch
from src.evaluation_utils import calculate_metrics, check_anomalous
from src.custom_types import Behavior, MTDTechnique
from tabulate import tabulate


def pretrain_ae_model(ae_data, split=0.8, lr=1e-4, momentum=0.9, num_epochs=100, num_std=2.5,
                      path="experiments/experiment_03/trained_models/ae_model.pth"):
    idx = int(len(ae_data) * split)
    train_ae_x = ae_data[:idx, :-1].astype(np.float32)
    valid_ae_x = ae_data[idx:, :-1].astype(np.float32)
    print(f"size train: {train_ae_x.shape}, size valid: {valid_ae_x.shape}")

    print("---Training AE---")
    ae = AutoEncoder(train_x=train_ae_x, valid_x=valid_ae_x)
    ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=lr, momentum=momentum), num_epochs=num_epochs)
    ae.determine_threshold(num_std=num_std)
    print(f"AE threshold: {ae.threshold}")
    ae.save_model(path=path)
    return train_ae_x, valid_ae_x


def pretrain_all_ds_as_ae_models(dtrain, ae_train_dict, dir="experiments/experiment_03/trained_models", num_std=1):
    """pretrains autoencoder models on 1. decision state normal,
    2. on each normal-mtd combination,
    3. on both decision and normal-mtd combination data"""
    all_train, all_valid = pretrain_ae_model(dtrain, path=f"{dir}/ae_model_ds.pth")
    for i, mtd in enumerate(ae_train_dict):
        path = f"{dir}/ae_model_{mtd.value}.pth"
        train_data, valid_data = pretrain_ae_model(ae_train_dict[mtd][:, :-1], path=path, num_std=num_std)
        all_train = np.vstack((all_train, train_data))
        all_valid = np.vstack((all_valid, valid_data))
        # for all afterstate model
        if i == 0:
            all_as_train, all_as_valid = train_data, valid_data
        else:
            all_as_train = np.vstack((all_as_train, train_data))
            all_as_valid = np.vstack((all_as_valid, valid_data))

    all_as_data = np.vstack((all_as_train, all_as_valid))
    all_as_data = np.hstack((all_as_data, np.ones((len(all_as_data), 1))))
    print("all as data: ", len(all_as_data))
    pretrain_ae_model(all_as_data, path=f"{dir}/ae_model_all_as.pth", num_std=num_std, num_epochs=100, lr=1e-4)

    all_data = np.vstack((all_train, all_valid))
    all_data = np.hstack((all_data, np.ones((len(all_data), 1))))
    print("all ds/as data: ", len(all_data))
    pretrain_ae_model(all_data, path=f"{dir}/ae_model_all_ds_as.pth", num_std=num_std, num_epochs=100, lr=1e-4)


def get_pretrained_ae(path, dims):
    pretrained_model = torch.load(path)
    ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
                                            pretrained_model['threshold'], in_features=dims, hidden_size=int(dims/2))
    print(f"ae_interpreter threshold: {ae_interpreter.threshold}")
    return ae_interpreter


def evaluate_ae_on_no_mtd_behavior(ae_interpreter: AutoEncoderInterpreter, test_data):
    res_dict = {}
    for b, d in test_data.items():
        y_test = np.array([0 if b == Behavior.NORMAL else 1] * len(d))
        y_predicted = ae_interpreter.predict(d[:, :-1].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        res_dict[b] = f'{(100 * acc):.2f}%'

    labels = ["Behavior"] + ["Accuracy"]
    results = []
    for b, a in res_dict.items():
        results.append([b.value, res_dict[b]])
    print(tabulate(results, headers=labels, tablefmt="latex"))


def evaluate_ae_on_afterstates(ae_interpreter: AutoEncoderInterpreter, test_data):
    res_dict = {}

    for t in test_data:
        isAnomaly = check_anomalous(t[0], t[1])
        y_test = np.array([isAnomaly] * len(test_data[t]))
        y_predicted = ae_interpreter.predict(test_data[t][:, :-2].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        res_dict[t] = f'{(100 * acc):.2f}%, {"anomaly" if isAnomaly else "normal"}'
    labels = ["Behavior", "MTD", "Accuracy", "Objective"]
    results = []
    for t, a in res_dict.items():
        res = a.split(",")
        results.append([t[0].value, t[1].value, res[0], res[1]])
    print(tabulate(results, headers=labels, tablefmt="latex"))


def evaluate_all_ds_as_ae_models(dtrain, atrain, dims, dir):
    for mtd in MTDTechnique:
        path = os.path.join(dir, f"/ae_model_{mtd.value}.pth")
        print(f"---Evaluating AE {mtd.value} ---")
        ae_interpreter = get_pretrained_ae(path=path, dims=dims)
        print("---Evaluation on decision behaviors train---")
        evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
        print("---Evaluation on afterstate behaviors train---")
        evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)

    print("Evaluating AE trained on all afterstates normal")
    path = f"{dir}/ae_model_all_as.pth"
    ae_interpreter = get_pretrained_ae(path=path, dims=dims)
    print("---Evaluation on decision behaviors train---")
    evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
    print("---Evaluation on afterstate behaviors train---")
    evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)

    print("Evaluating AE trained on all decision and afterstates normal")
    path = os.path.join(path, "/ae_model_all_ds_as.pth")
    ae_interpreter = get_pretrained_ae(path=path, dims=dims)
    print("---Evaluation on decision behaviors train---")
    evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
    print("---Evaluation on afterstate behaviors train---")
    evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)