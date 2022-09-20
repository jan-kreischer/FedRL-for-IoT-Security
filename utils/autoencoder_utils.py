import numpy as np
from autoencoder import AutoEncoder, AutoEncoderInterpreter
import torch
from utils.evaluation_utils import calculate_metrics
from custom_types import Behavior, MTDTechnique
from tabulate import tabulate


def split_ds_data_for_ae_and_rl(dtrain, n=200):
    normal_data = dtrain[Behavior.NORMAL]
    dtrain[Behavior.NORMAL] = normal_data[:n]
    return normal_data[n:], dtrain


def split_as_data_for_ae_and_rl(train_data, n=200):
    ae_dict = {}
    for mtd in MTDTechnique:
        normal_mtd_train = train_data[(Behavior.NORMAL, mtd)]
        train_data[(Behavior.NORMAL, mtd)] = normal_mtd_train[:n]
        ae_dict[mtd] = normal_mtd_train[n:]
    return ae_dict, train_data


def pretrain_ae_model(ae_data, split=0.8, lr=1e-4, momentum=0.8, num_epochs=300, num_std=1,
                      path="offline_prototype_3_ds_as_sampling/trained_models/ae_model.pth"):
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


def pretrain_all_afterstate_ae_models(ae_train_dict, dir="offline_prototype_3_ds_as_sampling/trained_models/", num_std=1):
    for i, mtd in enumerate(ae_train_dict):
        path = dir + "ae_model_" + str(mtd.value) + ".pth"
        if i == 0:
            all_train, all_valid = pretrain_ae_model(ae_train_dict[mtd][:, :-1], path=path, num_std=num_std)
        else:
            train_data, valid_data = pretrain_ae_model(ae_train_dict[mtd][:, :-1], path=path, num_std=num_std)
            all_train = np.vstack((all_train, train_data))
            all_valid = np.vstack((all_valid, valid_data))
    all_data = np.vstack((all_train, all_valid))
    all_data = np.hstack((all_data, np.ones((len(all_data), 1))))
    pretrain_ae_model(all_data, path=dir + "ae_model_all_as.pth", num_std=num_std)


def pretrain_all_ds_as_ae_models(dtrain, ae_train_dict, dir="offline_prototype_3_ds_as_sampling/trained_models/", num_std=1):
    """pretrains autoencoder models on 1. decision state normal,
    2. on each normal-mtd combination,
    3. on both decision and normal-mtd combination data"""
    all_train, all_valid = pretrain_ae_model(dtrain, path=dir + "ae_model_ds.pth")
    for i, mtd in enumerate(ae_train_dict):
        path = dir + "ae_model_" + str(mtd.value) + ".pth"
        train_data, valid_data = pretrain_ae_model(ae_train_dict[mtd][:, :-1], path=path, num_std=num_std)
        all_train = np.vstack((all_train, train_data))
        all_valid = np.vstack((all_valid, valid_data))

        all_data = np.vstack((all_train, all_valid))
        all_data = np.hstack((all_data, np.ones((len(all_data), 1))))
    pretrain_ae_model(all_data, path=dir + "ae_model_all_ds_as.pth", num_std=num_std)


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
    print(tabulate(results, headers=labels, tablefmt="pretty"))


def evaluate_ae_on_afterstates(ae_interpreter: AutoEncoderInterpreter, test_data):
    res_dict = {}

    for t in test_data:
        isAnomaly = check_anomalous(t[0], t[1])
        y_test = np.array([isAnomaly] * len(test_data[t]))
        y_predicted = ae_interpreter.predict(test_data[t][:, :-2].astype(np.float32))

        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        res_dict[t] = f'{(100 * acc):.2f}, {"anomaly" if isAnomaly else "normal"}%'
    labels = ["Behavior", "MTD", "Accuracy", "Objective"]
    results = []
    for t, a in res_dict.items():
        res = a.split(",")
        results.append([t[0].value, t[1].value, res[0], res[1]])
    print(tabulate(results, headers=labels, tablefmt="pretty"))


def evaluate_all_as_ae_models(dtrain, atrain, dims, dir):
    for mtd in MTDTechnique:
        path = dir + "ae_model_" + str(mtd.value) + ".pth"
        print("---Evaluating AE " + str(mtd.value) + "---")
        ae_interpreter = get_pretrained_ae(path=path, dims=dims)
        print("---Evaluation on decision behaviors train---")
        evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
        print("---Evaluation on afterstate behaviors train---")
        evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)

    print("Evaluating AE trained on all afterstates normal")
    path = dir + "ae_model_all_as.pth"
    ae_interpreter = get_pretrained_ae(path=path, dims=dims)
    print("---Evaluation on decision behaviors train---")
    evaluate_ae_on_no_mtd_behavior(ae_interpreter, test_data=dtrain)
    print("---Evaluation on afterstate behaviors train---")
    evaluate_ae_on_afterstates(ae_interpreter, test_data=atrain)


def check_anomalous(b: Behavior, m: MTDTechnique):
    if b == Behavior.NORMAL:
        return 0
    if (b == Behavior.ROOTKIT_BDVL or b == Behavior.ROOTKIT_BEURK) and m == MTDTechnique.ROOTKIT_SANITIZER:
        return 0
    if b == Behavior.RANSOMWARE_POC and (m == MTDTechnique.RANSOMWARE_DIRTRAP or m == MTDTechnique.RANSOMWARE_FILE_EXT_HIDE):
        return 0
    if (b == Behavior.CNC_BACKDOOR_JAKORITAR or b == Behavior.CNC_THETICK) and m == MTDTechnique.CNC_IP_SHUFFLE:
        return 0
    return 1
