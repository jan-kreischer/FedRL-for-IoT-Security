import numpy as np
from autoencoder import AutoEncoder, AutoEncoderInterpreter
import torch
from utils.evaluation_utils import calculate_metrics
from custom_types import Behavior
from tabulate import tabulate


# functions to learn autoencoders:

def pretrain_ae_model(ae_data, split=0.8, lr=1e-4, momentum=0.8, num_epochs=300, dir="offline_prototype_3_ds_as_sampling/",
                      model_name="ae_model.pth"):

    idx = int(len(ae_data) * split)
    train_ae_x = ae_data[:idx, :-1].astype(np.float32)
    valid_ae_x = ae_data[idx:, :-1].astype(np.float32)
    print(f"size train: {train_ae_x.shape}, size valid: {valid_ae_x.shape}")

    print("---Training AE---")
    ae = AutoEncoder(train_x=train_ae_x, valid_x=valid_ae_x)
    ae.train(optimizer=torch.optim.SGD(ae.get_model().parameters(), lr=lr, momentum=momentum), num_epochs=num_epochs)
    ae.determine_threshold()
    print(f"AE threshold: {ae.threshold}")
    ae.save_model(dir=dir, model_name=model_name)


def get_pretrained_ae(path, dims):
    pretrained_model = torch.load(path)
    ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
                                            pretrained_model['threshold'], in_features=dims)
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