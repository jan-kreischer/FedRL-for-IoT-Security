import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aggregation import Server
from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from participants import MLPParticipant
from utils import FederationUtils

if __name__ == "__main__":
    os.chdir("..")

    print("Similarity of attacks:\n"
          "Can the knowledge of one attack be used to detect another attack?")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    attacks = [val for val in Behavior if val not in normals]
    all_accs = []
    device = RaspberryPi.PI3_1GB
    test_devices = [(device, {beh: 75}) for beh in Behavior]
    eval_labels = [beh.value for beh in Behavior]
    train_labels = []
    for attack in attacks:
        FederationUtils.seed_random()
        train_sets, test_sets = DataHandler.get_all_clients_data(
            [(device, {Behavior.NORMAL: 250, attack: 250},
              {Behavior.NORMAL: 25, attack: 25})],
            test_devices)

        train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
        participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                        x_train, y_train, x_valid, y_valid in train_sets]
        server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
        server.train_global_model(aggregation_rounds=1, local_epochs=15)
        att_accs = []
        for i, (x_test, y_test) in enumerate(test_sets):
            y_predicted = server.predict_using_global_model(x_test)
            acc, f1, conf_mat = FederationUtils.calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
            att_accs.append(acc)
        all_accs.append(att_accs)
        train_labels.append(attack.value)

    FederationUtils.seed_random()
    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 250, Behavior.REPEAT: 125, Behavior.MIMIC: 125},
          {Behavior.NORMAL: 25, Behavior.REPEAT: 13, Behavior.MIMIC: 12})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = FederationUtils.calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("repeat, mimic")

    FederationUtils.seed_random()
    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 250, Behavior.FREEZE: 125, Behavior.MIMIC: 125},
          {Behavior.NORMAL: 25, Behavior.FREEZE: 13, Behavior.MIMIC: 12})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = FederationUtils.calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("freeze, mimic")

    FederationUtils.seed_random()
    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 250, Behavior.DELAY: 84, Behavior.FREEZE: 83, Behavior.NOISE: 83},
          {Behavior.NORMAL: 25, Behavior.DELAY: 9, Behavior.FREEZE: 8, Behavior.NOISE: 8})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = FederationUtils.calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("delay, freeze, noise")

    hm = sns.heatmap(np.array(all_accs), xticklabels=eval_labels, yticklabels=train_labels)
    # plt.title('Heatmap of Device ' + device.value, fontsize=15)
    plt.xlabel('Predicting', fontsize=12)
    plt.ylabel('Trained on normal vs ', fontsize=12)
    plt.show()
    hm.get_figure().savefig(f"data_plot_class_similarity_{device.value}.pdf")
