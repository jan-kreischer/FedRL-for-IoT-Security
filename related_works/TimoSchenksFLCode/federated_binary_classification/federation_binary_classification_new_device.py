import os
from copy import deepcopy
from typing import Dict

import numpy as np
from tabulate import tabulate

from aggregation import Server
from custom_types import Behavior, ModelArchitecture, Scaler, RaspberryPi
from data_handler import DataHandler
from participants import MLPParticipant
from utils import FederationUtils

if __name__ == "__main__":
    os.chdir("..")

    print("Starting demo experiment: Federated vs Centralized Binary Classification\n"
          "Training on a range of attacks and testing for each attack how well the joint model performs"
          " for a new device type unseen during training.\n")

    results, results_central = [], []
    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}

    for device in RaspberryPi:
        FederationUtils.seed_random()
        if device == RaspberryPi.PI4_2GB_WC:
            continue
        train_devices, test_devices = [], []
        for behavior in Behavior:
            test_devices.append((device, {behavior: 75}))
        if device == RaspberryPi.PI4_2GB_BC:
            for behavior in Behavior:
                test_devices.append((RaspberryPi.PI4_2GB_WC, {behavior: 75}))

        for device2 in RaspberryPi:
            if device2 != device and not (device == RaspberryPi.PI4_2GB_BC and device2 == RaspberryPi.PI4_2GB_WC):
                train_devices += FederationUtils.get_balanced_behavior_mlp_train_devices(device2)

        train_sets, test_sets = DataHandler.get_all_clients_data(
            train_devices,
            test_devices)

        # central
        train_sets_cen, test_sets_cen = deepcopy(train_sets), deepcopy(test_sets)
        train_sets_cen, test_sets_cen = DataHandler.scale(train_sets_cen, test_sets_cen, scaling=Scaler.MINMAX_SCALER)

        # copy data for federation and then scale
        train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
        train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

        # train federation
        participants = [MLPParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                        x_train, y_train, x_valid, y_valid in train_sets_fed]
        server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
        server.train_global_model(aggregation_rounds=15)

        # train central
        x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
        y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
        x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
        y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
        central_participant = [
            MLPParticipant(x_train_all, y_train_all, x_valid_all, y_valid_all,
                           batch_size_valid=1)]
        central_server = Server(central_participant, ModelArchitecture.MLP_MONO_CLASS)
        central_server.train_global_model(aggregation_rounds=1, local_epochs=15)

        for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
            y_predicted = server.predict_using_global_model(tfed[0])
            y_predicted_central = central_server.predict_using_global_model(tcen[0])
            behavior = list(test_devices[i][1].keys())[0]
            device = test_devices[i][0]

            acc, f1, _ = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
            acc_cen, f1_cen, _ = FederationUtils.calculate_metrics(tcen[1].flatten(),
                                                                   y_predicted_central.flatten().numpy())
            device_dict = res_dict[device] if device in res_dict else {}
            device_dict[behavior] = f'{acc * 100:.2f}% ({(acc - acc_cen) * 100:.2f}%)'

            res_dict[device] = device_dict

    for behavior in Behavior:
        results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])

    print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="latex"))
