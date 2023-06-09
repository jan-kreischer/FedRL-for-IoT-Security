import os
from copy import deepcopy
from typing import Dict

from tabulate import tabulate

from aggregation import Server
from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from participants import AutoEncoderParticipant
from utils import FederationUtils

if __name__ == "__main__":
    FederationUtils.seed_random()
    os.chdir("..")

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normal = Behavior.NORMAL
    train_devices = []
    test_devices = []

    results, results_central = [], []
    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}

    participants_per_device_type: Dict[RaspberryPi, int] = {
        RaspberryPi.PI3_1GB: 4,
        RaspberryPi.PI4_2GB_BC: 4,
        RaspberryPi.PI4_4GB: 4
    }

    for device in participants_per_device_type:
        train_devices += [(device, {normal: 1500}, {normal: 150})] * participants_per_device_type[device]
    for device in RaspberryPi:
        for behavior in Behavior:
            test_devices.append((device, {behavior: 150}))

    FederationUtils.print_participants(train_devices)

    train_sets, test_sets = DataHandler.get_all_clients_data(
        train_devices,
        test_devices)

    # central
    train_sets_cen, test_sets_cen = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_cen, test_sets_cen = DataHandler.scale(train_sets_cen, test_sets_cen, scaling=Scaler.MINMAX_SCALER)
    x_train_all, y_train_all, x_valid_all, y_valid_all = FederationUtils.aggregate_train_sets(train_sets_cen)

    # copy data for federation and then scale
    train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

    # train federation
    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets_fed]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
    server.train_global_model(aggregation_rounds=15)

    # train central
    central_participant = [
        AutoEncoderParticipant(x_train_all, y_train_all, x_valid_all, y_valid_all,
                               batch_size_valid=1)]
    central_server = Server(central_participant, ModelArchitecture.AUTO_ENCODER)
    central_server.train_global_model(aggregation_rounds=1, local_epochs=15)

    for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
        y_predicted = server.predict_using_global_model(tfed[0])
        y_predicted_central = central_server.predict_using_global_model(tcen[0])
        behavior = list(test_devices[i][1].keys())[0]
        device = test_devices[i][0]

        acc, f1, _ = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        acc_cen, f1_cen, _ = FederationUtils.calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
        device_dict = res_dict[device] if device in res_dict else {}
        device_dict[behavior] = f'{acc * 100:.2f}% ({(acc - acc_cen) * 100:.2f}%)'

        res_dict[device] = device_dict

    for behavior in Behavior:
        results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])

    print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="latex"))

    FederationUtils.print_thresholds(server, test_devices)
