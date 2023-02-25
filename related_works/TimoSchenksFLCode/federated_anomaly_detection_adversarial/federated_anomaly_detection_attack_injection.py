import os

import numpy as np
import pandas as pd
import torch

from aggregation import Server
from custom_types import Behavior, RaspberryPi, Scaler, ModelArchitecture, AggregationMechanism
from data_handler import DataHandler
from participants import AutoEncoderParticipant
from utils import FederationUtils

if __name__ == "__main__":
    cwd = os.getcwd()
    os.chdir("..")

    print("Starting demo experiment: Adversarial Impact (Attack Injection) on Federated Anomaly Detection")

    pis_to_inject = [pi for pi in RaspberryPi if pi != RaspberryPi.PI4_2GB_WC]

    max_adversaries = 4
    participants_per_device_type = 4  # max_adversaries

    test_devices = []

    for device in RaspberryPi:
        test_devices.append((device, {beh: 100 for beh in Behavior}))

    test_set_result_dict = {"device": [], "num_adversaries": [], "f1": [], "tp": [], "fp": [], "tn": [], "fn": [],
                            "aggregation": [], "injected": []}

    csv_result_path = cwd + os.sep + "attack_injection.csv"
    if os.path.isfile(csv_result_path):
        df = pd.read_csv(csv_result_path)
    else:
        for pi_to_inject in pis_to_inject:
            attack_devices = [
                (pi_to_inject, {Behavior.SPOOF: 300}, {Behavior.SPOOF: 30}),
                (pi_to_inject, {Behavior.MIMIC: 300}, {Behavior.MIMIC: 30}),
                (pi_to_inject, {Behavior.DELAY: 300}, {Behavior.DELAY: 30}),
                (pi_to_inject, {Behavior.DISORDER: 300}, {Behavior.DISORDER: 30})
            ]
            # Aggregation loop
            for agg in AggregationMechanism:
                # Adversary Loop -> here is the training
                for i in range(0, max_adversaries + 1):
                    FederationUtils.seed_random()
                    cp_filename = f'{cwd}{os.sep}attack_injection_{pi_to_inject.value}_{agg.value}_{str(i)}.pt'
                    train_devices = []
                    for device in RaspberryPi:
                        if device == RaspberryPi.PI4_2GB_WC:
                            continue
                        elif device == pi_to_inject:
                            train_devices += attack_devices[0:i]
                            for j in range(participants_per_device_type - i):
                                train_devices.append((device, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 150}))
                        else:
                            for j in range(participants_per_device_type):
                                train_devices.append((device, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 150}))
                    train_sets, test_sets = DataHandler.get_all_clients_data(
                        train_devices,
                        test_devices)

                    train_sets, test_sets = DataHandler.scale(train_sets, test_sets,
                                                              scaling=Scaler.MINMAX_SCALER)

                    # participants contains adversaries already
                    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                                    x_train, y_train, x_valid, y_valid in train_sets]
                    server = Server(participants, ModelArchitecture.AUTO_ENCODER,
                                    aggregation_mechanism=agg)
                    if not os.path.isfile(cp_filename):
                        server.train_global_model(aggregation_rounds=15)
                        torch.save(server.global_model.state_dict(), cp_filename)
                    else:
                        server.global_model.load_state_dict(torch.load(cp_filename))
                        server.load_global_model_into_participants()
                        print(f'Loaded model for device {pi_to_inject.value} and {str(i)} attackers and {agg.value}')

                    for j, (tset) in enumerate(test_sets):
                        y_predicted = server.predict_using_global_model(tset[0])
                        device = test_devices[j][0]
                        acc, f1, conf_mat = FederationUtils.calculate_metrics(tset[1].flatten(),
                                                                              y_predicted.flatten().numpy())
                        (tn, fp, fn, tp) = conf_mat.ravel()
                        test_set_result_dict['device'].append(device.value)
                        test_set_result_dict['num_adversaries'].append(i)
                        test_set_result_dict['f1'].append(f1 * 100)
                        test_set_result_dict['tp'].append(tp)
                        test_set_result_dict['tn'].append(tn)
                        test_set_result_dict['fp'].append(fp)
                        test_set_result_dict['fn'].append(fn)
                        test_set_result_dict['aggregation'].append(agg.value)
                        test_set_result_dict['injected'].append(pi_to_inject.value)

                    all_train, all_test = FederationUtils.aggregate_test_sets(test_sets)
                    y_predicted = server.predict_using_global_model(all_train)
                    acc, f1, conf_mat = FederationUtils.calculate_metrics(all_test.flatten(),
                                                                          y_predicted.flatten().numpy())
                    (tn, fp, fn, tp) = conf_mat.ravel()
                    test_set_result_dict['device'].append('All')
                    test_set_result_dict['num_adversaries'].append(i)
                    test_set_result_dict['f1'].append(f1 * 100)
                    test_set_result_dict['tp'].append(tp)
                    test_set_result_dict['tn'].append(tn)
                    test_set_result_dict['fp'].append(fp)
                    test_set_result_dict['fn'].append(fn)
                    test_set_result_dict['aggregation'].append(agg.value)
                    test_set_result_dict['injected'].append(pi_to_inject.value)
        df = pd.DataFrame.from_dict(test_set_result_dict)
        df.to_csv(csv_result_path, index=False)

    FederationUtils.visualize_adversaries_data_poisoning_pub(df, pis_to_inject,
                                                             # title='Attack Injection in Anomaly Detection\n',
                                                             title='',
                                                             row_title=lambda
                                                                 x: f'Injecting Attack to Device {"PI4" if x == RaspberryPi.PI4_4GB else "PI3" if x == RaspberryPi.PI3_1GB else "PI4_2"}\n',
                                                             save_dir='result_plot_homogeneous_anomaly_detection_inj_all.pdf')
