import os

import numpy as np
import torch
from tabulate import tabulate

from copy import deepcopy
from custom_types import Behavior, ModelArchitecture, AdversaryType, AggregationMechanism, Scaler
from data_handler import DataHandler
from aggregation import Server
from participants import AutoEncoderParticipant, RandomWeightAdversary, ExaggerateThresholdAdversary, \
    UnderstateThresholdAdversary, ModelCancelAdversary
from utils import FederationUtils

if __name__ == "__main__":
    os.chdir("..")
    FederationUtils.seed_random()

    print("Federated Anomaly Detection under Attack\n")

    # define federation composition and data contribution
    participants_per_arch = [2, 2, 2, 2]
    adversaries_per_arch = [2, 2, 0, 0]
    adversary_type = AdversaryType.MODEL_CANCEL_BC
    aggregation_mechanism = AggregationMechanism.FED_AVG
    normals = [(Behavior.NORMAL, 3000)]
    attacks = [val for val in Behavior if val not in [Behavior.NORMAL, Behavior.NORMAL_V2]]
    # attacks = [Behavior.DELAY, Behavior.DISORDER, Behavior.FREEZE]
    val_percentage = 0.1
    train_attack_frac = 1 / len(attacks) if len(normals) == 1 else 2 / len(attacks)  # enforce balancing per device
    num_behavior_test_samples = 200

    train_devices, test_devices = FederationUtils.select_federation_composition(participants_per_arch, normals, attacks, val_percentage,
                                                                train_attack_frac,
                                                                num_behavior_test_samples, is_anomaly=True)
    print("Training devices:", len(train_devices))
    print(train_devices)
    print("Testing devices:", len(test_devices))
    print(test_devices)

    incl_test = False
    incl_train = True
    incl_val = False
    print("Number of samples used per device type:", "\nincl. test samples - ", incl_test, "\nincl. val samples -",
          incl_val, "\nincl. train samples -", incl_train)
    sample_requirements = FederationUtils.get_sampling_per_device(train_devices, test_devices, incl_train, incl_val, incl_test)
    print(tabulate(sample_requirements, headers=["device"] + [val.value for val in Behavior] + ["Normal/Attack"],
                   tablefmt="pretty"))

    print("Train Federation")
    train_sets, test_sets = DataHandler.get_all_clients_data(train_devices, test_devices)
    train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

    # injecting model poisoning participants
    adversaries = []
    for i in range(len(participants_per_arch)):
        assert adversaries_per_arch[i] <= participants_per_arch[i], "There must be less adversaries than participants"
        adversaries += [1] * adversaries_per_arch[i] + [0] * (participants_per_arch[i] - adversaries_per_arch[i])
    assert len(train_sets_fed) == len(adversaries), "Unequal lenghts"

    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) if not is_adv else
                    RandomWeightAdversary(x_train, y_train, x_valid,
                                          y_valid) if adversary_type == AdversaryType.RANDOM_WEIGHT
                    else ModelCancelAdversary(x_train, y_train, x_valid,
                                              y_valid, n_honest=sum(participants_per_arch) - sum(adversaries_per_arch),
                                              n_malicious=sum(
                                                  adversaries_per_arch)) if adversary_type == AdversaryType.MODEL_CANCEL_BC
                    else ExaggerateThresholdAdversary(x_train, y_train, x_valid,
                                                      y_valid) if adversary_type == AdversaryType.EXAGGERATE_TRESHOLD
                    else UnderstateThresholdAdversary(x_train, y_train, x_valid, y_valid)
                    for (x_train, y_train, x_valid, y_valid), is_adv in zip(train_sets_fed, adversaries)]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER, aggregation_mechanism=aggregation_mechanism)
    server.train_global_model(aggregation_rounds=5)


    results, central_results = [], []
    for i, tfed in enumerate(test_sets_fed):
        y_predicted = server.predict_using_global_model(tfed[0])
        behavior = list(Behavior)[i % len(Behavior)]
        normal = normals[0][0].value if len(normals) == 1 else "normal/normal_v2"
        # federated results
        acc, _, conf_mat = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        results.append([test_devices[i][0], normal, behavior.value, f'{acc * 100:.2f}%'])

    print("Federated Results under Attack")
    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy'], tablefmt="pretty"))
