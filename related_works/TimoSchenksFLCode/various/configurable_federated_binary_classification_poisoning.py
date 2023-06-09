import numpy as np
import torch
from tabulate import tabulate

from copy import deepcopy
from custom_types import Behavior, ModelArchitecture, AdversaryType, AggregationMechanism, Scaler
from data_handler import DataHandler
from aggregation import Server
from participants import MLPParticipant, BenignLabelFlipAdversary, AttackLabelFlipAdversary, AllLabelFlipAdversary, \
    ModelCancelAdversary
from utils import FederationUtils
import os

if __name__ == "__main__":
    os.chdir("..")
    FederationUtils.seed_random()

    print("Federated Binary Classification under Attack\n")

    # define collective experiment config:
    participants_per_arch = [2, 2, 0, 2]
    adversaries_per_arch = [0, 2, 0, 0]
    n_malicious = sum(adversaries_per_arch)
    n_honest = sum(participants_per_arch) - n_malicious
    adversary_type = AdversaryType.ALL_LABEL_FLIP
    aggregation_mechanism = AggregationMechanism.FED_AVG
    normals = [(Behavior.NORMAL, 500)]
    # attacks = [val for val in Behavior if val not in [Behavior.NORMAL, Behavior.NORMAL_V2]]
    attacks = [Behavior.DISORDER] #Behavior.FREEZE, Behavior.NOISE]
    val_percentage = 0.1
    train_attack_frac = 1 / len(attacks) if len(normals) == 1 else 2 / len(attacks)  # enforce balancing per device
    num_behavior_test_samples = 100

    train_devices, test_devices = FederationUtils.select_federation_composition(participants_per_arch, normals, attacks, val_percentage,
                                                                train_attack_frac, num_behavior_test_samples)
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
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.STANDARD_SCALER)

    # injecting poisoning participants
    adversaries = []
    for i in range(len(participants_per_arch)):
        assert adversaries_per_arch[i] <= participants_per_arch[i], "There must be less adversaries than participants"
        adversaries += [1] * adversaries_per_arch[i] + [0] * (participants_per_arch[i] - adversaries_per_arch[i])
    assert len(train_sets_fed) == len(adversaries), "Unequal lenghts"

    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) if not is_adv else
                    BenignLabelFlipAdversary(x_train, y_train, x_valid,
                                             y_valid) if adversary_type == AdversaryType.BENIGN_LABEL_FLIP
                    else AttackLabelFlipAdversary(x_train, y_train, x_valid,
                                                  y_valid) if adversary_type == AdversaryType.ATTACK_LABEL_FLIP
                    else AllLabelFlipAdversary(x_train, y_train, x_valid,
                                               y_valid) if adversary_type == AdversaryType.ALL_LABEL_FLIP
                    else ModelCancelAdversary(x_train, y_train, x_valid, y_valid, n_honest, n_malicious)
                    for (x_train, y_train, x_valid, y_valid), is_adv in zip(train_sets_fed, adversaries)]

    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS, aggregation_mechanism=aggregation_mechanism)
    server.train_global_model(aggregation_rounds=5)

    results = []
    for i, tfed in enumerate(test_sets_fed):
        y_predicted = server.predict_using_global_model(tfed[0])
        behavior = list(Behavior)[i % len(Behavior)]
        normal = normals[0][0].value if len(normals) == 1 else "normal/normal_v2"
        # federated results
        acc, _, conf_mat = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        (tn, fp, fn, tp) = FederationUtils.get_confusion_matrix_vals_in_percent(acc, conf_mat, behavior)
        results.append(
            [test_devices[i][0], normal, behavior.value, f'{acc * 100:.2f}%', f'{tn * 100:.2f}%', f'{fp * 100:.2f}%',
             f'{fn * 100:.2f}%', f'{tp * 100:.2f}%'])

    print("Federated Results under Attack")
    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'tn', 'fp', 'fn', 'tp'],
                   tablefmt="pretty"))
