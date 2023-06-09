import random
from math import floor
from typing import Tuple, Any, List, Dict, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tabulate import tabulate

from aggregation import Server
from custom_types import RaspberryPi, Behavior, AggregationMechanism


class FederationUtils:
    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Any]:
        correct = np.count_nonzero(y_test == y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        cm_fed = confusion_matrix(y_test, y_pred)  # could also extract via tn, fp, fn, tp = confusion_matrix().ravel()
        return correct / len(y_pred), f1, cm_fed

    @staticmethod
    def get_confusion_matrix_vals_in_percent(acc, conf_mat, behavior):
        if acc == 1.0:
            if behavior in [Behavior.NORMAL, Behavior.NORMAL_V2]:
                tn, fp, fn, tp = 1, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 1
        else:
            tn, fp, fn, tp = conf_mat.ravel() / sum(conf_mat.ravel())
        return tn, fp, fn, tp

    @staticmethod
    def print_experiment_scores(y_test: np.ndarray, y_pred: np.ndarray, federated=True):
        if federated:
            print("\n\nResults Federated Model:")
        else:
            print("\n\nResults Centralized Model:")

        accuracy, f1, cm_fed = FederationUtils.calculate_metrics(y_test, y_pred)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Infected"])
              if len(np.unique(y_pred)) > 1
              else "only single class predicted, no report generated")
        print(f"Details:\nConfusion matrix \n[(TN, FP),\n(FN, TP)]:\n{cm_fed}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%, F1 score: {f1 * 100:.2f}%")

    @staticmethod
    def aggregate_train_sets(all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(tuple([data[0] for data in all_data])),
            np.concatenate(tuple([data[1] for data in all_data])),
            np.concatenate(tuple([data[2] for data in all_data])),
            np.concatenate(tuple([data[3] for data in all_data])))

    @staticmethod
    def aggregate_test_sets(all_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[
        np.ndarray, np.ndarray]:
        return (
            np.concatenate(tuple([data[0] for data in all_data])),
            np.concatenate(tuple([data[1] for data in all_data])))

    @staticmethod
    def print_thresholds(server: Server, test_devices: List[Tuple[RaspberryPi, Dict[Behavior, int]]]):
        devices = []
        behaviors = []
        for test_dev in test_devices:
            behavior = list(test_dev[1].keys())[0]
            devices += [test_dev[0].value] * test_dev[1][behavior]
            behaviors += [behavior.value] * test_dev[1][behavior]
        df = pd.DataFrame.from_dict(
            {"threshold": server.evaluation_thresholds, "device": devices,
             "behavior": behaviors})

        rows = []
        for behavior in Behavior:
            rows.append([behavior.value] + [
                f"{df[(df.device == dev.value) & (df.behavior == behavior.value)].median(numeric_only=True)[0]:.2f}" for
                dev in RaspberryPi])
        print(tabulate(rows, headers=["Behavior"] + [dev.value for dev in RaspberryPi], tablefmt="latex"))

    @staticmethod
    def visualize_adversaries_data_poisoning(df: pd.DataFrame, injected_pis: List[RaspberryPi],
                                             title: str, row_title: Callable[[RaspberryPi], str],
                                             save_dir: str):
        sns.set_theme(font_scale=1.75, style='whitegrid')
        # see https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
        fig = plt.figure(figsize=(21., 19.2))
        grid = plt.GridSpec(len(injected_pis), 1)

        for pi_to_inject in injected_pis:
            # create fake subplot just to title set of subplots
            fake = fig.add_subplot(grid[injected_pis.index(pi_to_inject)])
            # '\n' is important
            fake.set_title(row_title(pi_to_inject), fontweight='semibold', size=28)
            fake.set_axis_off()

            # create subgrid for two subplots without space between them
            # <https://matplotlib.org/2.0.2/users/gridspec.html>
            gs = GridSpecFromSubplotSpec(1, len(list(AggregationMechanism)),
                                         subplot_spec=grid[injected_pis.index(pi_to_inject)])

            agg_idx = 0
            for i, agg in enumerate(AggregationMechanism):
                ax = fig.add_subplot(gs[agg_idx])
                df_loop = df[(df.injected == pi_to_inject.value) & (df.aggregation == agg.value)]
                sns.barplot(
                    data=df_loop, ci=None,
                    x="device", y="f1", hue="num_adversaries",
                    alpha=.6, ax=ax
                )
                for j, (tick) in enumerate(ax.xaxis.get_major_ticks()):
                    if (j % 2) != (i % 2):
                        tick.set_visible(False)
                ax.set_ylim(0, 100)
                ax.set_title(f'{agg.value}', size=25)
                ax.get_legend().remove()

                ax.set_xlabel('Device')
                if agg_idx == 0:
                    ax.set_ylabel('F1 Score (%)')
                else:
                    ax.set_ylabel(None)
                agg_idx += 1

        # add legend
        handles, labels = fig.axes[4].get_legend_handles_labels()
        fig.axes[4].legend(handles, labels, bbox_to_anchor=(1, 1.03), title="# of Adversaries")
        fig.tight_layout()
        fig.suptitle(title, fontweight='bold', size=16)
        plt.show()
        fig.savefig(save_dir, dpi=100)

    # modify to remove different devices from the plotting
    @staticmethod
    def visualize_adversaries_data_poisoning_pub(df: pd.DataFrame, injected_pis_to_plot: List[RaspberryPi],
                                                 title: str, row_title: Callable[[RaspberryPi], str],
                                                 save_dir: str):
        sns.set_theme(font_scale=1.75, style='whitegrid')
        # see https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
        fig = plt.figure(figsize=(21., 19.2))

        injected_pis_to_plot.remove(RaspberryPi.PI4_2GB_BC)
        grid = plt.GridSpec(len(injected_pis_to_plot), 1)

        for pi_to_inject in injected_pis_to_plot:
            # create fake subplot just to title set of subplots
            fake = fig.add_subplot(grid[injected_pis_to_plot.index(pi_to_inject)])
            # '\n' is important
            fake.set_title(row_title(pi_to_inject), fontweight='semibold', size=28)
            fake.set_axis_off()

            # create subgrid for two subplots without space between them
            # <https://matplotlib.org/2.0.2/users/gridspec.html>
            gs = GridSpecFromSubplotSpec(1, len(list(AggregationMechanism)),
                                         subplot_spec=grid[injected_pis_to_plot.index(pi_to_inject)])

            agg_idx = 0
            for i, agg in enumerate(AggregationMechanism):
                ax = fig.add_subplot(gs[agg_idx])
                # leave out 2 of 5 possible groupings
                df_loop = df[(df.device != RaspberryPi.PI4_2GB_BC.value) & (df.device != RaspberryPi.PI4_2GB_WC.value)
                            & (df.injected == pi_to_inject.value) & (df.aggregation == agg.value)]
                sns.barplot(
                    data=df_loop, ci=None,
                    x="device", y="f1", hue="num_adversaries",
                    alpha=.6, ax=ax
                )
                labels = ["PI3", "PI4", "ALL"]
                ax.set_xticklabels(labels)
                ax.set_ylim(0, 100)
                ax.set_title(f'{agg.value}', size=25)
                ax.get_legend().remove()

                ax.set_xlabel('Device')
                if agg_idx == 0:
                    ax.set_ylabel('F1 Score (%)')
                else:
                    ax.set_ylabel(None)
                agg_idx += 1

        # add legend
        handles, labels = fig.axes[4].get_legend_handles_labels()
        fig.axes[4].legend(handles, labels, bbox_to_anchor=(1, 1.03), title="# of Adversaries")
        fig.tight_layout()
        fig.suptitle(title, fontweight='bold', size=16)
        plt.show()
        fig.savefig(save_dir, dpi=100)

    @staticmethod
    def visualize_adversaries_model_poisoning(df: pd.DataFrame,
                                              title: str, save_dir: str):
        sns.set_theme(font_scale=1.75, style='whitegrid')
        fig, axs = plt.subplots(nrows=1, ncols=len(list(AggregationMechanism)), figsize=(21., 6.4))
        axs = axs.ravel().tolist()

        for i, (agg) in enumerate(AggregationMechanism):
            df_loop = df[(df.aggregation == agg.value)].drop(
                ['aggregation'], axis=1)
            sns.barplot(
                data=df_loop, ci=None,
                x="device", y="f1", hue="num_adversaries",
                alpha=.6, ax=axs[i]
            )
            for j, (tick) in enumerate(axs[i].xaxis.get_major_ticks()):
                if (j % 2) != (i % 2):
                    tick.set_visible(False)
            axs[i].set_ylim(0, 100)
            axs[i].set_title(f'{agg.value}')
            axs[i].get_legend().remove()

            axs[i].set_xlabel('Device')
            if i == 0:
                axs[i].set_ylabel('F1 Score (%)')
            else:
                axs[i].set_ylabel(None)

        # add legend
        handles, labels = fig.axes[len(fig.axes) - 1].get_legend_handles_labels()
        fig.axes[len(fig.axes) - 1].legend(handles, labels, bbox_to_anchor=(1, 1.03),
                                           title="# of Adversaries")
        # fig.suptitle(title, fontweight='bold', size=16)
        fig.tight_layout()
        plt.show()
        fig.savefig(save_dir, dpi=100)


    @staticmethod
    def visualize_adversaries_model_poisoning_pub(df: pd.DataFrame,
                                              title: str, save_dir: str):
        sns.set_theme(font_scale=1.75, style='whitegrid')
        fig, axs = plt.subplots(nrows=1, ncols=len(list(AggregationMechanism)), figsize=(21., 6.4))
        axs = axs.ravel().tolist()

        for i, (agg) in enumerate(AggregationMechanism):
            df_loop = df[(df.device != RaspberryPi.PI4_2GB_BC.value) & (df.device != RaspberryPi.PI4_2GB_WC.value)
                          & (df.aggregation == agg.value)]
            sns.barplot(
                data=df_loop, ci=None,
                x="device", y="f1", hue="num_adversaries",
                alpha=.6, ax=axs[i]
            )
            labels = ["PI3", "PI4", "ALL"]
            axs[i].set_xticklabels(labels)
            axs[i].set_ylim(0, 100)
            axs[i].set_title(f'{agg.value}')
            axs[i].get_legend().remove()

            axs[i].set_xlabel('Device')
            if i == 0:
                axs[i].set_ylabel('F1 Score (%)')
            else:
                axs[i].set_ylabel(None)

        # add legend
        handles, labels = fig.axes[len(fig.axes) - 1].get_legend_handles_labels()
        fig.axes[len(fig.axes) - 1].legend(handles, labels, bbox_to_anchor=(1, 1.03),
                                           title="# of Adversaries")
        # fig.suptitle(title, fontweight='bold', size=16)
        fig.tight_layout()
        plt.show()
        fig.savefig(save_dir, dpi=100)

    @staticmethod
    def seed_random():
        random.seed(42)
        torch.random.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def print_participants(participants: List[Tuple[RaspberryPi, Dict[Behavior, int], Dict[Behavior, int]]]):
        all_behaviors = set().union(*[set(list(p[1].keys()) + list(p[2].keys())) for p in participants])
        headers = ['Participant ID', 'Device ID'] + [beh.value for beh in list(all_behaviors)]
        values = []
        for idx, (p) in enumerate(participants):
            row = [idx + 1, p[0].value]
            for beh in all_behaviors:
                row += [f'{p[1][beh] if beh in p[1] else 0} ({p[2][beh] if beh in p[2] else 0})']
            values.append(row)
        print(tabulate(values, headers=headers, tablefmt="latex"))

    @staticmethod
    def get_balanced_behavior_mlp_train_devices(device_type: RaspberryPi) -> List[
        Tuple[RaspberryPi, Dict[Behavior, int], Dict[Behavior, int]]]:
        return [(device_type, {Behavior.NORMAL: 250},
                 {Behavior.NORMAL: 25}),
                (device_type, {Behavior.NORMAL: 250, Behavior.DELAY: 250},
                 {Behavior.NORMAL: 25, Behavior.DELAY: 25}),
                (device_type, {Behavior.NORMAL: 250, Behavior.FREEZE: 250},
                 {Behavior.NORMAL: 25, Behavior.FREEZE: 25}),
                (device_type, {Behavior.NORMAL: 250, Behavior.NOISE: 250},
                 {Behavior.NORMAL: 25, Behavior.NOISE: 25})]


    # Assumption we test at most on what we train (attack types)
    @staticmethod
    def select_federation_composition(participants_per_arch: List, normals: List[Tuple[Behavior, int]],
                                      attacks: List[Behavior],
                                      val_percentage: float, attack_frac: float,
                                      num_behavior_test_samples: int, is_anomaly: bool = False) \
            -> Tuple[List[Tuple[Any, Dict[Behavior, Union[int, float]], Dict[Behavior, Union[int, float]]]], List[
                Tuple[Any, Dict[Behavior, int]]]]:
        assert len(list(RaspberryPi)) == len(participants_per_arch), "lengths must be equal"
        assert normals[0][1] == normals[1][1] if len(
            normals) == 2 else True, "equal amount of normal version samples required"
        # populate train and test_devices for
        train_devices, test_devices = [], []
        for i, num_p in enumerate(participants_per_arch):
            for p in range(num_p):

                # add all normal monitorings for the training + validation + testing per participant
                train_d, val_d = {}, {}
                for normal in normals:
                    train_d[normal[0]] = normal[1]
                    val_d[normal[0]] = floor(normal[1] * val_percentage)

                # add all attacks for training + validation per participant in case of binary classification training
                if not is_anomaly:
                    for attack in attacks:
                        train_d[attack] = floor(normals[0][1] * attack_frac)
                        val_d[attack] = floor(normals[0][1] * attack_frac * val_percentage)
                train_devices.append((list(RaspberryPi)[i], train_d, val_d))

                # populate test dictionary with all behaviors (only once per device type)
                if p == 0:
                    for b in list(Behavior):
                        test_d = {}
                        test_d[b] = num_behavior_test_samples
                        test_devices.append((list(RaspberryPi)[i], test_d))

        return train_devices, test_devices


    # helper function independent of how test or train_devices are created
    # can be used to plot exactly how many samples of each device are being used for training to estimate the oversampling
    @staticmethod
    def get_sampling_per_device(train_devices, test_devices, include_train=True, incl_val=True, include_test=False):
        devices_sample_reqs = []  # header
        for d in RaspberryPi:
            device_samples = [d.value]
            for b in Behavior:
                bcount = 0
                for dev, train_d, val_d in train_devices:
                    if dev == d:
                        if include_train:
                            if b in train_d:
                                bcount += train_d[b]
                        if incl_val:
                            if b in val_d:
                                bcount += val_d[b]
                if include_test:
                    for dev, test_d in test_devices:
                        if dev == d:
                            if b in test_d:
                                bcount += test_d[b]
                device_samples.append(bcount)
            normals = sum(device_samples[1:3])
            attacks = sum(device_samples[3:])
            device_samples.append(normals / attacks if attacks != 0 else None)
            devices_sample_reqs.append(device_samples)
        return devices_sample_reqs
