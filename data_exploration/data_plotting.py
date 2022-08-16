from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from custom_types import Behavior, RaspberryPi
from data_provider import DataProvider


class DataPlotter:
    @staticmethod
    def plot_behaviors(behaviors: List[Tuple[RaspberryPi, Behavior, str]], plot_name: Union[str, None] = None):
        # first find max number of samples
        all_data_parsed = DataProvider.parse_all_files_to_df(filter_outliers=False,
                                                             filter_suspected_external_events=False)
        max_number_of_samples = 0
        for behavior in behaviors:
            df_behavior = all_data_parsed.loc[
                (all_data_parsed['attack'] == behavior[1].value)]  # & (all_data_parsed['device'] == behavior[0].value)]
            if len(df_behavior) > max_number_of_samples:
                max_number_of_samples = len(df_behavior)
        cols_to_plot = [col for col in all_data_parsed if col not in ['attack']]

        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        for i in range(len(cols_to_plot)):
            for device, behavior, line_color in behaviors:
                df_b = all_data_parsed.loc[
                    (all_data_parsed['attack'] == behavior.value)]  # & (all_data_parsed['device'] == device.value)]
                if (df_b[cols_to_plot[i]] < 0).any():
                    df_b = df_b[(df_b[cols_to_plot[i]] > 0)]
                xes_b = [i for i in range(max_number_of_samples)]
                ys_actual_b = df_b[cols_to_plot[i]].tolist()
                ys_upsampled_b = [ys_actual_b[i % len(ys_actual_b)] for i in range(max_number_of_samples)]
                axs[i].set_yscale('log')
                label = "RP3" if device == RaspberryPi.PI3_1GB else "RP4"
                axs[i].plot(xes_b, ys_upsampled_b, color=line_color, label=(label + " " + str(behavior.value)))
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            # axs[i].set_ylabel("log features")
            # axs[i].set_xlabel("time steps")
            axs[i].legend(title='Device & Behavior')

        if plot_name is not None:
            fig.savefig(f'data_plot_{plot_name}.png', dpi=100)
            print(f'Saved {plot_name}')

    @staticmethod
    def plot_devices_as_kde(device: RaspberryPi):

        plot_name = f"all_behaviors_{device.value}_kde"
        all_data_parsed = DataProvider.parse_all_files_to_df(filter_outliers=True)

        all_data_parsed = all_data_parsed[all_data_parsed.attack == Behavior.ROOTKIT_BEURK.value]
        cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]
        all_data_parsed['Monitoring'] = all_data_parsed.apply(lambda row: f'{device.value} {row.attack}', axis=1)
        #all_data_parsed = all_data_parsed.drop(['device'], axis=1)
        all_data_parsed = all_data_parsed.reset_index()
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        palette = {f'{device.value} {Behavior.NORMAL.value}': "green",
                   f'{device.value} {Behavior.ROOTKIT_BDVL.value}': "lightgreen",
                   f'{device.value} {Behavior.ROOTKIT_BEURK.value}': "darkblue",
                   f'{device.value} {Behavior.RANSOMWARE_POC.value}': "orange",
                   f'{device.value} {Behavior.CNC_THETICK.value}': "grey",
                   f'{device.value} {Behavior.CNC_BACKDOOR_JAKORITAR.value}': "red"}
        for i in range(len(cols_to_plot)):
            axs[i].set_ylim([1e-4, 2])
            for behav in Behavior:
                if all_data_parsed[all_data_parsed.attack == behav.value][cols_to_plot[i]].unique().size == 1:
                    axs[i].axvline(all_data_parsed[all_data_parsed.attack == behav.value][cols_to_plot[i]].iloc[0],
                                   ymin=1e-4, ymax=2, color=palette[f'{device.value} {behav.value}'])
            sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], #palette=palette, hue="Monitoring",
                        common_norm=False, common_grid=True, ax=axs[i], cut=2,
                        log_scale=(False, True))  # False, True

        if plot_name is not None:
            fig.savefig(f'data_plot_{plot_name}.png', dpi=100)

    @staticmethod
    def plot_devices_as_kde_pub(device: RaspberryPi):

        plot_name = f"all_behaviors_{device.value}_kde"
        all_data_parsed = DataProvider.parse_all_files_to_df(filter_outliers=True)
        #all_data_parsed = all_data_parsed[all_data_parsed.device == device.value]
        cols_to_plot = [col for col in all_data_parsed if col not in ['attack']]
        dv = "RP3" if device == RaspberryPi.PI3_1GB else "RP4"
        all_data_parsed['Device & Behavior'] = all_data_parsed.apply(lambda row: f'{dv} {row.attack}',axis=1)
        #all_data_parsed = all_data_parsed.drop(['attack'], axis=1)
        all_data_parsed = all_data_parsed.reset_index()
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)

        palette = {f'{dv} {Behavior.NORMAL.value}': "green",
                   f'{dv} {Behavior.ROOTKIT_BDVL.value}': "lightgreen",
                   f'{dv} {Behavior.ROOTKIT_BEURK.value}': "darkblue",
                   f'{dv} {Behavior.RANSOMWARE_POC.value}': "orange",
                   f'{dv} {Behavior.CNC_THETICK.value}': "grey",
                   f'{dv} {Behavior.CNC_BACKDOOR_JAKORITAR.value}': "red"}
                   # f'{dv} {Behavior.MIMIC.value}': "violet",
                   # f'{dv} {Behavior.NOISE.value}': "turquoise",
                   # f'{dv} {Behavior.REPEAT.value}': "black",
                   # f'{dv} {Behavior.SPOOF.value}': "darkred"}
        for i in range(len(cols_to_plot)):
            axs[i].set_ylim([1e-4, 2])
            axs[i].set_xlabel("feature range")
            axs[i].set_ylabel("density")
            for b in Behavior:
                if all_data_parsed[all_data_parsed.attack == b.value][cols_to_plot[i]].unique().size == 1:
                    axs[i].axvline(all_data_parsed[all_data_parsed.attack == b.value][cols_to_plot[i]].iloc[0],
                                   ymin=1e-4, ymax=2, color=palette[f'{dv} {b.value}'])
            sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], #palette=palette, hue="Device & Behavior",
                        common_norm=False, common_grid=True, ax=axs[i], cut=2,
                        log_scale=(False, True))  # False, True

        if plot_name is not None:
            fig.savefig(f'data_plot_{plot_name}.pdf', dpi=100)

    # @staticmethod
    # def plot_behaviors_as_kde(device: RaspberryPi):
    #     for b in Behavior:
    #         plot_name = f"all_devices_{b.value}_kde"
    #         all_data_parsed = DataProvider.parse_all_files_to_df(filter_outliers=True)
    #         all_data_parsed = all_data_parsed[all_data_parsed.attack == b.value]
    #         cols_to_plot = [col for col in all_data_parsed if col not in ['attack']]
    #
    #         all_data_parsed['Device & Behavior'] = all_data_parsed.apply(lambda row: f'{device.value} {row.attack}', axis=1)
    #         all_data_parsed = all_data_parsed.drop(['attack'], axis=1)
    #         all_data_parsed = all_data_parsed.reset_index()
    #         fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
    #         axs = axs.ravel().tolist()
    #         fig.suptitle(plot_name)
    #         fig.set_figheight(len(cols_to_plot))
    #         fig.set_figwidth(50)
    #         palette = {f'{RaspberryPi.PI3_1GB.value} {b.value}': "red",
    #                    f'{RaspberryPi.PI4_2GB_WC.value} {b.value}': "blue"}
    #         for i in range(len(cols_to_plot)):
    #             axs[i].set_ylim([1e-4, 2])
    #             if all_data_parsed[cols_to_plot[i]].unique().size == 1:
    #                 continue
    #             # for device in RaspberryPi:
    #             #     if all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].unique().size == 1:
    #             #         axs[i].axvline(all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].iloc[0],
    #             #                        ymin=1e-4, ymax=2, color=palette[f'{device.value} {b.value}'])
    #             sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], palette=palette, hue="Device & Behavior",
    #                         common_norm=False, common_grid=True, ax=axs[i], cut=2,
    #                         log_scale=(False, True))  # False, True
    #
    #         if plot_name is not None:
    #             fig.savefig(f'data_plot_{plot_name}.png', dpi=100)

    # @staticmethod
    # def plot_behaviors_as_kde_pub():
    #     for behav in Behavior:
    #         plot_name = f"all_devices_{behav.value}_kde"
    #         all_data_parsed = DataHandler.parse_all_files_to_df(filter_outliers=True)
    #         all_data_parsed = all_data_parsed[all_data_parsed.attack == behav.value]
    #         cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]
    #
    #         all_data_parsed['Device & Behavior'] = all_data_parsed.apply(lambda
    #                                                                          row: f'{"RP3" if row.device == RaspberryPi.PI3_1GB.value else "RP4_1" if row.device == RaspberryPi.PI4_2GB_WC.value else "RP4_2" if row.device == RaspberryPi.PI4_2GB_BC.value else "RP4_3"} {row.attack}',
    #                                                                      axis=1)
    #
    #         all_data_parsed = all_data_parsed.drop(['attack'], axis=1)
    #         all_data_parsed = all_data_parsed.reset_index()
    #         fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
    #         axs = axs.ravel().tolist()
    #         fig.suptitle(plot_name)
    #         fig.set_figheight(len(cols_to_plot))
    #         fig.set_figwidth(50)
    #         palette = {f'RP3 {behav.value}': "red",
    #                    f'RP4_1 {behav.value}': "blue",
    #                    f'RP4_2 {behav.value}': "orange",
    #                    f'RP4_3 {behav.value}': "green"}
    #         for i in range(len(cols_to_plot)):
    #             axs[i].set_ylim([1e-4, 2])
    #             if all_data_parsed[cols_to_plot[i]].unique().size == 1:
    #                 continue
    #             for device in RaspberryPi:
    #                 if all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].unique().size == 1:
    #                     axs[i].axvline(all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].iloc[0],
    #                                    ymin=1e-4, ymax=2, color=palette[
    #                             f'{"RP3" if device == RaspberryPi.PI3_1GB else "RP4_1" if device == RaspberryPi.PI4_2GB_WC else "RP4_2" if device == RaspberryPi.PI4_2GB_BC else "RP4_3"}'
    #                             f' {behav.value}']
    #                                    )
    #             sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], palette=palette, hue="Device & Behavior",
    #                         common_norm=False, common_grid=True, ax=axs[i], cut=2,
    #                         log_scale=(False, True))  # False, True
    #
    #         if plot_name is not None:
    #             fig.savefig(f'data_plot_{plot_name}.pdf', dpi=100)

    #
    #
    #
    # @staticmethod
    # def plot_delay_and_normal_as_kde():
    #     plot_name = f"delay_normal_all_devices_hist"
    #     all_data_parsed = DataHandler.parse_all_files_to_df(filter_outliers=True)
    #     all_data_parsed = all_data_parsed[
    #         (all_data_parsed.attack == Behavior.DELAY.value) | (all_data_parsed.attack == Behavior.NORMAL.value)]
    #     cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]
    #     all_data_parsed['Monitoring'] = all_data_parsed.apply(lambda row: f'{row.device} {row.attack}', axis=1)
    #
    #     all_data_parsed = all_data_parsed.drop(['attack', 'device'], axis=1)
    #     all_data_parsed = all_data_parsed.reset_index()
    #     fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
    #     axs = axs.ravel().tolist()
    #     fig.suptitle(plot_name)
    #     fig.set_figheight(len(cols_to_plot))
    #     fig.set_figwidth(50)
    #     palette = {f'{RaspberryPi.PI3_1GB.value} {Behavior.NORMAL.value}': "red",
    #                f'{RaspberryPi.PI4_2GB_WC.value} {Behavior.NORMAL.value}': "blue",
    #                f'{RaspberryPi.PI4_2GB_BC.value} {Behavior.NORMAL.value}': "orange",
    #                f'{RaspberryPi.PI4_4GB.value} {Behavior.NORMAL.value}': "green",
    #                f'{RaspberryPi.PI3_1GB.value} {Behavior.DELAY.value}': "slategrey",
    #                f'{RaspberryPi.PI4_2GB_WC.value} {Behavior.DELAY.value}': "black",
    #                f'{RaspberryPi.PI4_2GB_BC.value} {Behavior.DELAY.value}': "lime",
    #                f'{RaspberryPi.PI4_4GB.value} {Behavior.DELAY.value}': "fuchsia"}
    #     for i in range(len(cols_to_plot)):
    #         axs[i].set_ylim([1e-4, 2])
    #         if all_data_parsed[cols_to_plot[i]].unique().size == 1:
    #             continue
    #         sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], palette=palette, hue="Monitoring",
    #                     common_norm=False, common_grid=True, ax=axs[i], cut=2,
    #                     log_scale=(False, True))  # False, True
    #
    #     if plot_name is not None:
    #         fig.savefig(f'data_plot_{plot_name}.png', dpi=100)
