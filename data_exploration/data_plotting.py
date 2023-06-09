from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import numpy as np
from docutils.nodes import title

from custom_types import Behavior, RaspberryPi, MTDTechnique
from data_provider import DataProvider, decision_state


class DataPlotter:

    @staticmethod
    def plot_decision_or_afterstate_behaviors_timeline(decision_states: List[Tuple[Behavior, str]] = [],
                                                       afterstates: List[Tuple[Behavior, MTDTechnique, str]] = [],
                                                       plot_name: Union[str, None] = None):
        all_data = DataProvider.parse_agent_data_files_to_df(filter_outliers=False,
                                                             filter_suspected_external_events=False)
        # find max num samples
        max_number_of_samples = 0
        for behavior in decision_states:
            df_behavior = all_data.loc[(all_data['attack'] == behavior[0].value)]
            if len(df_behavior) > max_number_of_samples:
                max_number_of_samples = len(df_behavior)
        for behavior in afterstates:
            df_behavior = all_data.loc[(all_data['attack'] == behavior[0].value) &
                                       (all_data['state'].str.contains(behavior[1].value))]
            if len(df_behavior) > max_number_of_samples:
                max_number_of_samples = len(df_behavior)

        cols_to_plot = [col for col in all_data if col not in ['attack', 'state']]
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)

        all_decision = all_data[all_data['state'] == decision_state]
        all_after = all_data[all_data['state'] != decision_state]
        for i in range(len(cols_to_plot)):
            for behavior, line_color in decision_states:
                df_b = all_decision.loc[(all_decision['attack'] == behavior.value)]
                if (df_b[cols_to_plot[i]] < 0).any():
                    df_b = df_b[(df_b[cols_to_plot[i]] > 0)]
                xes_b = [i for i in range(max_number_of_samples)]
                ys_actual_b = df_b[cols_to_plot[i]].tolist()
                ys_upsampled_b = [ys_actual_b[i % len(ys_actual_b)] for i in range(max_number_of_samples)]
                axs[i].set_yscale('log')
                axs[i].plot(xes_b, ys_upsampled_b, color=line_color, label=(decision_state + " " + str(behavior.value)))
            for behavior, mtd, line_color in afterstates:
                df_b = all_after.loc[(all_after['attack'] == behavior.value) &
                                     (all_after['state'].str.contains(mtd.value))]
                if (df_b[cols_to_plot[i]] < 0).any():
                    df_b = df_b[(df_b[cols_to_plot[i]] > 0)]
                xes_b = [i for i in range(max_number_of_samples)]
                ys_actual_b = df_b[cols_to_plot[i]].tolist()
                ys_upsampled_b = [ys_actual_b[i % len(ys_actual_b)] for i in range(max_number_of_samples)]
                axs[i].plot(xes_b, ys_upsampled_b, color=line_color, label=(mtd.value + " " + str(behavior.value)))

            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            axs[i].set_ylabel("log features")
            axs[i].set_xlabel("time steps")
            axs[i].legend(title='Behavior and MTD Results')

        fig.tight_layout()
        if plot_name is not None:
            fig.savefig(f'data_exploration/data_plot_{plot_name}.png', dpi=100)
            print(f'Saved {plot_name}')

    @staticmethod
    def plot_decision_or_afterstates_as_kde(decision_states: List[Tuple[Behavior, str]] = [],
                                            afterstates: List[Tuple[Behavior, MTDTechnique, str]] = [],
                                            plot_name: Union[str, None] = None):

        all_data = DataProvider.parse_agent_data_files_to_df(filter_outliers=False,
                                                             filter_suspected_external_events=False)

        cols_to_plot = [col for col in all_data if col not in ['attack', 'state']]
        all_data = all_data.reset_index()
        all_decision = all_data[all_data['state'] == decision_state]
        all_after = all_data[all_data['state'] != decision_state]
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        for i in range(len(cols_to_plot)):
            axs[i].set_ylim([1e-6, 3 * 1e-1])  # adapt limitations specifically for features
            axs[i].set_xlabel("feature range")
            axs[i].set_ylabel("density")
            for b, color in decision_states:
                series = all_decision[all_decision.attack == b.value][cols_to_plot[i]]
                if series.unique().size == 1:
                    axs[i].axvline(series.iloc[0], ymin=1e-4, ymax=2, color=color)  # palette[f'{dv} {b.value}'])
                    continue
                series = series[(np.isnan(series) == False) & (np.isinf(series) == False)]
                sns.kdeplot(data=all_decision[all_decision.attack == b.value], x=cols_to_plot[i],
                            color=color, common_norm=True, common_grid=True, ax=axs[i], cut=2,
                            label=f"{decision_state} {b.value}", log_scale=(False, True))  # False, True
            for b, mtd, color in afterstates:
                series = all_after[(all_after['attack'] == b.value) & (all_after['state'].str.contains(mtd.value))][
                    cols_to_plot[i]]
                if series.unique().size == 1:
                    axs[i].axvline(series.iloc[0], ymin=1e-4, ymax=2, color=color)  # palette[f'{dv} {b.value}'])
                    continue
                series = series[(np.isnan(series) == False) & (np.isinf(series) == False)]
                sns.kdeplot(data=all_after[
                    (all_after['attack'] == b.value) & (all_after['state'].str.contains(mtd.value))],
                            x=cols_to_plot[i], color=color, common_norm=True, common_grid=True, ax=axs[i], cut=2,
                            label=f"{mtd.value} {b.value}", log_scale=(False, True))  # False, True

            axs[i].legend(title="State & Behavior/MTD")
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            # axs[i].set(xlabel=None)

        fig.tight_layout()
        if plot_name is not None:
            fig.savefig(f'data_exploration/data_plot_{plot_name}.png', dpi=100)

    @staticmethod
    def plot_normals_kde(plot_name,num_behaviors=4, colors=["green", "red", "blue", "violet"]):
        ndata = DataProvider.parse_normals(filter_outliers=False,
                                           filter_suspected_external_events=False)
        print(len(ndata))
        cols_to_plot = [col for col in ndata if col not in ['attack', 'state']]
        all_data = ndata.reset_index()
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        for i in range(len(cols_to_plot)):
            axs[i].set_ylim([1e-6, 0.5])  # adapt limitations specifically for features
            axs[i].set_xlabel("feature range")
            axs[i].set_ylabel("density")
            for j, color in zip(range(num_behaviors), colors):
                series = all_data[all_data['state'].str.contains(str(j))][cols_to_plot[i]]
                if series.unique().size == 1:
                    print("unique")
                    axs[i].axvline(series.iloc[0], ymin=1e-4, ymax=2, color=color)  # palette[f'{dv} {b.value}'])
                    continue
                series = series[(np.isnan(series) == False) & (np.isinf(series) == False)]
                sns.kdeplot(data=all_data[all_data['state'].str.contains(str(j))], x=cols_to_plot[i],
                            color=color, common_norm=True, common_grid=True, ax=axs[i], cut=2,
                            label=f"{str(j)} normal", log_scale=(False, True))  # False, True
            axs[i].legend(title="Normal Behaviors")
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            # axs[i].set(xlabel=None)

        fig.tight_layout()
        if plot_name is not None:
            fig.savefig(f'data_exploration/data_plot_{plot_name}.png', dpi=100)

    @staticmethod
    def plot_behaviors(behaviors: List[Tuple[RaspberryPi, Behavior, str]], raw_behaviors: bool = True,
                       plot_name: Union[str, None] = None):

        all_data_parsed = DataProvider.parse_raw_behavior_files_to_df(filter_outliers=False,
                                                                      filter_suspected_external_events=False)
        # first find max number of samples
        max_number_of_samples = 0
        for behavior in behaviors:
            df_behavior = all_data_parsed.loc[
                (all_data_parsed['attack'] == behavior[1].value)]
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
            axs[i].set_ylabel("log features")
            axs[i].set_xlabel("time steps")
            axs[i].legend(title='Device & Behavior')

        fig.tight_layout()
        if plot_name is not None:
            fig.savefig(f'data_exploration/data_plot_{plot_name}.png', dpi=100)
            print(f'Saved {plot_name}')

    @staticmethod
    def plot_devices_as_kde(device: RaspberryPi):

        plot_name = f"all_behaviors_{device.value}_kde"
        all_data_parsed = DataProvider.parse_raw_behavior_files_to_df(filter_outliers=True)
        cols_to_plot = [col for col in all_data_parsed if col not in ['attack']]
        dv = "RP3" if device == RaspberryPi.PI3_1GB else "RP4"
        all_data_parsed['Device & Behavior'] = all_data_parsed.apply(lambda row: f'{dv} {row.attack}', axis=1)
        # all_data_parsed['Monitoring'] = all_data_parsed.apply(lambda row: f'{device.value} {row.attack}', axis=1)
        # all_data_parsed = all_data_parsed.drop(['device'], axis=1)
        all_data_parsed = all_data_parsed.reset_index()
        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        palette = {f'{dv} {Behavior.NORMAL.value}': "green",
                   f'{dv} {Behavior.ROOTKIT_BDVL.value}': "black",
                   f'{dv} {Behavior.ROOTKIT_BEURK.value}': "darkblue",
                   f'{dv} {Behavior.RANSOMWARE_POC.value}': "orange",
                   f'{dv} {Behavior.CNC_THETICK.value}': "grey",
                   f'{dv} {Behavior.CNC_BACKDOOR_JAKORITAR.value}': "red"}
        for i in range(len(cols_to_plot)):
            axs[i].set_ylim([1e-6, 2 * 1e-4])  # adapt limitations specifically for features
            axs[i].set_xlabel("feature range")
            axs[i].set_ylabel("density")
            for b in Behavior:
                series = all_data_parsed[all_data_parsed.attack == b.value][cols_to_plot[i]]
                if series.unique().size == 1:
                    axs[i].axvline(series.iloc[0], ymin=1e-4, ymax=2, color=palette[f'{dv} {b.value}'])
                    continue
                series = series[(np.isnan(series) == False) & (np.isinf(series) == False)]
                sns.kdeplot(data=all_data_parsed[all_data_parsed.attack == b.value], x=cols_to_plot[i], palette=palette,
                            hue="Device & Behavior",
                            common_norm=True, common_grid=True, ax=axs[i], cut=2, label=f"{dv} {b.value}",
                            log_scale=(False, True))  # False, True
            axs[i].legend(title="Device & Behavior")
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            # axs[i].set(xlabel=None)

        fig.tight_layout()
        if plot_name is not None:
            fig.savefig(f'data_exploration/data_plot_{plot_name}.png', dpi=100)

    @staticmethod
    def print_pca_scree_plot(n=30):
        pca = DataProvider.fit_pca()
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        acc_per_var = [per_var[i] + np.sum(per_var[:i]) for i in range(len(per_var))]

        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        xx = range(1, len(per_var) + 1)
        plt.plot(xx, acc_per_var, 'ro', label="accumulated explained variance")
        plt.bar(x=xx, height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.xticks(fontsize=6)
        plt.title('Scree Plot')
        plt.legend()
        plt.savefig(f"data_exploration/screeplot_n_{n}.png")

    # @staticmethod
    # def plot_behaviors_as_kde_pub():
    #     for behav in Behavior:
    #         plot_name = f"all_devices_{behav.value}_kde"
    #         all_data_parsed = DataHandler.parse_raw_behavior_files_to_df(filter_outliers=True)
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


