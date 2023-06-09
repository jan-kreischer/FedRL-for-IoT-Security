from typing import Dict
from custom_types import Behavior
from scipy import stats
from tabulate import tabulate
import numpy as np
import pandas as pd
import os

# paths to data
path = "behaviors_no_mtd"


data_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: f"../data/{path}/normal_samples_2022-06-13-11-25_50s",
    Behavior.RANSOMWARE_POC: f"../data/{path}/ransomware_samples_2022-06-20-08-49_50s",
    Behavior.ROOTKIT_BEURK: f"../data/{path}/rootkit_beurk_samples_2022-06-17-09-08_50s",
    Behavior.ROOTKIT_BDVL: f"../data/{path}/rootkit_bdvl_samples_2022-06-16-19-16_50s",
    Behavior.CNC_THETICK: f"../data/{path}/cnc_backdoor_jakoritar_samples_2022-06-18-09-35_50s",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"../data/{path}/cnc_thetick_samples_2022-06-19-16-54_50s"
}

time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ["alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
                    "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active", "dma_fence:dma_fence_init",
                    "udp:udp_fail_queue_rcv_skb"]


class DataManager:

    @staticmethod
    def parse_all_behavior_data(filter_suspected_external_events=True,
                              filter_constant_columns=True,
                              filter_outliers=True,
                              keep_status_columns=False) -> Dict[Behavior, np.ndarray]:
        #print(os.getcwd())
        file_name = f'../data/{path}/all_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        #if os.path.isfile(file_name):
         #   full_df = pd.read_csv(file_name)

        bdata = {}

        for attack in data_file_paths:
            df = pd.read_csv(data_file_paths[attack])

            if filter_suspected_external_events:
                # filter first hour of samples: 3600s / 50s = 72
                # and drop last measurement due to the influence of logging in, respectively out of the server
                df = df.iloc[72:-1]
            # filter for measurements where the device was connected
            df = df[df['connectivity'] == 1]

            # remove model-irrelevant columns
            if not keep_status_columns:
                df = df.drop(time_status_columns, axis=1)
            if filter_outliers:
                # drop outliers per measurement, indicated by (absolute z score) > 3
                df = df[(np.nan_to_num(np.abs(stats.zscore(df))) < 3).all(axis=1)]

            if filter_constant_columns:
                df = df.drop(all_zero_columns, axis=1)

            df['attack'] = attack
            bdata[attack] = df.to_numpy()
            #if not os.path.isfile(file_name): full_df = pd.concat([full_df, df])

        #full_df.to_csv(file_name, index_label=False)
        return bdata


    @staticmethod
    def parse_all_files_to_df(filter_suspected_external_events=True,
                                filter_constant_columns=True,
                                filter_outliers=True,
                                keep_status_columns=False) -> pd.DataFrame:

        file_name = f'../data/{path}/all_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"


        if os.path.isfile(file_name):
           return pd.read_csv(file_name)
        full_df = pd.DataFrame()

        for attack in data_file_paths:
            df = pd.read_csv(data_file_paths[attack])

            if filter_suspected_external_events:
                # filter first hour of samples: 3600s / 50s = 72
                # and drop last measurement due to the influence of logging in, respectively out of the server
                df = df.iloc[72:-1]
            # filter for measurements where the device was connected
            df = df[df['connectivity'] == 1]

            # remove model-irrelevant columns
            if not keep_status_columns:
                df = df.drop(time_status_columns, axis=1)
            if filter_outliers:
                # drop outliers per measurement, indicated by (absolute z score) > 3
                df = df[(np.nan_to_num(np.abs(stats.zscore(df))) < 3).all(axis=1)]

            if filter_constant_columns:
                df = df.drop(all_zero_columns, axis=1)

            df['attack'] = attack.value
            full_df = pd.concat([full_df, df])

        full_df.to_csv(file_name, index_label=False)
        return full_df

    @staticmethod
    def show_data_availability(raw=False):
        all_data = DataManager.parse_all_files_to_df(filter_outliers=not raw,
                                                          filter_suspected_external_events=not raw)

        print(f'Total data points: {len(all_data)}')
        drop_cols = [col for col in list(all_data) if col not in ['attack', 'block:block_bio_backmerge']]
        grouped = all_data.drop(drop_cols, axis=1).rename(columns={'block:block_bio_backmerge': 'count'}).groupby(
            ['attack'], as_index=False).count()
        labels = ['Behavior', 'Count']
        rows = []
        for behavior in Behavior:
            row = [behavior.value]
            cnt_row = grouped.loc[(grouped['attack'] == behavior.value)]
            row += [cnt_row['count'].iloc[0]]
            rows.append(row)
        print(tabulate(
            rows, headers=labels, tablefmt="pretty"))


