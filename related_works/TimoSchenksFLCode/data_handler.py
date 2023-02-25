import os
from collections import defaultdict
from math import floor
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate

from custom_types import RaspberryPi, Behavior, Scaler

class_map_binary: Dict[Behavior, int] = defaultdict(lambda: 1, {
    Behavior.NORMAL: 0,
    Behavior.NORMAL_V2: 0
})

class_map_multi: Dict[Behavior, int] = defaultdict(lambda: 0, {
    Behavior.DELAY: 1,
    Behavior.DISORDER: 2,
    Behavior.FREEZE: 3,
    Behavior.HOP: 4,
    Behavior.MIMIC: 5,
    Behavior.NOISE: 6,
    Behavior.REPEAT: 7,
    Behavior.SPOOF: 8
})

data_file_paths: Dict[RaspberryPi, Dict[Behavior, str]] = {
    RaspberryPi.PI3_1GB: {
        Behavior.NORMAL: "data/ras-3-1gb/samples_normal_2021-06-18-15-59_50s",
        Behavior.NORMAL_V2: "data/ras-3-1gb/samples_normal_v2_2021-06-23-16-54_50s",
        Behavior.DELAY: "data/ras-3-1gb/samples_delay_2021-07-01-08-30_50s",
        Behavior.DISORDER: "data/ras-3-1gb/samples_disorder_2021-06-30-23-54_50s",
        Behavior.FREEZE: "data/ras-3-1gb/samples_freeze_2021-07-01-14-11_50s",
        Behavior.HOP: "data/ras-3-1gb/samples_hop_2021-06-29-23-23_50s",
        Behavior.MIMIC: "data/ras-3-1gb/samples_mimic_2021-06-30-10-33_50s",
        Behavior.NOISE: "data/ras-3-1gb/samples_noise_2021-06-30-19-44_50s",
        Behavior.REPEAT: "data/ras-3-1gb/samples_repeat_2021-07-01-20-00_50s",
        Behavior.SPOOF: "data/ras-3-1gb/samples_spoof_2021-06-30-14-49_50s"
    },
    RaspberryPi.PI4_2GB_BC: {
        Behavior.NORMAL: "data/ras-4-black/samples_normal_2021-07-11-22-19_50s",
        Behavior.NORMAL_V2: "data/ras-4-black/samples_normal_v2_2021-07-17-15-38_50s",
        Behavior.DELAY: "data/ras-4-black/samples_delay_2021-06-30-14-03_50s",
        Behavior.DISORDER: "data/ras-4-black/samples_disorder_2021-06-30-09-44_50s",
        Behavior.FREEZE: "data/ras-4-black/samples_freeze_2021-06-29-22-50_50s",
        Behavior.HOP: "data/ras-4-black/samples_hop_2021-06-30-18-24_50s",
        Behavior.MIMIC: "data/ras-4-black/samples_mimic_2021-06-29-18-35_50s",
        Behavior.NOISE: "data/ras-4-black/samples_noise_2021-06-29-14-20_50s",
        Behavior.REPEAT: "data/ras-4-black/samples_repeat_2021-06-28-23-52_50s",
        Behavior.SPOOF: "data/ras-4-black/samples_spoof_2021-06-28-19-34_50s",
    },
    RaspberryPi.PI4_2GB_WC: {
        Behavior.NORMAL: "data/ras-4-white/samples_normal_2021-07-31-15-34_50s",
        Behavior.NORMAL_V2: "data/ras-4-white/samples_normal_v2_2021-07-18-15-27_50s",
        Behavior.DELAY: "data/ras-4-white/samples_delay_2021-06-30-14-04_50s",
        Behavior.DISORDER: "data/ras-4-white/samples_disorder_2021-06-30-09-45_50s",
        Behavior.FREEZE: "data/ras-4-white/samples_freeze_2021-06-29-22-51_50s",
        Behavior.HOP: "data/ras-4-white/samples_hop_2021-06-30-18-25_50s",
        Behavior.MIMIC: "data/ras-4-white/samples_mimic_2021-06-29-18-36_50s",
        Behavior.NOISE: "data/ras-4-white/samples_noise_2021-06-29-14-21_50s",
        Behavior.REPEAT: "data/ras-4-white/samples_repeat_2021-06-28-23-52_50s",
        Behavior.SPOOF: "data/ras-4-white/samples_spoof_2021-06-28-19-35_50s",
    },
    RaspberryPi.PI4_4GB: {
        Behavior.NORMAL: "data/ras-4-4gb/samples_normal_2021-07-09-09-56_50s",
        Behavior.NORMAL_V2: "data/ras-4-4gb/samples_normal_v2_2021-07-13-10-43_50s",
        Behavior.DELAY: "data/ras-4-4gb/samples_delay_2021-07-01-08-36_50s",
        Behavior.DISORDER: "data/ras-4-4gb/samples_disorder_2021-06-30-23-57_50s",
        Behavior.FREEZE: "data/ras-4-4gb/samples_freeze_2021-07-01-14-13_50s",
        Behavior.HOP: "data/ras-4-4gb/samples_hop_2021-06-29-23-25_50s",
        Behavior.MIMIC: "data/ras-4-4gb/samples_mimic_2021-06-30-10-00_50s",
        Behavior.NOISE: "data/ras-4-4gb/samples_noise_2021-06-30-19-48_50s",
        Behavior.REPEAT: "data/ras-4-4gb/samples_repeat_2021-07-01-20-06_50s",
        Behavior.SPOOF: "data/ras-4-4gb/samples_spoof_2021-06-30-14-54_50s"
    },
}

time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ["alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
                    "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active", "dma_fence:dma_fence_init",
                    "udp:udp_fail_queue_rcv_skb"]


class DataHandler:

    @staticmethod
    def __pick_from_all_data(all_data: pd.DataFrame, device: RaspberryPi, attacks: Dict[Behavior, int],
                             label_dict: Dict[Behavior, int], pick_ratios: Dict[str, float]) -> \
            Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        data_x, data_y = None, None
        picked_all = None
        for attack in attacks:
            df = all_data.loc[(all_data['attack'] == attack.value) & (all_data['device'] == device.value)]
            key = device.value + "-" + attack.value
            if pick_ratios[key] < 1.:
                picked = df.sample(n=floor(pick_ratios[key] * attacks[attack]))
                all_data = pd.concat([picked, all_data]).drop_duplicates(keep=False)
                sampled = picked.sample(n=attacks[attack], replace=True)
            else:
                # creates a ValueError in case the experiment takes more validation and testing data than available
                picked = df.sample(n=attacks[attack])
                sampled = picked
                all_data = pd.concat([picked, all_data]).drop_duplicates(keep=False)

            if data_x is None:
                data_x = sampled.drop(['attack', 'device'], axis=1).to_numpy()
                picked_all = picked
            else:
                data_x = np.concatenate((data_x, sampled.drop(['attack', 'device'], axis=1).to_numpy()))
                picked_all = pd.concat([picked, picked_all]).drop_duplicates(keep=False)

            sampled_y = np.array([label_dict[attack]] * attacks[attack])
            if data_y is None:
                data_y = sampled_y
            else:
                data_y = np.concatenate((data_y, sampled_y))
        data_y = data_y.reshape((len(data_y), 1))
        return all_data, data_x, data_y, picked_all

    @staticmethod
    def parse_all_files_to_df(filter_suspected_external_events=True,
                              filter_constant_columns=True,
                              filter_outliers=True,
                              keep_status_columns=False) -> pd.DataFrame:
        file_name = f'./data/all_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += '_keepstatus'
        file_name += f'.csv'

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        full_df = pd.DataFrame()
        for device in data_file_paths:
            for attack in data_file_paths[device]:
                df = pd.read_csv(data_file_paths[device][attack])

                if filter_suspected_external_events:
                    # special case: here the pi4 wc has entered some significantly different "normal behavior"
                    # (some features peak significantly, adding variance)
                    if device == RaspberryPi.PI4_2GB_WC and attack == Behavior.NORMAL:
                        df = df.drop(df.index[6300:7500])
                    # drop first and last measurement due to the influence of logging in, respectively out of the server
                    df = df.iloc[1:-1]
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

                df['device'] = device.value
                df['attack'] = attack.value
                full_df = pd.concat([full_df, df])
        full_df.to_csv(file_name, index_label=False)
        return full_df

    @staticmethod
    def show_data_availability(raw=False):
        all_data = DataHandler.parse_all_files_to_df(filter_outliers=not raw, filter_suspected_external_events=not raw)
        print(f'Total data points: {len(all_data)}')
        drop_cols = [col for col in list(all_data) if col not in ['device', 'attack', 'block:block_bio_backmerge']]
        grouped = all_data.drop(drop_cols, axis=1).rename(columns={'block:block_bio_backmerge': 'count'}).groupby(
            ['device', 'attack'], as_index=False).count()
        labels = ['Behavior']
        for device in RaspberryPi:
            labels += [device.value]
        rows = []
        for behavior in Behavior:
            row = [behavior.value]
            for device in RaspberryPi:
                cnt_row = grouped.loc[(grouped['attack'] == behavior.value) & (grouped['device'] == device.value)]
                row += [cnt_row['count'].iloc[0]]
            rows.append(row)
        print(tabulate(
            rows, headers=labels, tablefmt="pretty"))

    @staticmethod
    def get_all_clients_data(
            train_devices: List[Tuple[RaspberryPi, Dict[Behavior, int], Dict[Behavior, int]]],
            test_devices: List[Tuple[RaspberryPi, Dict[Behavior, int]]],
            multi_class=False) -> \
            Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:

        assert len(train_devices) > 0 and len(
            test_devices) > 0, "Need to provide at least one train and one test device!"

        all_data = DataHandler.parse_all_files_to_df()
        all_data_test = DataHandler.parse_all_files_to_df(filter_outliers=False)
        all_outliers = pd.concat([all_data, all_data_test]).drop_duplicates(keep=False)

        # Dictionaries that hold total request: e. g. we want 500 train data for a pi3 and delay
        # but may only have 100 -> oversample and prevent overlaps
        total_data_request_count_train: Dict[str, int] = defaultdict(lambda: 0)
        remaining_data_available_count: Dict[str, int] = defaultdict(lambda: 0)

        # determine how to label data
        label_dict = class_map_multi if multi_class else class_map_binary

        for device, attacks, validation_attacks in train_devices:
            for attack in attacks:
                total_data_request_count_train[device.value + "-" + attack.value] += attacks[attack]

        train_sets = []
        validation_sets = []
        test_sets = []

        # pick test sets
        for device, test_attacks in test_devices:
            all_data_test, test_x, test_y, picked = DataHandler.__pick_from_all_data(all_data_test, device,
                                                                                     test_attacks, label_dict,
                                                                                     defaultdict(lambda: 1))
            picked_outliers = pd.concat([picked, all_outliers]).drop_duplicates(keep=False)
            picked_non_outliers = pd.concat([picked, picked_outliers]).drop_duplicates(keep=False)
            all_data = pd.concat([all_data, picked_non_outliers]).drop_duplicates(keep=False)
            test_sets.append((test_x, test_y))

        # pick validation sets: same as test sets -> in refactoring can be merged
        for device, _, val_attacks in train_devices:
            all_data, val_x, val_y, _ = DataHandler.__pick_from_all_data(all_data, device, val_attacks, label_dict,
                                                                         defaultdict(lambda: 1))
            validation_sets.append((val_x, val_y))

        for __i, row in all_data.groupby(['device', 'attack']).count().iterrows():
            remaining_data_available_count[row.name[0] + "-" + row.name[1]] += row.cs

        train_ratio_dict = {}
        for key in total_data_request_count_train:
            train_ratio_dict[key] = float(remaining_data_available_count[key]) / total_data_request_count_train[key]

        # pick and sample train sets
        for device, attacks, _ in train_devices:
            all_data, train_x, train_y, _ = DataHandler.__pick_from_all_data(all_data, device, attacks, label_dict,
                                                                             train_ratio_dict)
            train_sets.append((train_x, train_y))

        return [(x, y, validation_sets[idx][0], validation_sets[idx][1]) for
                idx, (x, y) in enumerate(train_sets)], [(x, y) for x, y in test_sets]

    @staticmethod
    def scale(train_devices: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
              test_devices: List[Tuple[np.ndarray, np.ndarray]], central: bool = False,
              scaling=Scaler.STANDARD_SCALER) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        train_scaled = []
        test_scaled = []
        if central:
            assert len(train_devices) == 1, "Only single training device allowed in central mode!"
            scaler = StandardScaler() if scaling == Scaler.STANDARD_SCALER else MinMaxScaler()
            scaler.fit(train_devices[0][0])
            for x_train, y_train, x_val, y_val in train_devices:
                train_scaled.append((scaler.transform(x_train), y_train, scaler.transform(x_val), y_val))
            for x_test, y_test in test_devices:
                test_scaled.append((scaler.transform(x_test), y_test))
        else:
            if scaling == Scaler.STANDARD_SCALER:
                scalers = []
                for x_train, y_train, x_val, y_val in train_devices:
                    scaler = StandardScaler()
                    scaler.fit(x_train)
                    scalers.append(scaler)
                final_scaler = StandardScaler()
                final_scaler.scale_ = np.stack([s.scale_ for s in scalers], axis=1).mean(axis=1)
                final_scaler.mean_ = np.stack([s.mean_ for s in scalers], axis=1).mean(axis=1)
            else:
                final_scaler = MinMaxScaler(clip=False)
                final_scaler.fit(np.concatenate(tuple([x_train for x_train, y_train, x_val, y_val in train_devices])))
            for x_train, y_train, x_val, y_val in train_devices:
                train_scaled.append((final_scaler.transform(x_train), y_train, final_scaler.transform(x_val), y_val))
            for x_test, y_test in test_devices:
                test_scaled.append((final_scaler.transform(x_test), y_test))

        return train_scaled, test_scaled
