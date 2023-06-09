from typing import Dict
from src.custom_types import Behavior, MTDTechnique
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import joblib
import os
import pickle

relative_data_path = "data"

# raw behaviors without any MTD framework/Agent Components running
raw_behaviors_dir_rp3 = "raw_behaviors_no_agent_rp3"
raw_behaviors_file_paths_rp3: Dict[Behavior, str] = {
    Behavior.NORMAL: f"{relative_data_path}/{raw_behaviors_dir_rp3}/normal_expfs_online_samples_1_2022-08-20-09-16_5s.csv",
    Behavior.RANSOMWARE_POC: f"{relative_data_path}/{raw_behaviors_dir_rp3}/ransom_expfs_online_samples_1_2022-08-22-14-04_5s.csv",
    Behavior.ROOTKIT_BDVL: f"{relative_data_path}/{raw_behaviors_dir_rp3}/rootkit_bdvl_online_samples_1_2022-08-19-08-45_5s.csv",
    Behavior.ROOTKIT_BEURK: f"{relative_data_path}/{raw_behaviors_dir_rp3}/rootkit_beurk_online_samples_1_2022-09-01-18-12_5s.csv",
    Behavior.CNC_THETICK: f"{relative_data_path}/{raw_behaviors_dir_rp3}/cnc_thetick_online_samples_1_2022-08-30-16-11_5s.csv",
    # Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{raw_behaviors_dir_rp3}/cnc_backdoor_jakoritar_expfs_samples_1_2022-08-21-13-33_5s.csv"
    Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{raw_behaviors_dir_rp3}/cnc_backdoor_jakoritar_new_online_samples_1_2022-09-02-09-19_5s.csv",
    Behavior.CNC_OPT1: f"{relative_data_path}/{raw_behaviors_dir_rp3}/cnc_opt_1_file_extr_online_samples_1_2022-09-24-22-08_5s.csv",
    Behavior.CNC_OPT2: f"{relative_data_path}/{raw_behaviors_dir_rp3}/cnc_opt_2_sysinfo_online_samples_1_2022-09-24-14-04_5s.csv",
}

raw_behaviors_dir_rp4 = "raw_behaviors_no_agent_rp4"
raw_behaviors_file_paths_rp4: Dict[Behavior, str] = {
    Behavior.NORMAL: f"{relative_data_path}/{raw_behaviors_dir_rp4}/normal_samples_2022-06-13-11-25_50s.csv",
    Behavior.RANSOMWARE_POC: f"{relative_data_path}/{raw_behaviors_dir_rp4}/ransomware_samples_2022-06-20-08-49_50s.csv",
    Behavior.ROOTKIT_BEURK: f"{relative_data_path}/{raw_behaviors_dir_rp4}/rootkit_beurk_samples_2022-06-17-09-08_50s.csv",
    Behavior.ROOTKIT_BDVL: f"{relative_data_path}/{raw_behaviors_dir_rp4}/rootkit_bdvl_samples_2022-06-16-19-16_50s.csv",
    Behavior.CNC_THETICK: f"{relative_data_path}/{raw_behaviors_dir_rp4}/cnc_backdoor_jakoritar_samples_2022-06-18-09-35_50s.csv",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{raw_behaviors_dir_rp4}/cnc_thetick_samples_2022-06-19-16-54_50s.csv"
}

# behaviors with MTD framework/Agent Components running as per directory "online_prototype_monitoring"
decision_state = "decision"
decision_states_dir = "decision_states_online_agent"
decision_states_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: f"{relative_data_path}/{decision_states_dir}/normal_expfs_online_samples_1_2022-08-18-08-31_5s.csv",
    # Behavior.NORMAL: f"{relative_data_path}/{decision_states_dir}/normal_noexpfs_online_samples_1_2022-08-15-14-07_5s.csv",
    Behavior.RANSOMWARE_POC: f"{relative_data_path}/{decision_states_dir}/ransom_noexpfs_online_samples_1_2022-08-16-08-43_5s.csv",
    Behavior.ROOTKIT_BDVL: f"{relative_data_path}/{decision_states_dir}/rootkit_bdvl_online_samples_1_2022-08-12-16-40_5s.csv",
    # Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{decision_states_dir}/cnc_jakoritar_online_samples_1_2022-08-13-06-50_5s.csv"
    # Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{decision_states_dir}/cnc_backdoor_jakoritar_good_noexpfs_online_samples_1_2022-08-22-09-09_5s.csv",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"{relative_data_path}/{decision_states_dir}/cnc_backdoor_jakoritar_new_online_samples_1_2022-09-06-15-29_5s.csv",
    Behavior.ROOTKIT_BEURK: f"{relative_data_path}/{decision_states_dir}/rootkit_beurk_online_samples_1_2022-09-08-08-55_5s.csv",
    Behavior.CNC_THETICK: f"{relative_data_path}/{decision_states_dir}/cnc_thetick_online_samples_1_2022-09-12-10-27_5s.csv",
    Behavior.CNC_OPT1: f"{relative_data_path}/{decision_states_dir}/cnc_opt_1_file_extr_online_samples_1_2022-09-20-17-30_5s.csv",
    Behavior.CNC_OPT2: f"{relative_data_path}/{decision_states_dir}/cnc_opt_2_sysinfo_online_samples_1_2022-09-24-09-46_5s.csv"
}
afterstate = "after"
afterstates_dir = "afterstates_online_agent"
afterstates_file_paths: Dict[Behavior, Dict[MTDTechnique, str]] = {
    Behavior.NORMAL: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/normal_as_dirtrap_expfs_online_samples_2_2022-08-17-14-23_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/normal_as_filetypes_noexpfs_online_samples_2_2022-08-18-08-29_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/normal_as_removerk_noexpfs_online_samples_2_2022-08-17-08-17_5s.csv",
        # MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/normal_as_changeip_noexpfs_online_samples_2_2022-08-17-14-22_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/normal_as_changeip_new_online_samples_2_2022-09-12-08-01_5s.csv"
    },
    Behavior.RANSOMWARE_POC: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/ransom_as_dirtrap_expfs_online_samples_2_2022-08-16-09-33_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/ransom_as_filetypes_expfs_online_samples_2_2022-08-16-14-36_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/ransom_as_changeip_expfs_online_samples_2_2022-08-15-21-09_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/ransom_as_removerk_noexpfs_online_samples_2_2022-08-16-19-06_5s.csv"
    },
    Behavior.ROOTKIT_BDVL: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/rootkit_bdvl_as_dirtrap_samples_noexpfs_2022-08-12-17-02_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/rootkit_bdvl_as_filetypes_noexpfs_cip_online_samples_2_2022-08-12-21-14_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/rootkit_bdvl_as_changeip_expfs_cip_online_samples_2_2022-08-12-20-42_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/rootkit_bdvl_as_removerk_noexpfs_online_samples_2_2022-08-13-06-02_5s.csv"
    },
    Behavior.CNC_BACKDOOR_JAKORITAR: {
        # MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/old_cnc_jakoritar_as_dirtrap_expfs_online_samples_2_2022-08-15-08-59_5s.csv",
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/cnc_backdoor_jakoritar_as_dirtrap_online_samples_2_2022-09-07-09-06_5s.csv",
        # MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/old_cnc_jakoritar_as_filetypes_noexpfs_online_samples_2_2022-08-15-09-23_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/cnc_backdoor_jakoritar_as_filetypes_online_samples_2_2022-09-06-20-14_5s.csv",
        ## very different raw syscalls: MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/old_cnc_jakoritar_as_changeip_expfs_online_samples_2_2022-08-15-14-08_5s.csv",
        ## nan values: MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/cnc_backdoor_jakoritar_as_changeip_online_samples_2_2022-09-07-15-26_5s",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/cnc_backdoor_jakoritar_as_changeip_nohup_client_online_samples_2_2022-08-24-14-41_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/old_cnc_jakoritar_as_removerk_expfs_online_samples_2_2022-08-13-10-52_5s.csv",
    },
    Behavior.ROOTKIT_BEURK: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/rootkit_beurk_as_dirtrap_online_samples_2_2022-09-10-18-10_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/rootkit_beurk_as_filetypes_online_samples_2_2022-09-11-18-07_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/rootkit_beurk_as_changeip_online_samples_2_2022-09-09-14-27_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/rootkit_beurk_as_removerk_online_samples_2_2022-09-10-09-51_5s.csv"
    },
    Behavior.CNC_THETICK: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/cnc_thetick_as_dirtrap_online_samples_2_2022-09-13-08-15_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/cnc_thetick_as_filetypes_online_samples_2_2022-09-13-14-11_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/cnc_thetick_as_changeip_online_samples_2_2022-09-13-21-10_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/cnc_thetick_as_removerk_online_samples_2_2022-09-12-14-07_5s.csv",
    },
    Behavior.CNC_OPT1: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/cnc_opt_1_file_extr_as_dirtrap_online_samples_2_2022-09-21-14-33_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/cnc_opt_1_file_extr_as_filetypes_online_samples_2_2022-09-25-09-20_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/cnc_opt_1_file_extr_as_changeip_online_samples_2_2022-09-20-21-40_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/cnc_opt_1_file_extr_as_removerk_online_samples_2_2022-09-21-08-19_5s.csv",
    },
    Behavior.CNC_OPT2: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"{relative_data_path}/{afterstates_dir}/cnc_opt_2_sysinfo_as_dirtrap_online_samples_2_2022-09-22-20-22_5s.csv",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"{relative_data_path}/{afterstates_dir}/cnc_opt_2_sysinfo_as_filetypes_online_samples_2_2022-09-22-16-11_5s.csv",
        MTDTechnique.CNC_IP_SHUFFLE: f"{relative_data_path}/{afterstates_dir}/cnc_opt_2_sysinfo_as_changeip_online_samples_2_2022-09-23-19-30_5s.csv",
        MTDTechnique.ROOTKIT_SANITIZER: f"{relative_data_path}/{afterstates_dir}/cnc_opt_2_sysinfo_as_removerk_online_samples_2_2022-09-23-08-14_5s.csv"
    }

}
# ignore trying to divide by invalid value in scaling
np.seterr(divide='ignore', invalid='ignore')

# TODO: These columns are derived from data_availability.py -> check data
time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ['cpuNice', 'cpuHardIrq', 'alarmtimer:alarmtimer_fired', 'tasksStopped',
                    'alarmtimer:alarmtimer_start', 'cachefiles:cachefiles_create',
                    'cachefiles:cachefiles_lookup', 'cachefiles:cachefiles_mark_active',
                    'dma_fence:dma_fence_init', 'udp:udp_fail_queue_rcv_skb']
# suspected cyclic/unstable
cols_to_exclude = ['tasks', 'tasksSleeping', 'tasksZombie', 'tasksRunning',
                   'ramFree', 'ramUsed', 'ramCache', 'memAvail', 'numEncrypted',
                   'iface0RX', 'iface0TX', 'iface1RX', 'iface1TX'
                   ]

# suspected undesirably reactive, distorting afterstates
cols_to_exclude += ['cpuSystem', 'block:block_dirty_buffer', 'cpuSoftIrq', 'cs', 'cpu-migrations',
                    'irq:softirq_entry', 'kmem:kmem_cache_alloc', 'kmem:kmem_cache_free',
                    'random:urandom_read', 'raw_syscalls:sys_enter', 'raw_syscalls:sys_exit',
                    'sched:sched_switch', 'sched:sched_wakeup', 'skb:consume_skb', 'timer:hrtimer_start',
                    'writeback:global_dirty_state']

cols_to_exclude += ['cpuIdle', 'cpuIowait', 'block:block_bio_backmerge', 'block:block_touch_buffer', 'clk:clk_set_rate',
                    'irq:irq_handler_entry', 'jbd2:jbd2_start_commit', 'kmem:mm_page_alloc', 'kmem:mm_page_free',
                    'preemptirq:irq_enable', 'sock:inet_sock_set_state']


class DataProvider:

    @staticmethod
    def parse_no_mtd_behavior_data(filter_suspected_external_events=True,
                                   filter_constant_columns=True,
                                   filter_outliers=True,
                                   keep_status_columns=False,
                                   exclude_cols=True,
                                   decision=False, pi=3) -> Dict[Behavior, np.ndarray]:
        # print(os.getcwd())
        b_directory, b_file_paths = (decision_states_dir, decision_states_file_paths) if decision else \
            (raw_behaviors_dir_rp3, raw_behaviors_file_paths_rp3) if pi == 3 else \
                (raw_behaviors_dir_rp4, raw_behaviors_file_paths_rp4)
        file_name = f'../data/{b_directory}/all_data_filtered_external{"_decision" if decision else ""}' \
                    f'_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        full_df = pd.DataFrame()
        # bdata = {}

        for attack in b_file_paths:
            df = DataProvider.__get_filtered_df(b_file_paths[attack],
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                startidx=50,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns,
                                                exclude_cols=exclude_cols)
            df['attack'] = attack
            # bdata[attack] = df.to_numpy()
            if not os.path.isfile(file_name):
                full_df = pd.concat([full_df, df])

        # full_df.to_csv(file_name, index_label=False)
        return full_df  # , bdata

    @staticmethod
    def parse_mtd_behavior_data(filter_suspected_external_events=True,
                                filter_constant_columns=True,
                                filter_outliers=True,
                                keep_status_columns=False,
                                exclude_cols=True) -> Dict[Behavior, np.ndarray]:
        # function should return a dictionary of all the
        file_name = f'../data/{afterstates_dir}/all_afterstate_data_filtered_external' \
                    f'_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)

        full_df = pd.DataFrame()

        # adata = {}
        for asb in afterstates_file_paths:
            for mtd in afterstates_file_paths[asb]:
                df = DataProvider.__get_filtered_df(afterstates_file_paths[asb][mtd],
                                                    filter_suspected_external_events=filter_suspected_external_events,
                                                    filter_constant_columns=filter_constant_columns,
                                                    filter_outliers=filter_outliers,
                                                    keep_status_columns=keep_status_columns,
                                                    exclude_cols=exclude_cols)
                df['attack'] = asb
                df['state'] = mtd
                if not os.path.isfile(file_name):
                    full_df = pd.concat([full_df, df])

                # adata[(asb, mtd)] = df.to_numpy()
        # full_df.to_csv(file_name, index_label=False)
        return full_df  # ,adata

    @staticmethod
    def parse_agent_data_files_to_df(filter_suspected_external_events=True,
                                     filter_constant_columns=True,
                                     filter_outliers=True,
                                     keep_status_columns=False,
                                     exclude_cols=True) -> pd.DataFrame:

        #print(os.getcwd())
        file_name = f'./data/all_agent_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'
        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"
        if os.path.isfile(file_name):
            return pd.read_csv(file_name)

        full_df = pd.DataFrame()
        for dsb in decision_states_file_paths:
            df = DataProvider.__get_filtered_df(decision_states_file_paths[dsb],
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns,
                                                exclude_cols=exclude_cols)
            df['attack'] = dsb.value
            df['state'] = decision_state
            full_df = pd.concat([full_df, df])

        for asb in afterstates_file_paths:
            for mtd in afterstates_file_paths[asb]:
                df = DataProvider.__get_filtered_df(afterstates_file_paths[asb][mtd],
                                                    filter_suspected_external_events=filter_suspected_external_events,
                                                    filter_constant_columns=filter_constant_columns,
                                                    filter_outliers=filter_outliers,
                                                    keep_status_columns=keep_status_columns,
                                                    exclude_cols=exclude_cols)
                df['attack'] = asb.value
                df['state'] = f"{afterstate} {mtd.value}"
                full_df = pd.concat([full_df, df])

        full_df.to_csv(file_name, index_label=False)
        return full_df

    @staticmethod
    def parse_normals(filter_suspected_external_events=True,
                      filter_constant_columns=True,
                      filter_outliers=True,
                      keep_status_columns=False,
                      exclude_cols=True) -> pd.DataFrame:
        normal_paths = [
            f"{relative_data_path}/{decision_states_dir}/normal_noexpfs_online_samples_1_2022-08-15-14-07_5s.csv",
            f"{relative_data_path}/{decision_states_dir}/normal_expfs_online_samples_1_2022-08-18-08-31_5s.csv",
            f"{relative_data_path}/{decision_states_dir}/incompl_installs_normal_online_samples_1_2022-08-02-20-36_5s.csv",
            f"{relative_data_path}/{decision_states_dir}/incompl_installs_normal_online_samples_1_ssh_conn_open_2022-08-02-15-51_5s.csv"
        ]
        full_df = pd.DataFrame()
        for i, norm_p in enumerate(normal_paths):
            df = DataProvider.__get_filtered_df(norm_p,
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns,
                                                exclude_cols=exclude_cols)
            df['attack'] = Behavior.NORMAL.value
            df['state'] = decision_state + str(i)
            full_df = pd.concat([full_df, df])
        return full_df

    @staticmethod
    def __get_filtered_df(path, filter_suspected_external_events=True, startidx=10, endidx=-1,
                          filter_constant_columns=True,
                          filter_outliers=True,
                          keep_status_columns=False, exclude_cols=True):
        df = pd.read_csv(path)

        if filter_suspected_external_events:
            # filter first hour of samples: 3600s / 50s = 72
            # and drop last measurement due to the influence of logging in, respectively out of the server
            df = df.iloc[startidx:endidx]

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

        if exclude_cols:
            df = df.drop(cols_to_exclude, axis=1)

        assert df.isnull().values.any() == False, "behavior data should not contain NaN values"
        return df

    @staticmethod
    def get_scaled_train_test_split_anomaly_detection_afterstates(normal_split=0.7, scaling_minmax=True):
        ddf = DataProvider.parse_no_mtd_behavior_data(decision=True, filter_outliers=False)
        adf = DataProvider.parse_mtd_behavior_data(filter_outliers=False)

        # get decision state normal split
        normal_train, normal_test = DataProvider.__filter_train_split_for_outliers(ddf, Behavior.NORMAL, normal_split)
        # fit on normal train
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(normal_train[:, :-1])

        normal_val = normal_test[:int(2 * len(normal_test) / 3), :]
        normal_test = normal_test[int(2 * len(normal_test) / 3):, :]
        normal_train = np.vstack((normal_train, normal_val))
        scaled_ntrain = np.hstack((scaler.transform(normal_train[:, :-1]), np.expand_dims(normal_train[:, -1], axis=1)))

        test_ddata = {}
        for b in ddf["attack"].unique():
            if b == Behavior.NORMAL:
                test_ddata[b] = np.hstack((scaler.transform(normal_test[:, :-1]),
                                           np.expand_dims(normal_test[:, -1], axis=1)))
                continue
            data = ddf[ddf["attack"] == b].to_numpy()
            test_ddata[b] = np.hstack((scaler.transform(data[:, :-1]), np.expand_dims(data[:, -1], axis=1)))
        test_adata = {}
        for b in adf["attack"].unique():
            for mtd in adf[adf["attack"] == b]["state"].unique():
                data = adf[(adf["attack"] == b) & (adf["state"] == mtd)].to_numpy()
                test_adata[(b, mtd)] = np.hstack((scaler.transform(data[:, :-2]), data[:, -2:]))

        return scaled_ntrain, test_ddata, test_adata, scaler

    @staticmethod
    def get_scaled_scaled_train_test_split_with_afterstates(split=0.8, scaling_minmax=True, scale_normal_only=True):

        #  1 get both dicts for decision and afterstates
        ddf = DataProvider.parse_no_mtd_behavior_data(decision=True, filter_outliers=False)
        adf = DataProvider.parse_mtd_behavior_data(filter_outliers=False)
        # take split of all behaviors, concat, calc scaling, scale both train and test split
        # get behavior dicts for train and test
        train_filtered, df_test = DataProvider.__filter_train_split_for_outliers(ddf, Behavior.NORMAL, split)
        train_ddata = {}
        test_ddata = {}
        for b in ddf["attack"].unique():
            dtrain_filtered, df_test = DataProvider.__filter_train_split_for_outliers(ddf, b, split)
            train_ddata[b] = dtrain_filtered
            test_ddata[b] = df_test
            if b != Behavior.NORMAL and not scale_normal_only:
                train_filtered = np.vstack((train_filtered, train_ddata[b]))

        # repeat for afterstate data
        train_adata = {}
        test_adata = {}
        for b in adf["attack"].unique():
            for mtd in adf[adf["attack"] == b]["state"].unique():
                atrain_filtered, a_test = DataProvider.__filter_as_train_split_for_outliers(adf, b, mtd, split)
                train_adata[(b, mtd)] = atrain_filtered
                test_adata[(b, mtd)] = a_test
                if scale_normal_only and b == Behavior.NORMAL:
                    train_filtered = np.vstack((train_filtered, atrain_filtered[:, :-1]))

                if not scale_normal_only:
                    train_filtered = np.vstack((train_filtered, atrain_filtered[:, :-1]))

        # fit scaler on either just normal ds or all training data combined
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(train_filtered[:, :-1])

        # get behavior dicts for scaled train and test data
        scaled_dtrain = {}
        scaled_dtest = {}
        for b, d in train_ddata.items():
            scaled_dtrain[b] = np.hstack((scaler.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))
            scaled_dtest[b] = np.hstack(
                (scaler.transform(test_ddata[b][:, :-1]), np.expand_dims(test_ddata[b][:, -1], axis=1)))

        scaled_atrain = {}
        scaled_atest = {}
        for t, d in train_adata.items():
            b, m = t[0], t[1]
            scaled_atrain[(b, m)] = np.hstack((scaler.transform(d[:, :-2]), d[:, -2:]))
            scaled_atest[(b, m)] = np.hstack(
                (scaler.transform(test_adata[(b, m)][:, :-2]), test_adata[(b, m)][:, -2:]))

        return scaled_dtrain, scaled_dtest, scaled_atrain, scaled_atest, scaler

    @staticmethod
    def __filter_train_split_for_outliers(df: pd.DataFrame, b: Behavior, split=0.8):
        df = df[df["attack"] == b].drop(["attack"], axis=1)
        df_test = df.sample(frac=1 - split)  # .reset_index(drop=True)
        df_train = pd.concat([df, df_test]).drop_duplicates(keep=False)
        train_filtered = df_train[(np.nan_to_num(np.abs(stats.zscore(df_train))) < 3).all(axis=1)]
        train_filtered["attack"] = b
        df_test["attack"] = b
        train_filtered = train_filtered.to_numpy()
        df_test = df_test.to_numpy()
        return train_filtered, df_test

    @staticmethod
    def __filter_as_train_split_for_outliers(df: pd.DataFrame, b: Behavior, mtd: MTDTechnique, split):
        df = df[(df["attack"] == b) & (df["state"] == mtd)].drop(["attack", "state"], axis=1)
        df_test = df.sample(frac=1 - split)  # .reset_index(drop=True)
        df_train = pd.concat([df, df_test]).drop_duplicates(keep=False)
        train_filtered = df_train[(np.nan_to_num(np.abs(stats.zscore(df_train))) < 3).all(axis=1)]
        train_filtered["attack"], train_filtered["state"] = b, mtd
        df_test["attack"], df_test["state"] = b, mtd
        train_filtered = train_filtered.to_numpy()
        df_test = df_test.to_numpy()
        return train_filtered, df_test

    @staticmethod
    def get_scaled_train_test_split(split=0.8, scaling_minmax=True, scale_normal_only=True, decision=False, pi=3):
        """
        Method returns dictionaries mapping behaviors to scaled train and test data, as well as the scaler used
        Either decision states or raw behaviors can be utilized (decision flag) as no combinations
        with mtd need to be considered
        """
        print(os.getcwd())
        rdf = DataProvider.parse_no_mtd_behavior_data(decision=decision, pi=pi, filter_outliers=False)

        # take split of all behaviors, concat, calc scaling, scale both train and test split
        train_filtered, df_test = DataProvider.__filter_train_split_for_outliers(rdf, Behavior.NORMAL, split)

        # get behavior dicts for train and test
        train_bdata = {}
        test_bdata = {}
        for b in rdf["attack"].unique():
            dtrain_filtered, df_test = DataProvider.__filter_train_split_for_outliers(rdf, b, split)
            train_bdata[b] = dtrain_filtered
            test_bdata[b] = df_test
            if b != Behavior.NORMAL and not scale_normal_only:
                train_filtered = np.vstack((train_filtered, train_bdata[b]))

        # fit scaler on either just normal data (if scale_normal_only), or all training data combined
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(train_filtered[:, :-1])

        # get behavior dicts for scaled train and test data
        scaled_train = {}
        scaled_test = {}
        for b, d in train_bdata.items():
            scaled_train[b] = np.hstack((scaler.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))
            scaled_test[b] = np.hstack(
                (scaler.transform(test_bdata[b][:, :-1]), np.expand_dims(test_bdata[b][:, -1], axis=1)))

        # return also scaler in case of using the agent for online scaling
        return scaled_train, scaled_test, scaler

    @staticmethod
    def get_reduced_dimensions_with_pca_ds_as(dim=15, dir=""):
        ""
        dtrain, dtest, atrain, atest, scaler = DataProvider.get_scaled_scaled_train_test_split_with_afterstates(
            scaling_minmax=False)
        all_strain = dtrain[Behavior.NORMAL]
        for b in dtrain:
            if b != Behavior.NORMAL:
                all_strain = np.vstack((all_strain, dtrain[b]))
        for t, d in atrain.items():
            all_strain = np.vstack((all_strain, atrain[t][:, :-1]))

        pca = PCA(n_components=dim)
        pca.fit(all_strain[:, :-1])

        pca_dtrain = {}
        for b, d in dtrain.items():
            pca_dtrain[b] = np.hstack((pca.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))

        pca_dtest = {}
        for b, d in dtest.items():
            pca_dtest[b] = np.hstack((pca.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))

        pca_atrain = {}
        for t, d in atrain.items():
            pca_atrain[t] = np.hstack((pca.transform(d[:, :-2]), d[:, -2:]))

        pca_atest = {}
        for t, d in atest.items():
            pca_atest[t] = np.hstack((pca.transform(d[:, :-2]), d[:, -2:]))

        # save for later use for predictions preprocessing
        # scaler_file, pca_file = "scaler.gz", "pcafit.gz"
        scaler_file, pca_file = f"{dir}scalerdsas.obj", f"{dir}pcafitdsas.obj"

        if not os.path.isfile(scaler_file):
            joblib.dump(scaler, scaler_file[:-3] + "gz")
            with open(scaler_file, "wb") as sf:
                pickle.dump(scaler, sf)

        if not os.path.isfile(pca_file):
            joblib.dump(pca, pca_file[:-3] + "gz")
            with open(pca_file, "wb") as pf:
                pickle.dump(pca, pf)

        return pca_dtrain, pca_dtest, pca_atrain, pca_atest

    @staticmethod
    def get_reduced_dimensions_with_pca(dim=15, pi=3, normal_only=True):
        strain, stest, scaler = DataProvider.get_scaled_train_test_split(pi=pi, scaling_minmax=False,
                                                                         scale_normal_only=normal_only)
        all_strain = strain[Behavior.NORMAL]
        for b in strain:
            if b != Behavior.NORMAL:
                all_strain = np.vstack((all_strain, strain[b]))

        # fit pca on all
        pca = PCA(n_components=dim)
        pca.fit(all_strain[:, :-1])

        pca_train = {}
        for b, d in strain.items():
            pca_train[b] = np.hstack((pca.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))

        pca_test = {}
        for b, d in stest.items():
            pca_test[b] = np.hstack((pca.transform(d[:, :-1]), np.expand_dims(d[:, -1], axis=1)))

        # save for later use for predictions preprocessing
        # scaler_file, pca_file = "scaler.gz", "pcafit.gz"
        scaler_file, pca_file = "scaler.obj", "pcafit.obj"

        if not os.path.isfile(scaler_file):
            # joblib.dump(scaler, scaler_file)
            with open(scaler_file, "wb") as sf:
                pickle.dump(scaler, sf)

        if not os.path.isfile(pca_file):
            # joblib.dump(pca, pca_file)
            with open(pca_file, "wb") as pf:
                pickle.dump(pca, pf)

        return pca_train, pca_test

    @staticmethod
    def fit_pca(n=15, pi=3):
        strain, stest, scaler = DataProvider.get_scaled_train_test_split(pi=pi, scale_normal_only=False, scaling_minmax=False)
        all_strain = strain[Behavior.NORMAL]
        for b in strain:
            if b != Behavior.NORMAL:
                all_strain = np.vstack((all_strain, strain[b]))

        pca = PCA(n_components=n)
        pca.fit(all_strain[:, :-1])
        return pca

    @staticmethod
    def get_pca_loading_scores_dataframe(n=15):
        pca = DataProvider.fit_pca(n)
        loadings = pd.DataFrame(pca.components_,
                                columns=pd.read_csv(raw_behaviors_file_paths_rp3[Behavior.NORMAL]).drop(
                                    time_status_columns + all_zero_columns + cols_to_exclude, axis=1).columns,
                                index=["PC" + str(i) for i in range(1, n + 1)])
        return loadings

    @staticmethod
    def get_highest_weight_loading_scores_for_pc(n_pcs=15, pcn="PC1"):
        # maxCol = lambda x: max(x.min(), x.max(), key=abs)
        df = DataProvider.get_pca_loading_scores_dataframe(n_pcs)
        # df['max_loading_score'] = df.apply(maxCol, axis=1)
        sorted_pc = df.loc[pcn].reindex(df.loc[pcn].abs().sort_values(ascending=False).index)
        return sorted_pc

    @staticmethod
    def split_ds_data_for_ae_and_rl(dtrain, s=0.3):
        normal_data = dtrain[Behavior.NORMAL]
        dtrain[Behavior.NORMAL] = normal_data[:int(s * len(normal_data))]
        return normal_data[int(s * len(normal_data)):], dtrain

    @staticmethod
    def split_as_data_for_ae_and_rl(train_data, s=0.3):
        ae_dict = {}
        for mtd in MTDTechnique:
            normal_mtd_train = train_data[(Behavior.NORMAL, mtd)]
            train_data[(Behavior.NORMAL, mtd)] = normal_mtd_train[:int(s * len(normal_mtd_train))]
            ae_dict[mtd] = normal_mtd_train[int(s * len(normal_mtd_train)):]
        return ae_dict, train_data
