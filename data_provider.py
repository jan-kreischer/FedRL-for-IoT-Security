from typing import Dict
from custom_types import Behavior, MTDTechnique
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
import joblib
import os
import pickle

# raw behaviors without any MTD framework/Agent Components running
raw_behaviors_dir_rp3 = "raw_behaviors_no_agent_rp3"
raw_behaviors_file_paths_rp3: Dict[Behavior, str] = {
    Behavior.NORMAL: f"data/{raw_behaviors_dir_rp3}/normal_expfs_online_samples_1_2022-08-20-09-16_5s",
    Behavior.RANSOMWARE_POC: f"data/{raw_behaviors_dir_rp3}/ransom_expfs_online_samples_1_2022-08-22-14-04_5s",
    Behavior.ROOTKIT_BDVL: f"data/{raw_behaviors_dir_rp3}/rootkit_bdvl_online_samples_1_2022-08-19-08-45_5s",
    Behavior.ROOTKIT_BEURK: f"data/{raw_behaviors_dir_rp3}/rootkit_beurk_online_samples_1_2022-09-01-18-12_5s",
    Behavior.CNC_THETICK: f"data/{raw_behaviors_dir_rp3}/cnc_thetick_online_samples_1_2022-08-30-16-11_5s",
    # Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{raw_behaviors_dir_rp3}/cnc_backdoor_jakoritar_expfs_samples_1_2022-08-21-13-33_5s"
    Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{raw_behaviors_dir_rp3}/cnc_backdoor_jakoritar_new_online_samples_1_2022-09-02-09-19_5s"
}

raw_behaviors_dir_rp4 = "raw_behaviors_no_agent_rp4"
raw_behaviors_file_paths_rp4: Dict[Behavior, str] = {
    Behavior.NORMAL: f"data/{raw_behaviors_dir_rp4}/normal_samples_2022-06-13-11-25_50s",
    Behavior.RANSOMWARE_POC: f"data/{raw_behaviors_dir_rp4}/ransomware_samples_2022-06-20-08-49_50s",
    Behavior.ROOTKIT_BEURK: f"data/{raw_behaviors_dir_rp4}/rootkit_beurk_samples_2022-06-17-09-08_50s",
    Behavior.ROOTKIT_BDVL: f"data/{raw_behaviors_dir_rp4}/rootkit_bdvl_samples_2022-06-16-19-16_50s",
    Behavior.CNC_THETICK: f"data/{raw_behaviors_dir_rp4}/cnc_backdoor_jakoritar_samples_2022-06-18-09-35_50s",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{raw_behaviors_dir_rp4}/cnc_thetick_samples_2022-06-19-16-54_50s"
}

# behaviors with MTD framework/Agent Components running as per directory "online_prototype_monitoring"
decision_state = "decision"
decision_states_dir = "decision_states_online_agent"
decision_states_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: f"data/{decision_states_dir}/normal_expfs_online_samples_1_2022-08-18-08-31_5s",
    # Behavior.NORMAL: f"data/{decision_states_dir}/normal_noexpfs_online_samples_1_2022-08-15-14-07_5s",
    Behavior.RANSOMWARE_POC: f"data/{decision_states_dir}/ransom_noexpfs_online_samples_1_2022-08-16-08-43_5s",
    Behavior.ROOTKIT_BDVL: f"data/{decision_states_dir}/rootkit_bdvl_online_samples_1_2022-08-12-16-40_5s",
    # Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{decision_states_dir}/cnc_jakoritar_online_samples_1_2022-08-13-06-50_5s"
    Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{decision_states_dir}/cnc_backdoor_jakoritar_noexpfs_online_samples_1_2022-08-22-09-09_5s"
}
afterstate = "after"
afterstates_dir = "afterstates_online_agent"
afterstates_file_paths: Dict[Behavior, Dict[MTDTechnique, str]] = {
    Behavior.NORMAL: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"data/{afterstates_dir}/normal_as_dirtrap_expfs_online_samples_2_2022-08-17-14-23_5s",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"data/{afterstates_dir}/normal_as_filetypes_noexpfs_online_samples_2_2022-08-18-08-29_5s",
        MTDTechnique.ROOTKIT_SANITIZER: f"data/{afterstates_dir}/normal_as_removerk_noexpfs_online_samples_2_2022-08-17-08-17_5s",
        MTDTechnique.CNC_IP_SHUFFLE: f"data/{afterstates_dir}/normal_as_changeip_noexpfs_online_samples_2_2022-08-17-14-22_5s"
    },
    Behavior.RANSOMWARE_POC: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"data/{afterstates_dir}/ransom_as_dirtrap_expfs_online_samples_2_2022-08-16-09-33_5s",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"data/{afterstates_dir}/ransom_as_filetypes_expfs_online_samples_2_2022-08-16-14-36_5s",
        MTDTechnique.CNC_IP_SHUFFLE: f"data/{afterstates_dir}/ransom_as_changeip_expfs_online_samples_2_2022-08-15-21-09_5s",
        MTDTechnique.ROOTKIT_SANITIZER: f"data/{afterstates_dir}/ransom_as_removerk_noexpfs_online_samples_2_2022-08-16-19-06_5s"
    },
    Behavior.ROOTKIT_BDVL: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"data/{afterstates_dir}/rootkit_bdvl_as_dirtrap_samples_noexpfs_2022-08-12-17-02_5s",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"data/{afterstates_dir}/rootkit_bdvl_as_filetypes_noexpfs_cip_online_samples_2_2022-08-12-21-14_5s",
        MTDTechnique.CNC_IP_SHUFFLE: f"data/{afterstates_dir}/rootkit_bdvl_as_changeip_expfs_cip_online_samples_2_2022-08-12-20-42_5s",
        MTDTechnique.ROOTKIT_SANITIZER: f"data/{afterstates_dir}/rootkit_bdvl_as_removerk_noexpfs_online_samples_2_2022-08-13-06-02_5s"
    },
    Behavior.CNC_BACKDOOR_JAKORITAR: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"data/{afterstates_dir}/cnc_jakoritar_as_dirtrap_expfs_online_samples_2_2022-08-15-08-59_5s",
        MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"data/{afterstates_dir}/cnc_jakoritar_as_filetypes_noexpfs_online_samples_2_2022-08-15-09-23_5s",
        MTDTechnique.CNC_IP_SHUFFLE: f"data/{afterstates_dir}/cnc_jakoritar_as_changeip_expfs_online_samples_2_2022-08-15-14-08_5s",
        MTDTechnique.ROOTKIT_SANITIZER: f"data/{afterstates_dir}/cnc_jakoritar_as_removerk_expfs_online_samples_2_2022-08-13-10-52_5s",
    }
}

# TODO: These columns are derived from data_availability.py -> check data
time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ['cpuNice', 'cpuHardIrq', 'alarmtimer:alarmtimer_fired', 'tasksStopped',
                    'alarmtimer:alarmtimer_start', 'cachefiles:cachefiles_create',
                    'cachefiles:cachefiles_lookup', 'cachefiles:cachefiles_mark_active',
                    'dma_fence:dma_fence_init', 'udp:udp_fail_queue_rcv_skb']
cols_to_exclude = ['tasks', 'tasksSleeping', 'tasksZombie',
                   'ramFree', 'ramUsed', 'ramCache', 'memAvail', 'numEncrypted',
                   'iface0RX', 'iface0TX', 'iface1RX', 'iface1TX'
                   ]


# 'tasksRunning', 'tasksStopped' - included in zero cols


class DataProvider:

    @staticmethod
    def parse_no_mtd_behavior_data(filter_suspected_external_events=True,
                                   filter_constant_columns=True,
                                   filter_outliers=True,
                                   keep_status_columns=False, decision=False, pi=3) -> Dict[Behavior, np.ndarray]:
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

        # if os.path.isfile(file_name):
        #   full_df = pd.read_csv(file_name)
        # full_df = pd.DataFrame()
        bdata = {}

        for attack in b_file_paths:
            df = DataProvider.__get_filtered_df(b_file_paths[attack],
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                startidx=50,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns)
            df['attack'] = attack
            bdata[attack] = df.to_numpy()
            # if not os.path.isfile(file_name): full_df = pd.concat([full_df, df])

        # full_df.to_csv(file_name, index_label=False)
        return bdata

    @staticmethod
    def parse_mtd_behavior_data(filter_suspected_external_events=True,
                                filter_constant_columns=True,
                                filter_outliers=True,
                                keep_status_columns=False) -> Dict[Behavior, np.ndarray]:
        # function should return a dictionary of all the
        file_name = f'../data/{afterstates_dir}/all_afterstate_data_filtered_external' \
                    f'_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        # if os.path.isfile(file_name):
        #   full_df = pd.read_csv(file_name)
        # full_df = pd.DataFrame()
        adata = {}
        for asb in afterstates_file_paths:
            for mtd in afterstates_file_paths[asb]:
                df = DataProvider.__get_filtered_df(afterstates_file_paths[asb][mtd],
                                                    filter_suspected_external_events=filter_suspected_external_events,
                                                    filter_constant_columns=filter_constant_columns,
                                                    filter_outliers=filter_outliers,
                                                    keep_status_columns=keep_status_columns)
                df['attack'] = asb
                df['state'] = mtd
                # if not os.path.isfile(file_name): full_df = pd.concat([full_df, df])

                adata[(asb, mtd)] = df.to_numpy()
        # full_df.to_csv(file_name, index_label=False)
        return adata

    @staticmethod
    def parse_raw_behavior_files_to_df(filter_suspected_external_events=True,
                                       filter_constant_columns=True,
                                       filter_outliers=True,
                                       keep_status_columns=False, pi=3) -> pd.DataFrame:

        if pi == 3:
            rb_dir, rb_fpaths = raw_behaviors_dir_rp3, raw_behaviors_file_paths_rp3
        else:
            rb_dir, rb_fpaths = raw_behaviors_dir_rp4, raw_behaviors_file_paths_rp4

        print(os.getcwd())
        file_name = f'data/{rb_dir}/all_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        full_df = pd.DataFrame()

        for attack in rb_fpaths:
            df = DataProvider.__get_filtered_df(rb_fpaths[attack],
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                startidx=72,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns)
            df['attack'] = attack.value
            full_df = pd.concat([full_df, df])

        full_df.to_csv(file_name, index_label=False)
        return full_df

    @staticmethod
    def parse_agent_data_files_to_df(filter_suspected_external_events=True,
                                     filter_constant_columns=True,
                                     filter_outliers=True,
                                     keep_status_columns=False) -> pd.DataFrame:

        print(os.getcwd())
        file_name = f'data/all_agent_data_filtered_external_{str(filter_suspected_external_events)}' \
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
                                                keep_status_columns=keep_status_columns)
            df['attack'] = dsb.value
            df['state'] = decision_state
            full_df = pd.concat([full_df, df])

        for asb in afterstates_file_paths:
            for mtd in afterstates_file_paths[asb]:
                df = DataProvider.__get_filtered_df(afterstates_file_paths[asb][mtd],
                                                    filter_suspected_external_events=filter_suspected_external_events,
                                                    filter_constant_columns=filter_constant_columns,
                                                    filter_outliers=filter_outliers,
                                                    keep_status_columns=keep_status_columns)
                df['attack'] = asb.value
                df['state'] = f"{afterstate} {mtd.value}"
                full_df = pd.concat([full_df, df])

        full_df.to_csv(file_name, index_label=False)
        return full_df

    @staticmethod
    def parse_normals(filter_suspected_external_events=True,
                      filter_constant_columns=True,
                      filter_outliers=True,
                      keep_status_columns=False) -> pd.DataFrame:
        normal_paths = [
            f"data/{decision_states_dir}/normal_noexpfs_online_samples_1_2022-08-15-14-07_5s",
            f"data/{decision_states_dir}/normal_expfs_online_samples_1_2022-08-18-08-31_5s",
            f"data/{decision_states_dir}/incompl_installs_normal_online_samples_1_2022-08-02-20-36_5s",
            f"data/{decision_states_dir}/incompl_installs_normal_online_samples_1_ssh_conn_open_2022-08-02-15-51_5s"
        ]
        full_df = pd.DataFrame()
        for i, norm_p in enumerate(normal_paths):
            df = DataProvider.__get_filtered_df(norm_p,
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns)
            df['attack'] = Behavior.NORMAL.value
            df['state'] = decision_state + str(i)
            full_df = pd.concat([full_df, df])
        return full_df

    @staticmethod
    def __get_filtered_df(path, filter_suspected_external_events=True, startidx=10, endidx=-1,
                          filter_constant_columns=True, const_cols=all_zero_columns,
                          filter_outliers=True,
                          keep_status_columns=False, stat_cols=time_status_columns, cols_to_exclude=cols_to_exclude):

        df = pd.read_csv(path)
        assert df.isnull().values.any() == False, "behavior data should not contain NaN values"

        if filter_suspected_external_events:
            # filter first hour of samples: 3600s / 50s = 72
            # and drop last measurement due to the influence of logging in, respectively out of the server
            df = df.iloc[startidx:endidx]

        # filter for measurements where the device was connected
        df = df[df['connectivity'] == 1]

        # remove model-irrelevant columns
        if not keep_status_columns:
            df = df.drop(stat_cols, axis=1)

        if filter_outliers:
            # drop outliers per measurement, indicated by (absolute z score) > 3
            df = df[(np.nan_to_num(np.abs(stats.zscore(df))) < 3).all(axis=1)]

        if filter_constant_columns:
            df = df.drop(const_cols, axis=1)

        if len(cols_to_exclude) > 0:
            df = df.drop(cols_to_exclude, axis=1)

        return df

    # @staticmethod
    # def get_scaled_all_data(scaling_minmax=True):
    #     all_data = DataProvider.parse_raw_behavior_files_to_df().to_numpy()[:, :-1]
    #     scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
    #     scaler.fit(all_data)
    #
    #     bdata = DataProvider.parse_no_mtd_behavior_data()
    #     scaled_bdata = {}
    #     # return directory as
    #     for b in bdata:
    #         scaled_bdata[b] = np.hstack((scaler.transform(bdata[b][:, :-1]), np.expand_dims(bdata[b][:, -1], axis=1)))
    #
    #     return scaled_bdata

    @staticmethod
    def get_scaled_scaled_train_test_split_with_afterstates(split=0.8, scaling_minmax=True):
        #  1 get both dicts for decision and afterstates
        ddata = DataProvider.parse_no_mtd_behavior_data(decision=True)
        adata = DataProvider.parse_mtd_behavior_data()

        # take split of all behaviors, concat, calc scaling, scale both train and test split
        first_b = ddata[Behavior.NORMAL]
        np.random.shuffle(first_b)
        train = first_b[:int(split * first_b.shape[0]), :]
        # test = first_b[int(split * first_b.shape[0]):, :]
        # get behavior dicts for train and test
        train_ddata = {}
        test_ddata = {}
        for b, d in ddata.items():
            np.random.shuffle(d)
            d_train = d[:int(split * d.shape[0]), :]
            d_test = d[int(split * d.shape[0]):, :]

            train_ddata[b] = d_train
            test_ddata[b] = d_test
            if b != Behavior.NORMAL:
                train = np.vstack((train, d_train))
                # test = np.vstack((test, d_test))
        train_adata = {}
        test_adata = {}
        for t, d in adata.items():
            b, m = t[0], t[1]
            np.random.shuffle(d)
            a_train = d[:int(split * d.shape[0]), :]
            a_test = d[int(split * d.shape[0]):, :]

            train_adata[(b, m)] = a_train
            test_adata[(b, m)] = a_test
            if b != Behavior.NORMAL:
                train = np.vstack((train, a_train[:, :-1]))
                # test = np.vstack((test, d_test))

        # fit scaler on all training data combined
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(train[:, :-1])

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
    def get_scaled_train_test_split(split=0.8, scaling_minmax=True, decision=False, pi=3):
        """
        Method returns dictionaries mapping behaviors to scaled train and test data, as well as the scaler used
        Either decision states or raw behaviors can be utilized (decision flag) as no combinations
        with mtd need to be considered
        """

        bdata = DataProvider.parse_no_mtd_behavior_data(decision=decision, pi=pi)

        # take split of all behaviors, concat, calc scaling, scale both train and test split
        first_b = bdata[Behavior.NORMAL]
        np.random.shuffle(first_b)
        train = first_b[:int(split * first_b.shape[0]), :]
        # test = first_b[int(split * first_b.shape[0]):, :]

        # get behavior dicts for train and test
        train_bdata = {}
        test_bdata = {}
        for b, d in bdata.items():
            np.random.shuffle(d)
            d_train = d[:int(split * d.shape[0]), :]
            d_test = d[int(split * d.shape[0]):, :]

            train_bdata[b] = d_train
            test_bdata[b] = d_test
            if b != Behavior.NORMAL:
                train = np.vstack((train, d_train))
                # test = np.vstack((test, d_test))

        # fit scaler on all training data combined
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(train[:, :-1])

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
        dtrain, dtest, atrain, atest, scaler = DataProvider.get_scaled_scaled_train_test_split_with_afterstates()
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
            # joblib.dump(scaler, scaler_file)
            with open(scaler_file, "wb") as sf:
                pickle.dump(scaler, sf)

        if not os.path.isfile(pca_file):
            # joblib.dump(pca, pca_file)
            with open(pca_file, "wb") as pf:
                pickle.dump(pca, pf)

        return pca_dtrain, pca_dtest, pca_atrain, pca_atest

    @staticmethod
    def get_reduced_dimensions_with_pca(dim=15, pi=3):
        strain, stest, scaler = DataProvider.get_scaled_train_test_split(pi=pi)
        all_strain = strain[Behavior.NORMAL]
        for b in strain:
            if b != Behavior.NORMAL:
                all_strain = np.vstack((all_strain, strain[b]))

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
    def fit_pca(n=15):
        strain, stest, scaler = DataProvider.get_scaled_train_test_split(pi=3)
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
                                columns=pd.read_csv(raw_behaviors_file_paths_rp4[Behavior.CNC_BACKDOOR_JAKORITAR]).drop(
                                    time_status_columns, axis=1).drop(all_zero_columns, axis=1).columns,
                                index=["PC" + str(i) for i in range(1, n + 1)])
        return loadings

    @staticmethod
    def get_highest_weight_loading_scores_for_pc(n_pcs=15, pcn="PC1"):
        # maxCol = lambda x: max(x.min(), x.max(), key=abs)
        df = DataProvider.get_pca_loading_scores_dataframe(n_pcs)
        # df['max_loading_score'] = df.apply(maxCol, axis=1)
        sorted_pc = df.loc[pcn].reindex(df.loc[pcn].abs().sort_values(ascending=False).index)
        return sorted_pc
