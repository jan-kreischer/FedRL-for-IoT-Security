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
raw_behaviors_dir = "raw_behaviors_no_agent_rp4"
raw_behaviors_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: f"data/{raw_behaviors_dir}/normal_samples_2022-06-13-11-25_50s",
    Behavior.RANSOMWARE_POC: f"data/{raw_behaviors_dir}/ransomware_samples_2022-06-20-08-49_50s",
    Behavior.ROOTKIT_BEURK: f"data/{raw_behaviors_dir}/rootkit_beurk_samples_2022-06-17-09-08_50s",
    Behavior.ROOTKIT_BDVL: f"data/{raw_behaviors_dir}/rootkit_bdvl_samples_2022-06-16-19-16_50s",
    Behavior.CNC_THETICK: f"data/{raw_behaviors_dir}/cnc_backdoor_jakoritar_samples_2022-06-18-09-35_50s",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{raw_behaviors_dir}/cnc_thetick_samples_2022-06-19-16-54_50s"
}
# TODO: These columns are derived from data_availability.py -> check data
time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ['cpuNice', 'cpuHardIrq', 'alarmtimer:alarmtimer_fired',
                    'alarmtimer:alarmtimer_start', 'cachefiles:cachefiles_create',
                    'cachefiles:cachefiles_lookup', 'cachefiles:cachefiles_mark_active',
                    'dma_fence:dma_fence_init', 'udp:udp_fail_queue_rcv_skb']

# behaviors with MTD framework/Agent Components running as per directory "online_prototype_monitoring"
decision_state = "decision"
decision_states_dir = "decision_states_online_agent"
decision_states_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: f"data/{decision_states_dir}/normal_noexpfs_online_samples_1_2022-08-15-14-07_5s",
    Behavior.RANSOMWARE_POC: f"data/{decision_states_dir}/ransom_noexpfs_online_samples_1_2022-08-16-08-43_5s",
    Behavior.ROOTKIT_BDVL: f"data/{decision_states_dir}/rootkit_bdvl_online_samples_1_2022-08-12-16-40_5s",
    Behavior.CNC_BACKDOOR_JAKORITAR: f"data/{decision_states_dir}/cnc_jakoritar_online_samples_1_2022-08-13-06-50_5s"
}
afterstate = "after"
afterstates_dir = "afterstates_online_agent"
afterstates_file_paths: Dict[Behavior, Dict[MTDTechnique, str]] = {
    Behavior.NORMAL: {
        MTDTechnique.RANSOMWARE_DIRTRAP: f"data/{afterstates_dir}/normal_as_dirtrap_expfs_online_samples_2_2022-08-17-14-23_5s",
        #MTDTechnique.RANSOMWARE_FILE_EXT_HIDE: f"data/{afterstates_dir}/",
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


class DataProvider:

    @staticmethod
    def parse_all_behavior_data(filter_suspected_external_events=True,
                                filter_constant_columns=True,
                                filter_outliers=True,
                                keep_status_columns=False) -> Dict[Behavior, np.ndarray]:
        # print(os.getcwd())
        file_name = f'../data/{raw_behaviors_dir}/all_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        # if os.path.isfile(file_name):
        #   full_df = pd.read_csv(file_name)
        # full_df = pd.DataFrame()
        bdata = {}

        for attack in raw_behaviors_file_paths:
            df = DataProvider.__get_filtered_df(raw_behaviors_file_paths[attack],
                                                filter_suspected_external_events=filter_suspected_external_events,
                                                startidx=72,
                                                filter_constant_columns=filter_constant_columns,
                                                filter_outliers=filter_outliers,
                                                keep_status_columns=keep_status_columns)
            df['attack'] = attack
            bdata[attack] = df.to_numpy()
            # if not os.path.isfile(file_name): full_df = pd.concat([full_df, df])

        # full_df.to_csv(file_name, index_label=False)
        return bdata

    @staticmethod
    def parse_raw_behavior_files_to_df(filter_suspected_external_events=True,
                                       filter_constant_columns=True,
                                       filter_outliers=True,
                                       keep_status_columns=False) -> pd.DataFrame:

        print(os.getcwd())
        file_name = f'data/{raw_behaviors_dir}/all_data_filtered_external_{str(filter_suspected_external_events)}' \
                    f'_constant_{str(filter_constant_columns)}_outliers_{str(filter_outliers)}'

        if keep_status_columns:
            file_name += "_keepstatus"
        file_name += ".csv"

        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        full_df = pd.DataFrame()

        for attack in raw_behaviors_file_paths:
            df = DataProvider.__get_filtered_df(raw_behaviors_file_paths[attack],
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
    def __get_filtered_df(path, filter_suspected_external_events=True, startidx=10, endidx=-1,
                          filter_constant_columns=True, const_cols=all_zero_columns,
                          filter_outliers=True,
                          keep_status_columns=False, stat_cols=time_status_columns):

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

        return df

    @staticmethod
    def get_scaled_all_data(scaling_minmax=True):
        all_data = DataProvider.parse_raw_behavior_files_to_df().to_numpy()[:, :-1]
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(all_data)

        bdata = DataProvider.parse_all_behavior_data()
        scaled_bdata = {}
        # return directory as
        for b in bdata:
            scaled_bdata[b] = np.hstack((scaler.transform(bdata[b][:, :-1]), np.expand_dims(bdata[b][:, -1], axis=1)))

        return scaled_bdata

    @staticmethod
    def get_scaled_train_test_split(split=0.8, scaling_minmax=True):
        bdata = DataProvider.parse_all_behavior_data()

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
    def get_reduced_dimensions_with_pca(dim=15):
        strain, stest, scaler = DataProvider.get_scaled_train_test_split()
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
        strain, stest, scaler = DataProvider.get_scaled_train_test_split()
        all_strain = strain[Behavior.NORMAL]
        for b in strain:
            if b != Behavior.NORMAL:
                all_strain = np.vstack((all_strain, strain[b]))

        pca = PCA(n_components=n)
        pca.fit(all_strain[:, :-1])
        return pca

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
        plt.savefig(f"screeplot_n_{n}.pdf")

    @staticmethod
    def get_pca_loading_scores_dataframe(n=15):
        pca = DataProvider.fit_pca(n)
        loadings = pd.DataFrame(pca.components_,
                                columns=pd.read_csv(raw_behaviors_file_paths[Behavior.CNC_BACKDOOR_JAKORITAR]).drop(
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
