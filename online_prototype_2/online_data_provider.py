import pickle
from typing import Dict
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import joblib


time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ['cpuNice', 'cpuHardIrq', 'alarmtimer:alarmtimer_fired',
       'alarmtimer:alarmtimer_start', 'cachefiles:cachefiles_create',
       'cachefiles:cachefiles_lookup', 'cachefiles:cachefiles_mark_active',
       'dma_fence:dma_fence_init', 'udp:udp_fail_queue_rcv_skb']
    #
    #
    # ["alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
    #                 "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active", "dma_fence:dma_fence_init",
    #                 "udp:udp_fail_queue_rcv_skb"]

# choose which data to use for the scaler/pca
all_file = "all_data_filtered_external_True_constant_True_outliers_True.csv"
pca_file = "data_transforms/pcafit.gz"
scaler_file = "data_transforms/scaler.gz"

class OnlineDataProvider:
    @staticmethod
    def __parse_file_to_df(file_name, filter_suspected_external_events=True,
                              filter_constant_columns=True,
                              filter_outliers=True,
                              keep_status_columns=False) -> pd.DataFrame:

        df = pd.read_csv(file_name)

        if filter_suspected_external_events:
            # filter first 2 samples and last measurement
            df = df.iloc[2:-1]
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

        #df.to_csv(file_name + "preprocessed", index_label=False)
        return df

    @staticmethod
    def __get_scaler_and_pca(scaling_minmax=True, dim=15):
        all_filtered_data = pd.read_csv(all_file).to_numpy()[:, :-1]
        scaler = StandardScaler() if not scaling_minmax else MinMaxScaler()
        scaler.fit(all_filtered_data)
        pca = PCA(n_components=dim)
        pca.fit(all_filtered_data)

        if not os.path.isfile(scaler_file) or not os.path.isfile(pca_file):
            joblib.dump(scaler, scaler_file)
            joblib.dump(pca, pca_file)
            # with open(scaler_file, "wb") as sf:
            #     pickle.dump(scaler, sf)
            # with open(pca_file, "wb") as pf:
            #     pickle.dump(pca, pf)
        return scaler, pca

    @staticmethod
    def get_scale_and_pca_transformed_data(file_name):
        data = OnlineDataProvider.__parse_file_to_df(file_name).values
        if not os.path.isfile(pca_file) or not os.path.isfile(scaler_file):
            scaler, pca = OnlineDataProvider.__get_scaler_and_pca()
        else:
            scaler = joblib.load(scaler_file)
            pca = joblib.load(pca_file)
            # with open('data_transforms/scaler.obj', 'rb') as sf:
            #     scaler = pickle.load(sf)
            # with open('data_transforms/pcafit.obj', 'rb') as pf:
            #     pca = pickle.load(pf)

        return pca.transform(scaler.transform(data))



