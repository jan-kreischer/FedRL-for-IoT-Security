from typing import Dict
from scipy import stats
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#from tabulate import tabulate
import numpy as np
import pandas as pd
import os


time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ["alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
                    "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active", "dma_fence:dma_fence_init",
                    "udp:udp_fail_queue_rcv_skb"]



class DataManager:
    @staticmethod
    def parse_file_to_df(file_name, filter_suspected_external_events=True,
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