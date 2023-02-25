from typing import Dict
from custom_types import Behavior
from scipy import stats
from tabulate import tabulate
import numpy as np
import pandas as pd
import os
from data_manager import DataManager

# TODO:
#  sequence of staticmethods to
#  read in pool of data
#  preprocess?
#

# paths to data
data_file_paths: Dict[Behavior, str] = {
    Behavior.NORMAL: "../data/prototype_1/normal_samples_2022-06-13-11-25_50s",
    Behavior.RANSOMWARE_POC: "../data/prototype_1/ransomware_samples_2022-06-20-08-49_50s",
    Behavior.ROOTKIT_BEURK: "../data/prototype_1/rootkit_beurk_samples_2022-06-17-09-08_50s",
    Behavior.ROOTKIT_BDVL: "../data/prototype_1/rootkit_bdvl_samples_2022-06-16-19-16_50s",
    Behavior.CNC_THETICK: "../data/prototype_1/cnc_backdoor_jakoritar_samples_2022-06-18-09-35_50s",
    Behavior.CNC_BACKDOOR_JAKORITAR: "../data/prototype_1/cnc_thetick_samples_2022-06-19-16-54_50s"
}

time_status_columns = ["time", "timestamp", "seconds", "connectivity"]
all_zero_columns = ["alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
                    "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active", "dma_fence:dma_fence_init",
                    "udp:udp_fail_queue_rcv_skb"]

class EpisodeGenerator:

    @staticmethod
    def sample_random_state():
        pass


    @staticmethod
    def sample_next_state():
        pass

if __name__ == "__main__":
    DataManager.show_data_availability()