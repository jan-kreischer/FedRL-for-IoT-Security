import time
import os
import subprocess
import psutil
from abc import ABC, abstractmethod
from online_data_manager import DataManager
from anomaly_detector import AutoEncoderInterpreter
import torch


# TODO: make abstract (template method pattern) in case of multiple online methods
class OnlineRL():
    monitor_counter = 0
    start_str_datafile = "online_samples_"
    #start_str_datafile = "normal_samples_"

    def __init__(self):
        pass

    # TODO adapt to monitor every hour
    def learn_online(self):
        self.monitor(60)
        data = self.read_data()
        isAnomaly = self.interprete_data(data)
        if isAnomaly:
            while isAnomaly:
                action = self.dqn_predict(data)
                self.launch_mtd(action)
                # wait for n seconds
                time.sleep(180)
                data = self.monitor(60)
                isAnomaly = self.interprete_data(data)
                self.provide_feedback_and_update(data, isAnomaly)

    def monitor(self, t: int):

        OnlineRL.monitor_counter += 1

        # call monitoring shell script from python
        print("running rl_sampler subprocess")
        # subprocess.run(["./rl_sampler_online.sh", str(OnlineRL.monitor_counter)])
        p = subprocess.Popen(["./rl_sampler_online.sh", str(OnlineRL.monitor_counter), "&"])
        print(p.pid)

        time.sleep(t)

        # killing all related processes
        print("killing process")
        print(p.pid)
        kill(p.pid)

    def read_data(self):
        # use num to read "online_samples_{c}..."
        # find filename

        prefixed = [filename for filename in os.listdir('.') if
                    filename.startswith(OnlineRL.start_str_datafile + str(OnlineRL.monitor_counter))]

        print(prefixed)
        assert len(prefixed) == 1, "Only one file with counter number should exist"

        # read file and apply preprocessing
        fname = prefixed[0]
        print(fname)
        print(os.getcwd())
        data = DataManager.get_scale_and_pca_transformed_data(fname)
        print(data.shape)
        # TODO: apply scaling and pca as offline
        return data

    def interprete_data(self, data):
        pass

    def dqn_predict(self):
        pass

    def launch_mtd(self, n: int):
        pass

    def provide_feedback_and_update(self, data, isAnomaly):
        pass


def kill(pid):
    '''Kills all process'''
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


if __name__ == '__main__':
    # TODO:
    #  call monitoring, or at least read data -> last 10 samples (180s!)
    #  call AD/pretrained AE network and get results
    #  call DQN if anomaly is flagged, else wait till next monitoring (every hour?)
    #  call the MTD deployerframework with the MTD matching the action
    #  wait until MTD execution has finished, e.g. 2 mins
    #  call monitoring -> last 10 samples
    #  call AD on afterstate
    #  provide feedback according to afterstate being flagged normal to agent

    controller = OnlineRL()

    # uncomment before moving online
    # controller.monitor(180)
    OnlineRL.monitor_counter += 3
    # read the monitored data from file and apply all preset scalings and transforms
    data = controller.read_data()

    # run data through pretrained anomaly detector
    pretrained_model = torch.load("autoencoder_model.pth")
    ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
                                            pretrained_model['threshold'], in_features=data.shape[1])
    #print(f"ae_interpreter threshold: {ae_interpreter.threshold}")
    flagged_anomalies = ae_interpreter.predict(data)
    print(flagged_anomalies)

    # TODO: options:
    # - improve the accuracy of the anomaly detector
    # -> 1. change testdata: closing shh session in normal monitor -> python script call with nohup
    # TODO:compare data monitored offline and via this script for differences, exclude features? more pcs?

    # ---> flagged all as anomaly - conclusion: process python3 main.py mainly influences the normal behavior
    # -> 2. change traindata: retrain agent on more realistic data
    # ---> during monitoring python main.py must run in a similar stack situation

    # dependent on flagged majority deploy MTD or not
    # in case of majority abnormal
    if torch.sum(flagged_anomalies) / len(flagged_anomalies) > 0.5:
        #raise UserWarning("Should not happen! AE fails to predict majority of normal samples")
        controller.dqn_predict()
    else:
        # do nothing until next monitoring loop
        pass






