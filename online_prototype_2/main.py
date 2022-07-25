import time
import subprocess
import psutil
from abc import ABC, abstractmethod

# TODO: make abstract (template method pattern) in case of multiple online methods
class OnlineRL():
    monitor_counter = 0

    def __init__(self):
        pass

    # TODO adapt to monitor every hour
    def learn_online(self):
        data = self.monitor(60)
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
        #subprocess.run(["./rl_sampler_online.sh", str(OnlineRL.monitor_counter)])
        p = subprocess.Popen(["./rl_sampler_online.sh", str(OnlineRL.monitor_counter), "&"])
        print(p.pid)

        time.sleep(t)

        # killing all related processes
        print("killing process")
        print(p.pid)
        kill(p.pid)


    def read_data(self):
        pass


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
    #  call monitoring, or at least read data -> last 10 samples
    #  call AD/pretrained AE network and get results
    #  call DQN if anomaly is flagged, else wait till next monitoring (every hour?)
    #  call the MTD deployerframework with the MTD matching the action
    #  wait until MTD execution has finished, e.g. 2 mins
    #  call monitoring -> last 10 samples
    #  call AD on afterstate
    #  provide feedback according to afterstate being flagged normal to agent

    orchestrator = OnlineRL()
    orchestrator.monitor(15)


