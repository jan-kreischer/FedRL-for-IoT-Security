import time
import os
import subprocess
import psutil
import json
import jsonschema
from abc import ABC, abstractmethod
from online_data_manager import DataManager
from anomaly_detector import AutoEncoderInterpreter
from agent import Agent
import torch
import numpy as np



# TODO: make abstract (template method pattern) in case of multiple online methods
class OnlineRL():
    monitor_counter = 0
    start_str_datafile = "online_samples_"
    #start_str_datafile = "normal_samples_"

    def __init__(self, ae: AutoEncoderInterpreter, agent: Agent):
        self.ae_interpreter = ae
        self.agent = agent

    # TODO adapt to monitor every hour
    def learn_online(self):
        self.monitor(180)
        data = self.read_data()
        isAnomaly = self.interprete_data(data)
        if isAnomaly:
            while isAnomaly:
                action = self.choose_action(data)
                self.launch_mtd(action)
                # wait for n seconds
                time.sleep(180)
                data = self.monitor(180)
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
        return data

    def interprete_data(self, data):
        # print(f"ae_interpreter threshold: {ae_interpreter.threshold}")
        flagged_anomalies = ae_interpreter.predict(data)
        print(flagged_anomalies)
        return (torch.sum(flagged_anomalies) / len(flagged_anomalies) > 0.5).item()

    def choose_action(self, data):
        actions = []
        if np.random.random() > self.agent.epsilon:
            for state in data:
                actions.append(self.agent.take_greedy_action(state))
            print(actions)
            return max(set(actions), key=actions.count)
        else:
            for _ in data:
                return np.random.choice(self.agent.action_space)

    def launch_mtd(self, n: int):
        print("Launching MTD " + ACTIONS[n])
        # TODO:
        #  call the right MTD by integrating the json schema
        with open('config.json') as json_file:
            data = json.load(json_file)
            validate_config_file(data)

        # get commands from config file exactly:
        selected_mtd = None
        for mtd_conf in data[MTD_SOLUTIONS]:
            if mtd_conf[SCRIPT_NAME] == ACTIONS[n]:
                selected_mtd = mtd_conf
        print(selected_mtd)
        print(os.getcwd())
        os.chdir(selected_mtd[PATH])
        print(os.getcwd())

        mtd_params = ""
        try:
            mtd_params += str(selected_mtd[PARAMS])
        except KeyError:
            pass

        print("run: " + selected_mtd[RUN_PREFIX] + ' ' + str(selected_mtd[SCRIPT_NAME]) + ' ' + mtd_params)
        os.system(selected_mtd[RUN_PREFIX] + ' ' + str(selected_mtd[SCRIPT_NAME]) + ' ' + mtd_params)



    def provide_feedback_and_update(self, data, isAnomaly):
        pass


def kill(pid):
    '''Kills all process'''
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()

def validate_config_file(json_data):
    with open('config-schema.json') as json_schema_file:
        json_schema = json.load(json_schema_file)
        try:
            jsonschema.validate(instance=json_data, schema=json_schema, cls=jsonschema.Draft4Validator)
        except jsonschema.ValidationError as err:
            err.message = 'The config file does not follow the json schema and is therefore not compatible!'
            raise err
        return True


# jsonschema data
SCRIPT_NAME = 'ScriptName'
PATH = 'RelativePath'
#PRIORITY = 'Priority'
TYPE = 'Type'
MTD_SOLUTIONS = 'MTDSolutions'
#ATTACK_TYPES = 'AttackTypes'
#DEPL_POLICY = 'DeploymentPolicy'
#ALLOW_EXT = 'AllowAllExternalReports'
#WHITE_LIST = 'WhiteListForExternalReports'
PARAMS = 'Params'
RUN_PREFIX = 'RunWithPrefix'
#PORT = 'PortToUse'




DIMS = 15
# !!!Order is important!!!
ACTIONS = ("ChangeIpAddress.py", "RemoveRootkit.py",
           "CreateDummyFiles.py", "ChangeFileTypes.py")
GAMMA = 0.99
BATCH_SIZE = 100
BUFFER_SIZE = 500
MIN_REPLAY_SIZE = 100
EPSILON_START = 0.5#1.0
EPSILON_END = 0.01
TARGET_UPDATE_FREQ = 100
LEARNING_RATE = 1e-5
N_EPISODES = 5000
LOG_FREQ = 100


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

    pretrained_model = torch.load("autoencoder_model.pth")
    ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
                                            pretrained_model['threshold'], in_features=DIMS)

    # agent has no buffer yet
    agent = Agent(input_dims=DIMS, n_actions=len(ACTIONS), buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, eps_end=EPSILON_END)
    # get pretrained online and target dqn
    pretrained_online_net = torch.load("online_net_0.pth")
    pretrained_target_net = torch.load("target_net_0.pth")
    agent.online_net.load_state_dict(pretrained_online_net)
    agent.target_net.load_state_dict(pretrained_online_net)


    controller = OnlineRL(ae=ae_interpreter, agent=agent)
    controller.launch_mtd(0)
    exit(0)
    # uncomment before moving online
    # controller.monitor(180)
    OnlineRL.monitor_counter += 3
    # read the monitored data from file and apply all preset scalings and transforms
    data = controller.read_data()

    # run data through pretrained anomaly detector
    isAnomaly = controller.interprete_data(data)
    print(isAnomaly)
    if isAnomaly:
        action = controller.choose_action(data)
        print("chosen action: " + str(action))
        controller.launch_mtd(action)





    # TODO: options for improving the accuracy of the anomaly detector/dqn
    # -> 1. change testdata: closing shh session in normal monitor -> python script call with nohup
    # TODO:compare data monitored offline and via this script for differences, exclude features/ram/cpu which are intense for interpreters?
    #  more pcs?

    # ---> flagged all as anomaly - conclusion: process python3 main.py mainly influences the normal behavior
    # -> 2. change traindata: retrain agent on more realistic data
    # ---> during monitoring python main.py must run in a similar stack situation









