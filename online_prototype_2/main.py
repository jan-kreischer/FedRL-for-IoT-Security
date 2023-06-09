import time
import os
import subprocess
import psutil
import json
import jsonschema
from abc import ABC, abstractmethod
from collections import deque
from online_data_manager import DataManager
#from anomaly_detector import AutoEncoderInterpreter
#from agent import Agent
#import torch
import numpy as np

# TODO: add storage of agent networks after some episodes

# TODO: make abstract (template method pattern) in case of multiple online methods
class OnlineRL():
    monitor_counter = 0
    start_str_datafile = "online_samples_"

    # start_str_datafile = "normal_samples_"

    def __init__(self, ae=None, agent=None):#AutoEncoderInterpreter=None, agent: Agent=None):
        self.ae_interpreter = ae
        self.agent = agent

    # TODO: do something with learning data, progress monitoring
    def activate_learning(self, interval=900):
        episode_rewards = deque([0.0], maxlen=100)
        steps = 0
        while True:
            episode_reward, steps = self.start_monitoring_episode(steps)
            if episode_reward:
                episode_rewards.append(episode_reward)
            time.sleep(interval)

    def start_monitoring_episode(self, steps):
        self.monitor(180)
        decision_data = self.read_data()
        isAnomaly = self.interprete_data(decision_data)
        if isAnomaly:
            r = 0
            while isAnomaly:
                action = self.choose_action(decision_data)
                self.launch_mtd(action)
                time.sleep(180)  # wait for 3 mins -> make dependent on pid of mtd script?
                self.monitor(180)
                after_data = self.read_data()
                isAnomaly = self.interprete_data(after_data)
                r += self.provide_feedback_and_update(decision_data, action, after_data, isAnomaly)
                steps += 1
                if steps % TARGET_UPDATE_FREQ == 0:
                    self.agent.update_target_network()

                decision_data = after_data  # to reuse newly monitored data from afterstate
            print("successfully mitigated attack using action: " + str(ACTIONS[action]))
            return r, steps
        else:
            return None, steps

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

    # TODO: adapt to monitor again if no samples left after filtering
    def read_data(self):
        """should only be used after running self.montitor(n) first, since it relies
        on the monitor_counter to open the file online_samples_{c}"""
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
        flagged_anomalies = self.ae_interpreter.predict(data)
        print(flagged_anomalies)
        return False
        #return (torch.sum(flagged_anomalies) / len(flagged_anomalies) > 0.5).item()

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

        with open('config.json') as json_file:
            data = json.load(json_file)

        # get commands from config file exactly:
        selected_mtd = None
        for mtd_conf in data[MTD_SOLUTIONS]:
            if mtd_conf[SCRIPT_NAME] == ACTIONS[n]:
                selected_mtd = mtd_conf
        print(selected_mtd)
        old_dir = os.getcwd()
        print(old_dir)
        os.chdir(selected_mtd[PATH])
        print(os.getcwd())

        mtd_params = ""
        try:
            mtd_params += str(selected_mtd[PARAMS])
        except KeyError:
            pass

        print("run: " + selected_mtd[RUN_PREFIX] + ' ' + str(selected_mtd[SCRIPT_NAME]) + ' ' + mtd_params)
        os.system(selected_mtd[RUN_PREFIX] + ' ' + str(selected_mtd[SCRIPT_NAME]) + ' ' + mtd_params)
        os.chdir(old_dir)

    def provide_feedback_and_update(self, decision_data, action, after_data, isAnomaly, max_len=10):
        """feeds back [data] from afterstate behavior, along with a flag of whether it corresponds to normal behavior"""

        reward = 1 if isAnomaly else -1
        done = not isAnomaly

        # adding max monitored samples to replay_buffer
        ld, la = len(decision_data), len(after_data)
        m = min(ld, la)
        n_samples = max_len if m > max_len else m
        for i in range(n_samples):
            self.agent.replay_buffer.append((decision_data[i, :], action, reward,
                                             after_data[i, :], done))

        # call agent.learn
        self.agent.learn()
        return reward


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
TYPE = 'Type'
MTD_SOLUTIONS = 'MTDSolutions'
PARAMS = 'Params'
RUN_PREFIX = 'RunWithPrefix'

DIMS = 15
# !!!Order is important!!!
ACTIONS = ("ChangeIpAddress.py", "RemoveRootkit.py",
           "CreateDummyFiles.py", "ChangeFileTypes.py")
GAMMA = 0.99
BATCH_SIZE = 100
BUFFER_SIZE = 500
MIN_REPLAY_SIZE = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
TARGET_UPDATE_FREQ = 100
LEARNING_RATE = 1e-5
N_EPISODES = 5000
LOG_FREQ = 100

# TODO: initialize agent with full memory buffer -> as per last episode of offline training
if __name__ == '__main__':

    with open('config.json') as json_file:
        data = json.load(json_file)
        validate_config_file(data)

    # get pretrained anomaly detector
    # pretrained_model = torch.load("autoencoder_model.pth")
    # ae_interpreter = AutoEncoderInterpreter(pretrained_model['model_state_dict'],
    #                                         pretrained_model['threshold'], in_features=DIMS)
    #
    # get pretrained agent with defined parameters
    # pretrained_state = torch.load("trained_models/agent_0.pth")
    # pretrained_agent = Agent(input_dims=DIMS, n_actions=len(ACTIONS), buffer_size=BUFFER_SIZE,
    #                          batch_size=pretrained_state['batch_size'], lr=pretrained_state['lr'],
    #                          gamma=pretrained_state['gamma'], epsilon=pretrained_state['eps'],
    #                          eps_end=pretrained_state['eps_min'], eps_dec=pretrained_state['eps_dec'])
    # pretrained_agent.online_net.load_state_dict(pretrained_state['online_net_state_dict'])
    # pretrained_agent.target_net.load_state_dict(pretrained_state['target_net_state_dict'])
    # pretrained_agent.replay_buffer = pretrained_state['replay_buffer']
    #
    # controller = OnlineRL(ae=ae_interpreter, agent=agent)
    controller = OnlineRL()
    controller.launch_mtd(2)
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