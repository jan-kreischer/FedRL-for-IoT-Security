import torch
from torch import nn
import copy
from typing import List, Dict
import threading
import numpy as np
import json
import prototypes.prototype_01.sensor_environment
from prototypes.prototype_01.sensor_environment import SensorEnvironment
import prototypes.prototype_01.deep_q_network
import prototypes.prototype_01.agent
from prototypes.prototype_01.agent import Agent
import prototypes.prototype_01.client

class Server:
    def __init__(self, global_agent: Agent, test_data, nr_rounds = NR_ROUNDS, parallelized=False, verbose=True):
        self.clients = []
        self.global_agent = global_agent
        self.test_data = test_data
        self.nr_rounds = nr_rounds
        self.parallelized = parallelized
        self.verbose = verbose
        
    def aggregate_weights(self):
        print("=== AGGREGATING WEIGHTS ===")
        client_params = {client.client_id: client.get_weights() for client in self.clients}
        new_params = copy.deepcopy(next(iter(client_params.values())))  # names
        for name in new_params:
            new_params[name] = torch.zeros(new_params[name].shape)
        for client_id, params in client_params.items():
            client_weight = 1/len(self.clients)
            for name in new_params:
                new_params[name] += params[name] * client_weight  # averaging
        #set new parameters to global model
        self.global_agent.update_weights(new_params)
        #print(new_params)
        return new_params
         
    def broadcast_weights(self):
        """ Send to all clients """
        for client in self.clients:
            client.receive_weights(self.global_agent.get_weights())

    def add_client(self, client: Client):
        self.clients.append(client)
      
    def training_dist(self, verbose=False):
        for nr_round in range(self.nr_rounds):
            if self.verbose:
                print(f">>> SERVER TRAINING ROUND {nr_round + 1}/{self.nr_rounds} <<<")
            for client in self.clients:
                print(f"> AGENT {client.client_id} TRAINING ROUND {nr_round + 1}/{self.nr_rounds} <")
                client.receive_weights(self.global_agent.get_weights())
                
                if self.parallelized:
                    # Parallel training
                    threads = []
                    for client in self.clients:
                        #client.agent.model.share_memory()
                        t = threading.Thread(target=Client.train_agent, args=(client, 1000, 100))
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                else:
                    # Sequential training
                    for client in self.clients:
                        client.train_agent(1000, 100, verbose=verbose)
            
            self.aggregate_weights()
            #print(f">> EVALUATION TRAINING ROUND {nr_round + 1}/{self.nr_rounds} <<")
            for client in self.clients:
                prefix = f"ROUND-{(nr_round + 1):02d}_AGENT-{(client.client_id):02d}"
                client.plot_learning_curve(prefix + "_TRAINING-SUMMARY", save_results=True)
                client.plot_performance_evaluation(self.test_data, prefix + "_PERFORMANCE EVALUATION")

            # When every client is done training then we can aggregate weights
            print(f"GLOBAL AGENT")
            evaluate_agent(self.global_agent, self.test_data)
            
    def plot_learning_curves(self):
        for client in self.clients:
            episode_returns, eps_history = client.get_training_summary()
            plot_learning_curve(f"{client.client_id}", episode_returns, eps_history)
            
    def save_experiment_summary(self, path):
        experiment_summary = {}
        experiment_summary["server"] = {}
        experiment_summary["server"]["nr_clients"] = len(self.clients)
        experiment_summary["server"]["nr_rounds"] = self.nr_rounds
        experiment_summary["server"]["paralellized"] = self.parallelized
        experiment_summary["clients"] = {}
        for client in self.clients:
            experiment_summary["clients"][f"client_{client.client_id}"] = {}
            experiment_summary["clients"][f"client_{client.client_id}"]["gamma"] = client.agent.gamma
            experiment_summary["clients"][f"client_{client.client_id}"]["epsilon_max"] = client.agent.eps_max
            experiment_summary["clients"][f"client_{client.client_id}"]["epsilon_min"] = client.agent.eps_min
            experiment_summary["clients"][f"client_{client.client_id}"]["epsilon_decay"] = client.agent.eps_dec
            
        with open(os.path.join(path, "experiment_summary.txt"), "wt") as fp:
            json.dump(experiment_summary, fp, indent=2)