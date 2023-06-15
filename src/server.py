import os
import torch
from torch import nn
import copy
from typing import List, Dict
import threading
import numpy as np
import json
from tabulate import tabulate
import multiprocessing

from src.custom_types import Behavior, MTDTechnique, Execution, Evaluation, actions, mitigated_by

from src.agent import Agent
from src.client import Client
from src.autoencoder import AutoEncoder

class Server:
    def __init__(self,
                 global_agent: Agent,
                 test_data,
                 state_interpreter: AutoEncoder,
                ):
        self.clients = []
        self.global_agent = global_agent
        self.test_data = test_data
        self.state_interpreter = state_interpreter

        self.performance_evaluations = {}
        self.performance_evaluations[0] = {}
        self.performance_evaluations["rounds"] = []
        for behavior in Behavior:
            self.performance_evaluations[0][behavior] = []
        
        
    def aggregate_weights(self):
        client_parameters = {client.client_id: client.get_weights() for client in self.clients}
        client_weight = 1/len(self.clients)
        
        aggregated_weights = copy.deepcopy(next(iter(client_parameters.values())))  # names
        for parameter_name in aggregated_weights:
            aggregated_weights[parameter_name] = torch.zeros(aggregated_weights[parameter_name].shape)
        for client_parameter in client_parameters.values():
            for parameter_name in aggregated_weights:
                aggregated_weights[parameter_name] += client_parameter[parameter_name] * client_weight  # averaging

        self.global_agent.update_weights(aggregated_weights)
        
        
    def broadcast_weights(self):
        for client in self.clients:
            client.receive_weights(self.global_agent.get_weights())

    def add_client(self, client: Client):
        self.clients.append(client)
        
        self.performance_evaluations[client.client_id] = {}
        for behavior in Behavior:
            self.performance_evaluations[client.client_id][behavior] = []
    
    
    def run_federation(self, 
                       nr_rounds: int,
                       nr_episodes_per_round: int,
                       execution=Execution.SEQUENTIAL,
                       evaluations=[],
                       evaluation_frequency: int = 1,
                       verbose=True):
        
        self.verbose = verbose
        self.nr_rounds = nr_rounds
        
        evaluations = list(map(lambda x: x.name, evaluations))
        
        print(f"Training each of the {len(self.clients)} clients for a total of {nr_rounds*nr_episodes_per_round} episodes distributed over {nr_rounds} rounds with {nr_episodes_per_round} episodes per round.\n")
        # Performance evaluation for the randomly initialized agent
        # self.performance_evaluation(self.global_agent, self.test_data, 0)
        agents = list(map(lambda client: client.agent, self.clients))
        agents.append(self.global_agent)
        for agent in agents:
            self.performance_evaluation(agent, self.test_data, 0)
        
        for nr_round in range(1, nr_rounds+1):

            if verbose:
                print(f">>> SERVER TRAINING ROUND {nr_round}/{nr_rounds} <<<")
                
            for client in self.clients:
                client.receive_weights(self.global_agent.get_weights())
                   
            match execution:
                case Execution.MULTI_THREADED:
                    threads = []
                    for client in self.clients:
                        #client.agent.online_net.share_memory()
                        t = threading.Thread(target=Client.train_agent, args=(client, nr_episodes_per_round))
                        t.start()
                        threads.append(t)
                        
                    for t in threads:
                        t.join()
                        
                        
                case Execution.MULTI_PROCESSING:
                    threads = []
                    for client in self.clients:
                        thread = multiprocessing.Process(target=Client.train_agent, args=(client, nr_episodes_per_round))  
                        thread.start()
                        threads.append(thread)

                    for thread in threads:
                        thread.join() 
                       
                    
                case Execution.MULTI_PROCESSING_POOL:
                    pool = multiprocessing.Pool(processes=len(self.clients))
                    pool.starmap(Client.train_agent, map(lambda client: (client, nr_episodes_per_round), self.clients))
                    pool.close()
                    pool.join()
                    
                    
                case _:
                    for client in self.clients:
                        client.train_agent(nr_episodes_per_round)
                    
                    
            self.aggregate_weights()
            if nr_round % evaluation_frequency == 0:
                self.performance_evaluations['rounds'].append(nr_round)
                #agents = list(map(lambda client: client.agent, self.clients))
                self.performance_evaluation(client.agent, self.test_data, nr_round)
            
                for client in self.clients:
                    if Evaluation.PERFORMANCE_EVALUATION.name in evaluations or Evaluation.LOCAL_PERFORMANCE_EVALUATION.name in evaluations:
                        self.performance_evaluation(client.agent, self.test_data, nr_round)
  
                    if Evaluation.BEHAVIOR_EVALUATION.name in evaluations or Evaluation.LOCAL_BEHAVIOR_EVALUATION.name in evaluations:
                        self.behavior_action_evaluation(agent, self.test_data)
                
                if Evaluation.PERFORMANCE_EVALUATION.name in evaluations or Evaluation.GLOBAL_PERFORMANCE_EVALUATION.name in evaluations:
                    self.performance_evaluation(self.global_agent, self.test_data, nr_round)
                
                if Evaluation.BEHAVIOR_EVALUATION.name in evaluations or Evaluation.GLOBAL_BEHAVIOR_EVALUATION.name in evaluations:
                    self.behavior_action_evaluation(self.global_agent, self.test_data)
                
        if Evaluation.FINAL_PERFORMANCE_EVALUATION.name in evaluations:
            self.performance_evaluation(self.global_agent, self.test_data, nr_round)
            
        if Evaluation.LEARNING_CURVE.name in evaluations:
            for client in self.clients:
                client.plot_learning_curve(nr_round)

                     
    def final_training_accuracy(self):
        final_training_accuracy = list(self.global_agent.total_accuracies.values())[-1]
        final_mean_class_accuracy = list(self.global_agent.mean_class_accuracies.values())[-1]
        return final_training_accuracy, final_mean_class_accuracy
    
    def performance_evaluation(self, agent: Agent, test_data, nr_round, print_result=False):
        # check predictions with learnt dqn
        agent.online_net.eval()
        res_dict = {}
        objective_dict = {}
        with torch.no_grad():
            for b, d in test_data.items():
                #if b != Behavior.NORMAL:
                cnt_corr = 0
                cnt = 0
                if self.state_interpreter is not None:
                    for state in d:
                        cnt+=1
                        if b == Behavior.NORMAL:
                            #print(state[:-1].shape)
                            #print(state.shape)
                            #print(type(state))
                            is_normal = (torch.sum(self.state_interpreter.predict(state[:-1].astype(np.float32).reshape(1,-1), n_std=5)) / len(state)) <= 0.5
                            if is_normal:
                                 cnt_corr+=1
                        else:
                            is_abnormal = (torch.sum(self.state_interpreter.predict(state[:-1].astype(np.float32).reshape(1,-1), n_std=5)) / len(state)) <= 0.5
                            if is_abnormal:
                                action = agent.take_greedy_action(state[:-1])
                                if b in mitigated_by[action]:
                                    cnt_corr += 1
                            else:
                                print("misclassified as normal")
                else:
                    for state in d:
                        action = agent.take_greedy_action(state[:-1])
                        if b in mitigated_by[action]:
                            cnt_corr += 1
                        cnt += 1
                    
                res_dict[b] = (cnt_corr, cnt)

                for i in range(len(MTDTechnique)):
                    if b in mitigated_by[i]:
                        objective_dict[b] = actions[i]
               
        #objective_dict[Behavior.NORMAL] = MTDTechnique.CONTINUE
        objective_dict[Behavior.NORMAL] = "MTDTechnique.CONTINUE"
        labels = ("Behavior", "Accuracy", "Objective", "Nr. Samples")
        results = []
        
        #total_accuracy, mean_class_accuracy, total_number_samples = self.compute_accuracies(res_dict)
        
        number_correctly_classified = 0
        total_number_samples = 0
        class_accuracies = []
        
        for behavior, t in res_dict.items():
            #print("---")
            #print(f"behavior: {behavior} {behavior.value}")
            number_correctly_classified+=t[0]
            total_number_samples+=t[1]
            accuracy = t[0] / t[1]
            class_accuracies.append(accuracy)
            accuracy_in_percent = round(accuracy* 100, 2)
            #print(f"accuracy_in_percent: {accuracy_in_percent}")
            self.performance_evaluations[agent.agent_id][behavior].append(accuracy_in_percent)
            results.append((behavior, accuracy_in_percent, objective_dict[behavior], t[1]))
            agent.behavior_accuracies[behavior][nr_round] = accuracy
           
        total_accuracy = number_correctly_classified/total_number_samples
        agent.total_accuracies[nr_round] = total_accuracy
        
        mean_class_accuracy = np.mean(class_accuracies, axis=0)
        agent.mean_class_accuracies[nr_round] = mean_class_accuracy
        
        results.append(("GLOBAL MACRO ACCURACY", round(mean_class_accuracy*100, 2), "MISCELLANEOUS", total_number_samples))
        results.append(("GLOBAL MICRO ACCURACY", round(total_accuracy*100, 2), "MISCELLANEOUS", total_number_samples))
        
        if agent.agent_id == 0:
            print(f"{agent.get_name()} > Performance Evaluation after Round {nr_round}")
            print(tabulate(results, headers=labels, tablefmt="pipe"))
            print("")
        
    
    def behavior_action_evaluation(self, agent: Agent, test_data):
        n_behaviors = len(Behavior) - 1
        n_actions = len(MTDTechnique)
        matrix = np.zeros((n_behaviors, n_actions))
        
        behavior_action_counts = {}
        for behavior in Behavior:
            #if behavior != Behavior.NORMAL:
            behavior_action_counts[behavior] = np.zeros((1, n_actions))
        
        with torch.no_grad():
            for behavior, state_samples in test_data.items():
                #if behavior != Behavior.NORMAL:
                for state_sample in state_samples:
                    a_pred = agent.take_greedy_action(state_sample[:-1])
                    behavior_action_counts[behavior][0][a_pred]+=1
          
        labels = ["behavior"] + list(map(lambda x: x.value, actions))
        
        results = np.empty((0, n_actions+1))
        for behavior, action_counts in behavior_action_counts.items():
            row = np.append(np.array([behavior]).reshape(-1, 1), action_counts, axis=1)
            results = np.concatenate((results, row), axis=0)
        
        print(f"\n{agent.get_name()} > Behavior Action Evaluation")
        print(tabulate(results, headers=labels, tablefmt="pipe"))

        
    def print_performance_evaluations(self):
        for behavior in Behavior:
            for id in range(len(self.clients)+1):
                performance = self.performance_evaluations[id][behavior]
                rounds = self.performance_evaluations['rounds']
                plt.title(f"Mitigation Accuracy for {behavior}")
                plt.ylim([0,100])
                if id == 0:
                    plt.plot([*range(len(performance))], performance, linestyle='-', label=f"Global Agent") # plotting t, a separately 
                else:
                    plt.plot([*range(len(performance))], performance, linestyle='--', label=f"Agent {id}") # plotting t, a separately 
            plt.xlabel('Round')
            plt.ylabel('Accuracy [%]')
            plt.show()
            
    def show_misclassification_rates(self):
        for client in self.clients:
            print(f"Client {client.client_id}")
            client.environment.plot_fn_ratio()
            client.environment.plot_fp_ratio()
            print(f"---")
            
    def plot_learning_curves(self):
        for client in self.clients:
            client.plot_learning_curve(self.nr_rounds)
            
            
            
            
            
