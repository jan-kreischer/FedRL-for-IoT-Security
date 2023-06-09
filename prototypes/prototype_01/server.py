import os
import torch
from torch import nn
import copy
from time import time, time_ns
from src.custom_types import Behavior, MTDTechnique, actions, supervisor_map

from typing import List, Dict
import threading
import numpy as np
import json
from tabulate import tabulate
from datetime import date
from src.custom_types import MTDTechnique, Behavior
#from multiprocessing import Process
import multiprocessing

from agent import Agent
from client import Client
from enums import Execution, Evaluation
 
class Server:
    def __init__(self, global_agent: Agent, test_data, experiment_id, save_path, nr_rounds, nr_episodes_per_round):
        self.clients = []
        self.global_agent = global_agent
        self.test_data = test_data
        self.save_path = save_path
        self.experiment_id = experiment_id
        self.file_path = os.path.join(save_path, f"experiment-{experiment_id:02d}_summary.md")
        self.nr_rounds = nr_rounds
        self.nr_epochs_per_round = nr_episodes_per_round
        #self.parallelized = parallelized
        self.total_training_time = None
        self.round_training_times = []

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
    
    

    def run_federation(self, execution=Execution.SEQUENTIAL, evaluations=[], evaluation_frequency: int = 10, verbose=True, document_results=True):
        evaluations = list(map(lambda x: x.name, evaluations))

        if Evaluation.TRAINING_TIME.name in evaluations:
            start_time = time()
        
        if document_results:
            self.document("")
            self.document(f"# Prototype 1 (Experiment {self.experiment_id})")
            self.document("---")
            self.document("")
            self.document(f"Executed on {date.today().strftime('%d.%m.%Y')}")
            self.save_experiment_summary()
        
        for nr_round in range(1, self.nr_rounds+1):
            if nr_round % evaluation_frequency == 0:
                if document_results:
                    self.document("")
                    self.document('<div style="page-break-after: always;"></div>')
                    self.document("")
                    self.document("---")
                    self.document(f"### Training Round {nr_round}/{self.nr_rounds}")
                    self.document("")

            if verbose:
                print(f">>> SERVER TRAINING ROUND {nr_round}/{self.nr_rounds} <<<")
                
            for client in self.clients:
                client.receive_weights(self.global_agent.get_weights())
                   
            match execution:
                case Execution.MULTI_THREADED:
                    threads = []
                    for client in self.clients:
                        #client.agent.online_net.share_memory()
                        t = threading.Thread(target=Client.train_agent, args=(client, self.nr_epochs_per_round))
                        t.start()
                        threads.append(t)
                        
                    for t in threads:
                        t.join()
                        
                        
                case Execution.MULTI_PROCESSING:
                    threads = []
                    for client in self.clients:
                        thread = multiprocessing.Process(target=Client.train_agent, args=(client, self.nr_epochs_per_round))  
                        thread.start()
                        threads.append(thread)

                    for thread in threads:
                        thread.join() 
                       
                    
                case Execution.MULTI_PROCESSING_POOL:
                    pool = multiprocessing.Pool(processes=len(self.clients))
                    pool.starmap(Client.train_agent, map(lambda client: (client, self.nr_epochs_per_round), self.clients))
                    pool.close()
                    pool.join()
                    
                    
                case _:
                    for client in self.clients:
                        client.train_agent(self.nr_epochs_per_round)
                    
                    
            self.aggregate_weights()
            if nr_round % evaluation_frequency == 0:
                print(f"Evaluating round {nr_round}")
                self.performance_evaluations['rounds'].append(nr_round)

                for client in self.clients:
                    self.document(f"- Training Round {nr_round} on Client {client.client_id} took {round(client.get_training_time(), 2)}s")
                
                if Evaluation.LEARNING_CURVE.name in evaluations:
                    for client in self.clients:
                        filename = f"round-{self.prefix(nr_round)}_agent-{(client.client_id):02d}_learning-curve.png"
                        client.plot_learning_curve(filename, nr_round)
                        self.document(f"![graph]({filename})")

                agents = list(map(lambda client: client.agent, self.clients))
                agents.append(self.global_agent)
                for agent in agents:
                    #print(f"\n=== {client.agent.get_name()} - Evaluation ===\n")
                    if Evaluation.PERFORMANCE_EVALUATION.name in evaluations:
                        self.performance_evaluation(agent, self.test_data, nr_round)
  
                    if Evaluation.CONFUSION_MATRIX.name in evaluations:
                        self.confusion_matrix(agent, self.test_data)

                    if Evaluation.BEHAVIOR_ACTION_EVALUATION.name in evaluations:
                        self.behavior_action_evaluation(agent, self.test_data)
                        
                #print(f"\n=== {self.global_agent.get_name()} - Evaluation ===\n")
                #if Evaluation.PERFORMANCE_EVALUATION in evaluations:
                #    self.performance_evaluation(self.global_agent, self.test_data, nr_round)
                #if Evaluation.CONFUSION_MATRIX in evaluations:
                #    self.confusion_matrix(self.global_agent, self.test_data)
                #if Evaluation.BEHAVIOR_ACTION_EVALUATION in evaluations:
                #    self.behavior_action_evaluation(self.global_agent, self.test_data)
        
            if Evaluation.TRAINING_TIME.name in evaluations:
                round_time = time()
                time_elapsed = round_time - start_time
                self.round_training_times.append(time_elapsed)
                #print(f"Total time elapsed until end of round {nr_round}: {time_elapsed}s")    
        
        if Evaluation.TRAINING_TIME.name in evaluations:
            end_time = time()
            total_training_time = end_time - start_time
            print(f"Total training time with {len(self.clients)} clients: {total_training_time}")
            self.total_training_time = total_training_time
            
            if document_results:
                self.document(f"\n ### Total training time with {len(self.clients)}: {round(total_training_time, 2)}s")
             
    def final_training_accuracy(self):
        #list(my_dict.values())[-1]
        final_training_accuracy = list(self.global_agent.total_accuracies.values())[-1]
        final_mean_class_accuracy = list(self.global_agent.mean_class_accuracies.values())[-1]
        return final_training_accuracy, final_mean_class_accuracy
    
    def document(self, text):
        text = text.replace("_", "\_")
        if self.save_path:
            with open(self.file_path,'a') as f:
                # Add newline to text
                text += "  \n"
                f.write(text)   
                
    def document_block(self, text):
        text = text.replace("_", "\_")
        if self.save_path:
            with open(self.file_path,'a') as f:
                # Add newline to text
                text += "\n \n  \n \n"
                f.write(text)  
                            
                    
    def prefix(self, round: int): 
        prefix_length = len(str(self.nr_rounds))
        return f"{round:0{prefix_length}d}"
    
    def save_experiment_summary(self):
        experiment_summary = {}
        self.document("## Configuration")
        self.document("### Server")
        self.document(f"- nr_clients: {len(self.clients)}")
        self.document(f"- nr_rounds: {self.nr_rounds}")
        self.document(f"- nr_epochs_per_round: {self.nr_epochs_per_round}")
        #self.document(f"- parallelized: {self.parallelized}")
        self.document("")
        
        for client in self.clients:
            self.document(f"### Client {client.client_id}")
            self.document(f"- gamma: {client.agent.gamma}")
            self.document(f"- learning_rate: {client.agent.lr}")
            self.document(f"- batch_size: {client.agent.batch_size}")
            self.document(f"- epsilon_max: {client.agent.eps_max}")
            self.document(f"- epsilon_min: {client.agent.eps_min}")
            self.document(f"- epsilon_decay: {client.agent.eps_dec}")
            self.document(f"- input_dims: {client.agent.input_dims}")
            self.document(f"- output_dims: {client.agent.n_actions}")
            self.document("")

            self.document(f"Training Data Split")
            for key, value in client.environment.train_data.items():
                self.document(f"- {len(value)} samples of {key}")
                
            #client.plot_training_data_split()
            #self.document(f"![](behavior_sample_distribution_on_client-{client.client_id:02d}.png)")
            
        self.document(f"### Global Agent") 
        self.document(f"- id: {self.global_agent.agent_id}")
        self.document(f"- batch_size: {self.global_agent.batch_size}")
        self.document(f"- epsilon: 0")
        self.document(f"- batch_size: {self.global_agent.input_dims}")
        self.document(f"- batch_size: {self.global_agent.n_actions}")

            
    def performance_evaluation(self, agent: Agent, test_data, nr_round):
        print("PERFORMANCE EVALUATION")
        # check predictions with learnt dqn
        agent.online_net.eval()
        res_dict = {}
        objective_dict = {}
        with torch.no_grad():
            for b, d in test_data.items():
                if b != Behavior.NORMAL:
                    cnt_corr = 0
                    cnt = 0
                    for state in d:
                        action = agent.take_greedy_action(state[:-1])
                        if b in supervisor_map[action]:
                            cnt_corr += 1
                        cnt += 1
                    res_dict[b] = (cnt_corr, cnt)

                for i in range(len(actions)):
                    if b in supervisor_map[i]:
                        objective_dict[b] = actions[i]
        labels = ("Behavior", "Accuracy", "Objective")
        results = []
        
        total_accuracy, mean_class_accuracy = self.compute_accuracies(res_dict)
        agent.total_accuracies[nr_round] = total_accuracy
        agent.mean_class_accuracies[nr_round] = mean_class_accuracy
        
        for behavior, t in res_dict.items():
            #print("---")
            #print(f"behavior: {behavior} {behavior.value}")
            accuracy = t[0] / t[1] * 100
            accuracy_in_percent = round(accuracy, 2)
            #print(f"accuracy_in_percent: {accuracy_in_percent}")
            self.performance_evaluations[agent.agent_id][behavior].append(accuracy_in_percent)
            results.append((behavior.value, accuracy_in_percent, objective_dict[behavior].value))

        self.document(f"\n\n{agent.get_name()}\n")
        self.document(tabulate(results, headers=labels, tablefmt="pipe"))

        print(f"{agent.get_name()} > Performance Evaluation")
        print(tabulate(results, headers=labels, tablefmt="pipe"))
    
    def compute_accuracies(self, res_dict):
        class_accuracies = []
        number_correctly_classified = 0
        number_total_samples = 0

        for (class_correct, class_total) in res_dict.values():
            class_accuracy = class_correct/class_total
            class_accuracies.append(class_accuracy)

            number_correctly_classified+=class_correct
            number_total_samples+=class_total

        total_accuracy = number_correctly_classified/number_total_samples
        mean_class_accuracy = np.mean(class_accuracies, axis=0)
        return total_accuracy, mean_class_accuracy
        print(f"Total Accuracy: {round(total_accuracy*100, 2)}%")
        print(f"Mean Class Accuracy: {round(mean_class_accuracy*100, 2)}%")
    
    def confusion_matrix(self, agent: Agent, test_data):
        agent.online_net.eval()
        confusion_matrix = np.zeros((4,4))
          
        with torch.no_grad():
            for behavior, state_samples in test_data.items():
                if behavior != Behavior.NORMAL:
                    for state_sample in state_samples:
                        a_pred = agent.take_greedy_action(state_sample[:-1])
                        if behavior in supervisor_map[a_pred]:
                               confusion_matrix[a_pred][a_pred] +=1
                        else:
                            leftover_actions = [i for i in range(4) if i != a_pred]
                            for leftover_action in leftover_actions:
                                if behavior in supervisor_map[leftover_action]:
                                   confusion_matrix[leftover_action][a_pred] +=1
          
        mtds = list(map(lambda x: x.value, actions))
        results = np.concatenate((np.array(mtds).reshape((-1, 1)), confusion_matrix), axis=1)
        labels = ["MTD_true/ MTD_pred"] + mtds
        print("")
        print(f"{agent.get_name()} > Confusion Matrix")
        print(tabulate(results, headers=labels, tablefmt="pipe"))
        
        TPs = confusion_matrix.diagonal()
        TPnFPs = np.sum(confusion_matrix, axis=0)
        TPnFNs = np.sum(confusion_matrix, axis=1)
        precisions = TPs/TPnFPs
        recalls = TPs/TPnFNs
        f1s = 2*(np.multiply(precisions, recalls)/( precisions + recalls))

        labels = labels = ["MTD", "precision", "recall", "f1"]

        mtds = np.array(list(map(lambda x: x.value, actions))).reshape(-1,1)
        precisions = precisions.reshape(-1,1)
        recalls = precisions.reshape(-1,1)
        f1s = f1s.reshape(-1,1)

        metrics = np.hstack((mtds, precisions, recalls, f1s))
        print("")
        print(f"{agent.get_name()} > Precision & Recall")
        print(tabulate(metrics, headers=labels, tablefmt="pipe"))
        
    def behavior_action_evaluation(self, agent: Agent, test_data):
        n_behaviors = len(Behavior) - 1
        n_actions = len(actions)
        matrix = np.zeros((n_behaviors, n_actions))
        
        behavior_action_counts = {}
        for behavior in Behavior:
            if behavior != Behavior.NORMAL:
                behavior_action_counts[behavior] = np.zeros((1, n_actions))
        
        with torch.no_grad():
            for behavior, state_samples in test_data.items():
                if behavior != Behavior.NORMAL:
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