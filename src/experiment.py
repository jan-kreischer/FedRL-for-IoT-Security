import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from src.custom_types import Behavior


class Experiment:
    def __init__(self):
        print()
        
    def add_server(self, server):
        self.server = server
        
    def execute(self, 
                server,
                nr_rounds: int,
                nr_episodes_per_round: int,
                evaluations,
                evaluation_frequency=1,
                verbose=False):
        self.verbose = verbose
        if self.verbose:
            print(f"=== STARTING EXPERIMENT ===\n")
        self.server = server
        self.nr_rounds = nr_rounds
        self.nr_episodes_per_round = nr_episodes_per_round
        self.server.run_federation(nr_rounds=nr_rounds, nr_episodes_per_round=nr_episodes_per_round, evaluations=evaluations, evaluation_frequency=evaluation_frequency, verbose=verbose)
        
    def plot_test_accuracies(self, micro=True, show_episodes=False, y_threshold=99, show_individual_clients=True, y_log_scale=True):
        agents = list(map(lambda client: client.agent, self.server.clients))
        agents.append(self.server.global_agent)

        plt.figure(figsize=(9,6))
        if y_log_scale:
            plt.yscale("log") 
        else:
            plt.yscale("linear") 
            
        for agent in agents:
            agent_id = agent.agent_id

            label = agent.get_name()
            if micro:
                x = list(agent.total_accuracies.keys())
                y = np.array(list(agent.total_accuracies.values()))*100
            else:
                x = list(agent.mean_class_accuracies.keys())
                y = np.array(list(agent.mean_class_accuracies.values()))*100
            
            if len(x) == 0:
                continue
                
            # Should all individual clients be shown or only the global agent?
            if agent_id != 0 and not show_individual_clients:
                continue
    
            #if show_episodes:
            #    x = np.array(x)*self.nr_episodes_per_round
            #    x_label = "Number of Episodes trained per Client"
            #else:
            #    x_label = "Number of Rounds"
            x_label = "Number of Rounds"   

            if y_threshold!=None and agent.agent_id == 0:  
                #y_index = np.argmax(y>y_threshold)
                K = 2
                #y_index = np.flatnonzero(np.convolve(y>=y_threshold, np.ones(K,dtype=int))==K)[0]-K+1
                comp = np.flatnonzero(np.convolve(y>=y_threshold, np.ones(K,dtype=int))==K)
                if len(comp) > 0:
                    y_index = comp[0]-K+1
                else:
                    y_index = -1
            
            if agent_id == 0:
                color='blue'
                linestyle='solid'
                linewidth=2
            else:
                color='#666666'
                linestyle='dashdot'
                linewidth=1

            plt.xticks(x)
            plt.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            
        #if show_episodes:
        #    plt.xticks(rotation=45)

        plt.xlabel(x_label)
        plt.ylabel('Test Accuracy [%]')
        if y_log_scale:
            plt.ylim(1, 105) 
        else:
            plt.ylim(0, 105)
            
        plt.xlim(np.min(x), np.max(x)) 
        
        if micro:
            plt.title("Micro Test Accuracy over multiple Rounds")
        else:     
            plt.title("Macro Test Accuracy over multiple Rounds")
          
        #print(f"y_index: {y_index}")
        if y_threshold != None and (y_index != 0 or y[0] > y_threshold):
            plt.axvline(x=y_index, color='red', linestyle='dashed')
        #if y_threshold != None:
        #    plt.axhline(y=y_threshold, color='red', linestyle='dashed', label=f"{y_threshold}% Accuracy Threshold")
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_behavior_performances(self, show_episodes=False, y_threshold=99, show_individual_clients=True, y_log_scale=True):
        agents = list(map(lambda client: client.agent, self.server.clients))
        agents.append(self.server.global_agent)

        non_normal_behaviors = [behavior for behavior in Behavior if behavior is not Behavior.NORMAL]
        for behavior in non_normal_behaviors:
            plt.figure(figsize=(9,6))
            if y_log_scale:
                plt.yscale("log") 
            else:
                plt.yscale("linear") 
                
            for agent in agents:
                #print(agent.get_name())
                #print(agent.total_accuracies)
                #print(agent.mean_class_accuracies)
                #print("---------")
                agent_id = agent.agent_id

                label = agent.get_name()
                x = list(agent.behavior_accuracies[behavior].keys())
                if len(x) == 0:
                    continue
                    
                # Should all individual clients be shown or only the global agent?
                if agent_id != 0 and not show_individual_clients:
                    continue
                    
                #if show_episodes:
                #    x = np.array(x)*self.nr_episodes_per_round
                #    x_label = "Number of Episodes trained per Client"

                x_label = "Number of Rounds"
                y = np.array(list(agent.behavior_accuracies[behavior].values()))*100
                
                if y_threshold != None and agent.agent_id == 0: 
                    #y_index = np.argmax(y>y_threshold)
                    K = 2
                    comp = np.flatnonzero(np.convolve(y>=y_threshold, np.ones(K,dtype=int))==K)
                    if len(comp) > 0:
                        y_index = comp[0]-K+1
                    else:
                        y_index = -1

                if agent_id == 0:
                    color='blue'
                    linestyle='solid'
                    linewidth=2
                else:
                    color='#666666'
                    linestyle='dashdot'
                    linewidth=1
                    #plt.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                plt.xticks(x)
                plt.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

                #plt.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            #if show_episodes:
            #    plt.xticks(rotation=45)
            plt.xlabel(x_label)
            plt.ylabel('Test Accuracy [%]')
            #plt.grid()
            #ax = plt.axes()        
            #ax.yaxis.grid()
            if y_log_scale:
                plt.ylim(1, 105) 
            else:
                plt.ylim(0, 105)

            plt.xlim(np.min(x), np.max(x))

            plt.title(f"Test Accuracy for {behavior.name} over multiple Rounds")
            if y_threshold != None and (y_index != 0 or y[0] >= y_threshold):
                plt.axvline(x=y_index, color='red', linestyle='dashed')
            #if y_threshold != None:
            #    plt.axhline(y=y_threshold, color='red', linestyle='dashed', label=f"{y_threshold}% Accuracy Threshold")
            plt.legend(loc="lower right")
            plt.show()
            
            
    def show_learning_curves(self):
        self.server.plot_learning_curves()
        
    def show_experiment_graphs(self, y_threshold=0.99, show_individual_clients=True, y_log_scale=True):
        
        self.plot_test_accuracies(micro=True,
                                  show_episodes=False,
                                  y_threshold=y_threshold,
                                  show_individual_clients=show_individual_clients,
                                  y_log_scale=y_log_scale)
        
        self.plot_test_accuracies(micro=False, 
                                  show_episodes=False,
                                  y_threshold=y_threshold,
                                  show_individual_clients=show_individual_clients,
                                  y_log_scale=y_log_scale)
        
        self.plot_behavior_performances(show_episodes=False,
                                        y_threshold=y_threshold,
                                        show_individual_clients=show_individual_clients,
                                        y_log_scale=y_log_scale)
        
    def plot_sampling_distributions(self):
        environments = list(map(lambda client: client.environment, self.server.clients[:2]))
        for environment in environments:
            environment.plot_sampling_distribution()