import torch
from torch import nn, optim
import copy
import matplotlib.pyplot as plt
#%matplotlib inline
import prototypes.prototype_01.sensor_environment
from prototypes.prototype_01.sensor_environment import SensorEnvironment
import prototypes.prototype_01.agent
from prototypes.prototype_01.agent import Agent

class Client:
        # memory buffer is influenced by env.step -> resetting to previous action, which may result in unbalanced training
    def __init__(self, client_id: int, agent: Agent, environment: SensorEnvironment, save_path=""):
        self.client_id = client_id
        self.agent = agent
        self.environment = environment
        self.episode_returns = [] 
        self.eps_history = []
        self.save_path = save_path
    
    def init_replay_memory(self, min_size):
        obs = self.environment.reset()
        episode_action_memory = []
        i = 0
        while i < min_size:
            try:
                action = np.random.choice(list({0,1,2,3}.difference(episode_action_memory)))
                episode_action_memory.append(action)
            except ValueError:
                obs = self.environment.reset()
                episode_action_memory = []
                # results in slightly less entries than min_size
                print("exhausted all mtd techniques")
                continue
            i += 1

            new_obs, reward, done = self.environment.step(action)
            idx1 = -1 if obs[0, -1] in Behavior else -2
            idx2 = -1 if new_obs[0, -1] in Behavior else -2
            transition = (obs[:, :idx1], action, reward, new_obs[:, :idx2], done)
            self.agent.replay_buffer.append(transition)

            obs = new_obs
            if done:
                obs = self.environment.reset()
                episode_action_memory = []

    def train_agent(self, num_episodes, t_update_freq, verbose=False):
        episode_returns, eps_history = [], []
        step = 0
        for i in range(num_episodes):
            episode_return = 0
            episode_steps = 0
            done = False
            obs = self.environment.reset()
            while not done:
                idx1 = -1 if obs[0, -1] in Behavior else -2
                action = self.agent.choose_action(obs[:, :idx1])
                if action == -1:
                    print("Agent exhausted all MTD techniques upon behavior: ", obs[0, -1])
                    self.agent.episode_action_memory = set()
                    done = True
                    continue

                new_obs, reward, done = self.environment.step(action)
                idx2 = -1 if new_obs[0, -1] in Behavior else -2
                episode_return += reward
                self.agent.replay_buffer.append((obs[:, :idx1], action, reward,
                                            new_obs[:, :idx2], done))
                self.agent.reward_buffer.append(reward)
                if done:
                    self.agent.episode_action_memory = set()

                self.agent.learn()
                obs = new_obs

                episode_steps += 1
                # update target network
                step += 1
                if step % t_update_freq == 0:
                    self.agent.update_target_network()

                # if step % LOG_FREQ == 0:
                # print("Episode: ", i, "Step: ", step, ", Avg Reward: ", np.mean(agent.reward_buffer), "epsilon: ", agent.epsilon)

            self.episode_returns.append(episode_return / episode_steps)
            avg_episode_return = np.mean(episode_returns[-10:])
            self.eps_history.append(self.agent.epsilon)
            
            if verbose:
                print('| agent %d' % self.agent.agent_id,
                  '| episode ', i, '| episode_return %.2f' % episode_returns[-1],
                  '| average episode_return %.2f' % avg_episode_return,
                  '| epsilon %.2f' % self.agent.epsilon)
            #if i >= num_episodes - 6:
                #print(episode_returns[-10:])
                
            #self.episode_returns+=episode_returns
            #self.eps_history+=eps_history
        return episode_returns, eps_history
        

    def receive_weights(self, model_params):
        """ Receive aggregated parameters, update model """
        #self.agent.load_state_dict(copy.deepcopy(model_params))
        self.agent.update_weights(model_params)
        
    def get_weights(self):
        return self.agent.get_weights()
    
    def get_training_summary(self):
        return self.episode_returns, self.eps_history
    
    def plot_learning_curve(self, filename="", save_results=False):
        returns, epsilons = self.get_training_summary()
        #x = range(1, len(returns)+1)
        assert len(returns) == len(epsilons)
        x = [i + 1 for i in range(len(returns))]
        fig = plt.figure()
        #plt.ylim(0, 1)
        #fig.title(title)

        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        ax.set_title(filename)

        ax.plot(x, epsilons, color="C0")
        ax.set_xlabel("Episode", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

        N = len(returns)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(returns[max(0, t - 20):(t + 1)])

        ax2.scatter(x, running_avg, color="C1", s=2 ** 2)
        # ax2.xaxis.tick_top()
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        # ax2.set_xlabel('x label 2', color="C1")
        ax2.set_ylabel('Score', color="C1")
        ax2.set_ylim([0, 1])
        # ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        # ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")

        #fig.show()
        plt.show()
        if save_results:
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            plt.savefig(os.path.join(self.save_path, filename))

                        
    def plot_performance_evaluation(self, test_data, title="", save_results=False):
        # check predictions with learnt dqn
        self.agent.online_net.eval()
        res_dict = {}
        objective_dict = {}
        with torch.no_grad():
            for b, d in test_data.items():
                if b != Behavior.NORMAL:
                    cnt_corr = 0
                    cnt = 0
                    for state in d:
                        action = self.agent.take_greedy_action(state[:-1])
                        if b in supervisor_map[action]:
                            cnt_corr += 1
                        cnt += 1
                    res_dict[b] = (cnt_corr, cnt)

                for i in range(len(actions)):
                    if b in supervisor_map[i]:
                        objective_dict[b] = actions[i]
        labels = ("Behavior", "Accuracy", "Objective")
        results = []

        for b, t in res_dict.items():
            results.append((b.value, f'{(100 * t[0] / t[1]):.2f}%', objective_dict[b].value))
        print(title)
        print(tabulate(results, headers=labels, tablefmt="orgtbl"))