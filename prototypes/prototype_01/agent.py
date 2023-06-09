class Agent:
    def __init__(self, agent_id: int, input_dims: int, n_actions, batch_size,
                 lr, gamma, epsilon, eps_end=0.02, eps_dec=1e-4, buffer_size=100000, is_global_agent=False):
        self.agent_id = agent_id
        self.is_global_agent = is_global_agent
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_max = epsilon # Initial epsilon value
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]

        self.episode_action_memory = set()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0.0], maxlen=100)  # for printing progress

        self.batch_size = batch_size

        self.online_net = DeepQNetwork(lr, n_actions=n_actions,
                                       input_dims=input_dims,
                                       fc1_dims=60, fc2_dims=30)
        self.target_net = DeepQNetwork(lr, n_actions=n_actions,
                                       input_dims=input_dims,
                                       fc1_dims=60, fc2_dims=30)
        self.target_net.load_state_dict(self.online_net.state_dict())

    def choose_action(self, observation):
        try:
            if np.random.random() > self.epsilon:
                #
                action = self.take_greedy_action(observation)
                if action in self.episode_action_memory:
                    action = np.random.choice(list(set(self.action_space).difference(self.episode_action_memory)))
            else:
                action = np.random.choice(list(set(self.action_space).difference(self.episode_action_memory)))
            self.episode_action_memory.add(action)
        except ValueError:
            return -1
        return action

    def take_greedy_action(self, observation):
        state = torch.from_numpy(observation.astype(np.float32)).to(self.online_net.device)
        actions = self.online_net.forward(state)
        action = torch.argmax(actions).item()
        return action

    def learn(self):
        # init data batch from memory replay for dqn
        transitions = random.sample(self.replay_buffer, self.batch_size)
        b_obses = np.stack([t[0].astype(np.float32).squeeze(0) for t in transitions], axis=0)
        b_actions = np.asarray([t[1] for t in transitions]).astype(np.int64)
        b_rewards = np.asarray([t[2] for t in transitions]).astype(np.int16)
        b_new_obses = np.stack([t[3].astype(np.float32).squeeze(0) for t in transitions], axis=0)
        b_dones = np.asarray([t[4] for t in transitions]).astype(np.int16)
        t_obses = torch.from_numpy(b_obses).to(self.target_net.device)
        t_actions = torch.from_numpy(b_actions).to(self.target_net.device)
        t_rewards = torch.from_numpy(b_rewards).to(self.target_net.device)
        t_new_obses = torch.as_tensor(b_new_obses).to(self.target_net.device)
        t_dones = torch.as_tensor(b_dones).to(self.target_net.device)

        # compute targets
        target_q_values = self.target_net(t_new_obses)
        max_target_q_values = torch.max(target_q_values, dim=1)[0]

        targets = (t_rewards + self.gamma * (1 - t_dones) * max_target_q_values).unsqueeze(1)

        # compute loss
        q_values = self.online_net(t_obses)
        taken_action_q_values = torch.gather(input=q_values, dim=1, index=t_actions.unsqueeze(1))

        loss = self.online_net.loss(taken_action_q_values, targets).to(self.target_net.device)

        # gradient descent
        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()

        # epsilon decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_weights(self):
        print(f"SERVER <--- WEIGHTS --- {self.get_name()}")
        return copy.deepcopy(self.target_net.state_dict())
    
    def update_weights(self, model_params):
        print(f"SERVER --- WEIGHTS ---> {self.get_name()}")
        self.online_net.load_state_dict(copy.deepcopy(model_params))
        self.target_net.load_state_dict(copy.deepcopy(model_params))
        
    def get_name(self):
        if self.is_global_agent:
            return "GLOBAL AGENT"
        else:
            return f"AGENT {self.agent_id}"
    
    '''
    def save_agent_state(self, n: int, directory: str):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'batch_size': self.batch_size,
            'replay_buffer': self.replay_buffer,
            'reward_buffer': self.reward_buffer,
            'action_space': self.action_space,
            'gamma': self.gamma,
            'eps': self.epsilon,
            'eps_min': self.eps_min,
            'eps_dec': self.eps_dec,
            'lr': self.lr
        }, f"{directory}/trained_models/agent_{n}.pth")

        #torch.save(self.online_net.state_dict(), f"offline_prototype_2_raw_behaviors/trained_models/online_net_{n}.pth")
        #torch.save(self.target_net.state_dict(), f"offline_prototype_2_raw_behaviors/trained_models/target_net_{n}.pth")
    '''