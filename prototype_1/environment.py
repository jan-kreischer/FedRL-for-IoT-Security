from prototype_1.episode_generator import EpisodeGenerator

class SensorEnvironment:
    def __init__(self, monitor=None):
        self.monitor = monitor

    def step(self, action):
        """
        how to derive each?
        new_state: supervised sampling on the fly by episode_generator
        isTerminalState: true if episode_generator returns null
        reward: call calculate_reward(new_state)

        """
        new_state = None
        isTerminalState = False

        if self.monitor == None:
            # TODO: sample with episode_generator
        else:
            # TODO: sample with monitoring component

        return new_state, isTerminalState


    def reset(self):
        # TODO: take random sample of a behavior

        pass


    def calculate_reward_supervised(self, prev_state, action):
        """
        if action == supervisor_map[state.behavior]:
        then return positive
        else return negative

        this method can be exchanged for the online/unsupervised RL system with the autoencoder
        """
        return 0