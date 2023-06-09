import os

class Experiment:
    def __init__(self, base_path):
        self.base_path = base_path
    def get_path(self, experiment_id):
        path = os.path.join(self.base_path, f"experiments/experiment_{experiment_id}")
        if not os.path.isdir(path):
                os.makedirs(path)
        return path