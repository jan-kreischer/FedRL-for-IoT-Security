import os
import shutil

class Experiment:
    def __init__(self, base_path):
        self.base_path = base_path
        
    def get_experiment_path(self, experiment_id, experiment_version=0):
        path = os.path.join(self.base_path, f"experiments/experiment_{experiment_id:02d}")
        if experiment_version != 0:
            path = os.path.join(path, f"version_{experiment_version:02d}")
            
        #print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)          
            os.makedirs(path)
            
        return path