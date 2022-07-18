from data_manager import DataManager
from utils.utils import seed_random
import numpy as np



if __name__ == "__main__":
    seed_random()
    #rtrain, rtest = DataManager.get_reduced_dimensions_with_pca()
    #print(rtrain.shape)
    DataManager.print_pca_scree_plot(15)

    #d = DataManager.get_highest_weight_loading_scores_for_pc(pcn="PC1").head(5)** 2
    #print(d)
    #remove .head() in d & print(d.sum()) # should be one -> all features together contribute with a factor to the pca
