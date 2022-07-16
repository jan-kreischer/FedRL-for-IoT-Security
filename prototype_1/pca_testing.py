from data_manager import DataManager



if __name__ == "__main__":
    rtrain, rtest = DataManager.get_reduced_dimensions_with_pca()
    print(rtrain.shape)
    #DataManager.print_pca_scree_plot()