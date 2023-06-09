from data_manager import DataManager
from utils.utils import seed_random



if __name__ == "__main__":
    seed_random()
    #rtrain, rtest = DataManager.get_reduced_dimensions_with_pca()
    #print(rtrain.shape)
    #DataManager.print_pca_scree_plot(15)

    maxCol = lambda x: max(x.min(), x.max(), key=abs)
    df = DataManager.get_pca_loading_scores_dataframe(15)
    print(df)
    print(df.apply(maxCol, axis=0))
    print(df[['PC1']].idxmax())
