# https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
import random
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#import matplotlib.cm as cm
np.random.seed(123)
random.seed(123)

class plotScatters():

    def __init__(self):
        self.plots = []
        self.filenames = []

    def ploty(self, groups, labels, unique_labels, title, saving_path, index):

        self.tsne_results = groups
        self.labels = labels
        self.unique_labels = unique_labels
        self.title = title
        self.saving_path = saving_path
        self.index = index
        self.plt = plt
        self.filenames.append(saving_path)

        num_colors = len(self.unique_labels)
        cm = plt.get_cmap('tab20')
        #cm = plt.get_cmap('Spectral')# nipy_spectral
        #cmap = 'Diverging'#matplotlib.cm.get_cmap('Diverging')#'Spectral'
        #colors = [cm(float(k) / num_top_groups) for k in range(num_top_groups)]
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot(111)
        #colors = [cm(i) for i in range(num_colors)]


        for idx, i in enumerate(unique_labels):

            cluster_idx = np.where(labels == i)[0]

            self.ax.scatter(self.tsne_results[cluster_idx, 0], self.tsne_results[cluster_idx, 1], label=i, marker='.', cmap=cm)  # , vmin=0, vmax=50 )#)cmap='Diverging'color=cm

        self.plt.legend()
        self.plt.title('T-SNE')
        self.figure = plt.gcf()  # get current figure
        self.figure.set_size_inches(12, 9)
        # when saving, specify the DPI
        self.plt.savefig(self.saving_path, format='png', dpi=300)
        #plt.savefig(saving_path, format='png', dpi=400)

        if self.index == 1:
            fig, axs = plt.subplots(2, 2)
            for t, ax in enumerate(axs.flat):
                ax.set_axis_off()
                filename = self.filenames[t]
                #print('filename', filename)
                ax.imshow(mpimg.imread(filename))
            #plt.show()
            saving_path = self.saving_path
            #saving_path = saving_path+'_TSN_all_scatters.png'
            plt.savefig(saving_path, format='png', dpi=400)



def main():
    path = ' '
    name = '_tsne_'
    models = ['hdbscan4_1', 'Agg077']#, 'dbscan047_1'] #hdbscan8_2
    data1 = pd.read_csv(path + 'resultVggFaceGenderClassifierR1Epoch25_v17.csv')
    path_tsne = path + 'TSNEResultVggFaceGenderClassifier_V17.csv'

    title1 = models[0] + name
    saving_path1 = path + 'TSNEResultVggFaceGenderClassifierCosine_V17.png'
    for k in range(1):
        if k==0:
           data = data1
           title = title1
           saving_path = saving_path1

        len_groups = len(data.groupby('label'))
        #print('len_groups', len_groups)
        labels = np.array(data['label'], dtype=int)
        unique_labels = np.unique(labels)

        all_vectors = np.array(data.iloc[:, -128: ],dtype=float)
        len_data = data.shape[1]
        print('len_data', len_data)

        print(all_vectors[0, 0:3], all_vectors[0, -3:])

        #print('len_vector', len(all_vectors[0]))
        num_vectors = all_vectors.shape[0]

        all_vectors = np.array(all_vectors, dtype=float).reshape(num_vectors, 128)

        # as rules of thumb for perplexity should be N**0.5 and max_iteratio should be increase untill the scale of the TSNE map is about -50 to +50
        tsne = TSNE(n_components=2, random_state=1, perplexity=50, metric='cosine', init='pca', method='barnes_hut')

        grouped_result_data = data.groupby('label').size().sort_values(ascending=False)
        sorted_labels = grouped_result_data.index
        print('sorted_labels', sorted_labels[0:50])

        groups = tsne.fit_transform(X=all_vectors)
        if k == 0:
            plot_all = plotScatters()
        print(groups)
        result_data = pd.DataFrame(groups)
        result_data_all = pd.concat((result_data, pd.DataFrame(labels)), axis=1)
        result_data_all.columns = ['x', 'y', 'label']
        result_data_all.to_csv(path_tsne)
        plot_all.ploty(groups, labels, unique_labels, title, saving_path, index=k)

main()


