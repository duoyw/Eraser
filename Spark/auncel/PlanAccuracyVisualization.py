import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class PlanAccuracyVisualization:

    def __init__(self):
        super().__init__()
        self.join_conditions = []
        self.tables = []
        self.filter_cols = []
        self.join_types = []
        self.scan_types = []

    def draw_encord(self, encord_vals, metrics, is_2d=True, perplexity=None, file_name=None):
        if is_2d:
            self._draw_2d_figure(encord_vals, metrics, perplexity, file_name)
        else:
            self._draw_3d_figure(encord_vals, metrics)

    def _draw_2d_figure(self, encord_vals, metrics, perplexity=None, file_name=None):

        X_embedded = TSNE(n_components=2, learning_rate="auto", init='random', early_exaggeration=200,
                          perplexity=50 if perplexity is None else perplexity).fit_transform(
            np.array(encord_vals))
        plt.figure(figsize=(10, 8))
        # cmap = plt.cm.binary()

        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=metrics, cmap=plt.cm.get_cmap('plasma'))
        plt.colorbar(ticks=np.linspace(0, 1, 11))
        plt.clim(0, 1)
        file_name = file_name if file_name is not None else "prediction_distribution"
        plt.savefig("fig/{}.pdf".format(file_name), format="pdf")
        plt.show()
        plt.close()

    def _draw_3d_figure(self, encord_vals, metrics):
        X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(
            np.array(encord_vals))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=metrics)
        plt.show()


