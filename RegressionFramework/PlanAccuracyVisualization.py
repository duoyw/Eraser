from matplotlib import pyplot as plt

from RegressionFramework.Plan.PgPlan import PgScanPlanNode
from RegressionFramework.Plan.Plan import PlanNode, JoinPlanNode, ScanPlanNode

import numpy as np
from sklearn.manifold import TSNE

from RegressionFramework.utils import to_rgb_tuple


class PlanAccuracyVisualization:

    def __init__(self):
        super().__init__()
        self.join_conditions = []
        self.tables = []
        self.filter_cols = []
        self.join_types = []
        self.scan_types = []

    # def draw(self, plans):
    #     self.join_types, self.join_conditions, self.scan_types, self.tables, self.filter_cols = self.collect(plans)
    #
    #     encord_plans = []
    #     for plan in plans:
    #         encord_plans.append(self.encord_plan(plan))
    #     metrics=[plan.metric for plan in plans]
    #
    #     X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(np.array(encord_plans))
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=metrics, cmap=plt.cm.get_cmap('RdYlBu'))
    #     plt.colorbar(ticks=np.linspace(0, 1, 11))
    #     plt.clim(0, 1)
    #     plt.show()
    #     exit()

    def draw(self, plans):
        self.join_types, self.join_conditions, self.scan_types, self.tables, self.filter_cols = self.collect(plans)

        encord_plans = []
        for plan in plans:
            encord_plans.append(self.encord_plan(plan))
        metrics = [plan.metric for plan in plans]

        X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(
            np.array(encord_plans))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=metrics)
        plt.show()
        exit()

    def draw_encord(self, encord_vals, metrics, is_2d=True, perplexity=None, file_name=None):
        if is_2d:
            self._draw_2d_figure(encord_vals, metrics, perplexity, file_name)
        else:
            self._draw_3d_figure(encord_vals, metrics)

    # def _draw_2d_figure(self, encord_vals, metrics, perplexity=None, file_name=None):
    #
    #     X_embedded = TSNE(n_components=2, learning_rate="auto", init='random', early_exaggeration=200,
    #                       perplexity=50 if perplexity is None else perplexity).fit_transform(
    #         np.array(encord_vals))
    #     plt.figure(figsize=(10, 8))
    #     # cmap = plt.cm.binary()
    #
    #     colors = ['r' if i == 0 else 'g' for i in metrics]
    #     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=metrics, cmap=plt.cm.get_cmap('plasma'))
    #     plt.colorbar(ticks=np.linspace(0, 1, 11))
    #     plt.clim(0, 1)
    #     file_name = file_name if file_name is not None else "prediction_distribution"
    #     plt.savefig("RegressionFramework/fig/{}.pdf".format(file_name), format="pdf")
    #     plt.show()
    #     plt.close()

    def _draw_2d_figure(self, encord_vals, metrics, perplexity=None, file_name=None):
        # plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['mathtext.default'] = 'regular'
        X_embedded = TSNE(n_components=2, learning_rate="auto", init='random',
                          perplexity=50 if perplexity is None else perplexity).fit_transform(
            np.array(encord_vals))
        plt.figure(figsize=(10, 8))
        # cmap = plt.cm.binary()

        h_color = to_rgb_tuple("rgb(255,0,0)")
        l_color = to_rgb_tuple("rgb(132,184,68)")
        marker_size=70
        x =np.array([[X_embedded[i, 0], X_embedded[i, 1]] for i in range(len(X_embedded)) if metrics[i] == 1])
        plt.scatter(x[:, 0], x[:, 1], c=h_color, label="Accuracy > 0.7",s=marker_size)
        x =np.array([[X_embedded[i, 0], X_embedded[i, 1]] for i in range(len(X_embedded)) if metrics[i] == 0])
        plt.scatter(x[:, 0], x[:, 1],  c=l_color, label="Other",s=marker_size)
        file_name = file_name if file_name is not None else "prediction_distribution"
        plt.xticks([], [], rotation=0)
        plt.yticks([], [], rotation=0)
        plt.grid()
        plt.tight_layout()
        # plt.legend(fontsize=35)
        plt.savefig("RegressionFramework/fig/{}.pdf".format(file_name), format="pdf")
        plt.show()
        plt.close()

    def _draw_3d_figure(self, encord_vals, metrics):
        X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(
            np.array(encord_vals))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=metrics)
        plt.show()

    def collect(self, plans):
        join_conditions = set()
        tables = set()
        filter_cols = set()
        join_types = set()
        scan_types = set()

        def recurse(node: PlanNode):
            if node.is_join_node():
                node: JoinPlanNode = node
                join_conditions.add(node.get_join_key_str())
                join_types.add(node.get_join_type())
            elif node.is_scan_node():
                node: PgScanPlanNode = node
                scan_types.add(node.scan_type)
                tables.add(node.table_name)
                for predicate in node.predicates:
                    filter_cols.add(predicate[0])
            for child in node.children:
                recurse(child)

        for plan in plans:
            recurse(plan.root)
        return list(join_types), list(join_conditions), list(scan_types), list(tables), list(filter_cols)

    def encord_plan(self, plan):
        encord = []

        def recurse(node: PlanNode):
            if node.is_join_node():
                node: JoinPlanNode = node
                encord.append(self.encord_join_condition(node.get_join_key_str()))
                encord.append(self.encord_join_type(node.get_join_type()))
            elif node.is_scan_node():
                node: PgScanPlanNode = node
                encord.append(self.encord_scan_type(node.scan_type))
                encord.append(self.encord_table(node.table_name))
            for child in node.children:
                recurse(child)

        recurse(plan.root)
        return encord

    def encord_join_type(self, join_type):
        return self.join_types.index(join_type)

    def encord_join_condition(self, join_condition):
        return self.join_conditions.index(join_condition)

    def encord_scan_type(self, scan_type):
        return self.scan_types.index(scan_type)

    def encord_table(self, table):
        return self.tables.index(table)

    def encord_filter_col(self, col):
        return self.filter_cols.index(col)
