import json
import joblib
import os
from outer_module import feature, util
import perfguardConfig
from perfguard import *

config = PerfguardConfig.Config()


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")


class Get_Dataset():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.label = []
        self.NODE_INDEX = 0
        self.MAX_NODE_NUM = 0
        self._feature_generator = None

    def get_features(self):
        X1, X2 = self._load_pairwise_plans()
        # generate features,labels
        self._feature_generator = feature.FeatureGenerator()
        self._feature_generator.fit(X1 + X2)
        self.x1, self.y1 = self._feature_generator.transform(X1)
        self.x2, self.y2 = self._feature_generator.transform(X2)
        # plan_num,node_num,feature
        self.x1 = util.prepare_trees(self.x1, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x2 = util.prepare_trees(self.x2, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x1 = np.array(self.x1)
        self.x2 = np.array(self.x2)
        return self.x1, self.x2

    def get_labels(self):
        for i in range(len(self.y1)):
            if self.y1[i] < self.y2[i]:
                self.label.append(1)
            else:
                self.label.append(0)
        self.label = np.array(self.label)
        return self.label

    def get_two_adjaceny_matrix(self):
        # self.MAX_NODE_NUM = self.x1.shape[1]
        self.MAX_NODE_NUM1 = self.x1.shape[1]
        self.MAX_NODE_NUM2 = self.x2.shape[1]
        self.NODE_INDEX = 0
        X1, X2 = self._load_pairwise_plans()
        # plan_num,node_num,node_num
        adjaceny_matrix_list_x1 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 1) for plan_json in X1]
        adjaceny_matrix_list_x2 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 2) for plan_json in X2]
        adjaceny_matrix_list_x1 = np.array(adjaceny_matrix_list_x1)
        adjaceny_matrix_list_x2 = np.array(adjaceny_matrix_list_x2)
        return adjaceny_matrix_list_x1, adjaceny_matrix_list_x2

    def _load_pairwise_plans(self):
        X1, X2 = [], []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                arr = line.split("#####")
                arr = [json.loads(x) for x in arr]
                x1, x2 = self.get_training_pair(arr)
                X1 += x1
                X2 += x2
        return X1, X2

    def get_training_pair(self, candidates):
        assert len(candidates) >= 2
        X1, X2 = [], []

        i = 0
        while i < len(candidates) - 1:
            s1 = candidates[i]
            j = i + 1
            while j < len(candidates):
                s2 = candidates[j]
                X1.append(s1)
                X2.append(s2)
                j += 1
            i += 1
        return X1, X2

    def get_adjaceny_matrix(self, plan_json, flag):
        self.NODE_INDEX = 0
        MAX_NODE_NUM = 0
        if flag == 1:
            MAX_NODE_NUM = self.MAX_NODE_NUM1
        else:
            MAX_NODE_NUM = self.MAX_NODE_NUM2

        adjacecy_matrix = [[0] * MAX_NODE_NUM for _ in range(MAX_NODE_NUM)]
        self.dfs(plan_json, 0, adjacecy_matrix)
        return adjacecy_matrix

    def dfs(self, plan_json, node_index, adjacecy_matrix):
        self.NODE_INDEX += 1
        if 'Plans' in plan_json:
            plan_json = plan_json['Plans']
            for plan_json_ in plan_json:
                adjacecy_matrix[node_index][self.NODE_INDEX] = 1
                adjacecy_matrix[self.NODE_INDEX][node_index] = 1
                self.dfs(plan_json_, self.NODE_INDEX, adjacecy_matrix)
        else:
            return

    def transformer(self, x):
        return x.get_feature()

    def left_child(self, x):
        return x.get_left()

    def right_child(self, x):
        return x.get_right()

    def save(self, model, path, _input_feature_dim):
        os.makedirs(path, exist_ok=True)

        torch.save(model.state_dict(), _nn_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(_input_feature_dim, f)


class Get_Dataset_Test(Get_Dataset):
    def __init__(self, plan_dict):
        self.plan_dict = plan_dict
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.label = []
        self.NODE_INDEX = 0
        self.MAX_NODE_NUM = 0
        self._feature_generator = None

    def load_model(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            _input_feature_dim = joblib.load(f)

        model = PerfGuard(_input_feature_dim, config.embd_dim, config.tensor_dim, config.dropout).cuda(config.device)
        model = torch.nn.DataParallel(model, device_ids=config.GPU_LIST)
        model.load_state_dict(torch.load(_nn_path(path)))
        model.eval()

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)
        return model

    def get_features(self):
        X1, X2 = self._load_pairwise_plans()
        # generate features,labels
        feature_generator = self._feature_generator
        # feature_generator.fit(X1 + X2)
        self.x1, self.y1 = feature_generator.transform(X1)
        self.x2, self.y2 = feature_generator.transform(X2)
        # plan_num,node_num,feature
        self.x1 = util.prepare_trees(self.x1, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x2 = util.prepare_trees(self.x2, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x1 = np.array(self.x1)
        self.x2 = np.array(self.x2)
        return self.x1, self.x2

    def get_labels(self):
        for i in range(len(self.y1)):
            if self.y1[i] < self.y2[i]:
                self.label.append(1)
            else:
                self.label.append(0)
        self.label = np.array(self.label)
        return self.label

    def get_two_adjaceny_matrix(self):
        # self.MAX_NODE_NUM = self.x1.shape[1]
        self.MAX_NODE_NUM1 = self.x1.shape[1]
        self.MAX_NODE_NUM2 = self.x2.shape[1]
        self.NODE_INDEX = 0
        X1, X2 = self._load_pairwise_plans()
        # plan_num,node_num,node_num
        adjaceny_matrix_list_x1 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 1) for plan_json in X1]
        adjaceny_matrix_list_x2 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 2) for plan_json in X2]
        adjaceny_matrix_list_x1 = np.array(adjaceny_matrix_list_x1)
        adjaceny_matrix_list_x2 = np.array(adjaceny_matrix_list_x2)
        return adjaceny_matrix_list_x1, adjaceny_matrix_list_x2

    def _load_pairwise_plans(self):
        X1, X2 = [], []
        for qid in self.plan_dict.keys():
            if len(self.plan_dict[qid]) >= 2:
                x1, x2 = self.get_training_pair(self.plan_dict[qid])
                X1 += x1
                X2 += x2
        return X1, X2

    def get_training_pair(self, candidates):
        assert len(candidates) >= 2
        X1, X2 = [], []

        i = 0
        while i < len(candidates) - 1:
            s1 = candidates[i]
            j = i + 1
            while j < len(candidates):
                s2 = candidates[j]
                X1.append(s1)
                X2.append(s2)
                j += 1
            i += 1
        return X1, X2

    def get_adjaceny_matrix(self, plan_json, flag):
        self.NODE_INDEX = 0
        MAX_NODE_NUM = 0
        if flag == 1:
            MAX_NODE_NUM = self.MAX_NODE_NUM1
        else:
            MAX_NODE_NUM = self.MAX_NODE_NUM2

        adjacecy_matrix = [[0] * MAX_NODE_NUM for _ in range(MAX_NODE_NUM)]
        self.dfs(plan_json, 0, adjacecy_matrix)
        return adjacecy_matrix

    def dfs(self, plan_json, node_index, adjacecy_matrix):
        self.NODE_INDEX += 1
        if 'Plans' in plan_json:
            plan_json = plan_json['Plans']
            for plan_json_ in plan_json:
                adjacecy_matrix[node_index][self.NODE_INDEX] = 1
                adjacecy_matrix[self.NODE_INDEX][node_index] = 1
                self.dfs(plan_json_, self.NODE_INDEX, adjacecy_matrix)
        else:
            return

    def transformer(self, x):
        return x.get_feature()

    def left_child(self, x):
        return x.get_left()

    def right_child(self, x):
        return x.get_right()
