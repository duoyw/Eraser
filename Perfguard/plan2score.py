import joblib
import os
from Perfguard.perfguardConfig import *
from RegressionFramework.Common.Cache import Cache
from perfguard import PerfGuard
import torch
from outer_module import feature, util
import numpy as np
import json

config = Config()


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")


class Plan2Score:
    def __init__(self):
        self.MAX_NODE_NUM1 = None
        self.MAX_NODE_NUM2 = None

    def load_model(self, model_path):
        with open(_input_feature_dim_path(model_path), "rb") as f:
            _input_feature_dim = joblib.load(f)

        self.model = PerfGuard(_input_feature_dim, config.embd_dim, config.tensor_dim, config.dropout).cuda(
            config.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=config.GPU_LIST)
        self.model.load_state_dict(torch.load(_nn_path(model_path)))
        self.model.eval()

        with open(_feature_generator_path(model_path), "rb") as f:
            self.model._feature_generator = joblib.load(f)

    def get_features(self, plan_list1, plan_list2):
        feature_generator = self.model._feature_generator
        # feature_generator.fit(X1 + X2)
        self.x1, self.y1 = feature_generator.transform(plan_list1)
        self.x2, self.y2 = feature_generator.transform(plan_list2)
        # plan_num,node_num,feature
        self.x1 = util.prepare_trees(self.x1, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x2 = util.prepare_trees(self.x2, self.transformer, self.left_child, self.right_child, cuda=False,
                                     device=None)
        self.x1 = np.array(self.x1)
        self.x2 = np.array(self.x2)
        return self.x1, self.x2

    def get_two_adjaceny_matrix(self, plan_list1, plan_list2):
        self.MAX_NODE_NUM1 = self.x1.shape[1]
        self.MAX_NODE_NUM2 = self.x2.shape[1]
        self.NODE_INDEX = 0
        # plan_num,node_num,node_num
        adjaceny_matrix_list_x1 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 1) for plan_json in plan_list1]
        adjaceny_matrix_list_x2 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], 2) for plan_json in plan_list2]
        adjaceny_matrix_list_x1 = np.array(adjaceny_matrix_list_x1)
        adjaceny_matrix_list_x2 = np.array(adjaceny_matrix_list_x2)
        return adjaceny_matrix_list_x1, adjaceny_matrix_list_x2

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

    def predict(self, adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2):
        predict = self.model(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
        predict = predict.cpu().detach().numpy().tolist()
        predict_label = [1 if x > config.threshold else 0 for x in predict]
        return predict_label

    def transformer(self, x):
        return x.get_feature()

    def left_child(self, x):
        return x.get_left()

    def right_child(self, x):
        return x.get_right()


def load_perfguard_model(model_path, db):
    if db == "tpch":
        config.threshold = 0.5
    else:
        config.threshold = 0.7
    plan2score = Plan2Score()
    plan2score.load_model(model_path)
    return plan2score


def get_perfguard_result(plans1: list, plans2: list, model: Plan2Score):
    """

    :param plans1:
    :param plans2:
    :param model:
    :return: 1 : p1<=p2. 0: p1>p2
    """
    # model_path = 'model_pth/' + config.data + '_' + str(config.data_num)
    plans1 = [[p] for p in plans1]
    plans2 = [[p] for p in plans2]
    features1, features2 = model.get_features(plans1, plans2)
    adjaceny_matrix_list_x1, adjaceny_matrix_list_x2 = model.get_two_adjaceny_matrix(plans1, plans2)
    label_list = model.predict(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
    return label_list


if __name__ == "__main__":
    pass
    # # read plans
    # test_path = "/home/admin/wd_files/Lero/test_script/uncertainty_tools/data/stats_test"
    # qid2plans = {}
    # qid2plan_perfguard = {}
    # with open(test_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         arr = line.strip().split(config.SEP)
    #         qid = arr[0]
    #         plans = arr[1:]
    #         qid2plans[qid] = plans
    # # model
    # model_path = 'model_pth/' + config.data + '_' + str(config.data_num)
    # plan2score = Plan2Score()
    # plan2score.load_model(model_path)
    #
    # # predict
    # for qid in qid2plans.keys():
    #     plans = qid2plans[qid]
    #     tmp_plan = plans[0]
    #     for i in range(1, len(plans)):
    #         now_plan = plans[i]
    #         plan_list1 = [json.loads(tmp_plan)]
    #         plan_list2 = [json.loads(now_plan)]
    #         features1, features2 = plan2score.get_features(plan_list1, plan_list2)
    #         adjaceny_matrix_list_x1, adjaceny_matrix_list_x2 = plan2score.get_two_adjaceny_matrix(plan_list1,
    #                                                                                               plan_list2)
    #         label_list = plan2score.predict(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
    #
    #         if label_list[0] == 0:
    #             tmp_plan = now_plan
    #     # print("qid: {},plan:{}".format(qid,tmp_plan))
    #     qid2plan_perfguard[qid] = tmp_plan
    #
    # #
    # sum_latency_pg = 0
    # sum_latency_perfguard = 0
    # for qid in qid2plans.keys():
    #     sum_latency_pg += json.loads(qid2plans[qid][0])[0]['Execution Time'] / 1000
    #     sum_latency_perfguard += json.loads(qid2plan_perfguard[qid])[0]['Execution Time'] / 1000
    # print(sum_latency_pg)
    # print(sum_latency_perfguard)
