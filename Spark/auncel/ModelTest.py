import json
import math
import unittest

import os.path

import torch.nn
from numpy import mean

import model_config
from Common.Draw import draw
from auncel.Common.DotDrawer import draw_dot_spark_plan
from model_config import model_type, ModelType
from model_transformer import AuncelModelTransformerPairWise
from feature import json_str_to_json_obj
from model import AuncelModel
from test_script.config import DATA_BASE_PATH, PROJECT_BASE_PATH
from train import train
from utils import read_plans, read_config, to_tree_json, cal_plan_height_and_size
from pandas import DataFrame
import numpy as np


class MyTestCase(unittest.TestCase):
    training_type = 5

    def test_stats(self):
        # train_dataset_name = "stats10NodeTrainDatasetQ1-500_0929"
        train_dataset_name = "stats10NodeTrainDatasetQ1-1000_1002"
        # train_dataset_name = "stats10Q1000_train0911_wo508"
        # train_dataset_name = "stats10Q146_test0910_wo136"
        # train_dataset_name = "stats10Q10"
        # train_dataset_name = "stats10Q50"
        self.train(train_dataset_name, "stats")

    def test_train_stats_loop(self):
        # train_datasets = ["stats10NodeTrainDatasetQ1-200", "stats10NodeTrainDatasetQ1-400",
        #                   "stats10NodeTrainDatasetQ1-600", "stats10NodeTrainDatasetQ1-800",
        #                   "stats10NodeTrainDatasetQ1-1000"]
        train_datasets = ["stats10NodeTrainDatasetQ1-200_order_by_join", "stats10NodeTrainDatasetQ1-400_order_by_join",
                          "stats10NodeTrainDatasetQ1-600_order_by_join", "stats10NodeTrainDatasetQ1-800_order_by_join",
                          "stats10NodeTrainDatasetQ1-1000_order_by_join"]
        # train_datasets = ["stats10NodeTrainDatasetQ1-800_order_by_join",
        #                   "stats10NodeTrainDatasetQ1-1000_order_by_join"]
        for train_dataset_name in train_datasets:
            print("cur dataset is {}".format(train_dataset_name))
            self.train(train_dataset_name, "stats")

    def test_performance_stats(self):
        # train_dataset_name = "stats10NodeTrainDatasetQ1-1000"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-800"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-600"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-400"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-200"

        # train_dataset_name = "stats10NodeTrainDatasetQ1-1000_order_by_join"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-800_order_by_join"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-600_order_by_join"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-400_order_by_join"
        train_dataset_name = "stats10NodeTrainDatasetQ1-200_order_by_join"

        # show performance
        # train_dataset_name = "stats10Q10"
        # train_dataset_name = "stats10Q1000_train0911_wo508"
        # test_dataset_name = "stats10Q1000_train0911_wo508"
        # test_dataset_name = "stats10Q146_test0910_wo136"
        # test_dataset_name = "stats10Q146_test0910_wo136"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-500_0929"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-1000_1002"
        # test_dataset_name = "stats10NodeTrainDatasetQ1-1000_1002"
        # train_dataset_name = "stats10NodeTestDataset_0928"
        # test_dataset_name = "stats10NodeTestDataset_0928"
        test_dataset_name = "stats10NodeTestDataset_delete_larger"
        model_name = self.get_model_name(train_dataset_name)
        self.auncel_predict_performance(test_dataset_name, model_name, "stats")

    def test_train_tpcds_loop(self):
        # train_datasets = ["tpcdsQuery50_100_train_0914", "tpcdsQuery40_100_train_0914",
        #                   "tpcdsQuery30_100_train_0914", "tpcdsQuery20_100_train_0914",
        #                   "tpcdsQuery10_100_train_0914"]
        train_datasets = ["tpcdsQuery40_100_train_0914"]
        # train_datasets = ["tpcdsQuery20_100_train_0914",
        #                   "tpcdsQuery10_100_train_0914"]
        for train_dataset_name in train_datasets:
            print("cur dataset is {}".format(train_dataset_name))
            self.train(train_dataset_name, "stats")

    def test_tpcds(self):
        # train_dataset_name = "tpcdsQuery50_100_train_0914"
        train_dataset_name = "tpcdsQuery40_100_train_0914"
        # train_dataset_name = "tpcdsQuery30_100_train_0914"
        # train_dataset_name = "tpcdsQuery20_100_train_0914"
        # train_dataset_name = "tpcdsQuery10_100_train_0914"
        # train_dataset_name = "tpcdsQuery1_10_train"
        self.train(train_dataset_name, "tpcds")

    def test_performance_tpcds(self):
        # show performance
        train_dataset_name = "tpcdsQuery50_100_train_0914"
        # train_dataset_name = "tpcdsQuery40_100_train_0914"
        # train_dataset_name = "tpcdsQuery30_100_train_0914"
        # train_dataset_name = "tpcdsQuery20_100_train_0914"
        # train_dataset_name = "tpcdsQuery10_100_train_0914"
        test_dataset_name = "tpcdsQuery49_10_test_0914"
        # train_dataset_name = "tpcdsQuery10_10_train"
        # test_dataset_name = "tpcdsQuery1_10_train"
        model_name = self.get_model_name(train_dataset_name)
        self.auncel_predict_performance(test_dataset_name, model_name, "tpcds")

    def train(self, train_set_name, dataset_name, data_limit_ratio=1.0):
        train_data_path = self.get_data_path(train_set_name)
        if model_type == ModelType.TRANSFORMER:
            self.training_type = 5
        elif model_type == ModelType.TREE_CONV:
            self.training_type = 1
        elif model_type == ModelType.MSE_TREE_CONV:
            self.training_type = 0
        else:
            raise RuntimeError
        train(train_data_path, dataset_name=dataset_name, model_name=self.get_model_name(train_set_name),
              training_type=self.training_type, data_limit_ratio=data_limit_ratio)

    def get_model(self):
        if model_type == ModelType.TRANSFORMER:
            return AuncelModelTransformerPairWise(None, None, None, None)
        elif model_type == ModelType.TREE_CONV:
            return AuncelModel(None)
        elif model_type == ModelType.MSE_TREE_CONV:
            return AuncelModel(None)

    def auncel_predict_performance(self, test_set_name, model_name, dataset):
        # read test plan for query
        model_config.is_predict = True
        plans_for_query = read_plans(self.get_data_path(test_set_name))

        # load model
        auncel_model = self.get_model()
        auncel_model.load(model_name)

        # compare the performance of all method
        spark_plans = []
        auncel_plans = []
        best_plans = []
        heights = []
        nodes_size = []
        # qids = []
        for plans in plans_for_query:
            if len(plans) >= 2:
                spark_plans.append(json_str_to_json_obj(plans[0])["Execution Time"])
                auncel_plans.append(self.predict(plans, auncel_model, dataset)[1])
                # auncel_plans.append(self.choose_plan_by_compare(plans, auncel_model, dataset)[1])
                best_plans.append(self.find_best_plan(plans)[1])
                h, w = self.cal_plans_height_and_size(plans)
                heights.append(h)
                nodes_size.append(w)
                # qids.append([json_str_to_json_obj(plans[0])["Qid"]])

        # queries_name = ["Q{}".format(i) for i in qids]
        queries_name = ["Q{}".format(i) for i in range(0, len(spark_plans))]

        # print .csv
        df = DataFrame(
            {"sparkPlan": spark_plans,
             "auncelPlan": auncel_plans,
             "bestPlans": best_plans,
             "heights": heights,
             "nodes_size": nodes_size,
             "qn": queries_name}
        )
        df.to_csv(PROJECT_BASE_PATH + "performance.csv")
        print("valid query size is {}".format(len(queries_name)))

        # draw
        draw(queries_name, spark_plans, auncel_plans, best_plans, "overall")
        draw(queries_name, [sum(spark_plans) / len(spark_plans)], [sum(auncel_plans) / len(auncel_plans)],
             [sum(best_plans) / len(best_plans)], "average")

    def test_draw_dot_plan(self):
        with open("./drawPlan.json") as f:
            line = f.readlines()
            plan = to_tree_json(line[0])
            plan = json_str_to_json_obj(plan)
            dot = draw_dot_spark_plan(plan["Plan"])
            print(dot)

    def test_draw_dot_plan2(self):
        with open("./drawPlan.json") as f:
            line = f.readlines()
            plan = json_str_to_json_obj(line[0])
            dot = draw_dot_spark_plan(plan["Plan"])
            print(dot)

    def test_cal_sigmoid(self):
        def cal(x):
            return 1.0 / (1 + math.exp(x))

        def cal_exp_log(x):
            return math.log(cal(x))

        print("0 is {}".format(cal(0)))
        print("0.0138 is {}".format(cal(0.0138)))
        print("0.0011 is {}".format(cal(0.0011)))
        print("0.0011 is {}".format(cal(0.0011)))
        print("0.291 is {}".format(cal(0.291)))

        print("####log#####")
        print("0 is {}".format(cal(0)))
        print("0.0138 is {}".format(cal_exp_log(0.0138)))
        print("0.0011 is {}".format(cal_exp_log(0.0011)))
        print("0.0011 is {}".format(cal_exp_log(0.0011)))
        print("0.291 is {}".format(cal_exp_log(0.291)))

    def get_model_name(self, dataset_name):
        return "spark_{}_{}_model_0".format(dataset_name, model_type.name.lower())

    def predict(self, plans, auncel_model, dataset=None):
        best_plan_idx = -1
        best_latency = float('inf')
        for i, plan in enumerate(plans):
            feature = auncel_model.to_feature([plan], dataset)
            latency = auncel_model.predict(feature)[0][0]
            if latency < best_latency:
                best_plan_idx = i
                best_latency = latency

        latency = json_str_to_json_obj(plans[best_plan_idx])["Execution Time"]
        print("len is {}, choose is{}".format(len(plans), best_plan_idx))
        return best_plan_idx, latency

    def choose_plan_by_compare(self, plans, auncel_model, dataset=None):
        spark_score = None
        max_prob = -float('inf')
        max_prob_idx = -1
        for i, plan in enumerate(plans):
            # qid = json_str_to_json_obj(plan)["Qid"]
            score = auncel_model.predicts(auncel_model.to_feature([plan], dataset))[0][0]
            if i == 0:
                spark_score = score
            else:
                sigmoid = torch.nn.Sigmoid()
                prob = sigmoid(torch.tensor(np.array([spark_score - score]))).item()
                print(prob)

                if prob > max_prob:
                    max_prob_idx = i
                    max_prob = prob
        if max_prob < 0.7:
            max_prob_idx = 0
        latency = json_str_to_json_obj(plans[max_prob_idx])["Execution Time"]
        return max_prob_idx, latency

    def get_data_path(self, file_name):
        return os.path.join(DATA_BASE_PATH, file_name)

    def find_best_plan(self, plans):
        best_plan_idx = -1
        best_latency = float('inf')
        for i, plan in enumerate(plans):
            plan = json_str_to_json_obj(plan)
            if best_latency > plan["Execution Time"]:
                best_plan_idx = i
                best_latency = plan["Execution Time"]
        return best_plan_idx, best_latency

    def cal_plans_height_and_size(self, plans):
        heights = []
        nodes_size = []
        for i, plan in enumerate(plans):
            plan = json_str_to_json_obj(plan)
            h, w = cal_plan_height_and_size(plan["Plan"])
            heights.append(h)
            nodes_size.append(w)
        return mean(heights), mean(nodes_size)

    if __name__ == '__main__':
        unittest.main()
