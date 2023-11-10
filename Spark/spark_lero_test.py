import sys

sys.path.append("auncel/")
sys.path.append("./")
sys.path.append("../")
from drawConfig import regression_algo_name, font_size, get_plt, name_2_color, capitalize

import os
import unittest

import math
import numpy as np
import torch.nn
from pandas import DataFrame
from Spark.auncel.test_script.config import DATA_BASE_PATH

import auncel.model_config
from UncertantyModel.PairPlanGroupEstimate import PairPlanGroupEstimate
from UncertantyModel.SimilarModelEstimate import SimilarModelEstimate
from UncertantyModel.SingleModelConfidenceEstimate import SingleModelConfidenceEstimate
from UncertantyModel.UncertaintyEstimate import PlanGroupEstimate, ConfidenceEstimate
from auncel.Common.Cache import PredictTimeCache
from auncel.Common.Draw import draw_by_agg
from auncel.Common.PlanFactory import PlanFactory
from auncel.Common.TimeStatistic import TimeStatistic
from auncel.model import AuncelModel, AuncelModelPairConfidenceWise
from auncel.model_config import ModelType, model_type, uncertainty_threshold
from auncel.model_config import confidence_estimate_type, ConfidenceEstimateType
from auncel.model_transformer import AuncelModelTransformerPairWise
from auncel.train import train
from auncel.utils import read_plans, json_str_to_json_obj, flat_depth2_list, \
    add_to_json, get_confidence_model_name

input_ratio = None


class SparkLeroTest(unittest.TestCase):
    training_type = 5

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "lero"
        global input_ratio
        input_ratio = sys.argv[2] if len(sys.argv) == 3 else None

    def test_static_all_tpcds(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_tpcds()

    def test_tpcds(self):
        ratio = 3 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        if ratio == 1:
            train_dataset_name = "tpcdsQuery12_100_train"
        elif ratio == 2:
            train_dataset_name = "tpcdsQuery24_100_train"
        elif ratio == 3:
            train_dataset_name = "tpcdsQuery36_100_train"
        else:
            train_dataset_name = "tpcdsQuery50_100_train"

        test_dataset_name = "tpcdsQuery50_10_test"
        model_name = self.get_model_name(train_dataset_name)
        self.auncel_predict_performance(test_dataset_name, train_dataset_name, model_name, "tpcds")

    def test_dynamic_tpcds(self):
        train_dataset_name = "tpcdsQuery50_100_train"
        self.auncel_predict_dynamic_performance(train_dataset_name, "tpcds")

    def test_train_tpcds_dynamic(self):
        ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        train_dataset_name = "tpcdsQuery50_100_train"
        for ratio in ratios:
            self.train(train_dataset_name, "tpcds", data_limit_ratio=ratio)

    def test_train_tpcds(self):
        train_dataset_name = "tpcdsQuery12_100_train"
        # train_dataset_name = "tpcdsQuery24_100_train"
        # train_dataset_name = "tpcdsQuery36_100_train"
        # train_dataset_name = "tpcdsQuery50_100_train"
        self.train(train_dataset_name, "tpcds")

    def _add_plan_metric(self, plans_for_query, model: AuncelModel, sqls, db):
        print("add metric")
        all_plans = flat_depth2_list(plans_for_query)
        predicts = self.predict("ignore", all_plans, "empty", model, db, enable_cache=False)

        start = 0
        for i, plans in enumerate(plans_for_query):
            end = start + len(plans)
            predict_latencies = predicts[start:end]
            start = end
            total_count = 0
            correct_count = 0
            for i in range(len(predict_latencies)):
                p1 = plans[i]
                t1 = predict_latencies[i]
                for j in range(len(predict_latencies)):
                    # if i == j:
                    #     continue
                    p2 = plans[j]
                    t2 = predict_latencies[j]
                    if t1 <= t2 and p1["Execution Time"] <= p2["Execution Time"]:
                        correct_count += 1
                    total_count += 1
                p1["metric"] = correct_count / total_count if total_count > 0 else 0.5
                p1["predict"] = t1

    def predict(self, identifier, plans, model_name, auncel_model, dataset, enable_cache=True):
        cache = PredictTimeCache(identifier, model_name, enable=enable_cache)
        if cache.exist():
            predicts = cache.read()
        else:
            split_count = 3
            gap = int(len(plans) / split_count) + 1
            predicts = []
            for i in range(split_count):
                flat_plans = plans[0: gap] if i == 0 else plans[i * gap:min((i + 1) * gap, len(plans))]
                if len(flat_plans) > 0:
                    predicts += list(auncel_model.predict(auncel_model.to_feature(flat_plans, dataset)))
                    torch.cuda.empty_cache()
            for i in range(len(predicts)):
                assert len(predicts[i]) == 1
                predicts[i] = predicts[i][0]
            cache.save(predicts)
        return predicts

    def predict_by_queries(self, plans_for_queries, model_name, auncel_model, dataset):
        cache = PredictTimeCache("plans_for_queries", model_name, enable=True)
        if cache.exist():
            print("predict train plan latency trigger cache")
            predicts_for_queries = cache.read()
        else:
            print("predict train plan latency")
            predicts_for_queries = []
            for plans in plans_for_queries:
                predicts = []
                if len(plans) > 0:
                    predicts += list(auncel_model.predict(auncel_model.to_feature(plans, dataset)))
                for i in range(len(predicts)):
                    assert len(predicts[i]) == 1
                    predicts[i] = predicts[i].item()
                predicts_for_queries.append(predicts)
            print("predict train plan latency end")
            cache.save(predicts_for_queries)
        print("predict end")
        return predicts_for_queries

    def train(self, train_set_name, dataset_name, data_limit_ratio=None):
        train_data_path = self.get_data_path(train_set_name)
        if model_type == ModelType.TRANSFORMER:
            self.training_type = 5
        elif model_type == ModelType.TREE_CONV:
            self.training_type = 1
        elif model_type == ModelType.MSE_TREE_CONV:
            self.training_type = 0
        else:
            raise RuntimeError
        train(train_data_path, dataset_name=dataset_name,
              model_name=self.get_model_name(train_set_name, data_limit_ratio),
              training_type=self.training_type, data_limit_ratio=data_limit_ratio)

    def get_model(self):
        if model_type == ModelType.TRANSFORMER:
            return AuncelModelTransformerPairWise(None, None, None, None)
        elif model_type == ModelType.TREE_CONV:
            return AuncelModel(None)
        elif model_type == ModelType.MSE_TREE_CONV:
            return AuncelModel(None)
        else:
            raise RuntimeError

    def auncel_predict_dynamic_performance(self, train_set_name, dataset):
        # load model
        auncel.model_config.set_predict_status(True)

        print("Read candidate plans")
        plans_for_query = read_plans(self.get_data_path(train_set_name))

        gap = 300
        count = math.ceil(len(plans_for_query) / float(gap))

        regression_results = {}
        algo_results = []
        best_results = []
        spark_results = []
        uncertainty_thres = [0.6]
        data_ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(0, count - 0):
            data_ratio = data_ratios[i]
            print("current iteration is {}".format(i))
            print("Load Lero Model")
            model_name = self.get_model_name(train_set_name, data_ratio)
            auncel_model = self.get_model()
            auncel_model.load(model_name)

            min_pos = 0
            max_pos = max(i * gap, int(len(plans_for_query) * data_ratios[0]))
            cur_plans_for_query = plans_for_query[min_pos:max_pos]
            print("Build Eraser")
            estimate: ConfidenceEstimate = self.get_confidence_estimate(auncel_model, cur_plans_for_query, dataset,
                                                                        model_name, train_set_name)
            print("Start to select plan")
            test_plans_for_query = plans_for_query[i * gap:(i + 1) * gap]

            spark_results += self.get_spark_times(test_plans_for_query)
            best_results += self.get_best_times(test_plans_for_query)
            algo_results += self.choose_plan_by_lero(test_plans_for_query, auncel_model, dataset)

            latencies_for_queries = [None] * len(test_plans_for_query)
            plan_id_to_prob = {}
            for thres in uncertainty_thres:
                assert model_type == ModelType.TREE_CONV
                result = self.choose_plan_with_guard_pair_wise(test_plans_for_query, estimate,
                                                               auncel_model, thres, plan_id_to_prob,
                                                               latencies_for_queries,
                                                               dataset)
                if str(thres) not in regression_results:
                    regression_results[str(thres)] = []
                regression_results[str(thres)] += result[0]

        for thres in uncertainty_thres:
            algo_2_values = {
                self.algo: algo_results,
                "{}-{}".format(self.algo, regression_algo_name): regression_results[str(thres)],
                "Spark": spark_results
            }
            db = "tpcds"
            self.draw_scatter_chart(algo_2_values, db, "spark_dynamic_{}_{}".format(db, self.algo, str(thres)),
                                    self.algo.capitalize(), show_legend=True,
                                    width_inc=200)

    def auncel_predict_performance(self, test_set_name, train_set_name, model_name, dataset):
        auncel.model_config.set_predict_status(True)

        # load model
        print("Load Lero Model")
        auncel_model = self.get_model()
        auncel_model.load(model_name)

        print("Read candidate plans")
        plans_for_query = read_plans(self.get_data_path(test_set_name))
        plans_for_train_queries = read_plans(self.get_data_path(train_set_name))

        spark_plans = self.get_spark_times(plans_for_query)
        best_plans = self.get_best_times(plans_for_query)
        lero_plans = self.choose_plan_by_lero(plans_for_query, auncel_model, dataset)

        print("Build Eraser")
        estimate: ConfidenceEstimate = self.get_confidence_estimate(auncel_model, plans_for_train_queries, dataset,
                                                                    model_name, train_set_name)

        print("Start to select plan")
        uncertainty_thres = [0.6]

        auncel_plans = {}

        latencies_for_queries = [None] * len(plans_for_query)
        plan_id_to_prob = {}
        plan_id_to_predict_interval = {}
        for thres in uncertainty_thres:
            if confidence_estimate_type in [ConfidenceEstimateType.SINGLE_MODEL,
                                            ConfidenceEstimateType.SimilarModelEstimate]:
                res = self.choose_plan_with_guard_point_wise(plans_for_query, estimate,
                                                             auncel_model, thres, plan_id_to_prob,
                                                             latencies_for_queries, plan_id_to_predict_interval,
                                                             dataset)
                auncel_plans[str(thres)] = res[0]
            else:
                if model_type == ModelType.MSE_TREE_CONV:
                    res = self.choose_plan_with_guard_point_wise_by_interval(plans_for_query, estimate,
                                                                             auncel_model, thres, plan_id_to_prob,
                                                                             latencies_for_queries,
                                                                             plan_id_to_predict_interval, dataset)
                    auncel_plans[str(thres)] = res[0]

                elif model_type == ModelType.TREE_CONV:
                    res = self.choose_plan_with_guard_pair_wise(plans_for_query, estimate,
                                                                auncel_model, thres, plan_id_to_prob,
                                                                latencies_for_queries,
                                                                dataset)
                    auncel_plans[str(thres)] = res[0]

        queries_name = ["Q{}".format(i) for i in range(len(spark_plans))]

        self.draw(spark_plans, best_plans, lero_plans, auncel_plans, train_set_name, uncertainty_thres, queries_name)

    def draw(self, spark_plans, best_plans, lero_plans, auncel_plans, train_set_name, uncertainty_thres, queries_name,
             prefix=""):
        # print .csv
        df = DataFrame(
            {"Spark": spark_plans,
             "Lero": lero_plans,
             }
        )
        for thres in uncertainty_thres:
            df["Eraser"] = auncel_plans[str(thres)]
        df["qn"] = queries_name

        train_name_2_suffix = {
            "tpcdsQuery12_100_train": 1,
            "tpcdsQuery24_100_train": 2,
            "tpcdsQuery36_100_train": 3,
            "tpcdsQuery50_100_train": 4,
        }
        suffix = train_name_2_suffix[train_set_name]

        name = "spark_lero_test_{}".format(suffix)
        # df.to_csv(
        #     "../RegressionFramework/fig/{}{}_performance{}.csv".format("", name, ""))

        y_names = list(df.columns)
        y_names.remove("qn")
        draw_by_agg(df, y_names=y_names, agg="mean", file_name=name)

    def update_plan_with_accuracy(self, train_set_name, model_name, dataset):
        """
        computing predict accuracy
        :param train_set_name:
        :param model_name:
        :param dataset:
        :return: plan_jsons,each json contain an entry whose key is accuracy and value is accuracyValue
        """
        # load model
        auncel_model = self.get_model()
        auncel_model.load(model_name)
        print("read_plans start")
        plans_for_queries = read_plans(self.get_data_path(train_set_name))
        predicts_for_queries = self.predict_by_queries(plans_for_queries, model_name, auncel_model,
                                                       dataset)
        new_plans_for_queries = []
        # add accuracy
        for i in range(len(plans_for_queries)):
            plans = plans_for_queries[i]
            new_plans = []
            for j in range(len(plans)):
                predict = predicts_for_queries[i][j]
                new_plans.append(add_to_json("predict", predict, plans[j]))
            new_plans_for_queries.append(new_plans)
        return new_plans_for_queries

    def get_qids(self, plans):
        return [json_str_to_json_obj(p)["Qid"] for p in plans]

    def get_spark_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(json_str_to_json_obj(plans[0])["Execution Time"])
        return res

    def get_best_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(self.find_best_plan(plans)[1])
        return res

    def get_compare_spark_query_df(self, spark_plans, auncel_plans: dict):
        good_df = DataFrame()
        bad_df = DataFrame()
        for key, times in auncel_plans.items():
            key = "thres{}".format(key)
            good_count = 0
            bad_count = 0
            for i in range(len(spark_plans)):
                if times[i] < spark_plans[i]:
                    good_count += 1
                elif times[i] > spark_plans[i]:
                    bad_count += 1
            good_df[key] = [good_count]
            bad_df[key] = [bad_count]
        return good_df, bad_df

    def get_model_name(self, dataset_name, data_limit_ratio=None):
        suffix = "" if data_limit_ratio is None else "_{}".format(data_limit_ratio)
        return "spark_{}_{}_model_0{}".format(dataset_name, model_type.name.lower(), suffix)

    def choose_plan_by_compare(self, plans, auncel_model, dataset=None):
        spark_score = None
        max_prob = -float('inf')
        max_prob_idx = -1
        for i, plan in enumerate(plans):
            qid = json_str_to_json_obj(plan)["Qid"]
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
        if max_prob < uncertainty_threshold:
            max_prob_idx = 0
        latency = json_str_to_json_obj(plans[max_prob_idx])["Execution Time"]
        return max_prob_idx, latency

    def choose_plan_by_lero(self, plans_for_queries, auncel_model, dataset):
        res = []
        for i, plans in enumerate(plans_for_queries):
            latencies: list = list(auncel_model.predict(auncel_model.to_feature(plans, dataset)))
            idx = latencies.index(min(latencies))
            res.append(json_str_to_json_obj(plans[idx])["Execution Time"])
        return res

    def choose_plan_with_guard_pair_wise(self, plans_for_queries, estimate: ConfidenceEstimate, auncel_model, thres,
                                         plan_id_to_prob, latencies_for_queries, dataset=None):
        TimeStatistic.start("choose_plan_with_guard_pair_wise")

        res = []
        for i, plans in enumerate(plans_for_queries):
            id_to_win_count = {}
            if latencies_for_queries[i] is None:
                TimeStatistic.start("predict")
                latencies_for_queries[i] = auncel_model.predict(auncel_model.to_feature(plans_for_queries[i], dataset))
                TimeStatistic.end("predict")

            latencies = latencies_for_queries[i]

            for j in range(len(plans)):
                plan1 = plans[j]
                win_count = 0
                for k in range(len(plans)):
                    plan2 = plans[k]
                    if j == k:
                        continue
                    if latencies[j] < latencies[k]:
                        TimeStatistic.start("estimate")
                        key = "{}_{}_{}".format(i, j, k)
                        if key not in plan_id_to_prob:
                            prob = estimate.estimate(plan1, plan2)
                            plan_id_to_prob[key] = prob
                        else:
                            prob = plan_id_to_prob[key]
                        TimeStatistic.end("estimate")
                        # print("confidence is {}".format(prob))
                        if prob >= thres:
                            win_count += 1
                id_to_win_count[j] = win_count

            spark_win_count = id_to_win_count[0]
            id_win_count = sorted(id_to_win_count.items(), key=lambda x: x[1])
            choose_idx = id_win_count[-1][0]
            choose_idx = 0 if spark_win_count >= id_to_win_count[choose_idx] else choose_idx
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            spark_actual_latency = json_str_to_json_obj(plans[0])["Execution Time"]

            res.append(latency)
        TimeStatistic.end("choose_plan_with_guard_pair_wise")
        return res, 0, 0, None, 0

    def choose_plan_with_guard_point_wise(self, plans_for_queries, estimate: ConfidenceEstimate, auncel_model, thres,
                                          plan_id_to_prob, latencies_for_queries, plan_id_to_predict_interval,
                                          dataset=None):
        all_slow_than_spark_count = 0
        all_slow_thres_count = 0
        no_group_count = 0
        total_plan_count = 0
        reasons = []
        res = []

        for i, plans in enumerate(plans_for_queries):
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = auncel_model.predict(auncel_model.to_feature(plans_for_queries[i], dataset))
            latencies = latencies_for_queries[i]

            origin_latencies = latencies
            buff = []
            total_plan_count += len(plans)
            for j, plan in enumerate(plans):
                key = "{}_{}".format(i, j)
                if key not in plan_id_to_prob:
                    prob, min_predict, max_predict = estimate.estimate(plan, latencies[j])
                    plan_id_to_prob[key] = prob
                    plan_id_to_predict_interval[key] = (min_predict, max_predict)
                else:
                    prob = plan_id_to_prob[key]
                    min_predict, max_predict = plan_id_to_predict_interval[key]

                # print("confidence is {}".format(prob))
                if prob == -1:
                    no_group_count += 1

                buff.append((prob, latencies[j][0], j))

            origin_buff = buff
            reasons = []

            # decision
            assert len(latencies[0]) == 1
            spark_latency = latencies[0][0]

            all_slow_than_spark_count += 1 if len(list(filter(lambda x: x[1] < spark_latency, origin_buff))) == 0 else 0
            all_slow_thres_count += 1 if len(list(filter(lambda x: x[0] <= thres, origin_buff))) == 0 else 0

            buff = list(filter(lambda x: x[0] >= thres and x[1] < spark_latency, origin_buff))
            # buff = list(filter(lambda x: x[0] >= thres, buff))
            buff = sorted(buff, key=lambda x: x[1])

            choose_idx = buff[0][2] if len(buff) > 0 else 0
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            spark_actual_latency = json_str_to_json_obj(plans[0])["Execution Time"]

            res.append(latency)

        return res, all_slow_than_spark_count, no_group_count / total_plan_count, reasons, all_slow_thres_count

    def choose_plan_with_guard_point_wise_by_interval(self, plans_for_queries, estimate: ConfidenceEstimate,
                                                      auncel_model, thres, plan_id_to_prob, latencies_for_queries,
                                                      plan_id_to_predict_interval, dataset=None):
        all_slow_than_spark_count = 0
        no_group_count = 0
        total_plan_count = 0
        res = []

        for i, plans in enumerate(plans_for_queries):
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = auncel_model.predict(auncel_model.to_feature(plans_for_queries[i], dataset))
            latencies = latencies_for_queries[i]

            # [(min,max,mean latency)]
            multiple_latencies = []
            total_plan_count += len(plans)
            for j, plan in enumerate(plans):
                key = "{}_{}".format(i, j)
                if key not in plan_id_to_prob:
                    prob, min_predict, max_predict, mean_predict = estimate.estimate(plan, latencies[j])
                    plan_id_to_prob[key] = prob
                    plan_id_to_predict_interval[key] = (min_predict, max_predict, mean_predict)
                else:
                    prob = plan_id_to_prob[key]
                    min_predict, max_predict, mean_predict = plan_id_to_predict_interval[key]
                if prob == -1:
                    # print("confidence is {}".format(prob))
                    no_group_count += 1
                multiple_latencies.append((min_predict, max_predict, mean_predict, j))

            default_latency = multiple_latencies[0][2]

            choose = None
            for i in range(1, len(multiple_latencies)):
                target = multiple_latencies[i]
                if target[2] < default_latency and (choose is None or choose[2] > target[2]):
                    choose = target

            all_slow_than_spark_count += 1 if choose is None else 0
            choose_idx = choose[3] if choose is not None else 0
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            res.append(latency)

        return res, all_slow_than_spark_count, no_group_count / total_plan_count, [], 0

    def choose_plan_with_single_confidence_model(self, plans_for_queries, estimate: SingleModelConfidenceEstimate,
                                                 auncel_model, thres, plan_id_to_prob, latencies_for_queries,
                                                 dataset=None):
        total_plan_count = 0
        res = []

        for i, plans in enumerate(plans_for_queries):
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = auncel_model.predict(auncel_model.to_feature(plans_for_queries[i], dataset))
            latencies = latencies_for_queries[i]

            buff = []
            total_plan_count += len(plans)
            for j, plan in enumerate(plans):
                key = "{}_{}".format(i, j)
                if key not in plan_id_to_prob:
                    adjust_latency, adjust_ratio = estimate.estimate(plan, latencies[j])
                    plan_id_to_prob[key] = adjust_latency
                else:
                    adjust_latency = plan_id_to_prob[key]
                buff.append((j, adjust_latency))
            # decision
            assert len(latencies[0]) == 1
            spark_latency = buff[0][1]
            buff = sorted(buff, key=lambda x: x[1])

            choose_idx = buff[0][0] if buff[0][1] < spark_latency else 0
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]

            res.append(latency)

        return res, 0, 0, [], 0

    def reason_analysis(self, infos, plans, thres):
        assert len(infos) == len(plans)
        run_times = []
        for plan in plans:
            run_times.append(json_str_to_json_obj(plan)["Execution Time"])

        better_plans_idx = [i for i, time in enumerate(run_times) if time < run_times[0]]
        # better_plans_idx = [run_times.index(min(run_times))]

        larger_than_spark_count = 0
        prob_small_thres = 0
        no_smallest_latency = 0
        min_latency_bad_spark_count = 0

        min_predict_idx = sorted(infos, key=lambda x: x[1])[0][2]
        predict_spark_latency = infos[0][1]

        for better_plan_idx in better_plans_idx:
            infos = sorted(infos, key=lambda x: x[2])

            # better plan latency larger than spark
            larger_than_spark_count += 1 if infos[better_plan_idx][1] > predict_spark_latency else 0

            # better plan latency prob smaller than thres
            prob_small_thres += 1 if infos[better_plan_idx][0] < thres else 0

            # better plan latency(satisfy thres) is not the most small latency
            no_smallest_latency += 1 if min_predict_idx != better_plan_idx else 0

        # the plan with the smallest latency worse than spark
        actual_spark_latency = run_times[0]
        min_latency_bad_spark_count += 1 if run_times[min_predict_idx] > actual_spark_latency else 0

        min_latency_good_spark_count = 1 if run_times[min_predict_idx] < actual_spark_latency else 0
        min_latency_equal_spark_count = 1 if run_times[min_predict_idx] == actual_spark_latency else 0

        #
        min_predict_prob = infos[min_predict_idx][0]
        min_latency_bad_spark_count_and_larger_prob = 1 if min_latency_bad_spark_count > 0 and min_predict_prob > thres else 0
        min_latency_good_spark_count_and_larger_prob = 1 if min_latency_bad_spark_count > 0 and min_predict_prob > thres else 0
        min_latency_equal_spark_count_and_larger_prob = 1 if min_latency_equal_spark_count > 0 and min_predict_prob > thres else 0

        if len(better_plans_idx) > 0:
            size = len(better_plans_idx)
            return larger_than_spark_count / size, prob_small_thres / size, min_latency_bad_spark_count, \
                min_latency_bad_spark_count_and_larger_prob, \
                min_latency_good_spark_count, min_latency_good_spark_count_and_larger_prob, min_latency_equal_spark_count, \
                min_latency_equal_spark_count_and_larger_prob
        return 0, 0, min_latency_bad_spark_count, min_latency_bad_spark_count_and_larger_prob, \
            min_latency_good_spark_count, min_latency_good_spark_count_and_larger_prob, min_latency_equal_spark_count, \
            min_latency_equal_spark_count_and_larger_prob

    def get_confidence_estimate(self, auncel_model, plans_for_queries, dataset, model_name, train_set_name):
        if confidence_estimate_type == ConfidenceEstimateType.SINGLE_MODEL:
            auncel_model = self.get_confidence_model(train_set_name)
            return SingleModelConfidenceEstimate(auncel_model, dataset)
        elif confidence_estimate_type == ConfidenceEstimateType.SimilarModelEstimate:
            plans = flat_depth2_list(plans_for_queries)
            predicts = self.predict("confidence_model", plans, model_name, auncel_model, dataset)
            confidence_model = self.get_confidence_model(train_set_name)
            return SimilarModelEstimate(plans, predicts, dataset, confidence_model)
        else:
            plan_group_estimate: PlanGroupEstimate = self.build_plan_group_estimate(auncel_model, plans_for_queries,
                                                                                    dataset, model_name, train_set_name)
            # plan_group_estimate.draw_dot()
            # plan_group_estimate.save_leaf_nodes(get_group_plans_file_path(train_set_name,model_type))
            return plan_group_estimate

    def build_plan_group_estimate(self, auncel_model, plans_for_queries, dataset, model_name, train_set_name):
        confidence_model = self.get_confidence_model(train_set_name)
        all_flat_plans = flat_depth2_list(plans_for_queries)
        predicts = self.predict("target_model", all_flat_plans, model_name, auncel_model, dataset, enable_cache=False)
        if model_type == ModelType.MSE_TREE_CONV:
            return PlanGroupEstimate(all_flat_plans, predicts, model_name, train_set_name, dataset, confidence_model)
        elif model_type == ModelType.TREE_CONV:
            return PairPlanGroupEstimate(all_flat_plans, predicts, model_name, train_set_name, dataset,
                                         confidence_model)
        else:
            raise RuntimeError

    def get_confidence_model(self, train_set_name):
        auncel_model = None
        if confidence_estimate_type == ConfidenceEstimateType.SINGLE_MODEL:
            auncel_model = AuncelModel(None)
            auncel_model.load(get_confidence_model_name(train_set_name, auncel.model_config.confidence_model_type.name))
            print("use SINGLE_MODEL to get confidence")
        elif confidence_estimate_type == ConfidenceEstimateType.ADAPTIVE_MODEL:
            if auncel.model_config.confidence_model_type == ModelType.MSE_TREE_CONV:
                auncel_model = AuncelModel(None)
                print("use AuncelModel to get confidence")
            else:
                auncel_model = AuncelModelPairConfidenceWise(None, None)
                print("use AuncelModelPairConfidenceWise to get confidence")
            auncel_model.load(get_confidence_model_name(train_set_name, auncel.model_config.confidence_model_type.name))
            return auncel_model
        elif confidence_estimate_type == ConfidenceEstimateType.SimilarModelEstimate:
            auncel_model = AuncelModelPairConfidenceWise(None, None)
            auncel_model.load(get_confidence_model_name(train_set_name, auncel.model_config.confidence_model_type.name))
            print("use AuncelModelPairConfidenceWise to get confidence")
        return auncel_model

    def get_data_path(self, file_name):
        return os.path.join(DATA_BASE_PATH, file_name)

    def draw_plan(self, plan):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
            plan = PlanFactory.get_plan_instance(auncel.model_config.db_type, plan)
        return plan.draw_dot()

    def find_best_plan(self, plans):
        best_plan_idx = -1
        best_latency = float('inf')
        for i, plan in enumerate(plans):
            plan = json_str_to_json_obj(plan)
            if best_latency > plan["Execution Time"]:
                best_plan_idx = i
                best_latency = plan["Execution Time"]
        return best_plan_idx, best_latency

    def draw_scatter_chart(self, algo_2_values, db, file, title, x_ticks=None, width_inc=0, font_inc=0, x_gap=200,
                           show_legend=False):
        plt = get_plt()
        plt.figure(figsize=(10 + width_inc / 100, 10))
        for algo, values in algo_2_values.items():
            algo_alias = algo.split("-" if db == "tpcds" else "_")[0]
            color = name_2_color[algo_alias.lower()]
            # symbol = name_2_scatter_symbol[algo.lower()]
            line_dash = ":" if regression_algo_name in algo else "-"
            values = self.to_minute(values, db)

            # values=list(np.log(np.array(values)))
            plt.plot(list(range(len(values))), self.accumulate(values), linestyle=line_dash, label=capitalize(algo),
                     linewidth=10,
                     color=to_rgb_tuple(color))

        cur_font_size = font_size + font_inc - 10
        plt.ylabel("E2E Execution Total Time (Mins)", fontsize=cur_font_size)
        plt.xlabel("# of queries", fontsize=cur_font_size)
        plt.yticks(size=cur_font_size)
        plt.xticks(
            [i for i in range(len(values)) if i % x_gap == 0],
            [i for i in range(len(values)) if i % x_gap == 0],
            size=cur_font_size, weight='bold')
        if show_legend:
            # plt.legend()._spacing = 0.5
            plt.legend(loc='upper center', frameon=True, handletextpad=0.3, columnspacing=0.5,
                       bbox_to_anchor=(0.33, 1.03),
                       ncol=1,
                       fontsize=cur_font_size)
        plt.grid()
        plt.tight_layout()
        plt.savefig("../RegressionFramework/fig/{}.png".format(file), format="png")
        plt.show()

    def accumulate(self, values):
        new_arr = []
        for i in range(len(values)):
            new_arr.append(sum(values[:i + 1]))
        return new_arr

    def to_minute(self, values, db):
        if db == "tpcds":
            return [v / 60.0 for v in values]
        return [v / 60.0 / 1000 for v in values]


def to_rgb_tuple(color: str):
    return tuple([int(c) / 255 for c in color[4:-1].split(",")])


if __name__ == '__main__':
    unittest.main()
