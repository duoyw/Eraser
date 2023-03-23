import unittest

from Perfguard.perfguard import PerfGuard, PerfGuardModel
from RegressionFramework.Common.Cache import Cache
from RegressionFramework.Plan.Plan import Plan, json_str_to_json_obj
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.RegressionFramework import RegressionFramework, PerfRegressionFramework
from RegressionFramework.config import model_base_path
from RegressionFramework.utils import flat_depth2_list
from model_test import LeroTest
import numpy as np

from perfguard_result_cache import PerfguardResult, get_perfguard_result_manager
from plan2score import Plan2Score, get_perfguard_result, load_perfguard_model


class PerfguardTest(LeroTest):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "perfguard"

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_all_job(self):
        ratios = [1, 2, 3, 4]
        for ratio in ratios:
            self.test_job(ratio)

    def test_all_stats(self):
        ratios = [1, 2, 3, 4]
        for ratio in ratios:
            self.test_stats(ratio)

    def test_all_tpch(self):
        ratios = [1, 2, 3, 4]
        for ratio in ratios:
            self.test_tpch(ratio)

    def test_job(self, ratio=None):
        ratio = 1 if ratio is None else ratio
        train_file_name = "lero_job{}.log.training".format(ratio)
        sql_file_name = "job{}.txt".format(ratio)
        test_file_name = "job_test"
        model_path = self.get_model_path(train_file_name)

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "imdb", model_path)

    def test_stats(self, ratio=None):
        ratio = 1 if ratio is None else ratio
        train_file_name = "lero_stats{}.log.training".format(ratio)
        sql_file_name = "stats{}.txt".format(ratio)
        test_file_name = "stats_test"
        model_path = self.get_model_path(train_file_name)

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "stats", model_path)

    def test_tpch(self, ratio=None):
        ratio = 1 if ratio is None else ratio

        train_file_name = "lero_tpch{}.log.training".format(ratio)
        sql_file_name = "tpch{}.txt".format(ratio)
        test_file_name = "tpch_test"
        model_path = self.get_model_path(train_file_name)

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "tpch", model_path)

    def get_model_path(self, training_name: str):
        training_name = training_name.replace("lero", "perfguard")
        return model_base_path + "perfguard/test_model_{}".format(training_name)
        # return "{}perfguard/{}/{}".format(model_base_path, "model_pth", training_name)

    def load_model(self, model_name, db):
        return load_perfguard_model(model_name, db)

    def _add_plan_metric(self, plans_for_query, model: Plan2Score, sqls):
        self._add_id_to_plan(plans_for_query)
        print("add metric")
        for plans in plans_for_query:
            left = []
            right = []
            for i in range(len(plans)):
                for j in range(len(plans)):
                    left.append(plans[i])
                    right.append(plans[j])
            if len(left) <= 0:
                continue
            results = get_perfguard_result(left, right, model)
            total_count = 0
            correct_count = 0
            for i in range(len(plans)):
                p1 = plans[i]
                for j in range(len(plans)):
                    p2 = plans[j]
                    r = results[i * len(plans) + j]
                    if r and p1["Execution Time"] <= p2["Execution Time"]:
                        correct_count += 1
                    total_count += 1
                p1["metric"] = correct_count / total_count if total_count > 0 else 0.5

    def predict(self, model, plans):
        return None

    def select_plan_by_model(self, model, plans_for_query, model_path, sqls):
        times_for_query = []
        for plans in plans_for_query:
            best_idx = 0
            for i in range(1, len(plans)):
                result = get_perfguard_result([plans[best_idx]], [plans[i]], model)[0]
                if result == 0:
                    best_idx = i
            times_for_query.append(plans[best_idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model, regression_framework: RegressionFramework,
                                             plans_for_query, latencies_for_queries, thres, plan_id_2_confidence, sqls):
        times_for_query = []
        for qid, plans in enumerate(plans_for_query):
            print("cur query idx is {}".format(qid))

            best_idx = 0
            for i in range(1, len(plans)):
                best_plan = plans[best_idx]
                cur_plan = plans[i]
                result = get_perfguard_result([best_plan], [cur_plan], model)[0]
                if result == 0:
                    key = (best_plan["id"], cur_plan["id"])
                    if key in plan_id_2_confidence:
                        confidence = plan_id_2_confidence[key]
                    else:
                        confidence = regression_framework.evaluate(best_plan, cur_plan)
                        plan_id_2_confidence[key] = confidence
                    # print("confidence is {}".format(confidence))
                    if confidence >= thres:
                        best_idx = i
            # print("best_idx is {}".format(best_idx))
            times_for_query.append(plans[best_idx]["Execution Time"])
        return times_for_query

    def _init_regression_framework(self, train_plans, train_sqls, db, train_file_name, model):
        return PerfRegressionFramework(train_plans, train_sqls, db, train_file_name, model)


if __name__ == '__main__':
    unittest.main()
