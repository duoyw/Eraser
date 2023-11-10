import sys
import unittest

sys.path.append("test_script/")
sys.path.append("Hyperqo/")
sys.path.append("Perfguard/")

from RegressionFramework.Common.TimeStatistic import TimeStatistic
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.RegressionFramework import RegressionFramework, PerfRegressionFramework
from RegressionFramework.config import model_base_path
from lero_test import LeroTest

from Perfguard.perfguard_result_cache import get_perfguard_result_manager
from Perfguard.plan2score import Plan2Score, get_perfguard_result, load_perfguard_model

input_ratio = None


class PerfguardTest(LeroTest):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "perfguard"

    def test_static_all_job(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_job()

    def test_static_all_stats(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_stats()

    def test_static_all_tpch(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_tpch()

    def test_dynamic_job(self):
        train_file_name = "lero_job4.log.training"
        sql_file_name = "job4.txt"
        self.dynamic_performance(train_file_name, sql_file_name, "imdb")

    def test_dynamic_tpch(self):
        train_file_name = "lero_tpch4.log.training"
        sql_file_name = "tpch4.txt"
        self.dynamic_performance(train_file_name, sql_file_name, "tpch")

    def test_job(self):
        ratio = 4 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_job{}.log.training".format(ratio)
        sql_file_name = "job{}.txt".format(ratio)
        test_file_name = "job_test"

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "imdb")

    def test_stats(self):
        ratio = 4 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_stats{}.log.training".format(ratio)
        sql_file_name = "stats{}.txt".format(ratio)
        test_file_name = "stats_test"

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "stats")

    def test_tpch(self):
        ratio = 4 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_tpch{}.log.training".format(ratio)
        sql_file_name = "tpch{}.txt".format(ratio)
        test_file_name = "tpch_test"

        get_perfguard_result_manager().build(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "tpch")

    def get_dynamic_model_name(self, data_file_name: str, count):
        data_file_name = data_file_name.replace("lero", "perfguard")
        return super().get_dynamic_model_name(data_file_name, count)

    def get_model_name(self, training_name: str, model_with_generate_sql=False):
        training_name = training_name.replace("lero", "perfguard")
        return model_base_path + "perfguard/test_model_{}".format(training_name)

    def load_model(self, model_name, db):
        return load_perfguard_model(model_name, db)

    def _add_plan_metric(self, plans_for_query, model: Plan2Score, sqls):
        self._add_id_to_plan(plans_for_query)
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

    def select_plan_by_model(self, model, plans_for_query, model_path, sqls, db):
        times_for_query = []
        for plans in plans_for_query:
            best_idx = 0
            for i in range(1, len(plans)):
                result = get_perfguard_result([plans[best_idx]], [plans[i]], model)[0]
                if result == 0:
                    best_idx = i
            times_for_query.append(plans[best_idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model, regression_framework: RegressionFramework, plans_for_query,
                                             latencies_for_queries, thres, plan_id_2_confidence, sqls, ood_thres=None):
        times_for_query = []
        for qid, plans in enumerate(plans_for_query):
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
                        p1 = PlanFactory.get_plan_instance("pg", best_plan)
                        p2 = PlanFactory.get_plan_instance("pg", cur_plan)
                        TimeStatistic.start("{}_eraser_infer".format(self.algo))
                        confidence = regression_framework.evaluate(p1, p2)
                        TimeStatistic.end("{}_eraser_infer".format(self.algo))
                        plan_id_2_confidence[key] = confidence
                    # print("confidence is {}".format(confidence))
                    if confidence >= thres:
                        best_idx = i
            # print("best_idx is {}".format(best_idx))
            times_for_query.append(plans[best_idx]["Execution Time"])
        return times_for_query

    def _init_regression_framework(self, train_plans, plans_for_queries, train_sqls, db, train_file_name, model,
                                   mode="static",
                                   config_dict=None, forest=100):
        return PerfRegressionFramework(train_plans, train_sqls, db, train_file_name, model, mode=mode,
                                       config_dict=config_dict)


if __name__ == '__main__':
    unittest.main()
