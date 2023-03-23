import json
import unittest

from RegressionFramework.Common.dotDrawer import PlanDotDrawer
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.RegressionFramework import HyperQoRegressionFramework, RegressionFramework
from RegressionFramework.config import data_base_path, model_base_path
from RegressionFramework.utils import json_str_to_json_obj, flat_depth2_list, cal_ratio
from model_test import LeroTest
from plan2latency import get_hyperqo_result, load_hyperqo_model
from plans2best_plan import get_hyperqo_best_plan, load_hyperqo_best_plan_model


class HyperqoTest(LeroTest):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.test_model = None
        self.algo="hyperqo"

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_job(self):
        # train_file_name = "qo_job1.log.training"

        # train_file_name = "qo_job2.log.training"
        #
        # train_file_name = "qo_job3.log.training"
        #
        train_file_name = "qo_job4.log.training"

        test_file_name = "qo_job.log.test"
        sql_file_name = train_file_name

        model_path = self.get_model_path(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "imdb", model_path)

    def test_tpch(self):
        train_file_name = "qo_tpch1.log.training"

        # train_file_name = "lero_tpch2.log.training"

        # train_file_name = "lero_tpch3.log.training"

        # train_file_name = "lero_tpch4.log.training"

        test_file_name = "qo_job.log.test"
        sql_file_name = train_file_name

        model_path = self.get_model_path(train_file_name)
        self.performance(train_file_name, test_file_name, sql_file_name, "tpch", model_path)

    def get_data_file_path(self, data_file_name):
        return "{}{}/{}".format(data_base_path, "hyperqo", data_file_name)

    def get_model_path(self, training_name):
        return "{}hyperqo/{}".format(model_base_path, training_name)

    def read_sqls(self, data_file_path):
        sqls = []
        with open(data_file_path, "r") as f:
            for line in f.readlines():
                sqls.append(line.split("#####")[0])
        return sqls

    def _add_plan_metric(self, plans_for_query, model, sqls):
        print("add metric")
        for i, plans in enumerate(plans_for_query):
            latencies = get_hyperqo_result(plans, sqls[i], model)

            plan_objects = [PlanFactory.get_plan_instance("pg", p) for p in plans]
            PlanDotDrawer.get_plan_dot_str(plan_objects[0])
            for i, plan in enumerate(plans):
                plan["metric"] = cal_ratio(latencies[i], plan["Execution Time"])
                plan["predict"] = latencies[i]

    def select_plan_by_model(self, model, plans_for_query, model_path, sqls):
        model = load_hyperqo_best_plan_model(model_path) if self.test_model is None else self.test_model
        print("select plan by model")
        times_for_query = []
        for i, plans in enumerate(plans_for_query):
            plans = [json.dumps(p) for p in plans]
            times_for_query.append(get_hyperqo_best_plan(plans, sqls[i], model))
        return times_for_query

    def select_plan_by_lero_model_regression(self, model, regression_framework: RegressionFramework,
                                             plans_for_query, latencies_for_queries, thres, plan_id_2_confidence, sqls):
        select_times_for_query = []

        for i, plans in enumerate(plans_for_query):
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = get_hyperqo_result(plans, sqls[i], model)

            predict_latencies = latencies_for_queries[i]

            plan_idx_and_predict = []
            for j in range(len(plans)):
                plan = plans[j]
                predict = regression_framework.evaluate(plan, predict=predict_latencies[j])
                print(predict)
                if predict != -1:
                    plan_idx_and_predict.append((j, predict))

            # find best
            plan_idx_and_predict = sorted(plan_idx_and_predict, key=lambda x: x[1])
            choose_idx = plan_idx_and_predict[0][0] if len(plan_idx_and_predict) > 0 else 0
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            select_times_for_query.append(latency)
        return select_times_for_query

    def load_model(self, model_name, db):
        return load_hyperqo_model(model_path)

    def _init_regression_framework(self, train_plans, train_sqls, db, train_file_name, model):
        return HyperQoRegressionFramework(train_plans, train_sqls, db, train_file_name, model)


if __name__ == '__main__':
    unittest.main()
