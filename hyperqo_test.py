import json
import sys
import unittest

from Hyperqo.ImportantConfig import Config

sys.path.append("test_script/")
sys.path.append("Hyperqo/")
sys.path.append("Perfguard/")

from RegressionFramework.Common.TimeStatistic import TimeStatistic
from RegressionFramework.Common.dotDrawer import PlanDotDrawer
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.RegressionFramework import HyperQoRegressionFramework, RegressionFramework
from RegressionFramework.config import data_base_path, model_base_path
from RegressionFramework.utils import json_str_to_json_obj, relative_error
from lero_test import LeroTest
from Hyperqo.plan2latency import get_hyperqo_result, load_hyperqo_model
from Hyperqo.plans2best_plan import get_hyperqo_best_plan, load_hyperqo_best_plan_model

input_ratio = None


class HyperqoTest(LeroTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.test_model = None
        self.algo = "hyperqo"
        global input_ratio
        input_ratio = sys.argv[2] if len(sys.argv) == 3 else None
        self.config = Config()

    def test_static_all_job(self):
        assert self.config.database == "imdb"
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_job()

    def test_static_all_stats(self):
        assert self.config.database == "stats"
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_stats()

    def test_static_all_tpch(self):
        assert self.config.database == "tpch"
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_tpch()

    def test_job(self, ratio=None):
        assert self.config.database == "imdb"
        ratio = 1 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        self.evaluate("job", "imdb", ratio)

    def test_stats(self, ratio=None):
        assert self.config.database == "stats"
        ratio = 1 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        self.evaluate("stats", "stats", ratio)

    def test_tpch(self, ratio=None):
        assert self.config.database == "tpch"
        ratio = 1 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        self.evaluate("tpch", "tpch", ratio)

    def test_dynamic_job(self):
        assert self.config.database == "imdb"
        train_file_name = "lero_job4.log.training"
        sql_file_name = train_file_name
        self.dynamic_performance(train_file_name, sql_file_name, "imdb")

    def test_dynamic_tpch(self):
        assert self.config.database == "tpch"
        train_file_name = "lero_tpch4.log.training"
        sql_file_name = train_file_name
        self.dynamic_performance(train_file_name, sql_file_name, "tpch")

    def select_plan_by_lero_model_regression(self, model, regression_framework: RegressionFramework, plans_for_query,
                                             latencies_for_queries, thres, plan_id_2_confidence, sqls, ood_thres=None):
        select_times_for_query = []

        for i, plans in enumerate(plans_for_query):
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = get_hyperqo_result(plans, sqls[i], model)

            predict_latencies = latencies_for_queries[i]
            if predict_latencies is not None:
                plan_idx_and_predict = []
                for j in range(len(plans)):
                    plan = plans[j]
                    p = PlanFactory.get_plan_instance("pg", plan)
                    TimeStatistic.start("{}_eraser_infer".format(self.algo))
                    predict = regression_framework.evaluate(p, predict=predict_latencies[j])
                    TimeStatistic.end("{}_eraser_infer".format(self.algo))
                    if predict != -1:
                        plan_idx_and_predict.append((j, predict))

                # find best
                plan_idx_and_predict = sorted(plan_idx_and_predict, key=lambda x: x[1])
                choose_idx = plan_idx_and_predict[0][0] if len(plan_idx_and_predict) > 0 else 0
            else:
                choose_idx = 0
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            select_times_for_query.append(latency)
        return select_times_for_query

    def evaluate(self, workload, db, ratio):
        train_file_name = "lero_{}{}.log.training".format(workload, ratio)
        sql_file_name = train_file_name
        test_file_name = "lero_{}.log.test".format(workload)
        self.performance(train_file_name, test_file_name, sql_file_name, db)

    def get_data_file_path(self, data_file_name):
        return "{}{}/{}".format(data_base_path, "hyperqo", data_file_name)

    def get_dynamic_model_name(self, data_file_name: str, count):
        data_file_name = data_file_name.replace("lero", "qo")
        return super().get_dynamic_model_name(data_file_name, count)

    def get_model_name(self, training_name: str, model_with_generate_sql=False):
        training_name = training_name.replace("lero", "qo")
        return model_base_path + "hyperqo/test_model_{}".format(training_name)

    def get_model_path(self, training_name):
        return "{}hyperqo/{}".format(model_base_path, training_name)

    def read_sqls(self, data_file_path):
        sqls = []
        with open(data_file_path, "r") as f:
            for line in f.readlines():
                sqls.append(line.split("#####")[0])
        return sqls

    def _add_plan_metric(self, plans_for_query, model, sqls):
        for i, plans in enumerate(plans_for_query):
            latencies = get_hyperqo_result(plans, sqls[i], model)
            if latencies is None:
                latencies = [-100] * len(plans)

            plan_objects = [PlanFactory.get_plan_instance("pg", p) for p in plans]
            PlanDotDrawer.get_plan_dot_str(plan_objects[0])
            for i, plan in enumerate(plans):
                plan["metric"] = relative_error(latencies[i], plan["Execution Time"])
                plan["predict"] = latencies[i]

    def select_plan_by_model(self, model, plans_for_query, model_path, sqls, db):
        model = load_hyperqo_best_plan_model(model_path, db)
        print("select plan by model")
        times_for_query = []
        for i, plans in enumerate(plans_for_query):
            plans = [json.dumps(p) for p in plans]
            times_for_query.append(get_hyperqo_best_plan(plans, sqls[i], model))
        return times_for_query

    def load_model(self, model_name, db):
        return load_hyperqo_model(model_name, db)

    def _init_regression_framework(self, train_plans, plans_for_queries, train_sqls, db, train_file_name, model,
                                   mode="static",
                                   config_dict=None, forest=100):
        return HyperQoRegressionFramework(train_plans, train_sqls, db, train_file_name, model, mode=mode)


if __name__ == '__main__':
    unittest.main()
