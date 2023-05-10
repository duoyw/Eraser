import math
import unittest

from pandas import DataFrame

from RegressionFramework.Plan.Plan import json_str_to_json_obj
from RegressionFramework.EraserManager import EraserManager, LeroEraserManager
from RegressionFramework.config import data_base_path, model_base_path, betas
from RegressionFramework.Common.utils import flat_depth2_list, draw_by_agg, read_sqls
from model import LeroModelPairWise


class LeroTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "lero"

    def test_job_all(self):
        train_file_names = ["lero_job1.log.training", "lero_job2.log.training", "lero_job3.log.training",
                            "lero_job4.log.training"]
        sql_file_names = ["job1.txt", "job2.txt", "job3.txt", "job4.txt"]
        test_file_name = "job_test"
        for i in range(len(train_file_names)):
            self.performance(train_file_names[i], test_file_name, sql_file_names[i], "imdb")

    def test_job(self):
        # train_file_name = "lero_job1.log.training"
        # sql_file_name = "job1.txt"

        train_file_name = "lero_job2.log.training"
        sql_file_name = "job2.txt"

        # train_file_name = "lero_job3.log.training"
        # sql_file_name = "job3.txt"

        # train_file_name = "lero_job4.log.training"
        # sql_file_name = "job4.txt"

        test_file_name = "job_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "imdb")

    def test_dynamic_job(self):
        train_file_name = "lero_job4.log.training"
        sql_file_name = "job4.txt"
        self.dynamic_performance(train_file_name, sql_file_name, "imdb")

    def dynamic_performance(self, train_file_name, train_sql_file_name, db, save_suffix=""):
        train_sqls = self.read_sqls(self.get_data_file_path(train_sql_file_name))
        train_plans_for_query = self.read_data(self.get_data_file_path(train_file_name))
        self._add_id_to_plan(train_plans_for_query)

        # load lero model
        print("predict")
        gap = 100
        # gap = 50
        count = math.ceil(len(train_plans_for_query) / float(gap))

        regression_results = {}
        algo_results = []
        best_results = []
        pg_results = []
        candidate_thres = betas
        for i in range(0, count):
            print("cur iteration is {}".format(i))

            model_path = self.get_dynamic_model_name(train_file_name, i)
            model = self.load_model(model_path)

            min_pos = 0
            max_pos = max(i * gap, 1)
            tmp_train_plans_for_query = train_plans_for_query[min_pos:max_pos]
            train_plans = flat_depth2_list(tmp_train_plans_for_query)
            cur_train_sqls = train_sqls[min_pos: max_pos]
            self._add_plan_metric(train_plans_for_query[min_pos:max_pos], model, cur_train_sqls)
            regression_framework = self._init_eraser(train_plans, tmp_train_plans_for_query,
                                                     cur_train_sqls, db,
                                                     "dynamic" + train_file_name + "{}".format(i), model,
                                                     mode="dynamic")
            regression_framework.build()

            test_plans_for_query = train_plans_for_query[i * gap:(i + 1) * gap]
            test_sqls = train_sqls[i * gap:(i + 1) * gap]

            # algo result
            algo_results += self.select_plan_by_model(model, test_plans_for_query)
            best_results += self.select_best_plan_times(test_plans_for_query)
            pg_results += self.select_pg_plan_times(test_plans_for_query)

            # algo + regression result
            predict_latencies_for_queries = [None] * len(test_plans_for_query)
            for thres in candidate_thres:
                print("cur thres {}".format(thres))
                result = self.select_plan_by_lero_model_regression(model, regression_framework,
                                                                   test_plans_for_query,
                                                                   predict_latencies_for_queries, thres)
                if str(thres) not in regression_results:
                    regression_results[str(thres)] = []
                regression_results[str(thres)] += result
        if len(algo_results) != len(train_plans_for_query):
            raise RuntimeError
        self.draw(algo_results, regression_results, pg_results, best_results, candidate_thres, train_file_name, db,
                  save_prefix="dynamic_", save_suffix=save_suffix)

    def performance(self, train_file_name, test_file_name, train_sql_file_name, db, config_dict=None,
                    save_suffix="", save_prefix=""):

        # read training data
        train_sqls = self.read_sqls(self.get_data_file_path(train_sql_file_name))
        train_plans_for_query = self.read_data(self.get_data_file_path(train_file_name))

        # read test data
        test_sqls = self.read_sqls(self.get_data_file_path(test_file_name))
        test_plans_for_query = self.read_data(self.get_data_file_path(test_file_name))

        self._add_id_to_plan(test_plans_for_query)

        # load lero model
        print("predict")
        model_path = self.get_model_name(train_file_name)
        model = self.load_model(model_path)

        self._add_plan_metric(train_plans_for_query, model, train_sqls)

        # building regression framework
        print("build Eraser")
        train_plans = flat_depth2_list(train_plans_for_query)
        regression_framework = self._init_eraser(train_plans, train_plans_for_query, train_sqls, db,
                                                 train_file_name, model, config_dict=config_dict)
        regression_framework.build()

        # choose plans
        lero_results = self.select_plan_by_model(model, test_plans_for_query)

        predict_latencies_for_queries = [None] * len(test_plans_for_query)
        candidate_thres = betas
        regression_results = {}
        for thres in candidate_thres:
            print("cur thres {}".format(thres))
            result = self.select_plan_by_lero_model_regression(model, regression_framework,
                                                               test_plans_for_query,
                                                               predict_latencies_for_queries, thres)
            regression_results[str(thres)] = result

        best_results = self.select_best_plan_times(test_plans_for_query)
        pg_results = self.select_pg_plan_times(test_plans_for_query)
        self.draw(lero_results, regression_results, pg_results, best_results, candidate_thres, train_file_name, db,
                  save_suffix=save_suffix, save_prefix=save_prefix)

    def draw(self, algo_result, regression_results, pg_results, best_results, candidate_thres, train_file_name, db,
             save_prefix="", save_suffix=""):
        queries_name = ["Q{}".format(i) for i in range(len(algo_result))]
        # print .csv
        df = DataFrame(
            {"AlgoPlans": algo_result,
             "PgPlans": pg_results,
             "BestPlan": best_results,
             }
        )
        for thres in candidate_thres:
            df["regress-thres{}".format(thres)] = regression_results[str(thres)]

        df["qn"] = queries_name

        # lero_job4.log.training -> 4
        train_ratio = train_file_name.split(".")[0][-1]
        workload = db if db != "imdb" else "job"
        name = "{}_{}_test{}".format(self.algo, workload, train_ratio)
        df.to_csv(
            "RegressionFramework/fig/{}{}_performance{}.csv".format(save_prefix, name, save_suffix))

        # draw
        y_names = list(df.columns)
        y_names.remove("qn")
        draw_by_agg(df, y_names=y_names, agg="mean", file_name="{}{}{}".format(save_prefix, name, save_suffix))

    def _init_eraser(self, train_plans, plans_for_queries, train_sqls, db, train_file_name, model,
                     mode="static",
                     config_dict=None):
        return LeroEraserManager(train_plans, train_sqls, db, train_file_name, model, mode=mode,
                                 config_dict=config_dict, plans_for_queries=plans_for_queries)

    def _add_plan_metric(self, plans_for_query, model: LeroModelPairWise, sqls):
        print("add metric")
        for plans in plans_for_query:
            predict_latencies = self.predict(model, plans)
            total_count = 0
            correct_count = 0
            for i in range(len(predict_latencies)):
                p1 = plans[i]
                t1 = predict_latencies[i]
                for j in range(len(predict_latencies)):
                    if i == j:
                        continue
                    p2 = plans[j]
                    t2 = predict_latencies[j]
                    if t1 <= t2 and p1["Execution Time"] <= p2["Execution Time"]:
                        correct_count += 1
                    total_count += 1
                p1["metric"] = correct_count / total_count if total_count > 0 else 0.5
                p1["predict"] = t1

    def select_plan_by_model(self, model: LeroModelPairWise, plans_for_query):
        times_for_query = []
        for plans in plans_for_query:
            predicts = list(self.predict(model, plans))
            idx = predicts.index(min(predicts))
            times_for_query.append(plans[idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model: LeroModelPairWise, regression_framework: EraserManager,
                                             plans_for_query, latencies_for_queries, thres):
        select_times_for_query = []

        for i, plans in enumerate(plans_for_query):
            id_to_win_count = {}
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = model.predict(model.to_feature(plans_for_query[i]))
            predict_latencies = latencies_for_queries[i]
            for j in range(len(plans)):
                plan1 = plans[j]
                win_count = 0
                for k in range(len(plans)):
                    plan2 = plans[k]
                    if j == k:
                        continue
                    if predict_latencies[j] < predict_latencies[k]:
                        confidence = regression_framework.evaluate(plan1, plan2)
                        if confidence >= thres:
                            win_count += 1
                id_to_win_count[j] = win_count

            db_win_count = id_to_win_count[0]
            id_win_count = sorted(id_to_win_count.items(), key=lambda x: x[1])
            choose_idx = self._get_idx_min_predict_latency_with_max_count(id_win_count, predict_latencies)
            choose_idx = 0 if db_win_count >= id_to_win_count[choose_idx] else choose_idx
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            select_times_for_query.append(latency)

        return select_times_for_query

    def _get_idx_min_predict_latency_with_max_count(self, id_win_count, predict_latencies):
        """
        :param id_win_count: [(id,win_count),(),...], sorted count by ascending order
        :param predict_latencies: [latency1,...]
        :return:
        """
        count = id_win_count[-1][1]
        candidate_ids = []
        for items in id_win_count:
            if items[1] == count:
                candidate_ids.append(items[0])

        choose_idx = -1
        choose_predict_latency = math.inf
        for idx in candidate_ids:
            if predict_latencies[idx] < choose_predict_latency:
                choose_idx = idx
                choose_predict_latency = predict_latencies[idx]
        return choose_idx

    def get_model_name(self, data_file_name):
        return model_base_path + "test_model_on_0_{}".format(data_file_name)

    def get_dynamic_database_model_name_on_static_mode(self, data_file_name):
        data_file_name = data_file_name.replace("dynamic_database_", "")
        return model_base_path + "dynamic_database/test_model_on_0_{}".format(data_file_name)

    def get_dynamic_model_name(self, data_file_name: str, count):
        prefix = "dynamic_shuffle" if "shuffle" in data_file_name else "dynamic"
        data_file_name = data_file_name.replace("(shuffle)", "")
        data_file_name = data_file_name[0:-14] + str(count) + data_file_name[-13:]
        return model_base_path + "{}/dynamic_model_{}".format(prefix, data_file_name)

    def load_model(self, model_name):
        lero_model = LeroModelPairWise(None)
        lero_model.load(model_name)
        return lero_model

    def predict(self, model: LeroModelPairWise, plans):
        features = model.to_feature(plans)
        return model.predict(features)

    def read_data(self, data_file_path):
        plans_for_query = []
        with open(data_file_path, "r") as f:
            for line in f.readlines():
                plans = line.split("#####")[1:]
                plans = [json_str_to_json_obj(p) for p in plans]
                plans_for_query.append(plans)
        return plans_for_query

    def get_data_file_path(self, data_file_name):
        return data_base_path + data_file_name

    def read_sqls(self, data_file_path):
        return read_sqls(data_file_path)

    def _add_id_to_plan(self, plans_for_query):
        plan_id = 0
        for plans in plans_for_query:
            for plan in plans:
                plan["id"] = plan_id
                plan_id += 1

    def select_pg_plan_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(plans[0]["Execution Time"])
            else:
                res.append(plans[0]["Execution Time"])
        return res

    def select_best_plan_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(self.find_best_plan(plans)[1])
            else:
                res.append(plans[0]["Execution Time"])
        return res

    def find_best_plan(self, plans):
        best_plan_idx = -1
        best_latency = float('inf')
        for i, plan in enumerate(plans):
            plan = json_str_to_json_obj(plan)
            if best_latency > plan["Execution Time"]:
                best_plan_idx = i
                best_latency = plan["Execution Time"]
        return best_plan_idx, best_latency


if __name__ == '__main__':
    unittest.main()
