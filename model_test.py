import math
import unittest

from pandas import DataFrame

from RegressionFramework.Plan.Plan import json_str_to_json_obj
from RegressionFramework.utils import flat_depth2_list, draw_by_agg, read_sqls
from RegressionFramework.RegressionFramework import RegressionFramework, LeroRegressionFramework
from RegressionFramework.config import base_path, data_base_path, model_base_path
from model import LeroModelPairWise


class LeroTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "lero"

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_job_all(self):
        train_file_names = ["lero_job1.log.training", "lero_job2.log.training", "lero_job3.log.training",
                            "lero_job4.log.training"]
        sql_file_names = ["job1.txt", "job2.txt", "job3.txt", "job4.txt"]
        test_file_name = "job_test"
        for i in range(len(train_file_names)):
            self.performance(train_file_names[i], test_file_name, sql_file_names[i], "imdb")

    def test_stats_all(self):
        train_file_names = ["lero_stats1.log.training", "lero_stats2.log.training", "lero_stats3.log.training",
                            "lero_stats4.log.training"]
        sql_file_names = ["stats1.txt", "stats2.txt", "stats3.txt", "stats4.txt"]
        test_file_name = "stats_test"
        for i in range(len(train_file_names)):
            self.performance(train_file_names[i], test_file_name, sql_file_names[i], "stats")
    
    def test_tpch_all(self):
        train_file_names = ["lero_tpch1.log.training", "lero_tpch2.log.training", "lero_tpch3.log.training",
                            "lero_tpch4.log.training"]
        sql_file_names = ["tpch1.txt", "tpch2.txt", "tpch3.txt", "tpch4.txt"]
        test_file_name = "tpch_test"
        for i in range(len(train_file_names)):
            self.performance(train_file_names[i], test_file_name, sql_file_names[i], "tpch")

    def test_job(self):
        # train_file_name = "lero_job1.log.training"
        # sql_file_name = "job1.txt"

        # train_file_name = "lero_job2.log.training"
        # train_file_name = "lero_job3.log.training"

        train_file_name = "lero_job4.log.training"
        sql_file_name = "job4.txt"

        test_file_name = "job_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "imdb")

    def test_stats(self):
        # train_file_name = "lero_stats1.log.training"
        # sql_file_name = "stats1.txt"

        train_file_name = "lero_stats2.log.training"
        sql_file_name = "stats2.txt"

        # train_file_name = "lero_stats3.log.training"
        # sql_file_name = "stats3.txt"

        # train_file_name = "lero_stats4.log.training"
        # sql_file_name = "stats4.txt"

        test_file_name = "stats_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "stats")

    def test_tpch(self):
        # train_file_name = "lero_tpch1.log.training"
        # sql_file_name = "tpch1.txt"

        # train_file_name = "lero_tpch2.log.training"
        # sql_file_name = "tpch2.txt"

        # train_file_name = "lero_tpch3.log.training"
        # sql_file_name = "tpch3.txt"

        train_file_name = "lero_tpch4.log.training"
        sql_file_name = "tpch4.txt"

        test_file_name = "tpch_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "tpch")

    def test_both(self):
        train_file_name = "lero_tpch1.log.training"
        sql_file_name = "tpch4.txt"

        test_file_name = "tpch_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "tpch")

        train_file_name = "lero_stats4.log.training"
        sql_file_name = "stats4.txt"

        test_file_name = "stats_test"
        self.performance(train_file_name, test_file_name, sql_file_name, "stats")

    def performance(self, train_file_name, test_file_name, train_sql_file_name, db, model_path=None):
        train_sqls = self.read_sqls(self.get_data_file_path(train_sql_file_name))

        # read training data
        train_plans_for_query = self.read_data(self.get_data_file_path(train_file_name))

        # read test data
        test_plans_for_query = self.read_data(self.get_data_file_path(test_file_name))
        test_sqls = self.read_sqls(self.get_data_file_path(test_file_name))

        self._add_id_to_plan(test_plans_for_query)

        # load lero model
        print("predict")
        model = self.load_model(self.get_model_name(train_file_name) if model_path is None else model_path, db)
        self._add_plan_metric(train_plans_for_query, model, train_sqls)
        train_plans = flat_depth2_list(train_plans_for_query)

        # building regression framework
        print("build RegressionFramework")
        regression_framework = self._init_regression_framework(train_plans, train_sqls, db, train_file_name, model)
        regression_framework.build()

        # choose plans
        lero_results = self.select_plan_by_model(model, test_plans_for_query, model_path, test_sqls)

        predict_latencies_for_queries = [None] * len(test_plans_for_query)
        plan_id_2_confidence = {}
        candidate_thres = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        regression_results = {}
        for thres in candidate_thres:
            print("cur thres {}".format(thres))
            result = self.select_plan_by_lero_model_regression(model, regression_framework,
                                                               test_plans_for_query,
                                                               predict_latencies_for_queries, thres,
                                                               plan_id_2_confidence, test_sqls)
            regression_results[str(thres)] = result

        best_results = self.select_best_plan_times(test_plans_for_query)
        pg_results = self.select_pg_plan_times(test_plans_for_query)

        queries_name = ["Q{}".format(i) for i in range(len(lero_results))]
        # print .csv
        df = DataFrame(
            {"AlgoPlans": lero_results,
             # "RegressionPlans": regression_results,
             "PgPlans": pg_results,
             "BestPlan": best_results,
             }
        )
        for thres in candidate_thres:
            df["regress-thres{}".format(thres)] = regression_results[str(thres)]

        df["qn"] = queries_name

        # lero_job4.log.training -> 4
        train_ratio = train_file_name.split(".")[0][-1]
        name = "{}_{}{}".format(self.algo, test_file_name, train_ratio)
        df.to_csv(
            "RegressionFramework/fig/{}_performance.csv".format(name))

        # draw
        y_names = list(df.columns)
        y_names.remove("qn")
        # draw2(df, x_name="qn", y_names=y_names, file_name="overall")
        draw_by_agg(df, y_names=y_names, agg="mean", file_name="{}".format(name))

    def _init_regression_framework(self, train_plans, train_sqls, db, train_file_name, model):
        return LeroRegressionFramework(train_plans, train_sqls, db, train_file_name, model)

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

    def select_plan_by_model(self, model: LeroModelPairWise, plans_for_query, model_path, sqls):
        times_for_query = []
        for plans in plans_for_query:
            predicts = list(self.predict(model, plans))
            idx = predicts.index(min(predicts))
            times_for_query.append(plans[idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model: LeroModelPairWise, regression_framework: RegressionFramework,
                                             plans_for_query, latencies_for_queries, thres, plan_id_2_confidence, sqls):

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
                        # print("confidence is {}".format(confidence))
                        if confidence >= thres:
                            win_count += 1
                id_to_win_count[j] = win_count

            db_win_count = id_to_win_count[0]
            id_win_count = sorted(id_to_win_count.items(), key=lambda x: x[1])
            # choose_idx = id_win_count[-1][0]
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

    def load_model(self, model_name, db):
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

    @classmethod
    def to_plan_objects(cls, ):
        pass

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
