import unittest

from RegressionFramework.Plan.Plan import json_str_to_json_obj
from RegressionFramework.Plan.utils import flat_depth2_list
from RegressionFramework.RegressionFramework import RegressionFramework
from RegressionFramework.config import base_path, data_base_path
from model import LeroModelPairWise


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_stats(self):
        train_file_name = "lero_job1_.log.training"
        test_file_name = train_file_name
        # read training data
        train_plans = flat_depth2_list(self.read_data(train_file_name))

        # read test data
        test_plans_for_query = self.read_data(test_file_name)

        # load lero model
        model = self.load_model(self.get_model_name(train_file_name))
        predicts = self.predict(model, train_plans)

        # building regression framework
        regression_framework = RegressionFramework(train_plans, predicts)
        regression_framework.train()

        # choose plans

    def select_plan_by_lero_model(self, model: LeroModelPairWise, plans_for_query):
        times_for_query = []
        for plans in plans_for_query:
            predicts = self.predict(model, plans)
            idx = list.index(min(predicts))
            times_for_query.append(plans[idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model: LeroModelPairWise, regression_framework: RegressionFramework,
                                             plans_for_query):
        times_for_query = []
        for plans in plans_for_query:
            filter_plans = []
            for p in plans:
                if regression_framework.evaluate(p):
                    filter_plans.append(p)
            if len(filter_plans) != 0:
                predicts = self.predict(model, filter_plans)
                idx = list.index(min(predicts))
            else:
                idx = 0
            times_for_query.append(filter_plans[idx]["Execution Time"])
        return times_for_query

    # spark_plans = self.get_spark_times(plans_for_query)
    # best_plans = self.get_best_times(plans_for_query)

    def get_model_name(self, data_file_name):
        return base_path + "model/{}_model_test_model_on_0".format(data_file_name)

    def load_model(self, model_name):
        lero_model = LeroModelPairWise(None)
        lero_model.load(model_name)
        return lero_model

    def predict(self, model: LeroModelPairWise, plans):
        features = model.to_feature(plans)
        return model.predict(features)

    @classmethod
    def read_data(cls, data_file_path):
        file_path = data_base_path + data_file_path
        plans_for_query = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                plans = line.split("#####")
                plans = [json_str_to_json_obj(p) for p in plans]
                plans_for_query.append(plans)
        return plans_for_query

    @classmethod
    def to_plan_objects(cls, ):
        pass

    def get_pg_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(plans[0]["Execution Time"])
        return res

    def get_best_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(self.find_best_plan(plans)[1])
        return res


if __name__ == '__main__':
    unittest.main()
