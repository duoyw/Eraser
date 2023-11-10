import json
import os
import unittest

from auncel.Common.PlanCheck import PlanCheck
from auncel.Common.SqlStat import SqlStat
from auncel.test_script.config import DATA_BASE_PATH, SEP
from auncel.utils import read_plans, json_str_to_json_obj


class MyTestCase(unittest.TestCase):
    def test_plan_check(self):
        train_dataset_name = "stats10Q1000_train0911_wo508"
        # test_dataset_name = "stats10Q1000_train0911_wo508"
        # test_dataset_name = "stats10Q146_test0910_wo136"
        test_dataset_name = "stats10Q146_test0910_wo136"

        plan_check = PlanCheck()
        data_name = "stats10Q146_test0910_wo136"
        plans_for_query = read_plans(self.get_data_path(data_name))
        plans = []
        for p in plans_for_query:
            plans += p
        plan_check.check_same_plan(plans)

    def get_data_path(self, file_name):
        return os.path.join(DATA_BASE_PATH, file_name)

    def test_add_qid_to_plan(self):
        def add_qid(plans_str, qid):
            plans_str = plans_str.split(SEP)
            res = []
            for plan in plans_str:
                plan = json_str_to_json_obj(plan)
                plan["Qid"] = qid
                res.append(json.dumps([plan]))
            return SEP.join(res)

        data_names = ["stats10Q50"]
        # data_names=["tpcdsQuery10_20"]
        for data_name in data_names:
            print(data_name)
            with open(self.get_data_path(data_name+"_copy"), "r") as f:
                with open(self.get_data_path(data_name), "w") as wf:
                    line = f.readline()
                    while line is not None and line != "":
                        infos = line.split(SEP, 1)
                        query = infos[0]
                        plan_str = infos[1]
                        plan_str = add_qid(plan_str, query.split("query")[1])
                        wf.write(query + SEP + plan_str + "\n")
                        line = f.readline()


if __name__ == '__main__':
    unittest.main()
