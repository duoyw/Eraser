import json

from Common.DotDrawer import SparkPlanDotDrawer
from model_config import db_type
from Common.PlanFactory import PlanFactory

def json_str_to_json_obj(json_data):
    if isinstance(json_data, str):
        origin = json_data
        json_data = json_data.strip().strip("\\n")
        json_obj = json.loads(json_data)
        if type(json_obj) == list:
            assert len(json_obj) == 1
            json_obj = json_obj[0]
            assert type(json_obj) == dict
        return json_obj
    return json_data


class PlanFilter:
    def __init__(self, plans):
        if isinstance(plans[0], str):
            self.plans = [json_str_to_json_obj(p) for p in plans]
        self.plans = [PlanFactory.get_plan_instance(db_type, self.plans[i], i) for i in range(len(plans))]

        self.identifier2plans = {}

    def compress(self):
        for plan in self.plans:
            identifier = self._get_plan_identifier(plan)
            if identifier not in self.identifier2plans:
                self.identifier2plans[identifier] = []
            self.identifier2plans[identifier].append(plan)

        filter_plans = [plans[0] for plans in self.identifier2plans.values()]
        print("compress plan ratio is {}, origin is {}, cur is {}".format(1-float(len(filter_plans)) / len(self.plans),
              len(self.plans), len(filter_plans)))
        return filter_plans

    def _get_plan_identifier(self, plan):
        keys = []
        self._recurse(plan.root, keys)
        return ".".join(keys)

    def _recurse(self, node, keys: list):
        keys.append(node.get_identifier())
        for child in node.children:
            self._recurse(child, keys)
