from auncel.QueryFormer.model.database_util import formatFilter
from auncel.model_config import FILTER_TYPES
from auncel.sparkFeature import SCAN_TYPES, JOIN_TYPES
from auncel.utils import json_str_to_json_obj, extract_table_name, extract_join_key


class PlanCheck:

    def check_same_plan(self, plan_strs):
        plan_to_infos = {}
        for plan in plan_strs:
            plan = json_str_to_json_obj(plan)
            feature = self.build_plan_features(plan)
            feature = " ".join(feature)
            if feature not in plan_to_infos:
                plan_to_infos[feature] = []

            qid = plan["Qid"]
            time = plan["Execution Time"]
            plan_to_infos[feature].append((qid, time, plan))

        # show error plan
        for feature, info in plan_to_infos.items():
            if len(info) > 1:
                print("#########same count is {}#########".format(len(info)))
                for qid, time, plan in info:
                    print("qid is {}, time is{}".format(qid,time))
        print()

    def build_plan_features(self, plan):
        def recurse(node):
            feature = [self.extract_feature(node)]
            if "Plans" in node:
                children = node["Plans"]
                for child in children:
                    feature+=recurse(child)
            return feature

        return recurse(plan["Plan"])

    def extract_feature(self, node):
        op = node["class"]
        row_count = node["rowCount"]
        if op in SCAN_TYPES:
            table = extract_table_name(node)
            return "scan:{}_{}".format(table, row_count)
        elif op in FILTER_TYPES:
            return "filter_{}_{}".format(formatFilter(node, True), row_count)
        elif op in JOIN_TYPES:
            left, right = extract_join_key(node)
            return "{}_{}_{}_{}".format(op, left, right, row_count)
        else:
            return "{}_{}".format(op, row_count)
