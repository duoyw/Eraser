import re

from Common.PlanConfig import SparkNodeConfig
from auncel.utils import extract_table_name, json_str_to_json_obj


class SqlStat:
    @classmethod
    def stat_table_dist(cls, plans_queries):
        info_queries = []
        for plans_query in plans_queries:
            plan = json_str_to_json_obj(plans_query[0])
            tables = sorted(list(cls.recuse(plan["Plan"])))
            qid = plan["Qid"]
            info_queries.append((tables, qid))

        tables_to_qid = {}
        for info in info_queries:
            qid = info[1]
            key = "-".join(info[0])
            if key not in tables_to_qid:
                tables_to_qid[key] = []
            tables_to_qid[key].append(qid)

        # print
        for k, v in tables_to_qid.items():
            print("table is {}, count is {}, qid is {}".format(k, len(v), v))

    @classmethod
    def recuse(cls, node):
        tables = set()
        if "Plans" in node:
            children = node["Plans"]
            for child in children:
                tables.update(cls.recuse(child))
        if node["class"] in SparkNodeConfig.SCAN_TYPES:
            tables.add(extract_table_name(node))
        return tables
