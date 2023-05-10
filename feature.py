import json
from abc import ABCMeta, abstractmethod

import numpy as np

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


def json_str_to_json_obj(json_data):
    if isinstance(json_data, str):
        json_obj = json.loads(json_data)
        if type(json_obj) == list:
            assert len(json_obj) == 1
            json_obj = json_obj[0]
            assert type(json_obj) == dict
        return json_obj
    return json_data

"""
X1和对应X2都来自于同一个查询的不同计划
X1 = [p0,p0,p1,...],X2 = [p1,p2,p2,...]
trees = X1+X2 = [p0,p0,p1,...,p1,p2,p2,...]
pi = [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]
"""
class FeatureGenerator():

    def __init__(self) -> None:
        self.normalizer = None
        self.feature_parser = None

    def fit(self, trees):
        # 将所有计划的对应变量都存入以下列表或集合
        exec_times = []

        startup_costs = []
        total_costs = []
        rows = []

        input_relations = set()
        rel_type = set()

        def recurse(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            rel_type.add(n["Node Type"])
            if "Relation Name" in n:
                # base table
                input_relations.add(n["Relation Name"])

            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)
        # 遍历每一个计划
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            recurse(json_obj["Plan"])

        startup_costs = np.array(startup_costs)
        total_costs = np.array(total_costs)
        rows = np.array(rows)
        
        # 对数化
        startup_costs = np.log(startup_costs + 1)
        total_costs = np.log(total_costs + 1)
        rows = np.log(rows + 1)

        # 求最大最小值
        startup_costs_min = np.min(startup_costs)
        startup_costs_max = np.max(startup_costs)
        total_costs_min = np.min(total_costs)
        total_costs_max = np.max(total_costs)
        rows_min = np.min(rows)
        rows_max = np.max(rows)

        print("RelType : ", rel_type)

        # min-max标准化
        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Execution Time": exec_times_max, "Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        else:
            self.normalizer = Normalizer(
                {"Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        self.feature_parser = AnalyzeJsonParser(self.normalizer, list(input_relations))
    # tree = [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]
    """
     [{"Plan": {"Node Type": "Aggregate", "Strategy": "Plain", "Partial Mode": "Simple", "Parallel Aware": false, "Startup Cost": 29698.47, "Total Cost": 29698.48, 
     "Plan Rows": 1, "Plan Width": 8, "Actual Startup Time": 474.185, "Actual Total Time": 474.189, "Actual Rows": 1, "Actual Loops": 1, "Output": ["count(*)"], 
     "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, "Join Type": "Inner", "Startup Cost": 13035.52, 
     "Total Cost": 28477.89, "Plan Rows": 488235, "Plan Width": 0, "Actual Startup Time": 205.014, "Actual Total Time": 420.058, "Actual Rows": 896180, "Actual Loops": 1, 
     "Inner Unique": false,  "Hash Cond": "(v.postid = p.id)", "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "votes", "Schema": "public", "Alias": "v", "Startup Cost": 0.0, "Total Cost": 5410.64, "Plan Rows": 328064, 
     "Plan Width": 4, "Actual Startup Time": 0.009, "Actual Total Time": 37.697, "Actual Rows": 328064, "Actual Loops": 1, 
     "Output": ["v.postid"]}, {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, 
     "Startup Cost": 10789.5, "Total Cost": 10789.5, "Plan Rows": 136881, "Plan Width": 8, "Actual Startup Time": 204.569, 
     "Actual Total Time": 204.571, "Actual Rows": 209583, "Actual Loops": 1, "Output": ["ph.postid", "p.id"], 
     "Hash Buckets": 131072, "Original Hash Buckets": 131072, "Hash Batches": 4, "Original Hash Batches": 2, 
     "Peak Memory Usage": 3130, "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Join Type": "Inner", "Startup Cost": 4381.46, "Total Cost": 10789.5, "Plan Rows": 136881, "Plan Width": 8, 
     "Actual Startup Time": 30.065, "Actual Total Time": 162.566, "Actual Rows": 209583, "Actual Loops": 1, 
     "Output": ["ph.postid", "p.id"], "Inner Unique": true, "Hash Cond": "(p.owneruserid = u.id)", 
     "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Join Type": "Inner", "Startup Cost": 3177.14, "Total Cost": 9219.91, "Plan Rows": 139140, 
     "Plan Width": 12, "Actual Startup Time": 20.782, "Actual Total Time": 111.892, "Actual Rows": 213642, 
     "Actual Loops": 1, "Output": ["ph.postid", "p.id", "p.owneruserid"], "Inner Unique": true, "Hash Cond": "(ph.postid = p.id)", 
     "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, "Relation Name": "posthistory", 
     "Schema": "public", "Alias": "ph", "Startup Cost": 0.0, "Total Cost": 5246.87, "Plan Rows": 303187, "Plan Width": 4, 
     "Actual Startup Time": 0.003, "Actual Total Time": 23.737, "Actual Rows": 303187, "Actual Loops": 1, 
     "Output": ["ph.id", "ph.posthistorytypeid", "ph.postid", "ph.creationdate", "ph.userid"]}, 
     {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, "Startup Cost": 2649.52, 
     "Total Cost": 2649.52, "Plan Rows": 42210, "Plan Width": 8, "Actual Startup Time": 20.527, "Actual Total Time": 20.528, "Actual Rows": 42291, 
     "Actual Loops": 1, "Output": ["p.id", "p.owneruserid"], "Hash Buckets": 65536, "Original Hash Buckets": 65536, "Hash Batches": 1, "Original Hash Batches": 1, 
     "Peak Memory Usage": 2162, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "posts", "Schema": "public", "Alias": "p", "Startup Cost": 0.0, "Total Cost": 2649.52, "Plan Rows": 42210, "Plan Width": 8, 
     "Actual Startup Time": 0.011, "Actual Total Time": 14.106, "Actual Rows": 42291, "Actual Loops": 1, "Output": ["p.id", "p.owneruserid"], 
     "Filter": "((p.score >= '-1'::integer) AND (p.commentcount >= 0) AND (p.commentcount <= 11) AND (p.posttypeid = 1))", "Rows Removed by Filter": 49685}]}]}, 
     {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, "Startup Cost": 700.25, "Total Cost": 700.25, "Plan Rows": 40325, "Plan Width": 4, 
     "Actual Startup Time": 9.181, "Actual Total Time": 9.181, "Actual Rows": 40325, "Actual Loops": 1, "Output": ["u.id"], "Hash Buckets": 65536, "Original Hash Buckets": 65536, 
     "Hash Batches": 1, "Original Hash Batches": 1, "Peak Memory Usage": 1930, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "users", "Schema": "public", "Alias": "u", "Startup Cost": 0.0, "Total Cost": 700.25, "Plan Rows": 40325, "Plan Width": 4, "Actual Startup Time": 0.01, 
     "Actual Total Time": 3.841, "Actual Rows": 40325, "Actual Loops": 1, "Output": ["u.id"]}]}]}]}]}]}, 
     
     "Planning Time": 83.305, "Triggers": [], "Execution Time": 474.25}]""" 
     
    def transform(self, trees):
        local_features = []
        y = []
        # 遍历每一个计划
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if type(json_obj["Plan"]) != dict:
                json_obj["Plan"] = json.loads(json_obj["Plan"])
            local_feature = self.feature_parser.extract_feature(
                json_obj["Plan"])
            local_features.append(local_feature)

            if "Execution Time" in json_obj:
                label = float(json_obj["Execution Time"])
                if self.normalizer.contains("Execution Time"):
                    label = self.normalizer.norm(label, "Execution Time")
                y.append(label)
            else:
                y.append(None)
        return local_features, y


class SampleEntity():
    def __init__(self, node_type: np.ndarray, startup_cost: float, total_cost: float,
                 rows: float, width: int,
                 left, right,
                 startup_time: float, total_time: float,
                 input_tables: list, encoded_input_tables: list) -> None:
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.rows = rows
        self.width = width
        self.left = left
        self.right = right
        self.startup_time = startup_time
        self.total_time = total_time
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables

    def __str__(self):
        return "{%s, %s, %s, %s, %s, [%s], [%s], %s, %s, [%s], [%s]}" % (self.node_type,
                                                                        self.startup_cost, self.total_cost, self.rows,
                                                                        self.width, self.left, self.right,
                                                                        self.startup_time, self.total_time,
                                                                        self.input_tables, self.encoded_input_tables)

    def get_feature(self):
        # return np.hstack((self.node_type, np.array([self.width, self.rows])))
        return np.hstack((self.node_type, np.array(self.encoded_input_tables), np.array([self.width, self.rows])))

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def subtrees(self):
        trees = []
        trees.append(self)
        if self.left is not None:
            trees += self.left.subtrees()
        if self.right is not None:
            trees += self.right.subtrees()
        return trees


class Normalizer():
    def __init__(self, mins: dict, maxs: dict) -> None:
        self._mins = mins
        self._maxs = maxs

    def norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to normalize " + name)

        return (np.log(x + 1) - self._mins[name]) / (self._maxs[name] - self._mins[name])

    def inverse_norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to inversely normalize " + name)

        return np.exp((x * (self._maxs[name] - self._mins[name])) + self._mins[name]) - 1

    def contains(self, name):
        return name in self._mins and name in self._maxs


class FeatureParser(metaclass=ABCMeta):

    @abstractmethod
    def extract_feature(self, json_data) -> SampleEntity:
        pass

    """
     [{"Plan": {"Node Type": "Aggregate", "Strategy": "Plain", "Partial Mode": "Simple", "Parallel Aware": false, "Startup Cost": 29698.47, "Total Cost": 29698.48, 
     "Plan Rows": 1, "Plan Width": 8, "Actual Startup Time": 474.185, "Actual Total Time": 474.189, "Actual Rows": 1, "Actual Loops": 1, "Output": ["count(*)"], 
     "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, "Join Type": "Inner", "Startup Cost": 13035.52, 
     "Total Cost": 28477.89, "Plan Rows": 488235, "Plan Width": 0, "Actual Startup Time": 205.014, "Actual Total Time": 420.058, "Actual Rows": 896180, "Actual Loops": 1, 
     "Inner Unique": false,  "Hash Cond": "(v.postid = p.id)", 
     
     
     "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "votes", "Schema": "public", "Alias": "v", "Startup Cost": 0.0, "Total Cost": 5410.64, "Plan Rows": 328064, 
     "Plan Width": 4, "Actual Startup Time": 0.009, "Actual Total Time": 37.697, "Actual Rows": 328064, "Actual Loops": 1, 
     "Output": ["v.postid"]}, 
     
     {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, 
     "Startup Cost": 10789.5, "Total Cost": 10789.5, "Plan Rows": 136881, "Plan Width": 8, "Actual Startup Time": 204.569, 
     "Actual Total Time": 204.571, "Actual Rows": 209583, "Actual Loops": 1, "Output": ["ph.postid", "p.id"], 
     "Hash Buckets": 131072, "Original Hash Buckets": 131072, "Hash Batches": 4, "Original Hash Batches": 2, 
     "Peak Memory Usage": 3130, "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Join Type": "Inner", "Startup Cost": 4381.46, "Total Cost": 10789.5, "Plan Rows": 136881, "Plan Width": 8, 
     "Actual Startup Time": 30.065, "Actual Total Time": 162.566, "Actual Rows": 209583, "Actual Loops": 1, 
     "Output": ["ph.postid", "p.id"], "Inner Unique": true, "Hash Cond": "(p.owneruserid = u.id)", 
     "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Join Type": "Inner", "Startup Cost": 3177.14, "Total Cost": 9219.91, "Plan Rows": 139140, 
     "Plan Width": 12, "Actual Startup Time": 20.782, "Actual Total Time": 111.892, "Actual Rows": 213642, 
     "Actual Loops": 1, "Output": ["ph.postid", "p.id", "p.owneruserid"], "Inner Unique": true, "Hash Cond": "(ph.postid = p.id)", 
     "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, "Relation Name": "posthistory", 
     "Schema": "public", "Alias": "ph", "Startup Cost": 0.0, "Total Cost": 5246.87, "Plan Rows": 303187, "Plan Width": 4, 
     "Actual Startup Time": 0.003, "Actual Total Time": 23.737, "Actual Rows": 303187, "Actual Loops": 1, 
     "Output": ["ph.id", "ph.posthistorytypeid", "ph.postid", "ph.creationdate", "ph.userid"]}, 
     {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, "Startup Cost": 2649.52, 
     "Total Cost": 2649.52, "Plan Rows": 42210, "Plan Width": 8, "Actual Startup Time": 20.527, "Actual Total Time": 20.528, "Actual Rows": 42291, 
     "Actual Loops": 1, "Output": ["p.id", "p.owneruserid"], "Hash Buckets": 65536, "Original Hash Buckets": 65536, "Hash Batches": 1, "Original Hash Batches": 1, 
     "Peak Memory Usage": 2162, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "posts", "Schema": "public", "Alias": "p", "Startup Cost": 0.0, "Total Cost": 2649.52, "Plan Rows": 42210, "Plan Width": 8, 
     "Actual Startup Time": 0.011, "Actual Total Time": 14.106, "Actual Rows": 42291, "Actual Loops": 1, "Output": ["p.id", "p.owneruserid"], 
     "Filter": "((p.score >= '-1'::integer) AND (p.commentcount >= 0) AND (p.commentcount <= 11) AND (p.posttypeid = 1))", "Rows Removed by Filter": 49685}]}]}, 
     {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": false, "Startup Cost": 700.25, "Total Cost": 700.25, "Plan Rows": 40325, "Plan Width": 4, 
     "Actual Startup Time": 9.181, "Actual Total Time": 9.181, "Actual Rows": 40325, "Actual Loops": 1, "Output": ["u.id"], "Hash Buckets": 65536, "Original Hash Buckets": 65536, 
     "Hash Batches": 1, "Original Hash Batches": 1, "Peak Memory Usage": 1930, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
     "Relation Name": "users", "Schema": "public", "Alias": "u", "Startup Cost": 0.0, "Total Cost": 700.25, "Plan Rows": 40325, "Plan Width": 4, "Actual Startup Time": 0.01, 
     "Actual Total Time": 3.841, "Actual Rows": 40325, "Actual Loops": 1, "Output": ["u.id"]}]}]}]}]}]}, 
     
     "Planning Time": 83.305, "Triggers": [], "Execution Time": 474.25}]""" 
# the json file is created by "EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING, SUMMARY, FORMAT JSON) ..."


     
class AnalyzeJsonParser(FeatureParser):

    def __init__(self, normalizer: Normalizer, input_relations: list) -> None:
        self.normalizer = normalizer
        self.input_relations = input_relations

    def extract_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]))

        node_type = op_to_one_hot(json_rel['Node Type'])
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        return SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations))

    def encode_relation_names(self, l):
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in l:
            if name not in self.input_relations:
                # -1 means UNKNOWN
                encode_arr[-1] += 1
            else:
                encode_arr[list(self.input_relations).index(name)] += 1
        return encode_arr


def op_to_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    if op_name not in OP_TYPES:
        arr[OP_TYPES.index(UNKNOWN_OP_TYPE)] = 1
    else:
        arr[OP_TYPES.index(op_name)] = 1
    return arr
