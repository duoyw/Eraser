import json
import numpy as np

from Common.PlanConfig import SparkNodeConfig
from Common.PlanFactory import PlanFactory
from UncertantyModel.SparkPlan import SparkFilterPlanNode, SparkJoinPlanNode
from auncel.Common.DotDrawer import draw_dot_spark_plan
from auncel.QueryFormer.model.database_util import formatFilter
from auncel.model_config import diff_join_type, FILTER_TYPES, max_predicate_num, ALIAS_TO_TABLE, db_type
from utils import extract_table_name, extract_join_key, flatten_list, extract_column_with_prefix, \
    extract_filter_operator, extract_filter_literal, extract_filter_predicate, combine_table_col, json_str_to_json_obj, \
    cal_accuracy, is_number
from feature import AnalyzeJsonParser, Normalizer, FeatureGenerator, SampleEntity

SCAN_TYPES = SparkNodeConfig.SCAN_TYPES
UNKNOWN_OP_TYPE = "Unknown"
JOIN_TYPES = SparkNodeConfig.JOIN_TYPES
AGGREGATE_TYPES = ["org.apache.spark.sql.execution.aggregate.HashAggregateExec"]
OTHER_TYPES = ["org.apache.spark.sql.execution.UnionExec",
               "org.apache.spark.sql.execution.ExpandExec",
               "org.apache.spark.sql.execution.window.WindowExec",
               "org.apache.spark.sql.execution.ProjectExec", "org.apache.spark.sql.execution.FilterExec",
               "org.apache.spark.sql.execution.SortExec", "org.apache.spark.sql.execution.TakeOrderedAndProjectExec"]

OP_TYPES = [UNKNOWN_OP_TYPE] + SCAN_TYPES + JOIN_TYPES + AGGREGATE_TYPES + OTHER_TYPES

SCAN_DICT = {
    "org.apache.spark.sql.execution.FileSourceScanExec": "org.apache.spark.sql.execution.datasources.LogicalRelation"}
JOIN_DICT = {
    "org.apache.spark.sql.execution.joins.BroadcastHashJoinExec": "org.apache.spark.sql.catalyst.plans.logical.Join",
    "joins.ShuffledHashJoinExec": "org.apache.spark.sql.catalyst.plans.logical.Join",
    "joins.SortMergeJoinExec": "org.apache.spark.sql.catalyst.plans.logical.Join",
    "joins.CartesianProductExec": "org.apache.spark.sql.catalyst.plans.logical.Join"}
AGGREGATE_DICT = {
    "org.apache.spark.sql.execution.aggregate.HashAggregateExec": "org.apache.spark.sql.catalyst.plans.logical.Aggregate"}
OTHER_DICT = {"org.apache.spark.sql.execution.ProjectExec": "org.apache.spark.sql.catalyst.plans.logical.Project",
              "org.apache.spark.sql.execution.FilterExec": "org.apache.spark.sql.catalyst.plans.logical.Filter"}
OP_DICT = {**SCAN_DICT, **JOIN_DICT, **AGGREGATE_DICT, **OTHER_DICT}


class SparkFeatureGenerator(FeatureGenerator):

    def __init__(self, dataset_name) -> None:
        super().__init__()
        self.join_key_tuples = None
        self.filter_predicate = None
        self.dataset_name = dataset_name

    def fit(self, trees):
        trees = list(set(trees))
        exec_times = []
        rows = []
        widths = []
        unique_input_relations = set()
        join_key_tuples = set()
        rel_type = set()
        filter_predicate = {
            "col": set(),
            "op": set(),
            "col_values": {}
        }
        swing = 1
        swing_level = 1

        def recurse(node):
            operator_name = node["class"]
            if operator_name not in OP_TYPES:
                raise RuntimeError("finding new Type, please deal with it, the type is {}".format(operator_name))
            input_relations = []
            if node["class"] in SCAN_TYPES:
                # base table
                input_relations.append(extract_table_name(node))

            if node["class"] in JOIN_TYPES:
                spark_join_node: SparkJoinPlanNode = PlanFactory.get_plan_node_instance(db_type, node)
                # base table
                join_key_tuples.add(tuple(spark_join_node.join_key))

            if node["class"] in FILTER_TYPES:
                filter_spark_node: SparkFilterPlanNode = PlanFactory.get_plan_node_instance(db_type, node)
                predicate = filter_spark_node.predicates
                tables = filter_spark_node.tables
                for i in range(len(predicate)):
                    infos = predicate[i]
                    if is_number(infos[2]):
                        col = infos[0]
                        filter_predicate["col"].add(col)
                        filter_predicate["op"].add(infos[1])
                        if col not in filter_predicate["col_values"]:
                            filter_predicate["col_values"][col] = set()
                        filter_predicate["col_values"][col].add(float(infos[2]))

            if "Plans" in node:
                for child in node["Plans"]:
                    input_relations += recurse(child)

            if "rowCount" in node:
                row = node["rowCount"]
            else:
                row = 0.0

            width = node["sizeInBytes"]
            rel_type.add(operator_name)
            rows.append(row)
            widths.append(float(width))

            return input_relations

        for i, tree in enumerate(trees):
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            swing = json_obj["Swing"]
            swing_level = json_obj["Level"]
            unique_input_relations.update(recurse(json_obj["Plan"]))

            if i % (len(trees) / 10) == 0:
                print("In SparkFeatureGenerator fit, total plan is {}, cur is {}".format(len(trees), i))

        filter_predicate["col"] = list(filter_predicate["col"])
        filter_predicate["op"] = list(filter_predicate["op"])
        rows = np.array(rows)
        rows = np.log(rows + 1)
        widths = np.array(widths)
        widths = np.log(widths + 1)

        rows_min = np.min(rows)
        rows_max = np.max(rows)
        widths_min = np.min(widths)
        widths_max = np.max(widths)

        print("RelType : ", rel_type)

        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            # exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "rowCount": rows_min, "sizeInBytes": widths_min},
                {"Execution Time": exec_times_max, "rowCount": rows_max, "sizeInBytes": widths_max})
        else:
            self.normalizer = Normalizer(
                {"rowCount": rows_min, "sizeInBytes": widths_min},
                {"rowCount": rows_max, "sizeInBytes": widths_max})

        self.add_filter_value_to_norm(filter_predicate, self.normalizer)

        self.feature_parser = SparkAnalyzeJsonParser(self.normalizer, list(unique_input_relations),
                                                     flatten_list(list(join_key_tuples)), filter_predicate["col"],
                                                     filter_predicate["op"], self.dataset_name)
        self.join_key_tuples = join_key_tuples

        self.filter_predicate = filter_predicate

    def add_filter_value_to_norm(self, filter_predicate, normalizer: Normalizer):
        cols = filter_predicate["col"]
        cols_values = filter_predicate["col_values"]
        for col in cols:
            values = cols_values[col]
            min_val = np.min(np.array(list(values)))
            max_val = np.max(np.array(list(values)))
            normalizer.add(col, min_val, max_val)

    def get_tables(self):
        return self.feature_parser.input_relations

    def get_col_min_max_vals(self):
        col_min_max_vals = {}
        cols = self.filter_predicate["col"]
        cols_values = self.filter_predicate["col_values"]
        for col in cols:
            values = list(cols_values[col])
            col_min_max_vals[col] = (np.min(np.array(values)), np.max(np.array(values)))
        return col_min_max_vals

    def get_join_keys(self):
        return self.join_key_tuples

    def transform(self, trees):
        local_features = []
        y = []
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
                    label = self.normalizer.norm_no_log(label, "Execution Time")
                y.append(label)
            else:
                y.append(None)
        return local_features, y

    def transform_confidence_y(self, trees):
        y = []
        for tree in trees:
            plan_json = json_str_to_json_obj(tree)
            key = "predict"
            if key not in plan_json:
                raise RuntimeError()
            predict = float(plan_json[key])

            actual_time = plan_json["Execution Time"]
            accuracy = cal_accuracy(predict, actual_time)
            y.append(accuracy)
        return y


# the json file is created by "EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING, SUMMARY, FORMAT JSON) ..."
class SparkAnalyzeJsonParser(AnalyzeJsonParser):

    def __init__(self, normalizer: Normalizer, input_relations: list, join_keys: list, filter_cols: list,
                 filter_ops: list, dataset_name) -> None:
        super().__init__(normalizer, input_relations)
        self.join_keys = join_keys
        self.filter_cols = filter_cols
        self.filter_ops = filter_ops
        self.dataset_name = dataset_name

    def extract_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []
        left_input_row = 0.0
        right_input_row = 0.0

        rows = self.normalizer.norm(0.0 if "rowCount" not in json_rel else float(json_rel['rowCount']), 'rowCount')
        width = self.normalizer.norm(float(json_rel['sizeInBytes']), 'sizeInBytes')
        # width = int(json_rel['sizeInBytes'])

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_feature(children[0])
            left_input_row = left.rows
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
                right_input_row = right.rows
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]),
                                     self.encode_join_keys(None, None), 0, 0, UNKNOWN_OP_TYPE,
                                     self.encode_filter_col([]), self.encode_filter_op([]),
                                     self.encode_filter_values([]))

        if json_rel["class"] in JOIN_TYPES:
            spark_join_node: SparkJoinPlanNode = PlanFactory.get_plan_node_instance(db_type, json_rel)
            # base table
            left_join_key = spark_join_node.join_key[0]
            right_join_key = spark_join_node.join_key[1]
        else:
            # left_input_row=0.0
            left_join_key = None
            right_join_key = None

        operator = self.get_op(json_rel)
        node_type = op_to_one_hot(operator)
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None

        if operator in SCAN_TYPES:
            input_relations.append(extract_table_name(json_rel))

        filter_cols = []
        filter_ops = []
        filter_values = []
        if operator in FILTER_TYPES:
            filter_spark_node: SparkFilterPlanNode = PlanFactory.get_plan_node_instance(db_type, json_rel)
            predicate = filter_spark_node.predicates
            for i in range(len(predicate)):
                infos = predicate[i]
                if is_number(infos[2]):
                    col = infos[0]
                    filter_cols.append(col)
                    filter_ops.append(infos[1])
                    filter_values.append(self.normalizer.norm_no_log(float(infos[2]), col))

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        return SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations),
                            self.encode_join_keys(left_join_key, right_join_key),
                            left_input_row, right_input_row, operator,
                            self.encode_filter_col(filter_cols), self.encode_filter_op(filter_ops),
                            self.encode_filter_values(filter_values))

    def encode_join_keys(self, left_join_key, right_join_key):
        arr = np.zeros(len(self.join_keys))
        if left_join_key is None and right_join_key is None:
            return arr
        if left_join_key not in self.join_keys or right_join_key not in self.join_keys:
            # print(RuntimeError("unknown join_key"))
            pass
        else:
            arr[self.join_keys.index(left_join_key)] = 1
            arr[self.join_keys.index(right_join_key)] = 1
        return arr

    def encode_filter_col(self, filter_cols):
        filter_cols = filter_cols[0:max_predicate_num]
        arr = np.zeros((max_predicate_num, len(self.filter_cols)))
        for i, col in enumerate(filter_cols):
            if col is None:
                continue
            if col not in self.filter_cols:
                pass
                # print(RuntimeError("unknown filter_col"))
            else:
                arr[i][self.filter_cols.index(col)] = 1
        return arr.flatten()

    def encode_filter_op(self, filter_ops):
        filter_ops = filter_ops[0:max_predicate_num]
        arr = np.zeros((max_predicate_num, len(self.filter_ops)))
        for i, op in enumerate(filter_ops):
            if op is None:
                continue
            if op not in self.filter_ops:
                # print("unknown filter_op")
                pass
            else:
                arr[i][self.filter_ops.index(op)] = 1
        return arr.flatten()

    def encode_filter_values(self, filter_values):
        filter_values = filter_values[0:max_predicate_num]
        arr = np.zeros(max_predicate_num)
        for i, value in enumerate(filter_values):
            arr[i] = filter_values[i]
        return arr

    def get_op(self, json_rel):
        operator = json_rel['class']
        if operator in JOIN_TYPES and not diff_join_type:
            operator = JOIN_TYPES[0]
        return operator


def op_to_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    if op_name not in OP_TYPES:
        arr[OP_TYPES.index(UNKNOWN_OP_TYPE)] = 1
        # raise RuntimeError("unknown operator")
    else:
        arr[OP_TYPES.index(op_name)] = 1
    return arr
