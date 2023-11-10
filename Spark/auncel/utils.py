import configparser
import json
import os
import random
import time
from copy import copy

import numpy as np
import torch

from Common.Cache import ParsePlanCache
from feature import JOIN_TYPES, SCAN_TYPES
from model_config import IGNORE_TYPES, OPERATOR_TYPE, ATTRIBUTE_REFERENCE_TYPE, LITERAL_TYPE, TransformerConfig, \
    ALIAS_TO_TABLE, IN_TYPE, INSET_TYPE, model_type
from plan_cpmpress import PlanCompress
from Spark.auncel.test_script.config import DATA_BASE_PATH, SEP


def read_config():
    config = configparser.ConfigParser()
    config.read("server.conf")

    if "auncel" not in config:
        print("server.conf does not have a [auncel] section.")
        exit(-1)

    config = config["auncel"]
    return config


def print_log(s, log_path, print_to_std_out=False):
    os.system("echo \"" + str(s) + "\" >> " + log_path)
    if print_to_std_out:
        print(s)


# Auncel guides the optimizer to generate different plans by changing cardinalities,
# but row count will be used as the input feature when predicting the plan score.
# So we need to restore all the row counts to the original values before feeding the model.
class PlanCardReplacer():
    def __init__(self, table_array, rows_array) -> None:
        self.table_array = table_array
        self.rows_array = rows_array
        self.SCAN_TYPES = SCAN_TYPES
        self.JOIN_TYPES = JOIN_TYPES
        self.SAME_CARD_TYPES = ["Hash", "Materialize",
                                "Sort", "Incremental Sort", "Limit"]
        self.OP_TYPES = ["Aggregate", "Bitmap Index Scan"] + \
                        self.SCAN_TYPES + self.JOIN_TYPES + self.SAME_CARD_TYPES
        self.table_idx_map = {}
        for arr in table_array:
            for t in arr:
                if t not in self.table_idx_map:
                    self.table_idx_map[t] = len(self.table_idx_map)

        self.table_num = len(self.table_idx_map)
        self.table_card_map = {}
        for i in range(len(table_array)):
            arr = table_array[i]
            card = rows_array[i]
            hash = self.encode_input_tables(arr)
            # print(hash, card, arr)
            if hash in self.table_card_map:
                pass
                # if self.table_card_map[hash] != card:
                #     print("hash conflict")
                #     print(self.table_card_map)
                #     print(hash)
                #     print(card)
                #     if abs(card - self.table_card_map[hash]) / self.table_card_map[hash] > 0.001:
                #         raise Exception("hash conflict")
            else:
                self.table_card_map[hash] = card

    def replace(self, plan):
        input_card = None
        input_tables = []
        output_card = None

        if "Plans" in plan:
            children = plan['Plans']
            child_input_tables = None
            if len(children) == 1:
                child_input_card, child_input_tables = self.replace(children[0])
                input_card = child_input_card
                input_tables += child_input_tables
            else:
                for child in children:
                    _, child_input_tables = self.replace(child)
                    input_tables += child_input_tables

        node_type = plan['Node Type']
        if node_type in self.JOIN_TYPES:
            tag = self.encode_input_tables(input_tables)
            if tag not in self.table_card_map:
                print(input_tables)
                raise Exception("Unknown tag " + str(tag))
            card = self.table_card_map[tag]
            plan['Plan Rows'] = card
            output_card = card
        elif node_type in self.SAME_CARD_TYPES:
            if input_card is not None:
                plan['Plan Rows'] = input_card
                output_card = input_card
        elif node_type in self.SCAN_TYPES:
            input_tables.append(plan['Relation Name'])
        elif node_type not in self.OP_TYPES:
            raise Exception("Unknown node type " + node_type)

        return output_card, input_tables

    def encode_input_tables(self, input_table_list):
        l = [0 for _ in range(self.table_num)]
        for t in input_table_list:
            l[self.table_idx_map[t]] += 1

        hash = 0
        for i in range(len(l)):
            hash += l[i] * (10 ** i)
        return hash


def get_tree_signature(json_tree):
    signature = {}
    if "Plans" in json_tree:
        children = json_tree['Plans']
        if len(children) == 1:
            signature['L'] = get_tree_signature(children[0])
        else:
            assert len(children) == 2
            signature['L'] = get_tree_signature(children[0])
            signature['R'] = get_tree_signature(children[1])

    node_type = json_tree['Node Type']
    if node_type in SCAN_TYPES:
        signature["T"] = json_tree['Relation Name']
    elif node_type in JOIN_TYPES:
        signature["J"] = node_type[0]
    return signature


class OptState:
    def __init__(self, card_picker, plan_card_replacer, dump_card=False) -> None:
        self.card_picker = card_picker
        self.plan_card_replacer = plan_card_replacer
        if dump_card:
            self.card_list_with_score = []
            self.visited_trees = set()


def to_tree_json(spark_plan):
    spark_plan = json_str_to_json_obj(spark_plan)
    plan = copy(spark_plan)
    if isinstance(spark_plan["Plan"], list):
        plan["Plan"], _ = _to_tree_json(spark_plan["Plan"], 0)
    return json.dumps([plan])


def _to_tree_json(targets, index=0):
    node = targets[index]
    num_children = node["num-children"]

    all_child_node_size = 0
    if num_children == 0:
        # +1 is self
        return node, all_child_node_size + 1

    left_node, left_size = _to_tree_json(targets, index + all_child_node_size + 1)
    node["Plans"] = [left_node]
    all_child_node_size += left_size

    if num_children == 2:
        right_node, right_size = _to_tree_json(targets, index + all_child_node_size + 1)
        node["Plans"].append(right_node)
        all_child_node_size += right_size

    return node, all_child_node_size + 1


def extract_table_name(file_scan_operator):
    if "tableIdentifier" not in file_scan_operator:
        raise RuntimeError("please input file_scan_operator")
    return file_scan_operator["tableIdentifier"]["table"]


def extract_join_key(spark_join_node):
    try:
        if "leftKeys" not in spark_join_node or "rightKeys" not in spark_join_node:
            raise RuntimeError("please input join_operator")
        return spark_join_node["leftKeys"][0][0]["name"], spark_join_node["rightKeys"][0][0]["name"]
    except:
        return None, None


def read_plans(file):
    """
    :param file: each row including multiple plan whit seperator "######"
    :return: a two dim list, first dim is query, second dim is each plan of a query
    """
    is_compress = False
    is_correct_card = False
    plans_for_query = []
    cache = ParsePlanCache(file, is_compress, is_correct_card, enable=True)
    if cache.exist():
        # print("read_plans trigger cache")
        return cache.read()
    with open(file, "r") as f:
        # print("read_plans do not trigger cache")
        for line in f.readlines():
            plans = line.split("#####")[1:]
            plans = [to_tree_json(plan) for plan in plans]
            if is_compress:
                plans = [compress_plan(plan) for plan in plans]
            if is_correct_card:
                plans = [correct_card(plan) for plan in plans]
            if len(plans) > 1:
                plans_for_query.append(plans)
        cache.save(plans_for_query)
    return plans_for_query


def read_accuracy_plans(file):
    """
    :param file: each row including multiple plan whit seperator "######"
    :return: a two dim list, first dim is query, second dim is each plan of a query
    """
    plans_for_query = []
    with open(file, "r") as f:
        print("read_accuracy_plans")
        for line in f.readlines():
            plans = line.split("#####")[1:]
            if len(plans) > 1:
                plans_for_query.append(plans)
    return plans_for_query


def get_cache_plan(file, is_compress, is_correct_card):
    file = file + "_PlanCache"
    if os.path.exists(file):
        with open(file, "r") as f:
            key = f.readline().strip("\n")
            if key != get_cache_plan_key(is_compress, is_correct_card):
                return None
            value = f.readline()
            return json.loads(value)
    return None


def save_cache_plan(plans_for_query, file, is_compress, is_correct_card):
    file = file + "_PlanCache"
    with open(file, "w") as f:
        key = get_cache_plan_key(is_compress, is_correct_card)
        value = json.dumps(plans_for_query)
        f.write(key + "\n")
        f.write(value)


def get_cache_plan_key(is_compress, is_correct_card):
    return "is_compress={}, is_correct_card={}".format(is_compress, is_correct_card)


def compress_plan(plan):
    plan = json_str_to_json_obj(plan)
    plan_compress = PlanCompress(IGNORE_TYPES)
    plan["Plan"] = plan_compress.compress(plan["Plan"])
    return json.dumps([plan])


def _load_pairwise_plans(path, limit_ratio=None):
    X1, X2 = [], []
    plans_for_query = read_plans(path)
    if limit_ratio is not None:
        plans_for_query = plans_for_query[0:int(len(plans_for_query) * limit_ratio)]
    for arr in plans_for_query:
        arr = list(sorted(set(arr), key=arr.index))
        if len(arr) >= 2:
            x1, x2 = get_training_pair(arr)
            X1 += x1
            X2 += x2

    assert len(X1) > 0
    return X1, X2


def _load_accuracy_pairwise_plans(path):
    X1, X2 = [], []
    plans_for_query = read_accuracy_plans(path)
    for arr in plans_for_query:
        arr = list(sorted(set(arr), key=arr.index))
        if len(arr) >= 2:
            x1, x2 = get_training_pair(arr)
            X1 += x1
            X2 += x2

    assert len(X1) > 0
    return X1, X2


def _load_accuracy_pairwise_plans_cross_plan(path, k):
    X1, X2 = [], []
    struct_2_plans = {}
    with open(path) as f:
        line = f.readline()
        while line is not None and line != "":
            values = line.split(SEP)
            name = values[0]
            if name not in struct_2_plans:
                struct_2_plans[name] = []
            plans = values[1:]
            plans = plans[0:min(len(plans), k)]
            struct_2_plans[name] += plans
            # all_plans += plans[0:min(len(plans), k)]
            line = f.readline()

    for plans in struct_2_plans.values():
        if len(plans) >= 2:
            x1, x2 = get_training_pair(plans)
            X1 += x1
            X2 += x2
    return X1, X2


def unzip(target):
    """
    :param x: [(x1,y1),(x2,y2)...]
    :return: x=[x1,x2,...],y=[y1,y2]
    """
    x = [t[0] for t in target]
    y = [t[1] for t in target]
    return x, y


def _load_accuracy_pointwise_plans(path):
    X1, X2 = _load_accuracy_pairwise_plans(path)
    return list(set(X1 + X2))


def _load_pointwise_plans(path):
    X1, X2 = _load_pairwise_plans(path)
    return list(set(X1 + X2))


def get_training_pair(candidates):
    assert len(candidates) >= 2
    X1, X2 = [], []

    i = 0
    while i < len(candidates) - 1:
        s1 = candidates[i]
        j = i + 1
        while j < len(candidates):
            s2 = candidates[j]
            X1.append(s1)
            X2.append(s2)
            j += 1
        i += 1
    return X1, X2


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


def flatten_list(targets):
    if targets is None:
        return targets
    return [t for sub in list(targets) for t in sub]


def join_key_identifier(key1, key2):
    return "{}####{}".format(key1, key2)


def extract_filter_predicate(node):
    """
    :param node:
    :return: [(col op value)(...)], [table]
    """
    empty = None, None
    conditions = node["condition"]
    if len(conditions) < 3:
        return empty
    all_classes = list(map(lambda x: x["class"], conditions))
    if LITERAL_TYPE not in all_classes or IN_TYPE in all_classes or INSET_TYPE in all_classes:
        return empty

    predicates = []
    prefixes = []
    pos = 0
    for i in range(TransformerConfig.max_predicate_num):
        if pos > len(all_classes) or all_classes[pos:].count(LITERAL_TYPE) == 0:
            break
        try:
            pos = all_classes.index(LITERAL_TYPE, pos)

            # find first AttributeReference that record column
            column, prefix = extract_column_with_prefix(conditions[pos - 1])

            # find operator such as =, > ,<
            op = extract_filter_operator(conditions[pos - 2])

            # find Literal
            value, data_type = extract_filter_literal(conditions[pos])
            pos += 1
            if data_type != "string":
                # predicates.append("({} {} {})".format(column, op, value))
                predicates.append([column, op, value])
                prefixes.append(prefix)
        except:
            break

    if len(predicates) < 1:
        return empty
    return predicates, prefixes


def extract_column_with_prefix(attribute_reference_node):
    assert attribute_reference_node["class"] == ATTRIBUTE_REFERENCE_TYPE
    column = attribute_reference_node["name"]

    # may be table, or table alias, a list

    prefix = attribute_reference_node["qualifier"][1:-1].split(",")[-1]
    return column, prefix


def extract_filter_operator(expressions_node):
    """
    :param expressions_node:
    :return: find operator such as =, > ,<
    """
    assert expressions_node["class"] in OPERATOR_TYPE

    op = OPERATOR_TYPE[expressions_node["class"]]
    return op


def extract_filter_literal(literal_node):
    assert literal_node["class"] == LITERAL_TYPE
    value = literal_node["value"]
    data_type = literal_node["dataType"]
    if data_type == "timestamp":
        t = time.strptime(value, "%Y-%m-%d %H:%M:%S")
        value = time.mktime(t)
    elif data_type == "date":
        t = time.strptime(value, "%Y-%m-%d")
        value = time.mktime(t)
    return value, data_type


def correct_card(plan):
    plan = json_str_to_json_obj(plan)
    recurse_correct_card(plan["Plan"], plan["Swing"], plan["Level"])
    return json.dumps([plan])


def recurse_correct_card(node, swing, swing_level):
    if swing <= 1.0:
        return
    input_relations = []
    if node["class"] in SCAN_TYPES:
        # base table
        input_relations.append(extract_table_name(node))

    if "Plans" in node:
        for child in node["Plans"]:
            input_relations += recurse_correct_card(child, swing, swing_level)

    if "rowCount" in node:
        row = node["rowCount"]
    else:
        row = 0.0

    width = node["sizeInBytes"]
    if len(input_relations) == swing_level:
        row /= float(swing)
        width /= float(swing)
        node["rowCount"] = row
        node["sizeInBytes"] = width
    # print("operator_name is {}, input_relations size is {}".format(operator_name, input_relations))

    return input_relations


def combine_table_col(table_alias, col, dataset_name):
    table = table_alias
    table = ALIAS_TO_TABLE[dataset_name][table] if table in ALIAS_TO_TABLE[
        dataset_name] else table
    return table + "." + col


def to_one_hot_tensor(input_tensor: torch.Tensor, max_num):
    shape = input_tensor.shape
    device = input_tensor.device

    input_tensor.reshape(shape)
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.reshape((-1,))
    assert input_tensor.dim() == 1

    input_tensor = input_tensor.cpu()
    res = []
    for i, ele in enumerate(input_tensor):
        res.append(to_one_hot(max_num, int(ele.item())))

    res = torch.tensor(res, device=device)
    res = res.reshape((*list(shape), max_num))
    return res


def to_one_hot(max_num, cur_pos):
    arr = np.zeros(max_num)
    assert cur_pos < max_num
    arr[cur_pos] = 1
    return arr


def cal_plan_height_and_size(root):
    def recuse(node):
        cur_max_height = 0
        cur_max_node_size = 0
        if "Plans" in node:
            children = node["Plans"]
            for child in children:
                c_h, c_size = recuse(child)
                cur_max_height = max(cur_max_height, c_h)
                cur_max_node_size += c_size
        return cur_max_height + 1, cur_max_node_size + 1

    return recuse(root)


def add_to_json(key, value, target):
    if isinstance(target, str):
        target = json_str_to_json_obj(target)
    target[key] = value
    return json.dumps(target)


def flat_depth2_list(targets: list):
    """
    :param targets: [[],[],[]]
    :return: []
    """
    res = []
    for values in targets:
        res += values
    return res


def add_list_by_pos(t1, t2):
    assert len(t1) == len(t2)
    return [t1[i] + t2[i] for i in range(len(t1))]


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def swap(a, b):
    return b, a


# for confidence, on tree_conv model
def get_confidence_model_name(dataset_name, model_type_name):
    return "confidence_{}_{}_model".format(dataset_name, model_type_name.lower())


def get_plans_with_accuracy_file_path(dataset_name):
    return os.path.join(DATA_BASE_PATH, "accuracy/accuracy_plan_{}_{}".format(dataset_name, model_type.name.lower()))


def get_group_plans_file_path(train_set_name, model_type):
    return os.path.join(DATA_BASE_PATH, "leaf_group_{}_{}".format(train_set_name, model_type.name.lower()))


def save_accuracy_plans_to_file(file, plans_for_queries):
    with open(file, "w") as f:
        i = 0
        for plans in plans_for_queries:
            line = ["query{}".format(i)]
            line += plans
            line = SEP.join(line)
            f.writelines(line)
            f.write("\n")


def cal_accuracy(predict, actual):
    assert predict >= 0
    return min(predict / actual, 2.0)
    # return max(0.0, 1 - abs(predict - actual) / actual)


def cal_ratio(predict, actual):
    assert predict >= 0
    return min(predict / actual, 2.0)
