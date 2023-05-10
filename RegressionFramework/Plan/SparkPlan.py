import time

from RegressionFramework.Plan.Plan import PlanNode, FilterPlanNode, JoinPlanNode, ScanPlanNode, ProjectPlanNode, Plan
from RegressionFramework.Plan.PlanConfig import SparkNodeConfig
from RegressionFramework.Common.utils import is_number


class SparkPlanNodeMixIn(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

    @classmethod
    def get_node_type(cls, node_json=None):
        return node_json["class"]

    @classmethod
    def get_expression_type(cls, expression):
        return expression["class"]

    @classmethod
    def _get_attr_expression_info(cls, expression):
        assert cls._is_attr_expression(expression)
        name = expression["name"]
        qualifier = cls._get_qualifier(expression)
        return cls._get_combine_name_qualifier(name, qualifier), expression["dataType"]

    @classmethod
    def _get_qualifier(cls, expression):
        qualifier = expression["qualifier"]
        if isinstance(qualifier, list) and len(qualifier) == 0:
            return ""
        qualifier = expression["qualifier"][1:-1]
        return qualifier.split(",")[-1]

    @classmethod
    def _get_literal_expression_info(cls, expression):
        assert cls._is_literal_expression(expression)
        return expression["value"], expression["dataType"]

    @classmethod
    def _get_combine_name_qualifier(cls, name: str, qualifier: list):
        return name + qualifier[-1] if len(qualifier) > 0 else name

    @classmethod
    def _is_broadcast_nested_loop_join_type(cls, node):
        return node["class"] == SparkNodeConfig.BroadcastNestedLoopJoinType

    @classmethod
    def _is_attr_expression(cls, expression):
        return expression["class"] == SparkNodeConfig.ATTRIBUTE_REFERENCE_TYPE

    @classmethod
    def _is_isnull_expression(cls, expression):
        return expression["class"] == SparkNodeConfig.iS_NULL_TYPE

    @classmethod
    def _is_literal_expression(cls, expression):
        return expression["class"] == SparkNodeConfig.LITERAL_TYPE

    @classmethod
    def _is_substring_expression(cls, expression):
        return expression["class"] == SparkNodeConfig.SUBSTRING_TYPE

    @classmethod
    def _is_coalesce_expression(cls, expression):
        return expression["class"] == SparkNodeConfig.COALESCE_TYPE

    @classmethod
    def _get_literal_expression_infos(cls, expression):
        assert cls._is_literal_expression(expression)
        value = expression["value"]
        data_type = expression["dataType"]
        if data_type == "timestamp":
            t = time.strptime(value, "%Y-%m-%d %H:%M:%S")
            value = time.mktime(t)
        elif data_type == "date":
            t = time.strptime(value, "%Y-%m-%d")
            value = time.mktime(t)
        return value, data_type


class SparkFilterPlanNode(SparkPlanNodeMixIn, FilterPlanNode):
    def parse_predicates(self):
        predicates, tables = self.extract_filter_predicate(self.node_json)
        if predicates is None:
            return [], []
        for i, predicate in enumerate(predicates):
            predicate[0] = tables[i] + "_" + predicate[0]
            value = predicate[2]
            if is_number(value):
                predicate[2] = float(value)
        return predicates, tables

    def extract_filter_predicate(self, node):
        """
        :param node:
        :return: [(col op value)(...)], [table]
        """
        empty = None, None
        conditions = node["condition"]
        all_classes = list(map(lambda x: x["class"], conditions))

        predicates = []
        prefixes = []
        pos = 0
        while True:
            if pos > len(all_classes) or all_classes[pos:].count(
                    SparkNodeConfig.LITERAL_TYPE) == 0 or SparkNodeConfig.IN_SET_TYPE in all_classes \
                    or SparkNodeConfig.IN_TYPE in all_classes:
                break
            pos = all_classes.index(SparkNodeConfig.LITERAL_TYPE, pos)

            # find first AttributeReference that record column
            column, _ = self._get_attr_expression_info(conditions[pos - 1])
            qualifier = self._get_qualifier(conditions[pos - 1])

            # find operator such as =, > ,<
            op = self._simplify_op(self.get_node_type(conditions[pos - 2]))

            # find Literal
            value, data_type = self._get_literal_expression_infos(conditions[pos])
            predicates.append([column, op, value])
            prefixes.append(qualifier)
            pos += 1

        if len(predicates) < 1:
            return empty

        return predicates, prefixes

    def _simplify_op(self, node_type):
        """
        :param node_type:  "org.apache.spark.sql.catalyst.expressions.EqualTo"...
        :return: =,<,>
        """
        if node_type not in SparkNodeConfig.OPERATOR_TYPE:
            return ""
        return SparkNodeConfig.OPERATOR_TYPE[node_type]

    def extract_column_with_prefix(self, attribute_reference_node):
        assert attribute_reference_node["class"] == SparkNodeConfig.ATTRIBUTE_REFERENCE_TYPE
        column = attribute_reference_node["name"]

        # may be table, or table alias, a list

        prefix = attribute_reference_node["qualifier"][1:-1].split(",")[-1]
        return column, prefix

    def _get_literal_expression_infos(self, literal_node):
        assert literal_node["class"] == SparkNodeConfig.LITERAL_TYPE
        value = literal_node["value"]
        data_type = literal_node["dataType"]
        if data_type == "timestamp":
            t = time.strptime(value, "%Y-%m-%d %H:%M:%S")
            value = time.mktime(t)
        elif data_type == "date":
            t = time.strptime(value, "%Y-%m-%d")
            value = time.mktime(t)
        return value, data_type


# class SparkJoinPlanNode(SparkPlanNodeMixIn, JoinPlanNode):
#     def get_join_type(self):
#         return self.node_type
#
#     def get_join_key(self):
#         keys = ["None", "None"]
#         node = self.node_json
#         try:
#             if "leftKeys" not in node or "rightKeys" not in node:
#                 raise RuntimeError("please input join_operator")
#             keys[0] = node["leftKeys"][0][0]["name"]
#             keys[1] = node["rightKeys"][0][0]["name"]
#         except:
#             print("get_join_key error")
#         keys = sorted(keys)
#         return "{}_{}".format(keys[0], keys[1])


class SparkJoinPlanNode(SparkPlanNodeMixIn, JoinPlanNode):
    def get_join_type(self):
        return self.node_type

    def _parse_join_key(self):
        node = self.node_json
        left_keys = []
        right_keys = []
        if self._is_broadcast_nested_loop_join_type(node):
            if "condition" in node:
                left_key, right_key = self._get_join_key_for_broadcast_nested_loop_join(node["condition"])
                left_keys.append(left_key)
                right_keys.append(right_key)
        else:
            left_infos = node["leftKeys"]
            right_infos = node["leftKeys"]
            assert len(left_infos) == len(right_infos)
            for i in range(len(left_infos)):
                left_info = left_infos[i]
                right_info = right_infos[i]
                left_keys.append(self._join_key_parse_type_assign(left_info))
                right_keys.append(self._join_key_parse_type_assign(right_info))
        return ".".join(left_keys), ".".join(right_keys)

    def print_join_node_expression(self, keys):
        print("##############")
        for key in keys:
            print(self.get_expression_type(key[0]))

    def _get_join_key_for_broadcast_nested_loop_join(self, conditions):
        assert len(conditions) == 10
        left_key = self._get_join_key_for_substring(conditions[2:6])
        right_key = self._get_join_key_for_substring(conditions[6:10])
        return left_key, right_key

    def _join_key_parse_type_assign(self, key_infos):
        first_expression_type = key_infos[0]
        if self._is_substring_expression(first_expression_type) or (
                len(key_infos) > 1 and self._is_substring_expression(key_infos[1])):
            return self._get_join_key_for_substring(key_infos)
        elif len(key_infos) == 1 and self._is_attr_expression(first_expression_type):
            return self._get_join_key_for_value(key_infos)
        elif len(key_infos) == 1 and self._is_isnull_expression(first_expression_type):
            return None
        elif self._is_coalesce_expression(first_expression_type):
            return self._get_join_key_for_coalesce(key_infos)
        else:
            raise RuntimeError

    def _get_join_key_for_substring(self, infos: list):
        origin_size = len(infos)
        assert origin_size == 4 or origin_size == 6 or origin_size == 3 or origin_size == 5

        origin = infos
        if self._is_substring_expression(origin[1]):
            infos = origin[1:]

        value2 = ""
        value3 = ""
        assert self._is_substring_expression(infos[0])
        name, _ = self._get_attr_expression_info(infos[1])
        value1, _ = self._get_literal_expression_info(infos[2])
        if len(infos) > 3:
            value2, _ = self._get_literal_expression_info(infos[3])
        if len(infos) > 4:
            value3, _ = self._get_literal_expression_info(infos[4])
        return "substring_{}_{}_{}_{}, ".format(name, value1, value2, value3)

    def _get_join_key_for_coalesce(self, infos: list):
        origin_size = len(infos)
        assert origin_size == 3

        assert self._is_coalesce_expression(infos[0])
        name, _ = self._get_attr_expression_info(infos[1])
        value1, _ = self._get_literal_expression_info(infos[2])
        return "coalesce_{}_{}, ".format(name, value1)

    def _get_join_key_for_value(self, infos: list):
        assert len(infos) == 1

        key, _ = self._get_attr_expression_info(infos[0])
        return "key={}, ".format(key)


class SparkScanPlanNode(SparkPlanNodeMixIn, ScanPlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

    def get_table_name(self):
        if "tableIdentifier" not in self.node_json:
            raise RuntimeError("please input file_scan_operator")
        return self.node_json["tableIdentifier"]["table"]


class SparkProjectPlanNode(SparkPlanNodeMixIn, ProjectPlanNode):

    def _parse_project_cols(self):
        cols = []
        node_json = self.node_json
        project_infos = node_json["projectList"]
        for info in project_infos:
            assert len(info) == 1
            name, data_type = self._get_attr_expression_info(info[0])
            cols.append(name)
        return cols


class SparkOtherPlanNode(SparkPlanNodeMixIn, PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)


class SparkPlan(Plan):
    def __init__(self, plan_json, plan_id, predict=None):
        super().__init__(plan_json, plan_id, predict)

    @classmethod
    def to_node(cls, node_json, node_id=None):
        node_type = cls.get_node_type(node_json)
        if node_type in SparkNodeConfig.SCAN_TYPES:
            plan_node = SparkScanPlanNode(node_json, node_id)
        elif node_type in SparkNodeConfig.JOIN_TYPES:
            plan_node = SparkJoinPlanNode(node_json, node_id)
        elif node_type in SparkNodeConfig.FILTER_TYPES:
            plan_node = SparkFilterPlanNode(node_json, node_id)
        # elif node_type in SparkNodeConfig.PROJECT_TYPES:
        #     plan_node = SparkProjectPlanNode(node_json, node_id)
        else:
            plan_node = SparkOtherPlanNode(node_json, node_id)
        return plan_node

    def _to_plan_node(self, node_json, node_id, node_id_to_node):
        plan_node = self.to_node(node_json, node_id)

        assert node_id not in node_id_to_node
        node_id_to_node[node_id] = plan_node

        cur_max_node_id = node_id
        if "Plans" in node_json:
            children = node_json["Plans"]
            for child in children:
                child_node, cur_max_node_id = self._to_plan_node(child, cur_max_node_id + 1,
                                                                 node_id_to_node)
                plan_node.children.append(child_node)
        return plan_node, cur_max_node_id

    @classmethod
    def get_node_type(cls, node_json):
        return SparkPlanNodeMixIn.get_node_type(node_json)
