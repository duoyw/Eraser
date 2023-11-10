import numpy as np

from RegressionFramework.Plan.Plan import PlanNode, JoinPlanNode, Plan, FilterPlanNode
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.utils import cal_ratio

ignore_node_type = ["Hash", "Sort", "Bitmap Index Scan", "Aggregate", "Limit"]


class StaticConfig:
    def __init__(self, struct_enable=True, scan_type_enable=False, table_name_enable=False,
                 join_type_enable=False, join_key_enable=False,
                 filter_enable=False, filter_col_enable=False, filter_op_enable=False, filter_value_enable=False):
        self.struct_enable = struct_enable
        self.scan_type_enable = scan_type_enable
        self.table_name_enable = table_name_enable
        self.join_type_enable = join_type_enable
        self.join_key_enable = join_key_enable
        self.filter_enable = filter_enable
        self.filter_col_enable = filter_col_enable
        self.filter_op_enable = filter_op_enable
        self.filter_value_enable = filter_value_enable


class StaticPlanGroup:
    def __init__(self, config=None):
        if config is None:
            config = StaticConfig()
        self.struct_enable = config.struct_enable
        self.scan_type_enable = config.scan_type_enable
        self.table_name_enable = config.table_name_enable
        self.join_type_enable = config.join_type_enable
        self.join_key_enable = config.join_key_enable
        self.filter_enable = config.filter_enable
        self.filter_col_enable = config.filter_col_enable
        self.filter_op_enable = config.filter_op_enable
        self.filter_value_enable = config.filter_value_enable

        self.key_to_group = {}

    def build(self, plans, model_name=None, train_set_name=None):
        for idx, plan in enumerate(plans):
            key = self.get_group_key(plan)
            if key not in self.key_to_group:
                self.key_to_group[key] = Group()
            group: Group = self.key_to_group[key]
            group.add_plan(plan)

    def get_group(self, plan):
        key = self.get_group_key(plan)
        if key in self.key_to_group:
            return self.key_to_group[key]
        return None

    def evaluate(self, plan):
        key = self.get_group_key(plan)
        if key in self.key_to_group:
            return True
        return False

    def get_group_key(self, plan):
        key = []
        self._recurse_plan(plan.root, key)
        # self._recurse_plan_simplify(plan.root, key)
        return "".join(key)

    def get_all_groups(self):
        return list(self.key_to_group.values())

    def _recurse_plan(self, node: PlanNode, key: list):
        node_type = node.node_type

        if node.is_filter_node():
            key.append(self.get_filter_key(node))
        elif node.is_scan_node():
            key.append(self.get_table_key(node))
        elif node.is_join_node():
            key.append(self.get_join_key(node))
        else:
            if node_type not in ignore_node_type:
                key.append(node_type)

        if len(node.children) > 0:
            children = node.children
            # children.sort(key=lambda x: x.node_type)
            for idx, child in enumerate(children):
                key.append(str(idx))
                self._recurse_plan(child, key)

    def _recurse_plan_simplify(self, node: PlanNode, key: list):
        node_type = node.node_type

        if node.is_filter_node():
            key.append(self.get_filter_key(node)[0:3])
        elif node.is_scan_node():
            key.append(self.get_table_key(node)[0:3])
        elif node.is_join_node():
            key.append(self.get_join_key(node)[0:3])
        else:
            if node_type not in ignore_node_type:
                key.append(node_type[0:3])

        if len(node.children) > 0:
            children = node.children
            # children.sort(key=lambda x: x.node_type)
            for idx, child in enumerate(children):
                key.append(str(idx))
                self._recurse_plan_simplify(child, key)

    def get_table_key(self, node: PlanNode):
        table_type = "type"
        table_name = "name"
        if self.scan_type_enable:
            table_type = node.get_node_type(node.node_json)
        if self.table_name_enable:
            table_name = node.get_table_name()
        return "table_{}_{}".format(table_type, table_name)

    def get_join_key(self, node: JoinPlanNode):
        join_key = "key"
        join_type = "type"
        if self.join_type_enable:
            join_type = node.get_join_type()
        if self.join_key_enable:
            join_key = node.get_join_key_str()
        return "join_{}_{}".format(join_key, join_type)

    def get_filter_key(self, node: FilterPlanNode):
        predicates = node.predicates
        if predicates is not None and self.filter_enable:
            key = []
            for predicate in predicates:
                key.append("{}_{}_{}".format(
                    predicate[0] if self.filter_col_enable else "col",
                    predicate[1] if self.filter_op_enable else "op",
                    predicate[2] if self.filter_value_enable else "value",
                ))
            return "".join(key)
        return ""


class Group:
    all_groups = []
    id = 0

    def __str__(self, *args, **kwargs):
        return "plan_size is {}, variance is {}".format(len(self.plans), self._variance)

    def __init__(self, plans=None):
        self.plans = []
        self._variance = None
        self.min_ratio = float("inf")
        self.max_ratio = -float("inf")
        self.ratios = []
        self.id = Group.id
        Group.id += 1
        Group.all_groups.append(self)

        if plans is not None:
            for p in plans:
                self.add_plan(p)

    def add_plan(self, plan):
        self._variance = None
        self.plans.append(plan)
        ratio = plan.metric
        self.min_ratio = min(self.min_ratio, ratio)
        self.max_ratio = min(self.max_ratio, ratio)
        self.ratios.append(ratio)

    def adjust_predict(self, predict):
        predict = predict[0]
        mean_predict = np.array([predict / r for r in self.ratios]).mean()
        assert not np.isnan(mean_predict)
        return predict / self.max_ratio, predict / self.min_ratio, mean_predict

    def size(self):
        return len(self.plans)

    def variance(self):
        if self._variance is not None:
            return self._variance

        return self.variance_no_cache()

    def variance_no_cache(self):
        assert len(self.plans) > 0
        self._variance = np.var(np.array(self.ratios))
        return self._variance

    def confidence(self):
        assert len(self.plans) > 0
        return np.mean(np.array(self.ratios))

    def confidence_range(self):
        assert len(self.plans) > 0
        ratios = np.array(self.ratios)
        return np.min(ratios), np.mean(ratios), np.max(ratios)

    def draw(self):
        res = []
        for plan in self.plans:
            res.append(plan.draw_dot())
        return res
