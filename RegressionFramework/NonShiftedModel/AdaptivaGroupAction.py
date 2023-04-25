from bisect import bisect_left

from RegressionFramework.Common.dotDrawer import PlanDotDrawer
from RegressionFramework.NonShiftedModel.AdaptivaGroupTree import TreeNode
from RegressionFramework.NonShiftedModel.PlansManeger import PlansManager
from RegressionFramework.Plan.Plan import FilterPlanNode, JoinPlanNode, PlanNode
from RegressionFramework.Plan.PlanConfig import PgNodeConfig
from RegressionFramework.config import db_node_config, GroupEnable


class Action:
    def __init__(self, plan_node_id, plans_manager: PlansManager, score):
        self.plan_node_id = plan_node_id
        self.plans_manager: PlansManager = plans_manager
        self.score = score

    def name(self):
        return ""

    def __hash__(self):
        return hash((type(self), self.plan_node_id))

    def __eq__(self, other):
        return type(self) == type(other) and self.plan_node_id == other.plan_node_id

    def split(self, tree_node: TreeNode):
        left_tree_node, right_tree_node = self._split_no_add_action(tree_node)
        self.add_actions_left(left_tree_node, tree_node)
        self.add_actions_right(right_tree_node, tree_node)
        return left_tree_node, right_tree_node

    def _split_no_add_action(self, tree_node: TreeNode):
        left_group_plans = []
        right_group_plans = []
        plans = tree_node.plans
        for plan in plans:
            try:
                plan_node = self.plans_manager.get_node(plan.plan_id, self.plan_node_id)
                if self.is_left_group(plan_node):
                    left_group_plans.append(plan)
                else:
                    right_group_plans.append(plan)
            except:
                pass
        return TreeNode(left_group_plans), TreeNode(right_group_plans)

    def fake_split(self, tree_node: TreeNode):
        return  self._split_no_add_action(tree_node)


    @classmethod
    def update_actions(cls, plan_manager: PlansManager, root: TreeNode):
        plans = root.plans

        node_id_to_col_to_count = {}
        node_id_to_col_to_op_to_count = {}
        node_id_to_col_to_op_value_to_count = {}
        node_id_to_join_type_to_count = {}
        node_id_to_join_key_to_count = {}
        node_id_to_scan_type_to_count = {}
        node_id_to_table_name_to_count = {}
        node_id_to_project_col_to_count = {}

        for plan in plans:
            for plan_node in plan.get_all_nodes():
                plan_id = plan.plan_id
                node_id = plan_node.node_id
                node_type = plan_node.node_type

                if node_type in db_node_config.FILTER_TYPES:
                    cols_to_ops_to_values = plan_manager.get_all_filter_infos(plan_id, node_id)
                    cols = [] if len(cols_to_ops_to_values) == 0 else cols_to_ops_to_values.keys()
                    for col in cols:
                        cls.init_dict_and_inc(node_id_to_col_to_count, node_id, col)
                        for op in cols_to_ops_to_values[col].keys():
                            cls.init_dict_and_inc(node_id_to_col_to_op_to_count, node_id, col, op)
                            for value in cols_to_ops_to_values[col][op]:
                                cls.init_dict_and_inc(node_id_to_col_to_op_value_to_count, node_id, col, op, value)

                if node_type in db_node_config.JOIN_TYPES:
                    if not GroupEnable.join_key_enable:
                        join_keys = plan_manager.get_all_join_keys(plan_id, node_id)
                        for split_join_key in join_keys:
                            cls.init_dict_and_inc(node_id_to_join_key_to_count, node_id, split_join_key)

                    if not GroupEnable.join_type_enable:
                        join_types = plan_manager.get_all_join_types(plan_id, node_id)
                        for split_join_type in join_types:
                            cls.init_dict_and_inc(node_id_to_join_type_to_count, node_id, split_join_type)

                if node_type in db_node_config.SCAN_TYPES:
                    if not GroupEnable.scan_type_enable:
                        table_types = plan_manager.get_all_table_types(plan_id, node_id)
                        for split_table_type in table_types:
                            cls.init_dict_and_inc(node_id_to_scan_type_to_count, node_id, split_table_type)

                    if not GroupEnable.table_name_enable:
                        table_names = plan_manager.get_all_table_names(plan_id, node_id)
                        for split_table_name in table_names:
                            cls.init_dict_and_inc(node_id_to_table_name_to_count, node_id, split_table_name)

                # if node_type in db_node_config.PROJECT_TYPES:
                #     cols = list(set(plan_manager.get_all_project_cols(plan_id, node_id)))
                #     for col in cols:
                #         cls.init_dict_and_inc(node_id_to_project_col_to_count, node_id, col)

        plan_size = len(plans)
        for node_id in node_id_to_col_to_count.keys():
            for col, count in node_id_to_col_to_count[node_id].items():
                if count < plan_size:
                    root.add_action(FilterColAction(node_id, col, plan_manager, count / plan_size))
                else:
                    for op, op_count in node_id_to_col_to_op_to_count[node_id][col].items():
                        if op_count < plan_size:
                            root.add_action(FilterOpAction(node_id, col, op, plan_manager, count / plan_size))
                        else:
                            values = node_id_to_col_to_op_value_to_count[node_id][col][op]
                            values = sorted(list(values))
                            for i in range(1, len(values)):
                                value = values[i]
                                value_count = node_id_to_col_to_op_value_to_count[node_id][col][op][value]
                                if value_count < plan_size:
                                    root.add_action(
                                        FilterValueAction(node_id, col, op, value, plan_manager, count / plan_size))
        for node_id in node_id_to_join_key_to_count.keys():
            cls._aux_add_action(root, plan_manager, node_id_to_join_key_to_count, node_id, plan_size,
                                GroupEnable.join_key_enable, JoinKeyAction)
        for node_id in node_id_to_join_type_to_count.keys():
            cls._aux_add_action(root, plan_manager, node_id_to_join_type_to_count, node_id, plan_size,
                                GroupEnable.join_type_enable, JoinTypeAction)
        for node_id in node_id_to_scan_type_to_count.keys():
            cls._aux_add_action(root, plan_manager, node_id_to_scan_type_to_count, node_id, plan_size,
                                GroupEnable.scan_type_enable, TableTypeAction)
        for node_id in node_id_to_table_name_to_count.keys():
            cls._aux_add_action(root, plan_manager, node_id_to_table_name_to_count, node_id, plan_size,
                                GroupEnable.table_name_enable, TableNameAction)
        for node_id in node_id_to_project_col_to_count.keys():
            cls._aux_add_action(root, plan_manager, node_id_to_project_col_to_count, node_id, plan_size,
                                True, ProjectAction)

    @classmethod
    def _aux_add_action(cls, root: TreeNode, plan_manager, node_id_to_value_to_count, node_id, plan_size,
                        enable, action):
        if not enable:
            for value, count in node_id_to_value_to_count[node_id].items():
                if count < plan_size:
                    root.add_action(
                        action(node_id, value, plan_manager, count / plan_size))

    @classmethod
    def init_dict_and_inc(cls, node_id_to_value_to_count: dict, node_id, value, value2=None, value3=None):
        if node_id not in node_id_to_value_to_count:
            node_id_to_value_to_count[node_id] = {}
        if value2 is None:
            if value not in node_id_to_value_to_count[node_id]:
                node_id_to_value_to_count[node_id][value] = 0
            node_id_to_value_to_count[node_id][value] = node_id_to_value_to_count[node_id][value] + 1
        elif value3 is None:
            if value not in node_id_to_value_to_count[node_id]:
                node_id_to_value_to_count[node_id][value] = {}
            if value2 not in node_id_to_value_to_count[node_id][value]:
                node_id_to_value_to_count[node_id][value][value2] = 0
            node_id_to_value_to_count[node_id][value][value2] = node_id_to_value_to_count[node_id][value][value2] + 1
        else:
            if value not in node_id_to_value_to_count[node_id]:
                node_id_to_value_to_count[node_id][value] = {}
            if value2 not in node_id_to_value_to_count[node_id][value]:
                node_id_to_value_to_count[node_id][value][value2] = {}
            if value3 not in node_id_to_value_to_count[node_id][value][value2]:
                node_id_to_value_to_count[node_id][value][value2][value3] = 0
            node_id_to_value_to_count[node_id][value][value2][value3] = \
                node_id_to_value_to_count[node_id][value][value2][
                    value3] + 1


    def add_actions_left(self, target: TreeNode, origin: TreeNode, ignore_actions=None, ignore_action_types=None):
        if target.empty():
            return
        self.update_actions(self.plans_manager, target)

    def add_actions_right(self, target: TreeNode, origin: TreeNode, ignore_actions=None, ignore_action_types=None):
        if target.empty():
            return
        self.update_actions(self.plans_manager, target)

    @classmethod
    def is_ignore_action(cls, action, ignore_actions=None):
        if ignore_actions is None:
            return False
        return action in ignore_actions

    @classmethod
    def is_ignore_action_types(cls, action, ignore_action_types=None):
        if ignore_action_types is None:
            return False
        return type(action) in ignore_action_types

    def node_id_equal(self, action):
        return action.plan_node_id == self.plan_node_id

    def is_left_group(self, plan_node):
        raise RuntimeError


class RangeAction(Action):
    def __init__(self, plan_node_id, plans_manager: PlansManager, split_value, values, score):
        super().__init__(plan_node_id, plans_manager, score)
        self.sorted_values = sorted(list(set(values)))
        self.split_value = split_value

    def __hash__(self):
        return hash((type(self), self.plan_node_id, self.split_value))

    def __eq__(self, other):
        return super().__eq__(
            other) and self.split_value == other.split_value and self.sorted_values == other.sorted_values

    def is_left_group(self, plan_node: PlanNode):
        split_pos = bisect_left(self.sorted_values, self.split_value)
        cur_pos = bisect_left(self.sorted_values, self.get_cur_value(plan_node))
        if cur_pos <= split_pos:
            return True
        return False

    def get_cur_value(self, plan_node: PlanNode):
        raise RuntimeError

    def get_next_action(self, value, values):
        raise RuntimeError


class OnceAction(Action):

    def __init__(self, target_value, plan_node_id, plans_manager: PlansManager, score):
        super().__init__(plan_node_id, plans_manager, score)
        self.target_value = target_value

    def __hash__(self):
        return hash((type(self), self.plan_node_id, self.target_value))

    def name(self):
        return ""

    def __eq__(self, other):
        return super().__eq__(other) and self.target_value == other.target_value

    def is_left_group(self, plan_node):
        cur_value = self.get_cur_value(plan_node)
        if self.target_value == cur_value:
            return True
        return False

    def get_cur_value(self, plan_node):
        raise RuntimeError


class ProjectAction(OnceAction):
    def __init__(self, plan_node_id, col, plans_manager: PlansManager, score):
        super().__init__(col, plan_node_id, plans_manager, score)
        self.col = col

    # def is_left_group(self, plan_node: ProjectPlanNode):
    #     project_cols = set(plan_node.project_cols)
    #     if self.col in project_cols:
    #         return True
    #     return False


class FilterColAction(OnceAction):
    def __init__(self, plan_node_id, col, plans_manager: PlansManager, score):
        super().__init__(col, plan_node_id, plans_manager, score)
        self.col = col

    def name(self):
        return "F_Col_{}".format(self.col)

    def is_left_group(self, plan_node: FilterPlanNode):
        predicates = plan_node.predicates
        for predicate in predicates:
            if self.col == predicate[0]:
                return True
        return False


class FilterOpAction(Action):
    def __init__(self, plan_node_id, col, op, plans_manager: PlansManager, score):
        super().__init__(plan_node_id, plans_manager, score)
        self.op = op
        self.col = col

    def __hash__(self):
        return hash((type(self), self.plan_node_id, self.col, self.op))

    def name(self):
        return "F_Col_{}_Op_{}".format(self.col, self.op.split(".")[-1])

    def __eq__(self, other):
        return super().__eq__(other) and self.col == other.col and self.op == other.op

    def is_left_group(self, plan_node: FilterPlanNode):
        predicates = plan_node.predicates
        for predicate in predicates:
            col = predicate[0]
            op = predicate[1]
            if self.col == col and self.op == op:
                return True
        return False


class FilterValueAction(OnceAction):
    def __init__(self, plan_node_id, col, op, split_value, plans_manager: PlansManager, score):
        super().__init__(split_value, plan_node_id, plans_manager, score)
        self.col = col
        self.op = op

    def name(self):
        return "F_Col_{}_Op_{}_value_{}".format(self.col, self.op.split(".")[-1], self.target_value)

    def __hash__(self):
        return hash((type(self), self.plan_node_id, self.target_value, self.col, self.op))

    def __eq__(self, other):
        return super().__eq__(
            other) and self.col == other.col and self.op == other.op and self.target_value == other.target_value

    def is_left_group(self, plan_node: FilterPlanNode):
        predicates = plan_node.predicates
        for predicate in predicates:
            col = predicate[0]
            op = predicate[1]
            value = predicate[2]
            if self.col == col and self.op == op and value == self.target_value:
                return True
        return False


class JoinKeyAction(OnceAction):
    def __init__(self, plan_node_id, split_key, plans_manager: PlansManager, score):
        super().__init__(split_key, plan_node_id, plans_manager, score)

    def get_cur_value(self, plan_node: JoinPlanNode):
        return plan_node.get_join_key_str()

    def name(self):
        return "J_Key_{}".format(self.target_value)


class JoinTypeAction(OnceAction):
    def __init__(self, plan_node_id, split_join_type, plans_manager: PlansManager, score):
        super().__init__(split_join_type, plan_node_id, plans_manager, score)

    def get_cur_value(self, plan_node: PlanNode):
        return plan_node.get_join_type()

    def name(self):
        return "J_Type_{}".format(self.target_value.split(".")[-1])


class TableTypeAction(OnceAction):
    def __init__(self, plan_node_id, table_type, plans_manager: PlansManager, score):
        super().__init__(table_type, plan_node_id, plans_manager, score)

    def name(self):
        return "T_Type_{}".format(self.target_value.split(".")[-1])

    def get_cur_value(self, plan_node: PlanNode):
        return plan_node.get_scan_type()


class TableNameAction(OnceAction):
    def __init__(self, plan_node_id, table_name, plans_manager: PlansManager, score):
        super().__init__(table_name, plan_node_id, plans_manager, score)

    def get_cur_value(self, plan_node: PlanNode):
        return plan_node.get_table_name()

    def name(self):
        return "T_Name_{}".format(self.target_value)
