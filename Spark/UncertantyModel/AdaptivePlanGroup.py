import math
from queue import Queue

from Common.Cache import AdaptiveGroupCache
from Common.PlanConfig import SparkNodeConfig
from Common.TimeStatistic import TimeStatistic
from Plan import PlanNode, Plan
from UncertantyModel.AdaptivaGroupAction import FilterColAction, JoinKeyAction, JoinTypeAction, TableTypeAction, \
    TableNameAction, Action
from UncertantyModel.AdaptivaGroupTree import TreeNode
from UncertantyModel.PlansManeger import PlansManager
from UncertantyModel.StaticPlanGroup import StaticPlanGroup, Group
from model_config import GroupEnable


class AdaptivePlanGroup(StaticPlanGroup):
    def __init__(self, delta=0.001, struct_enable=GroupEnable.struct_enable,
                 scan_type_enable=GroupEnable.scan_type_enable, table_name_enable=GroupEnable.table_name_enable,
                 join_type_enable=GroupEnable.join_type_enable, join_key_enable=GroupEnable.join_key_enable,
                 filter_enable=GroupEnable.filter_enable, filter_col_enable=GroupEnable.filter_col_enable,
                 filter_op_enable=GroupEnable.filter_op_enable, filter_value_enable=GroupEnable.filter_value_enable):
        super().__init__(struct_enable, scan_type_enable, table_name_enable, join_type_enable, join_key_enable,
                         filter_enable, filter_col_enable, filter_op_enable, filter_value_enable)
        self.db_node_config = SparkNodeConfig
        self.key_to_static_root = {}
        self.key_to_plan_manager = {}
        self.delta = delta

    def group(self, plans, model_name=None, train_set_name=None):
        cache = AdaptiveGroupCache(model_name, train_set_name,
                                   self.struct_enable, self.scan_type_enable, self.table_name_enable,
                                   self.join_type_enable, self.join_key_enable, self.filter_enable,
                                   self.filter_col_enable,
                                   self.filter_op_enable, self.filter_value_enable, enable=False)
        if cache.exist():
            res = cache.read()
            self.key_to_static_root = res[0]
        else:
            super().group(plans)
            i = 0
            for key, group in self.key_to_group.items():
                group: Group = group
                TimeStatistic.start("manager")
                root = TreeNode(group.plans)
                plan_manager = PlansManager(group.plans)
                TimeStatistic.end("manager")
                self.key_to_static_root[key] = root
                self.key_to_plan_manager[key] = plan_manager

                # add available action
                Action.update_actions(plan_manager, root)

                leaf_nodes = Queue()
                leaf_nodes.put(root)
                count = 0
                while leaf_nodes.qsize() != 0:
                    leaf: TreeNode = leaf_nodes.get()
                    if self.is_stop(leaf):
                        continue
                    TimeStatistic.start("_split_leaf_node")
                    left_child, right_child, action = self._split_leaf_node(leaf)
                    leaf.split_action = action
                    TimeStatistic.end("_split_leaf_node")

                    assert len(left_child) + len(right_child) == len(leaf)
                    if not (not left_child.empty() and not right_child.empty()):
                        left_child, right_child, action = self._split_leaf_node(leaf)
                    if not (not left_child.empty() and not right_child.empty()):
                        left_child, right_child, action = self._split_leaf_node(leaf)
                    assert not left_child.empty() and not right_child.empty()
                    leaf.add_left_child(left_child)
                    leaf_nodes.put(left_child)

                    leaf.add_right_child(right_child)
                    leaf_nodes.put(right_child)
                    count += 1
                i += 1
            cache.save([self.key_to_static_root])

        # self.stat_leaf_groups()

    def remove_plan_json_for_cache(self, node: PlanNode):
        pass

    def _recurse_remove_plan_json_for_cache(self, node: PlanNode):
        node.node_json = None
        for child in node.children:
            self._recurse_remove_plan_json_for_cache(child)

    def stat_leaf_groups(self):
        leaf_groups = self.get_all_groups()
        var_and_group = [(group.variance(), len(group), group) for group in leaf_groups]
        sorted_by_var = sorted(var_and_group, key=lambda x: -x[0])
        sorted_by_len = sorted(var_and_group, key=lambda x: -x[1])
        sorted_by_var[30][2].variance_no_cache()
        p1: Plan = sorted_by_var[0][2].plans[0]
        p2 = sorted_by_var[0][2].plans[1]
        p1.draw_dot()
        p2.draw_dot()
        return var_and_group

    def stat_all_group_confidences(self):
        groups = self.get_all_groups()
        confidences = []
        for group in groups:
            confidences.append((group.confidence(), group))
        confidences = sorted(confidences, key=lambda x: -x[0])
        return confidences

    def is_stop(self, tree_node: TreeNode):
        if tree_node.variance() < self.delta or len(tree_node.actions) == 0:
            return True
        return False

    def get_all_groups(self):
        leaf_nodes = []
        for root in self.key_to_static_root.values():
            leaf_nodes += self._recurse_for_all_leafs(root)
        return leaf_nodes

    def get_all_groups_for_static_root(self):
        """
        :return: [[static_1_plans],[static_2_plans],]
        """
        leaf_nodes = []
        for root in self.key_to_static_root.values():
            leaf_nodes.append(self._recurse_for_all_leafs(root))
        return leaf_nodes

    def _recurse_for_all_leafs(self, tree_node: TreeNode):
        if tree_node.is_leaf():
            return [tree_node]
        leafs = []
        leafs += [] if tree_node.left_child is None else self._recurse_for_all_leafs(tree_node.left_child)
        leafs += [] if tree_node.right_child is None else self._recurse_for_all_leafs(tree_node.right_child)
        return leafs

    def get_group(self, plan: Plan):
        statis_key = super().get_group_key(plan)
        if statis_key not in self.key_to_static_root:
            return None
        tree_node: TreeNode = self.key_to_static_root[statis_key]

        while not tree_node.is_leaf():
            action: Action = tree_node.split_action
            node_id_to_node = plan.node_id_to_node
            plan_node = node_id_to_node[action.plan_node_id]

            if action.is_left_group(plan_node):
                tree_node = tree_node.left_child
            else:
                tree_node = tree_node.right_child

        if tree_node.empty():
            print("tree_node is empty")
            return None
        return tree_node

    def get_split_path(self, plan: Plan):
        statis_key = super().get_group_key(plan)
        if statis_key not in self.key_to_static_root:
            return None
        tree_node: TreeNode = self.key_to_static_root[statis_key]

        paths = []
        while not tree_node.is_leaf():
            action: Action = tree_node.split_action
            paths.append(action)
            node_id_to_node = plan.node_id_to_node
            plan_node = node_id_to_node[action.plan_node_id]

            if action.is_left_group(plan_node):
                tree_node = tree_node.left_child
            else:
                tree_node = tree_node.right_child

        if tree_node.empty():
            print("tree_node is empty")
            return None
        return paths

    def _split_leaf_node(self, tree_node: TreeNode):
        best_score = math.inf
        best_action = None
        for action in tree_node.actions:
            TimeStatistic.start("score_if_split")
            score = action.score_if_split(tree_node)
            TimeStatistic.end("score_if_split")
            if score < best_score:
                best_score = score
                best_action = action
        assert best_action is not None
        TimeStatistic.start(" best_action.split")
        child1, child2 = best_action.split(tree_node)
        TimeStatistic.end(" best_action.split")
        return child1, child2, best_action
