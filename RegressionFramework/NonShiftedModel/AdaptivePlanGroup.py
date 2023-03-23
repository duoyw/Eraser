import math
from queue import Queue

from RegressionFramework.Common.dotDrawer import GroupTreeDotDrawer
from RegressionFramework.NonShiftedModel.AdaptivaGroupAction import Action
from RegressionFramework.NonShiftedModel.AdaptivaGroupTree import TreeNode
from RegressionFramework.NonShiftedModel.PlansManeger import PlansManager
from RegressionFramework.Plan.Plan import PlanNode, Plan
from RegressionFramework.Plan.PlanConfig import PgNodeConfig
from RegressionFramework.StaticPlanGroup import StaticPlanGroup, Group


class AdaptivePlanGroup:
    def __init__(self, delta=0.001, min_ele_count=5, static_config=None):
        super().__init__()
        self.static_group = StaticPlanGroup(static_config)
        self.db_node_config = PgNodeConfig
        self.key_to_static_root = {}
        self.key_to_plan_manager = {}
        self.delta = delta
        # the least number of a leaf
        self.min_ele_count = min_ele_count

    def build(self, plans, model_name=None, train_set_name=None):
        print("AdaptivePlanGroup start")
        self.static_group.build(plans)
        key_to_group = self.static_group.key_to_group
        i = 0
        for key, group in key_to_group.items():
            # log
            if i % (len(key_to_group) // min(5, len(key_to_group))) == 0:
                print("cur statis group is {}, total is {}".format(i, len(key_to_group)))

            group: Group = group
            root = TreeNode(group.plans)
            plan_manager = PlansManager(group.plans)
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
                left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count)
                leaf.split_action = action

                assert len(left_child) + len(right_child) == len(leaf)
                if not (not left_child.empty() and not right_child.empty()):
                    left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count)
                if not (not left_child.empty() and not right_child.empty()):
                    left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count)
                assert not left_child.empty() and not right_child.empty()
                leaf.add_left_child(left_child)
                leaf_nodes.put(left_child)

                leaf.add_right_child(right_child)
                leaf_nodes.put(right_child)
                count += 1
            i += 1
        print("AdaptivePlanGroup end")

        self.stat_all_leafs()

    def stat_all_leafs(self):
        leaf_groups = self.get_all_leafs()
        var_and_group = [(group.variance(), len(group), group) for group in leaf_groups]
        sorted_by_var = sorted(var_and_group, key=lambda x: -x[0])
        sorted_by_len = sorted(var_and_group, key=lambda x: -x[1])
        return var_and_group

    def draw_dot(self):
        dot = GroupTreeDotDrawer.get_plan_dot_str(self)
        return dot

    def is_stop(self, tree_node: TreeNode):
        if tree_node.variance() < self.delta or len(tree_node.actions) == 0:
            return True
        return False

    def get_all_leafs(self):
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
        statis_key = self.static_group.get_group_key(plan)
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
        statis_key = self.static_group.get_group_key(plan)
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

    def _split_leaf_node(self, tree_node: TreeNode, min_ele_count):
        best_score = math.inf
        best_action = None
        best_is_satisfy_ele_count = False
        for action in tree_node.actions:
            score, l_size, r_size = action.score_if_split(tree_node)
            if best_action is None:
                best_score, best_action = score, action

            cur_is_satisfy_ele_count = self._is_satisfy_ele_count(min_ele_count, l_size, r_size)
            if not best_is_satisfy_ele_count:
                if cur_is_satisfy_ele_count:
                    best_score, best_action = score, action
                    best_is_satisfy_ele_count = True
                elif best_score > score:
                    best_score, best_action = score, action
            else:
                if cur_is_satisfy_ele_count and best_score > score:
                    best_score, best_action = score, action

        assert best_action is not None
        child1, child2 = best_action.split(tree_node)
        return child1, child2, best_action

    def _is_satisfy_ele_count(self, min_ele_count, l_size, r_size):
        return l_size >= min_ele_count and r_size >= min_ele_count
