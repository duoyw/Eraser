import math
from queue import Queue

from RegressionFramework.Common.dotDrawer import GroupTreeDotDrawer
from RegressionFramework.SegmentModel.AdaptivaGroupAction import Action
from RegressionFramework.SegmentModel.AdaptivaGroupTree import TreeNode
from RegressionFramework.SegmentModel.PlansManeger import PlansManager
from RegressionFramework.Plan.Plan import Plan
from RegressionFramework.Plan.PlanConfig import PgNodeConfig
from RegressionFramework.SegmentModel.StaticPlanGroup import StaticPlanGroup, Group
from RegressionFramework.Common.utils import flat_depth2_list
from model import LeroModelPairWise


class AdaptivePlanGroup:
    """
    Segment model
    """

    def __init__(self, min_ele_count, delta=0.001, static_config=None, beta=0.6):
        super().__init__()
        self.static_group = StaticPlanGroup(static_config)
        self.db_node_config = PgNodeConfig
        self.key_to_static_root = {}
        self.key_to_plan_manager = {}
        # the variance thres to control split (cancel)
        self.delta = delta
        # the least number of a leaf
        self.min_ele_count = min_ele_count
        self.beta = beta
        self.score_computer: ScoreComputer = None

    def build(self, plans_for_queries, model_name=None, train_set_name=None, model=None):
        self.score_computer = ScoreComputer(model)

        print("SegmentModel building start")
        plans = flat_depth2_list(plans_for_queries)
        self.static_group.build(plans)
        key_to_group = self.static_group.key_to_group
        i = 0
        for structure_iter, group in key_to_group.items():
            # if i % (len(key_to_group) // min(10, len(key_to_group))) == 0:
            #     print("cur statis group is {}, total is {}".format(i, len(key_to_group)))
            group: Group = group
            plans = group.plans
            for p in plans:
                p.structure = self.static_group.get_group_key(p)
            root = TreeNode(group.plans)
            plan_manager = PlansManager(group.plans)
            self.key_to_static_root[structure_iter] = root
            self.key_to_plan_manager[structure_iter] = plan_manager

            # add available action
            Action.update_actions(plan_manager, root)

            leaf_nodes = Queue()
            leaf_nodes.put(root)
            count = 0
            while leaf_nodes.qsize() != 0:
                leaf: TreeNode = leaf_nodes.get()
                if self.is_stop(leaf):
                    continue
                left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count, root,
                                                                        plans_for_queries, structure_iter)
                if action is None:
                    continue

                leaf.split_action = action

                assert len(left_child) + len(right_child) == len(leaf)
                if not (not left_child.empty() and not right_child.empty()):
                    left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count, root,
                                                                            plans_for_queries, structure_iter)
                if not (not left_child.empty() and not right_child.empty()):
                    left_child, right_child, action = self._split_leaf_node(leaf, self.min_ele_count, root,
                                                                            plans_for_queries, structure_iter)
                assert not left_child.empty() and not right_child.empty()
                leaf.add_left_child(left_child)
                leaf_nodes.put(left_child)

                leaf.add_right_child(right_child)
                leaf_nodes.put(right_child)
                count += 1
            i += 1
        print("SegmentModel building end")

        self.stat_all_leafs()

    def stat_all_leafs(self):
        leaf_groups = self.get_all_leafs()
        var_and_group = [(group.variance(), len(group), group) for group in leaf_groups]
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

    def get_group_with_maximal_plans(self):
        groups = self.key_to_static_root.values()
        target = None
        for group in groups:
            group: TreeNode = group
            if target is None or (target.size() < group.size() and group.variance() > 0.022):
                target = group
        return target

    def get_group(self, plan: Plan):
        statis_key = self.static_group.get_group_key(plan)
        if statis_key not in self.key_to_static_root:
            return None
        root: TreeNode = self.key_to_static_root[statis_key]
        return self.get_group_with_root(root, plan)

    @classmethod
    def get_group_with_root(cls, root, plan: Plan):
        tree_node: TreeNode = root

        while not tree_node.is_leaf():
            action: Action = tree_node.split_action
            if cls.is_left_group(action, plan):
                tree_node = tree_node.left_child
            else:
                tree_node = tree_node.right_child

        if tree_node.empty():
            raise RuntimeError
        return tree_node

    @classmethod
    def is_left_group(cls, action: Action, plan):
        node_id_to_node = plan.node_id_to_node
        if action.plan_node_id not in node_id_to_node:
            raise RuntimeError
        plan_node = node_id_to_node[action.plan_node_id]
        if action.is_left_group(plan_node):
            return True
        return False

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

    def _split_leaf_node(self, tree_node: TreeNode, min_ele_count, root: TreeNode, plans_for_queries, structure_iter):
        best_score = math.inf
        best_action = None
        best_is_satisfy_ele_count = False
        plan_id_2_group = {}
        for i, action in enumerate(tree_node.actions):
            left_tree_node, right_tree_node = action.fake_split(tree_node)
            if left_tree_node.size() < self.min_ele_count or right_tree_node.size() < self.min_ele_count:
                continue
            score = self._score(root, tree_node, left_tree_node, right_tree_node, action, plans_for_queries, self.beta,
                                structure_iter, plan_id_2_group)
            l_size, r_size = left_tree_node.size(), right_tree_node.size()
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

        if best_action is None:
            return None, None, None
        child1, child2 = best_action.split(tree_node)
        return child1, child2, best_action

    def _is_satisfy_ele_count(self, min_ele_count, l_size, r_size):
        return l_size >= min_ele_count and r_size >= min_ele_count

    def _score(self, root: TreeNode, cur_tree_node: TreeNode, left_tree_node: TreeNode, right_tree_node: TreeNode,
               action: Action, plans_for_queries, beta, structure_iter, plan_id_2_group):
        score = 0
        if not left_tree_node.empty():
            score += left_tree_node.variance()
        if not right_tree_node.empty():
            score += right_tree_node.variance()
        return score


class AdaptivePlanGroupOptimization(AdaptivePlanGroup):

    def __init__(self, min_ele_count, delta=0.001, static_config=None, beta=0.6):
        super().__init__(min_ele_count, delta, static_config, beta)

    def _score(self, root: TreeNode, cur_tree_node: TreeNode, left_tree_node: TreeNode, right_tree_node: TreeNode,
               action: Action, plans_for_queries, beta, structure_iter, plan_id_2_group):
        score = self.score_computer.score(root, cur_tree_node, left_tree_node, right_tree_node, action,
                                          plans_for_queries, beta, structure_iter, plan_id_2_group)
        return score

    def is_stop(self, tree_node: TreeNode):
        if len(tree_node.plans) < self.min_ele_count or len(tree_node.actions) == 0:
            return True
        return False


class ScoreComputer:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.group_confidence = {}

    def score(self, root: TreeNode, cur_tree_node: TreeNode, left_tree_node: TreeNode, right_tree_node: TreeNode,
              action: Action, plans_for_queries, beta, structure_iter, plan_id_2_group):
        select_times_for_query = []
        for plans in plans_for_queries:
            id_to_win_count = {}
            for i, plan1 in enumerate(plans):
                if plan1.structure != structure_iter:
                    continue
                win_count = 0
                g1 = self._get_group_with_root(root, plan1, cur_tree_node, left_tree_node, right_tree_node, action,
                                               plan_id_2_group)

                for j in range(i + 1, len(plans)):
                    plan2 = plans[j]
                    if plan2.structure != structure_iter:
                        continue
                    g2 = self._get_group_with_root(root, plan2, cur_tree_node, left_tree_node, right_tree_node, action,
                                                   plan_id_2_group)
                    if plan1.predict < plan2.predict:
                        confidence = self._pair_group_confidence(g1, g2)
                        if confidence >= beta:
                            win_count += 1
                id_to_win_count[i] = win_count

            if len(id_to_win_count) == 0:
                choose_idx = 0
            else:
                id_win_count = sorted(id_to_win_count.items(), key=lambda x: x[1])
                choose_idx = self._get_idx_min_predict_latency_with_max_count(id_win_count, [p.predict for p in plans])
            latency = plans[choose_idx].execution_time
            select_times_for_query.append(latency)
        # compute score
        assert len(select_times_for_query) > 0
        return sum(select_times_for_query)

    def _get_group_with_root(self, root, plan: Plan, cur_tree_node: TreeNode, left_tree_node: TreeNode,
                             right_tree_node: TreeNode, action: Action, plan_id_2_group):
        if plan.plan_id in plan_id_2_group:
            return plan_id_2_group[plan.plan_id]
        leaf_node = AdaptivePlanGroup.get_group_with_root(root, plan)
        if leaf_node.id == cur_tree_node.id:
            if AdaptivePlanGroup.is_left_group(action, plan):
                return left_tree_node
            else:
                return right_tree_node
        plan_id_2_group[plan.plan_id] = leaf_node
        return leaf_node

    def _get_idx_min_predict_latency_with_max_count(self, id_win_count, predict_latencies):
        """
        :param id_win_count: [(id,win_count),(),...], sorted count by ascending order
        :param predict_latencies: [latency1,...]
        :return:
        """
        count = id_win_count[-1][1]
        candidate_ids = []
        for items in id_win_count:
            if items[1] == count:
                candidate_ids.append(items[0])

        choose_idx = -1
        choose_predict_latency = math.inf
        for idx in candidate_ids:
            if predict_latencies[idx] < choose_predict_latency:
                choose_idx = idx
                choose_predict_latency = predict_latencies[idx]
        return choose_idx

    def _pair_group_confidence(self, group1: Group, group2: Group):
        is_same_group = group1.id == group2.id
        key = self._get_group_confidence_key(group1, group2)
        if key in self.group_confidence:
            return self.group_confidence[key]

        plans1 = group1.plans
        plans2 = group2.plans

        total_count = 0
        true_count = 0
        for i, plan1 in enumerate(plans1):
            start = i + 1 if is_same_group else 0
            for j in range(start, len(plans2)):
                plan2: Plan = plans2[j]
                if plan1.execution_time <= plan2.execution_time and plan1.predict <= plan2.predict:
                    true_count += 1
                elif plan1.execution_time >= plan2.execution_time and plan1.predict >= plan2.predict:
                    true_count += 1
                total_count += 1
        if total_count != 0:
            confidence = true_count / total_count
        else:
            confidence = 1.0
        self.group_confidence[key] = confidence
        return confidence

    def _get_group_confidence_key(self, g1: TreeNode, g2: TreeNode):
        res = [g1.id, g2.id]
        res.sort()
        return "{}_{}".format(res[0], res[1])


# class LeroScoreComputer(ScoreComputer):
# 
#     def select_plan_from_model(self, plans_for_queries):
#         model: LeroModelPairWise = self.model
#         times_for_query = []
#         for plans in plans_for_queries:
#             predicts = list(self._predict(model, plans))
#             idx = predicts.index(min(predicts))
#             times_for_query.append(plans[idx]["Execution Time"])
#         return times_for_query
# 
#     def _predict(self, model: LeroModelPairWise, plans):
#         features = model.to_feature(plans)
#         return model.predict(features)
