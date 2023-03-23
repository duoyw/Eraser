import numpy as np

from RegressionFramework.StaticPlanGroup import Group


class TreeNode(Group):
    def __init__(self, plans):
        super().__init__(plans)
        self.actions = set()
        self.left_child = None
        self.right_child = None
        self.split_action = None

    def add_action(self, action):
        self.actions.add(action)

    def __len__(self):
        return len(self.plans)

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def add_left_child(self, child):
        self.left_child = child

    def add_right_child(self, child):
        self.right_child = child

    def empty(self):
        return len(self.plans) == 0

    def get_plan_nodes(self, node_id):
        return
