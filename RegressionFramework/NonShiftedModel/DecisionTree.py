from chefboost import Chefboost

from RegressionFramework.Plan.PgPlan import PgScanPlanNode
from RegressionFramework.Plan.Plan import PlanNode, ScanPlanNode, JoinPlanNode, Plan
from pandas import DataFrame

from RegressionFramework.StaticPlanGroup import StaticPlanGroup, Group


class Forest:
    def __init__(self):
        self.plans = None
        self.model = None
        self.group = StaticPlanGroup()
        self.group_id_2_Tree = {}

    def build(self, plans):
        self.group.build(plans)
        all_groups = self.group.get_all_groups()
        for group in all_groups:
            group: Group = group
            tree = PlanDecisionTree()
            tree.build(group.plans)
            self.group_id_2_Tree[group.id] = tree

    def evaluate(self, plan1, plan2):
        group1: Group = self.group.get_group(plan1)
        group2: Group = self.group.get_group(plan2)

        if group1 is None or group2 is None:
            raise RuntimeError

        return self._pair_group_confidence(group1, group2)

    def _pair_group_confidence(self, group1: Group, group2: Group):
        is_same_group = group1.id == group2.id
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
        return confidence


class PlanDecisionTree:
    def __init__(self):
        self.plans = None
        self.model = None

    def build(self, plans):
        self.plans = plans
        features, columns = self._to_features(plans)

        # all plan's column is same
        columns = columns[0]
        columns.append("Decision")
        df = DataFrame(features, columns=columns)

        config = {'algorithm': 'Regression'}
        self.model = Chefboost.fit(df, config=config, target_label='Decision')

    # def _find_best_split(self):
    #     pass
    #
    # def

    def _to_features(self, plans):
        features = []
        columns = []
        for plan in plans:
            plan_features = []
            plan_columns = []
            self._recurse_plan(plan.root, plan_features, plan_columns)
            plan_features.append(plan.execution_time)
            features.append(plan_features)
            columns.append(plan_columns)
        return features, columns

    def _recurse_plan(self, node: PlanNode, features: list, columns: list):
        def _append_column(cols, k):
            cols.append("{}_{}".format(k, len(cols)))

        if node.is_scan_node():
            node: PgScanPlanNode = node
            _append_column(columns, "scanType")
            features.append(node.scan_type)
            _append_column(columns, "tbName")
            features.append(node.table_name)
            if node.predicates is not None:
                for predicate in node.predicates:
                    _append_column(columns, "filterCol")
                    features.append(predicate[0])

        elif node.is_join_node():
            node: JoinPlanNode = node
            _append_column(columns, "joinType")

            features.append(node.join_type)
            for key in node.join_key:
                _append_column(columns, "joinKey")
                features.append(key)

        for child in node.children:
            self._recurse_plan(child, features, columns)
