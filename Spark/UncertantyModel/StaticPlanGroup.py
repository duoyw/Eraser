from Plan import PlanNode, FilterPlanNode, Plan, JoinPlanNode
import numpy as np

from utils import cal_ratio


class StaticPlanGroup:
    def __init__(self, struct_enable=True, scan_type_enable=True, table_name_enable=True,
                 join_type_enable=True, join_key_enable=False,
                 filter_enable=True, filter_col_enable=True, filter_op_enable=True, filter_value_enable=False):
        self.struct_enable = struct_enable

        self.scan_type_enable = scan_type_enable
        self.table_name_enable = table_name_enable

        self.join_type_enable = join_type_enable
        self.join_key_enable = join_key_enable

        self.filter_enable = filter_enable
        self.filter_col_enable = filter_col_enable
        self.filter_op_enable = filter_op_enable
        self.filter_value_enable = filter_value_enable

        self.key_to_group = {}

    def group(self, plans, model_name=None, train_set_name=None):
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

    def get_group_key(self, plan):
        key = []
        self._recurse_plan(plan.root, key)
        return "".join(key)

    def get_all_groups(self):
        return list(self.key_to_group.values())

    def stat_all_group_confidences(self):
        confidences = []
        for group in self.key_to_group.values():
            confidences.append((group.confidence(), group))
        confidences = sorted(confidences, key=lambda x: -x[0])
        return confidences

    def stat_groups_variance(self, confidence_larger=0.5, plan_count_larger=2):
        res = []
        for group in self.key_to_group.values():
            if group.confidence() > confidence_larger and group.size() > plan_count_larger:
                res.append((group.variance(), group))
        res = sorted(res, key=lambda x: x[0])
        return res

    def _recurse_plan(self, node: PlanNode, key: list):
        node_type = node.node_type

        if node.is_filter_node():
            key.append(self.get_filter_key(node))
        elif node.is_scan_node():
            key.append(self.get_table_key(node))
        elif node.is_join_node():
            key.append(self.get_join_key(node))
        else:
            key.append(node_type)

        if len(node.children) > 0:
            children = node.children
            for idx, child in enumerate(children):
                key.append(str(idx))
                self._recurse_plan(child, key)

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
    def __init__(self, plans=None):
        if plans is not None:
            self.plans = plans
        else:
            self.plans = []
        self._variance = None

    def add_plan(self, plan):
        self._variance = None
        self.plans.append(plan)

    def confidence(self):
        accuracy = 0.0
        for plan in self.plans:
            plan: Plan = plan
            time = plan.execution_time
            diff = abs(plan.predict - time)
            accuracy += 1.0 - min(diff / time, 1.0)
        return accuracy / len(self.plans)

    def is_error_predict_bias(self, predict):
        predicts = [p.predict for p in self.plans]
        aver_predict = np.array(predicts).mean()
        std_predict = np.array(predicts).std()
        if std_predict < 0.00001 and abs(predict - aver_predict) / aver_predict > 0.1:
            return False
        return False

    def adjust_predict(self, predict):
        predicts = [p.predict for p in self.plans]
        execution_times = [p.execution_time for p in self.plans]
        if len(self.plans) == 1:
            aver_predict = np.array(predicts).mean()
            aver_execution_times = np.array(execution_times).mean()
            if abs(predict - aver_predict) / aver_predict < 0.1:
                return aver_execution_times
        return predict

    def size(self):
        return len(self.plans)

    def variance(self):
        if self._variance is not None:
            return self._variance

        return self.variance_no_cache()

    def variance_no_cache(self):
        assert len(self.plans) > 0
        ratios = []
        for plan in self.plans:
            time = plan.execution_time
            ratios.append(plan.predict / time)

        self._variance = np.var(np.array(ratios))
        return self._variance

    def draw(self):
        res = []
        for plan in self.plans:
            res.append(plan.draw_dot())
        return res

    def compare(self):
        """
        :return: less count, more count. less count: the number of predict value lower than actual
        """
        less_count = 0
        less_sum = 0
        more_count = 0
        more_sum = 0
        for plan in self.plans:
            predict = plan.predict
            actual = plan.execution_time
            diff = abs(predict - actual)
            if predict < actual:
                less_count += 1
                less_sum += diff
            elif predict > actual:
                more_count += 1
                more_sum += diff
        return less_sum, less_count, more_sum, more_count




# class Group:
#     def __init__(self, plans=None):
#         self.plans = []
#         self._variance = None
#         self.min_ratio = float("inf")
#         self.max_ratio = -float("inf")
#         self.ratios = []
#
#         if plans is not None:
#             for p in plans:
#                 self.add_plan(p)
#
#     def add_plan(self, plan):
#         self._variance = None
#         self.plans.append(plan)
#         ratio = cal_ratio(plan.predict, plan.execution_time)
#         self.min_ratio = min(self.min_ratio, ratio)
#         self.max_ratio = min(self.max_ratio, ratio)
#         self.ratios.append(ratio)
#
#     def confidence(self):
#         accuracy = 0.0
#         for plan in self.plans:
#             plan: Plan = plan
#             time = plan.execution_time
#             diff = abs(plan.predict - time)
#             accuracy += 1.0 - min(diff / time, 1.0)
#         return accuracy / len(self.plans)
#
#     def is_error_predict_bias(self, predict):
#         predicts = [p.predict for p in self.plans]
#         aver_predict = np.array(predicts).mean()
#         std_predict = np.array(predicts).std()
#         if std_predict < 0.00001 and abs(predict - aver_predict) / aver_predict > 0.1:
#             return False
#         return False
#
#     def adjust_predict(self, predict):
#         # predicts = [p.predict for p in self.plans]
#         # execution_times = [p.execution_time for p in self.plans]
#         # if len(self.plans) == 1:
#         #     aver_predict = np.array(predicts).mean()
#         #     aver_execution_times = np.array(execution_times).mean()
#         #     if abs(predict - aver_predict) / aver_predict < 0.1:
#         #         return aver_execution_times
#         # return predict
#
#         predict = predict[0]
#         mean_predict = np.array([predict / r for r in self.ratios]).mean()
#         assert not np.isnan(mean_predict)
#         return predict / self.max_ratio, predict / self.min_ratio, mean_predict
#
#     def size(self):
#         return len(self.plans)
#
#     def variance(self):
#         if self._variance is not None:
#             return self._variance
#
#         return self.variance_no_cache()
#
#     def variance_no_cache(self):
#         assert len(self.plans) > 0
#         ratios = []
#         for plan in self.plans:
#             time = plan.execution_time
#             ratios.append(plan.predict / time)
#
#         self._variance = np.var(np.array(ratios))
#         return self._variance
#
#     def draw(self):
#         res = []
#         for plan in self.plans:
#             res.append(plan.draw_dot())
#         return res
#
#     def compare(self):
#         """
#         :return: less count, more count. less count: the number of predict value lower than actual
#         """
#         less_count = 0
#         less_sum = 0
#         more_count = 0
#         more_sum = 0
#         for plan in self.plans:
#             predict = plan.predict
#             actual = plan.execution_time
#             diff = abs(predict - actual)
#             if predict < actual:
#                 less_count += 1
#                 less_sum += diff
#             elif predict > actual:
#                 more_count += 1
#                 more_sum += diff
#         return less_sum, less_count, more_sum, more_count
