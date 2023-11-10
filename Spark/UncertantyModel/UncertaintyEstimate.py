import json

from Common.DotDrawer import draw_dot_spark_plan, GroupTreeDotDrawer
from Common.PlanFactory import PlanFactory
from Plan import Plan
from UncertantyModel.AdaptivePlanGroup import AdaptivePlanGroup
from UncertantyModel.StaticPlanGroup import StaticPlanGroup, Group
from feature import json_str_to_json_obj
from model import AuncelModel, AuncelModelPairConfidenceWise
from model_config import confidence_estimate_type, ConfidenceEstimateType, db_type, GroupEnable, uncertainty_threshold, \
    confidence_model_type, ModelType
from test_script.config import SEP
from utils import cal_accuracy


class ConfidenceEstimate:
    pass


class PlanGroupEstimate(ConfidenceEstimate):
    def __init__(self, plans, predicts, model_name, train_set_name, dataset=None, confidence_model: AuncelModel = None):
        if isinstance(plans[0], str):
            self.plans = [json_str_to_json_obj(p) for p in plans]
        self.plans = [PlanFactory.get_plan_instance(db_type, self.plans[i], i, predicts[i]) for i in range(len(plans))]
        self.confidence_model = confidence_model
        self.dataset = dataset
        if confidence_estimate_type == ConfidenceEstimateType.STATIC:
            self.plan_group = StaticPlanGroup(struct_enable=GroupEnable.struct_enable,
                                              scan_type_enable=GroupEnable.scan_type_enable,
                                              table_name_enable=GroupEnable.table_name_enable,
                                              join_type_enable=GroupEnable.join_type_enable,
                                              join_key_enable=GroupEnable.join_key_enable,
                                              filter_enable=GroupEnable.filter_enable,
                                              filter_col_enable=GroupEnable.filter_col_enable,
                                              filter_op_enable=GroupEnable.filter_op_enable,
                                              filter_value_enable=GroupEnable.filter_value_enable)
            # self.plan_group = StaticPlanGroup()
        elif confidence_estimate_type == ConfidenceEstimateType.ADAPTIVE or \
                confidence_estimate_type == ConfidenceEstimateType.ADAPTIVE_MODEL:
            self.plan_group = AdaptivePlanGroup(struct_enable=GroupEnable.struct_enable,
                                                scan_type_enable=GroupEnable.scan_type_enable,
                                                table_name_enable=GroupEnable.table_name_enable,
                                                join_type_enable=GroupEnable.join_type_enable,
                                                join_key_enable=GroupEnable.join_key_enable,
                                                filter_enable=GroupEnable.filter_enable,
                                                filter_col_enable=GroupEnable.filter_col_enable,
                                                filter_op_enable=GroupEnable.filter_op_enable,
                                                filter_value_enable=GroupEnable.filter_value_enable)
        else:
            raise RuntimeError

        self._init(model_name, train_set_name)
        self.plan_id_to_confidence_predict = {}

    def _init(self, model_name, train_set_name):
        self.plan_group.group(self.plans, model_name, train_set_name)

    def estimate(self, plan, predict):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
        plan = PlanFactory.get_plan_instance(db_type, plan, predict=predict)
        group: Group = self.get_group(plan)

        if group is None:
            return -1, predict, predict, predict

        if self.confidence_model is not None:
            valid_candidate_plans = self.filter_plans_by_confidence_model(plan, group.plans)
            if len(valid_candidate_plans) == 0:
                return -2, predict, predict, predict
            group = Group(plans=valid_candidate_plans)

        if group.is_error_predict_bias(plan.predict):
            return -2, predict, predict, predict

        return group.confidence(), *(group.adjust_predict(predict))

    def filter_plans_by_confidence_model(self, query_plan: Plan, candidate_plans):
        if confidence_model_type == ModelType.MSE_TREE_CONV:
            return self.filter_plans_by_single_confidence_model(query_plan, candidate_plans)
        elif confidence_model_type == ModelType.TREE_CONV:
            return self.filter_plans_by_pair_confidence_model(query_plan, candidate_plans)
        else:
            print("self.confidence_model type is {}".format(type(self.confidence_model)))
            raise RuntimeError

    def filter_plans_by_pair_confidence_model(self, query_plan: Plan, candidate_plans):
        model: AuncelModelPairConfidenceWise = self.confidence_model
        query_plan_predict = model.predict_confidence(model.to_feature([query_plan.plan_json], self.dataset))[0]

        if candidate_plans[0].plan_id not in self.plan_id_to_confidence_predict:
            candidate_plans_str = [p.plan_json for p in candidate_plans]
            candidate_plans_predicts = list(
                model.predict_confidence(model.to_feature(candidate_plans_str, self.dataset)))
            for i, candidate_plan in enumerate(candidate_plans):
                self.plan_id_to_confidence_predict[candidate_plan.plan_id] = candidate_plans_predicts[i]
        else:
            candidate_plans_predicts = []
            for i, candidate_plan in enumerate(candidate_plans):
                candidate_plans_predicts.append(self.plan_id_to_confidence_predict[candidate_plan.plan_id])

        filter_count = 0
        # true positive ratio
        tp_ratio = 0
        # false negative ratio
        fn_count = 0
        valid_plan = []
        for i, candidate_plan in enumerate(candidate_plans):
            candidate_plans_predict = candidate_plans_predicts[i]
            if model.is_same_buckets(query_plan_predict, candidate_plans_predict):
                valid_plan.append(candidate_plan)
            else:
                query_plan_accuracy = cal_accuracy(query_plan_predict, query_plan.execution_time)
                candidate_plan_accuracy = cal_accuracy(candidate_plans_predict, candidate_plan.execution_time)
                if abs(query_plan_accuracy - candidate_plan_accuracy) < model.diff_thres:
                    fn_count += 1
                filter_count += 1
        print("filter_plans_by_confidence_model count is {}, total is {}".format(filter_count, len(candidate_plans)))
        return valid_plan

    def filter_plans_by_single_confidence_model(self, query_plan: Plan, candidate_plans):
        model: AuncelModel = self.confidence_model

        if candidate_plans[0].plan_id not in self.plan_id_to_confidence_predict:
            candidate_plans_str = [p.plan_json for p in candidate_plans]
            candidate_plans_predicts = list(
                model.predict_confidence(model.to_feature(candidate_plans_str, self.dataset)))
            for i, candidate_plan in enumerate(candidate_plans):
                self.plan_id_to_confidence_predict[candidate_plan.plan_id] = candidate_plans_predicts[i]
        else:
            candidate_plans_predicts = []
            for i, candidate_plan in enumerate(candidate_plans):
                candidate_plans_predicts.append(self.plan_id_to_confidence_predict[candidate_plan.plan_id])

        filter_count = 0
        # true positive ratio
        tp_count = 0
        # false negative ratio
        fn_count = 0
        valid_plan = []
        for i, candidate_plan in enumerate(candidate_plans):
            candidate_plans_predict = candidate_plans_predicts[i]
            if candidate_plans_predict >= uncertainty_threshold:
                valid_plan.append(candidate_plan)
                if cal_accuracy(candidate_plan.predict, candidate_plan.execution_time) < uncertainty_threshold:
                    tp_count += 1
            else:
                if cal_accuracy(candidate_plan.predict, candidate_plan.execution_time) > uncertainty_threshold:
                    fn_count += 1
                filter_count += 1
        print("filter_plans_by_confidence_model count is {}, total is {}".format(filter_count, len(candidate_plans)))
        print("false negative count is {}, total is {}".format(fn_count, len(candidate_plans)))
        print("true positive count is {}, total is {}".format(tp_count, len(candidate_plans)))
        return valid_plan

    def get_group(self, plan):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
            plan = PlanFactory.get_plan_instance(db_type, plan)
        return self.plan_group.get_group(plan)

    def get_adaptive_split_path(self, plan):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
            plan = PlanFactory.get_plan_instance(db_type, plan)
        if isinstance(self.plan_group, AdaptivePlanGroup):
            return AdaptivePlanGroup(self.plan_group).get_split_path(plan)
        else:
            print("get_adaptive_split_path is only used by AdaptivePlanGroup")
            return None

    def draw_self_and_group(self, plan):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
        self_dot_str = draw_dot_spark_plan(plan["Plan"])

        group: Group = self.get_group(plan)
        if group is None:
            return self_dot_str, None
        return self_dot_str, group.draw()

    def statistic_predict_label_relation(self):
        less_sum = 0
        less_count = 0
        more_sum = 0
        more_count = 0
        for group in self.plan_group.key_to_group.values():
            res = group.compare()
            less_sum += res[0]
            less_count += res[1]
            more_sum += res[2]
            more_count += res[3]
        return less_sum, less_count, more_sum, more_count

    def stat_all_group_confidences(self):
        return self.plan_group.stat_all_group_confidences()

    def save_leaf_nodes(self, file):
        # [[static1_plans],[static2_plans]...]
        res = {}
        leaf_groups_for_static_trees = self.get_all_groups_for_static_root()
        i = 0
        with open(file, "w") as f:
            for groups in leaf_groups_for_static_trees:
                for group in groups:
                    line = "struct_{}{}".format(i, SEP) + SEP.join([p.get_plan_json_str() for p in group.plans])
                    f.write(line + "\n")
                i += 1
        exit()

    def draw_dot(self):
        return GroupTreeDotDrawer.get_plan_dot_str(self.plan_group)

    def get_all_groups_for_static_root(self):
        return self.plan_group.get_all_groups_for_static_root()


class FirstLossEstimate:
    pass
