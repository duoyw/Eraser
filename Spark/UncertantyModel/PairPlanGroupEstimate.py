from Common.PlanFactory import PlanFactory
from Plan import Plan
from UncertantyModel.StaticPlanGroup import Group
from UncertantyModel.UncertaintyEstimate import PlanGroupEstimate
from model_config import db_type
from utils import json_str_to_json_obj, swap


class PairPlanGroupEstimate(PlanGroupEstimate):
    def __init__(self, plans, predicts, model_name, train_set_name,dataset=None, confidence_model=None):
        super().__init__(plans, predicts, model_name, train_set_name,dataset, confidence_model)
        groups = self.plan_group.get_all_groups()
        i = 0
        for group in groups:
            group.id = i
            i += 1

        self._key_to_pair_group_confidence = {}

    def estimate(self, plan1, plan2):
        if isinstance(plan1, str):
            plan1 = json_str_to_json_obj(plan1)
        if isinstance(plan2, str):
            plan2 = json_str_to_json_obj(plan2)

        plan1 = PlanFactory.get_plan_instance(db_type, plan1, predict=None)
        plan2 = PlanFactory.get_plan_instance(db_type, plan2, predict=None)

        group1: Group = self.get_group(plan1)
        group2: Group = self.get_group(plan2)

        if group1 is None or group2 is None:
            return -1

        return self._pair_group_confidence(group1, group2)

    def _pair_group_confidence(self, group1: Group, group2: Group):
        pair_group_key = self._get_pair_group_key(group1, group2)
        if pair_group_key in self._key_to_pair_group_confidence:
            return self._key_to_pair_group_confidence[pair_group_key]

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
        self._key_to_pair_group_confidence[pair_group_key] = confidence
        return confidence

    def _get_pair_group_key(self, group1: Group, group2: Group):
        id1 = group1.id
        id2 = group2.id
        if id1 > id2:
            id1, id2 = swap(id1, id2)
        return "{}_{}".format(id1, id2)
