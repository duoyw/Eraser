from RegressionFramework.NonShiftedModel.AdaptivePlanGroup import AdaptivePlanGroup
from RegressionFramework.NonShiftedModel.DecisionTree import Forest
from RegressionFramework.Plan.Plan import Plan
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.ShiftedPlanProducer.ShiftedManager import ShiftedManager, LeroShiftedManager, \
    PerfguardShiftedManager
from RegressionFramework.StaticPlanGroup import StaticPlanGroup, Group, StaticConfig
from perfguard_result_cache import perfguard_result
from plan2score import get_perfguard_result


class RegressionFramework:
    def __init__(self, plans, sqls, db, training_set_name, model, algo, leaf_ele_min_count=5):
        self._plans = plans
        # only used in single model
        self._sqls = sqls
        self.model = model
        self.db = db

        if type(self._plans[0]) != Plan:
            self._plans = self._to_plan_objects(self._plans)

        self.shifted_manager = None

        self.iod_models = []

        self.leaf_ele_min_count = leaf_ele_min_count

    def build(self):
        self.shifted_manager.build(self._plans, self._sqls)
        self.iod_models.append(AdaptivePlanGroup(min_ele_count=self.leaf_ele_min_count,
                                                 static_config=StaticConfig()))
        self.iod_models.append(AdaptivePlanGroup(min_ele_count=self.leaf_ele_min_count,
                                                 static_config=StaticConfig(scan_type_enable=True,
                                                                            table_name_enable=True)))
        self.iod_models.append(AdaptivePlanGroup(min_ele_count=self.leaf_ele_min_count,
                                                 static_config=StaticConfig(scan_type_enable=True,
                                                                            table_name_enable=True,
                                                                            join_type_enable=True)))
        self.iod_models.append(AdaptivePlanGroup(min_ele_count=self.leaf_ele_min_count,
                                                 static_config=StaticConfig(scan_type_enable=True,
                                                                            table_name_enable=True,
                                                                            join_type_enable=True,
                                                                            join_key_enable=True)))
        self.iod_models.append(AdaptivePlanGroup(min_ele_count=self.leaf_ele_min_count,
                                                 static_config=StaticConfig(scan_type_enable=True,
                                                                            table_name_enable=True,
                                                                            join_type_enable=True, join_key_enable=True,
                                                                            filter_enable=True,
                                                                            filter_col_enable=True)))
        for iod_model in self.iod_models:
            iod_model.build(self._plans)
        pass

    def _to_plan_objects(self, plans, predicts=None):
        objects = []
        for i, plan in enumerate(plans):
            objects.append(self._to_plan_object(plan, i, predicts[i] if predicts is not None else None))
        return objects

    def _to_plan_object(self, plan, idx=None, predict=None):
        return PlanFactory.get_plan_instance("pg", plan, idx, predict)

    def evaluate(self, plan1, plan2=None, predict=None):
        raise RuntimeError

    def _confidence_for_new_structure(self):
        return -1


class HyperQoRegressionFramework(RegressionFramework):

    def __init__(self, plans, sqls, db, training_set_name, model, algo="hyperqo"):
        super().__init__(plans, sqls, db, training_set_name, model, algo)

    def evaluate(self, plan1, plan2=None, predict=None):
        plan1 = self._to_plan_object(plan1)

        # ood
        is_filter_1 = self.shifted_manager.is_filter(plan1)
        is_filter_2 = self.shifted_manager.is_filter(plan2)
        if is_filter_1 or is_filter_2:
            return -1

        ratios = []

        for i in range(len(self.iod_models)):
            g1: Group = self.iod_models[i].get_group(plan1)

            if g1 is None:
                continue
            elif g1.size() < self.leaf_ele_min_count:
                continue
            else:
                ratio = g1.confidence()
                if ratio != 0 and ratio != 2:
                    ratios.append(ratio)
        if len(ratios) == 0:
            return -1

        aver = float(sum(ratios)) / len(ratios)
        return predict / aver


class LeroRegressionFramework(RegressionFramework):
    def __init__(self, plans, sqls, db, training_set_name, model, algo="lero"):
        # {(p1.id,p2.id):latency}
        super().__init__(plans, sqls, db, training_set_name, model, algo)
        self.plans_2_predicts = {}
        self.shifted_manager = LeroShiftedManager(db, training_set_name, model, algo)

    def evaluate(self, plan1, plan2=None, predict=None):
        plan1 = self._to_plan_object(plan1)
        plan2 = self._to_plan_object(plan2)

        # ood
        is_filter_1 = self.shifted_manager.is_filter(plan1)
        is_filter_2 = self.shifted_manager.is_filter(plan2)
        if is_filter_1 or is_filter_2:
            return -1
        confidences = []

        for i in range(len(self.iod_models)):
            g1: Group = self.iod_models[i].get_group(plan1)
            g2: Group = self.iod_models[i].get_group(plan2)

            if g1 is None or g2 is None:
                confidences.append(self._confidence_for_new_structure())
            elif g1.size() < self.leaf_ele_min_count or g2.size() < self.leaf_ele_min_count:
                confidences.append(1.1)
            else:
                # iod detect
                confidences.append(self._pair_group_confidence(g1, g2))
        if len(confidences) == 0:
            return -1
        return float(sum(confidences)) / len(confidences)

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
                if plan1.execution_time <= plan2.execution_time and self._select(plan1, plan2) == 0:
                    true_count += 1
                elif plan1.execution_time >= plan2.execution_time and self._select(plan1, plan2) == 1:
                    true_count += 1
                total_count += 1
        if total_count != 0:
            confidence = true_count / total_count
        else:
            confidence = 1.0
        return confidence

    def _select(self, plan1: Plan, plan2):
        return 0 if plan1.predict <= plan2.predict else 1


class PerfRegressionFramework(LeroRegressionFramework):

    def __init__(self, plans, sqls, db, training_set_name, model, algo="perfguard"):
        super().__init__(plans, sqls, db, training_set_name, model, algo)
        self.group_id_2_confidence = {}
        self.shifted_manager = PerfguardShiftedManager(db, training_set_name, model, algo)

    def _pair_group_confidence(self, group1: Group, group2: Group):
        is_same_group = group1.id == group2.id
        id1 = group1.id
        id2 = group2.id
        plans1 = group1.plans
        plans2 = group2.plans

        key = (id1, id2) if id1 < id2 else (id2, id1)
        if key in self.group_id_2_confidence:
            return self.group_id_2_confidence[key]

        results = self._compare(plans1, plans2)
        total_count = 0
        true_count = 0
        for i, plan1 in enumerate(plans1):
            start = i + 1 if is_same_group else 0
            for j in range(start, len(plans2)):
                plan2: Plan = plans2[j]
                cmp_result = results[i * len(plans2) + j]
                if plan1.execution_time <= plan2.execution_time and cmp_result == 1:
                    true_count += 1
                elif plan1.execution_time >= plan2.execution_time and cmp_result == 0:
                    true_count += 1
                total_count += 1
        if total_count != 0:
            confidence = true_count / total_count
        else:
            confidence = 1.0
        self.group_id_2_confidence[key] = confidence
        return confidence

    def _compare(self, plans1, plans2):
        plans1 = [p.plan_json for p in plans1]
        plans2 = [p.plan_json for p in plans2]

        left = []
        right = []
        for p1 in plans1:
            for p2 in plans2:
                left.append(p1)
                right.append(p2)
        return get_perfguard_result(left, right, self.model)

    def _confidence_for_new_structure(self):
        if self.db == "tpch":
            return 1
        else:
            return -1
