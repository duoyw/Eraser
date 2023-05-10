import copy

from RegressionFramework.SegmentModel.AdaptivePlanGroup import AdaptivePlanGroup, AdaptivePlanGroupOptimization
from RegressionFramework.Plan.Plan import Plan
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.UnexpectedPlanExplorer.ShiftedManager import LeroShiftedManager
from RegressionFramework.SegmentModel.StaticPlanGroup import Group, StaticConfig
from RegressionFramework.Common.utils import flat_depth2_list
from RegressionFramework.config import min_leaf_ele_count, decision_for_new_structure


class EraserManager:
    def __init__(self, plans, sqls, db, training_set_name, model, algo, mode, leaf_ele_min_count,
                 plans_for_queries=None):
        self._plans = plans
        self.plans_for_queries = copy.copy(plans_for_queries)
        # only used in single model
        self._sqls = sqls
        self.model = model
        self.db = db
        self.algo = algo

        # static or dynamic
        self.mode = mode

        plan_idx = 0
        if plans_for_queries is not None and type(plans_for_queries[0][0]) != Plan:
            for i in range(len(plans_for_queries)):
                plan_objects = []
                for plan in plans_for_queries[i]:
                    plan_objects.append(self._to_plan_object(plan, plan_idx, None))
                    plan_idx += 1
                self.plans_for_queries[i] = plan_objects
            self._plans = flat_depth2_list(self.plans_for_queries)
        else:
            if type(self._plans[0]) != Plan:
                self._plans = self._to_plan_objects(self._plans)

        self.shifted_manager = None

        self.iod_models = []

        self.leaf_ele_min_count = leaf_ele_min_count

    def iod_model_factory(self, min_ele_count, static_config):
        return AdaptivePlanGroup(min_ele_count=min_ele_count, static_config=static_config)

    def build(self):
        self.shifted_manager.build(self._plans, self._sqls)
        self.iod_models.append(self.iod_model_factory(min_ele_count=self.leaf_ele_min_count,
                                                      static_config=StaticConfig()))
        self.iod_models.append(self.iod_model_factory(min_ele_count=self.leaf_ele_min_count,
                                                      static_config=StaticConfig(scan_type_enable=True,
                                                                                 table_name_enable=True)))
        self.iod_models.append(self.iod_model_factory(min_ele_count=self.leaf_ele_min_count,
                                                      static_config=StaticConfig(scan_type_enable=True,
                                                                                 table_name_enable=True,
                                                                                 join_type_enable=True)))
        self.iod_models.append(self.iod_model_factory(min_ele_count=self.leaf_ele_min_count,
                                                      static_config=StaticConfig(scan_type_enable=True,
                                                                                 table_name_enable=True,
                                                                                 join_type_enable=True,
                                                                                 join_key_enable=True)))
        self.iod_models.append(self.iod_model_factory(min_ele_count=self.leaf_ele_min_count,
                                                      static_config=StaticConfig(scan_type_enable=True,
                                                                                 table_name_enable=True,
                                                                                 join_type_enable=True,
                                                                                 join_key_enable=True,
                                                                                 filter_enable=True,
                                                                                 filter_col_enable=True)))
        self._build_iod_model()

    def _build_iod_model(self):
        for iod_model in self.iod_models:
            iod_model.build(self._plans, self.model)

    def _to_plan_objects(self, plans, predicts=None):
        objects = []
        for i, plan in enumerate(plans):
            objects.append(self._to_plan_object(plan, i, predicts[i] if predicts is not None else None))
        return objects

    def _to_plan_object(self, plan, idx=None, predict=None):
        return PlanFactory.get_plan_instance("pg", plan, idx, predict)

    def evaluate(self, plan1, plan2=None, predict=None, ood_thres=None):
        raise RuntimeError

    def _confidence_for_new_structure(self):
        raise RuntimeError


class LeroEraserManager(EraserManager):
    def __init__(self, plans, sqls, db, training_set_name, model, algo="lero", mode="static", config_dict=None,
                 plans_for_queries=None):
        super().__init__(plans, sqls, db, training_set_name, model, algo, mode, min_leaf_ele_count,
                         plans_for_queries=plans_for_queries)
        self.plans_2_predicts = {}
        self.shifted_manager = LeroShiftedManager(db, training_set_name, model, algo)
        self.config_dict = config_dict

    def iod_model_factory(self, min_ele_count, static_config):
        return AdaptivePlanGroupOptimization(min_ele_count, static_config=static_config)

    def _build_iod_model(self):
        for iod_model in self.iod_models:
            iod_model.build(self.plans_for_queries, self.model)

    def evaluate(self, plan1, plan2=None, predict=None, ood_thres=None):
        plan1 = self._to_plan_object(plan1)
        plan2 = self._to_plan_object(plan2)

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
                # conservative strategy
                confidences.append(1.1)
            else:
                confidences.append(self._pair_group_confidence(g1, g2))
        assert len(confidences) > 0
        return float(sum(confidences)) / len(confidences)

    def _confidence_for_new_structure(self):
        return -1

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
