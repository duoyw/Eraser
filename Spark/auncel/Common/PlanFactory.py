from UncertantyModel.SparkPlan import SparkPlan
from model_config import DbType


class PlanFactory:
    @classmethod
    def get_plan_instance(cls, db_type: DbType, plan_json, plan_id=None, predict=None):
        if db_type == DbType.SPARK:
            return SparkPlan(plan_json, plan_id, predict)
        else:
            raise RuntimeError

    @classmethod
    def get_plan_node_instance(cls, db_type: DbType, node_json, plan_id=None, predict=None):
        if db_type == DbType.SPARK:
            return SparkPlan.to_node(node_json)
        else:
            raise RuntimeError
