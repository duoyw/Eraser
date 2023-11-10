import json

from test_script.config import SEP
from utils import flat_depth2_list, get_training_pair


def _load_accuracy_pairwise_plans_cross_plan_with_filter(path, k):
    from Common.PlanFilter import PlanFilter
    X1, X2 = [], []
    struct_2_plans = {}
    with open(path) as f:
        line = f.readline()
        while line is not None and line != "":
            values = line.split(SEP)
            name = values[0]
            if name not in struct_2_plans:
                struct_2_plans[name] = []
            plans = values[1:]
            # plans = plans[0:min(len(plans), k)]
            struct_2_plans[name] += plans
            # all_plans += plans[0:min(len(plans), k)]
            line = f.readline()

    for plans in struct_2_plans.values():
        plan_filter = PlanFilter(plans)
        plans = plan_filter.compress()
        plans = [json.dumps(p.plan_json) for p in plans]
        if len(plans) >= 2:
            x1, x2 = get_training_pair(plans)
            X1 += x1
            X2 += x2
    return X1, X2
