from RegressionFramework.Common.Cache import Cache


class PerfguardResult:
    def __init__(self):
        self.perfguard_predict_cache = None

        # {(id1,id2):0/1}
        self.plan_id_2_result = {}

    def build(self, train_set_name):
        self.perfguard_predict_cache = Cache("perfguard_predict_{}".format(train_set_name), enable=False)

        # {(id1,id2):0/1}
        self.plan_id_2_result = {}
        if self.perfguard_predict_cache.exist():
            self.plan_id_2_result = self.perfguard_predict_cache.read()

    def exist(self):
        return len(self.plan_id_2_result) > 0

    def add(self, id1, id2, r):
        key = (id1, id2) if id1 < id2 else (id2, id1)
        self.plan_id_2_result[key] = r

    def get_result(self, id1, id2):
        key = self._get_key(id1, id2)
        return self.plan_id_2_result[key]

    def _get_key(self, id1, id2):
        return (id1, id2) if id1 < id2 else (id2, id1)

    def save(self):
        self.perfguard_predict_cache.save(self.plan_id_2_result)


perfguard_result = PerfguardResult()


def get_perfguard_result_manager():
    return perfguard_result
