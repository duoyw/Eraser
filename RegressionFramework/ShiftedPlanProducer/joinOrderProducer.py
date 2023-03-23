import random

from RegressionFramework.ShiftedPlanProducer.JoinKeySearch import JoinKeySearch
from RegressionFramework.ShiftedPlanProducer.statistic import Statistic
from RegressionFramework.config import overlap_thres


class JoinOrderProducer:
    def __init__(self, statistic: Statistic, db):
        self.statistic = statistic
        self.paths = None
        self.random = random.Random()
        self.join_key_search = None
        self.db = db

    def build(self):
        print("JoinOrderProducer build")
        # update paths
        self.join_key_search = JoinKeySearch(self.db, self.statistic)
        self.join_key_search.build()
        self.paths = self.join_key_search.get_paths()

    def get_paths(self):
        return self.paths

    def get_candidate_join_keys(self):
        return self.join_key_search.get_candidate_join_keys()

    def generate_with_specific_length(self, length, k, only_shifted_table=True):
        """
        we preferentially generate choose from shifted tables
        :param length:
        :param only_shifted_table:
        :param k: the number of join orders
        :return:
        """
        raise RuntimeError

    def generate_with_specific_table(self, k, table):
        """
        the join order must contain specify table, other table is selected from shifted table
        :return:
        """
        raise RuntimeError

    def generate_with_specific_join_key(self, k, join_key):
        raise RuntimeError


class BruteForceJoinOrderProducer(JoinOrderProducer):
    def __init__(self, statistic: Statistic, db):
        super().__init__(statistic, db)
        self.length_2_non_shifted_paths = {}
        self.length_2_mix_paths = {}
        self.table_2_paths = {}

        # { "c1#c2":paths,...}
        self.join_key_2_paths = {}

    def build(self):
        super().build()

        for path in self.paths:
            if len(path) == 1 and path[0] == '':
                continue
            length = len(path)
            if self._is_non_shifted_path(path):
                if length not in self.length_2_non_shifted_paths:
                    self.length_2_non_shifted_paths[length] = []
                self.length_2_non_shifted_paths[length].append(path)
            else:
                if length not in self.length_2_mix_paths:
                    self.length_2_mix_paths[length] = []
                self.length_2_mix_paths[length].append(path)

            # organize table -> paths that contains table
            self._update_table_2_paths(path)
            self._update_join_key_2_paths(path)
        print("")

    def _update_table_2_paths(self, path):
        for join_key in path:
            for col in join_key:
                table = col.split(".")[0]
                if table not in self.table_2_paths:
                    self.table_2_paths[table] = []
                self.table_2_paths[table].append(path)

    def _update_join_key_2_paths(self, path):
        for join_key in path:
            c0 = join_key[0]
            c1 = join_key[1]
            key = "{}#{}".format(c0, c1)
            if key not in self.join_key_2_paths:
                self.join_key_2_paths[key] = []
            self.join_key_2_paths[key].append(path)

    def generate_with_specific_length(self, length, k, only_shifted_table=True):
        if length in self.length_2_non_shifted_paths:
            paths = self.length_2_non_shifted_paths[length]
        elif length in self.length_2_mix_paths:
            paths = self.length_2_mix_paths[length]
        else:
            return []
        paths = list.copy(paths)
        self.random.shuffle(paths)
        return paths[0:min(k, len(paths))]

    def generate_with_specific_table(self, k, table):
        """
        the join order must contain specify table, other table is selected from non shifted table
        :return:
        """
        if table not in self.table_2_paths:
            return None
        paths = self.table_2_paths[table]
        self.random.shuffle(paths)
        return paths[0:min(k, len(paths))]

    def generate_with_specific_join_key(self, k, join_key):
        """
        the join order must contain specify join key, other key is selected from non shifted keys
        :return:
        """
        key="{}#{}".format(join_key[0], join_key[1])
        if key not in self.join_key_2_paths:
            return None
        paths = self.join_key_2_paths[key]
        filter_paths = []
        for path in paths:
            if not self._is_non_shifted_path(path, {join_key[0], join_key[1]}):
                filter_paths.append(path)
        self.random.shuffle(filter_paths)
        return filter_paths[0:min(k, len(filter_paths))]

    def _is_non_shifted_path(self, path, ignore_set=None):
        non_shifted_join_keys = self.statistic.get_non_shifted_join_cols()
        for join_key in path:
            for col in join_key:
                if ignore_set is not None and col in ignore_set:
                    continue
                if col not in non_shifted_join_keys:
                    return False
        return True


class ScoreJoinOrderProducer(JoinOrderProducer):
    pass
