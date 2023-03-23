import bisect
import itertools
import random

from RegressionFramework.ShiftedPlanProducer.statistic import Statistic
from RegressionFramework.config import max_join_keys_size, overlap_thres, min_join_keys_size


class Entry:
    def __init__(self, c1, c2, score):
        self.c1 = c1
        self.c2 = c2
        self.score = score

    def __getitem__(self, item):
        return self.score

    def __str__(self):
        return "c1={}, c2={}, score={}".format(self.c1, self.c2, self.score)


class Graph:
    def __init__(self):
        self.vertexes = set()
        self.vertex_2_backwards = {}

    def _clear(self):
        self.vertexes = set()
        self.vertex_2_backwards = {}

    def build(self, entries):
        self._clear()
        for entry in entries:
            self._add(entry)

    def find_paths(self):
        all_paths = []
        for v in self.vertexes:
            paths = []
            self._recurse(v, list(), paths)
            for path in paths:
                join_order = []
                for i in range(len(path) - 1):
                    join_order.append((path[i], path[i+1]))
                if len(join_order) > 1:
                    all_paths.append(join_order)
        return all_paths

    def _recurse(self, vertex, cur_path: list, paths: list):
        if vertex in cur_path:
            return
        cur_path.append(vertex)
        paths.append(cur_path)
        for backward in self.vertex_2_backwards[vertex]:
            self._recurse(backward, list.copy(cur_path), paths)

    def _add(self, entry: Entry):
        c1 = entry.c1
        c2 = entry.c2
        for c in [c1, c2]:
            self.vertexes.add(c)
            if c not in self.vertex_2_backwards:
                self.vertex_2_backwards[c] = set()
        self.vertex_2_backwards[c1].add(c2)


class JoinKeySearch:
    """
    find candidate join keys with requirements
    """

    def __init__(self, db, statistic: Statistic, thres=overlap_thres, max_count=max_join_keys_size,
                 min_count=min_join_keys_size):
        self.statistic = statistic
        self.entries = None
        self.overlap_thres = thres
        self.max_count = max_count
        self.min_count = min_count
        self.paths = []
        self.random = random.Random()
        self.table_2_entries = {}
        self.db = db

    def build(self):
        self._rank()
        self._compress_join_keys()
        self._add_non_shifted_join_keys()

        if self.db == "tpch" or self.db == "stats":
            self.paths = self._find_paths()
        else:
            if len(self.entries) > 0:
                graph = Graph()
                graph.build(self.entries)
                self.paths = graph.find_paths()

    def _delete_no_match_type(self):
        entries = []
        for e in self.entries:
            ty1 = self.statistic.get_column_type(e.c1)
            ty2 = self.statistic.get_column_type(e.c2)
            if ty1 == ty2:
                entries.append(e)
        return entries

    def _add_non_shifted_join_keys(self):
        join_keys = self.statistic.get_non_shifted_join_keys()
        for key in join_keys:
            self.entries.append(Entry(key[0], key[1], 1.0))

    def _compress_join_keys(self):
        self.entries = self._delete_no_match_type()
        scores = [e.score for e in self.entries]
        idx = bisect.bisect_right(scores, self.overlap_thres)
        entries = self.entries[idx:]
        if len(entries) > self.max_count:
            self.random.shuffle(entries)
            entries = entries[-self.max_count:]
        elif len(entries) < self.min_count:
            entries = self.entries[-self.min_count:]
        self.entries = entries

    def get_paths(self):
        """
        :return: [[t1.c,t2.c,...],[]]
        """
        return self.paths

    def get_candidate_join_keys(self):
        """
        :return: [(c1,c2),(),...]
        """
        return [(e.c1, e.c2) for e in self.entries]

    def _rank(self):
        all_tables = list(self.statistic.get_all_tables())

        # col_pairs: [(col1,col2,score),()]
        entries = []
        for i in range(0, len(all_tables)):
            for j in range(i + 1, len(all_tables)):
                t1 = all_tables[i]
                t2 = all_tables[j]
                cols1 = self.statistic.get_category_columns(t1)
                cols2 = self.statistic.get_category_columns(t2)

                for c1 in cols1:
                    for c2 in cols2:
                        v1 = self.statistic.get_domains(t1, c1)
                        v2 = self.statistic.get_domains(t2, c2)

                        if c1 > c2:
                            entries.append(Entry(c2, c1, self._compute_score(v1, v2)))
                        else:
                            entries.append(Entry(c1, c2, self._compute_score(v1, v2)))

        self.entries = sorted(entries, key=lambda a: a[2])

    def _sort_join_key(self, c1, c2):
        cs = sorted([c1, c2])
        return cs[0], cs[1]

    def _compute_score(self, values1, values2):
        values1 = set(values1)
        values2 = set(values2)
        if len(values1.union(values2)) > 0:
            return float(len(values1.intersection(values2))) / len(values1.union(values2))
        return 0

    def _collect_tables_pair(self):
        for e in self.entries:
            t1 = e.c1.split(".")[0]
            t2 = e.c2.split(".")[0]
            if t1 == "" or t2 == "":
                continue
            key = (t1, t2)
            if key not in self.table_2_entries:
                self.table_2_entries[key] = []
            key2 = (t2, t1)
            if key2 not in self.table_2_entries:
                self.table_2_entries[key2] = []
            self.table_2_entries[key].append(e)

    def _find_paths(self):
        self._collect_tables_pair()
        paths = []
        tables = self.statistic.get_all_tables()
        for length in range(1, len(tables) + 1):
            table_paths = list(itertools.permutations(tables, length))
            # table_paths = self.random.sample(table_paths, min(100, len(table_paths)))
            for table_path in table_paths:
                self._recurse_find_path(table_path, 0, [], paths)
        return paths

    def _recurse_find_path(self, tables, idx, cur_path, all_paths):
        if idx >= len(tables) - 1:
            if len(cur_path) > 0:
                all_paths.append(cur_path)
            return

        left_table = tables[idx]
        right_table = tables[idx + 1]
        if (left_table, right_table) not in self.table_2_entries:
            return
        entries = self.table_2_entries[(left_table, right_table)]
        for entry in entries:
            new_path = list.copy(cur_path)
            new_path.append((entry.c1, entry.c2))
            self._recurse_find_path(tables, idx + 1, new_path, all_paths)
