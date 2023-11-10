import random

from RegressionFramework.ShiftedPlanProducer.statistic import Statistic
from RegressionFramework.utils import join, is_number


class SqlTemplate:
    def __init__(self, statistic: Statistic, db, max_predicates=3, max_projects=3):
        self.statistic = statistic
        self.max_predicates = max_predicates
        self.max_projects = max_projects
        self.random = random.Random()
        self.db = db

    def generate(self, join_order, project_cols=None, only_predicate: tuple = None):
        """

        :param join_order:
        :param project_cols:
        :param only_predicate: (col,op,value)
        :return:
        """
        tables = self._extract_tables(join_order)

        if project_cols is None:
            project_cols = self._generate_project_cols(tables)

        if only_predicate is not None:
            predicates = [only_predicate]
        else:
            predicates = self._generate_predicates(tables)
        predicates = self._delete_predicates(predicates)
        self._add_single_quotes_to_filter_val(predicates)
        # [[col,op,value],[]] -> [col_op_value,col_op_value2,...]
        predicates = [join("", p) for p in predicates]

        # [T1.c,T2.c,...] -> [ 'T1.c=T2.c', ...]
        join_keys = self._to_join_keys(join_order)

        sql = "SELECT {} FROM {} WHERE {} {} {}".format(
            ",".join(project_cols),
            ",".join(tables),
            " AND ".join(join_keys),
            " AND " if len(join_keys) > 0 and len(predicates) > 0 else "",
            " AND ".join(predicates)
        )
        if self.db == "tpch":
            # because all sql in benchmark contain a limit 10.
            sql += " limit 10;"
        else:
            sql += ";"
        return sql

    def _delete_predicates(self, predicates):
        filter_predicates = []
        for p in predicates:
            val = str(p[2]).strip("\'")
            if "\'" not in val:
                filter_predicates.append(p)
        return filter_predicates

    def _add_single_quotes_to_filter_val(self, predicates):
        for i in range(len(predicates)):
            p = predicates[i]
            col = p[0]
            val = p[2]
            ty = self.statistic.get_column_type(col)
            if ty == "string":
                val = str(val)
                val = val.strip("\'")
                # lineitem.l_shipdate<'lineitem.l_commitdate'
                if not ("date" in val and "." in val):
                    val = "\'{}\'".format(val)
            elif ty == "date":
                # lineitem.l_shipdate<'lineitem.l_commitdate'
                if not ("date" in val and "." in val):
                    val = "\'{}\'".format(val)
            predicates[i] = [p[0], p[1], val]

    def _to_join_keys(self, join_order):
        """
        :param join_order: like: [(T1.c,T2.c),()...]
        :return:
        """
        join_keys = []
        for join_key in join_order:
            join_keys.append("{}={}".format(join_key[0], join_key[1]))
        return join_keys

    def _extract_tables(self, join_order):
        """
        :param join_order: like: [(T1.c,T2.c),()...]
        :return:
        """
        tables = set()
        for join_key in join_order:
            for col in join_key:
                tables.add(col.split(".")[0])
        return list(tables)

    def _generate_project_cols(self, tables):
        cols = []
        for table in tables:
            if self._is_shifted_tables(table):
                cs = self._random_select_project_cols(table)
            else:
                cs = self.statistic.get_non_shifted_project_cols(table)
                if cs is None:
                    cs = self._random_select_project_cols(table)
            assert cs is not None
            cols += cs
        cols = list(set(cols))
        return self._random_pick_items(self.max_projects, cols)

    def _is_shifted_tables(self, table):
        shifted_tables = self.statistic.get_shifted_tables()
        return table in shifted_tables

    def _random_select_project_cols(self, table):
        columns = list.copy(self.statistic.get_columns(table))
        columns = self._random_pick_items(self.max_projects, columns)
        if self.db == "imdb":
            return ["MIN({})".format(c) for c in columns]
        elif self.db == "stats":
            return ["count(*)"]
        elif self.db == "tpch":
            return columns
        else:
            raise RuntimeError

    def _generate_predicates(self, tables):
        predicates = []
        for table in tables:
            if self._is_shifted_tables(table):
                p = self._random_select_predicates(table)
            else:
                p = self.statistic.get_shifted_predicates(table)
                if p is None:
                    p = self._random_select_predicates(table)
            assert p is not None
            p = self._distinct_predicates_col(p)
            predicates += p
        return self._random_pick_items(self.max_predicates, predicates)

    def _distinct_predicates_col(self, predicates):
        """

        :param predicates: [[col,op,val],...]
        :return:
        """
        col_2_predicates = {}
        for p in predicates:
            col = p[0]
            if col not in col_2_predicates:
                col_2_predicates[col] = []
            col_2_predicates[col].append(p)

        res = []
        for col, ps in col_2_predicates.items():
            res.append(self._random_pick_items(1, ps)[0])
        return res

    def _random_select_predicates(self, table):
        cols = self.statistic.get_columns(table)
        cols = self._random_pick_items(self.max_predicates, cols)
        predicates = []
        for col in cols:
            ty = self.statistic.get_column_type(col)
            domains = self.statistic.get_domains(table, col)
            if len(domains) == 0:
                continue
            val = self._random_pick_items(1, domains)[0]
            if ty == "string":
                op = "="
            else:
                op = self._random_pick_items(1, ["<", ">", "="])[0]
            predicates.append([col, op, val])
        return predicates

    def _random_pick_items(self, max_size, targets):
        targets = list.copy(list(targets))
        self.random.shuffle(targets)
        assert len(targets) > 0
        count = min(max_size, self.random.randint(1, len(targets) + 1))
        return targets[0:count]
