import random

import sqlparse

from RegressionFramework.ShiftedPlanProducer.histogram import Histogram
from RegressionFramework.ShiftedPlanProducer.joinOrderProducer import JoinOrderProducer, BruteForceJoinOrderProducer
from RegressionFramework.ShiftedPlanProducer.sqlTemplate import SqlTemplate
from RegressionFramework.ShiftedPlanProducer.statistic import Statistic
from RegressionFramework.config import overlap_thres, sqls_for_each_join_key, histogram_bin_size, sqls_for_each_bin
from RegressionFramework.utils import json_str_to_json_obj
from test_script.utils import run_query


class BaseSqlProducer:
    def __init__(self, join_order_producer: JoinOrderProducer, statistic: Statistic, sql_template: SqlTemplate, db):
        self.join_order_producer = join_order_producer
        self.statistic = statistic
        self.sql_template = sql_template
        self.random = random.Random()
        self.db = db

    def _check_sql(self, sql):
        # parsed = sqlparse.parse(sql)
        # if len(parsed) <= 0:
        #     raise RuntimeError
        try:
            run_query("EXPLAIN {}".format(sql), None, self.db,2000)
        except Exception as e:
            if "timeout" not in str(e):
                print(e)
                raise RuntimeError("Syntax Error: {}".format(sql))

class StructureSqlProducer(BaseSqlProducer):
    """
    for generating the sql with shifted structure that has a new length
    """

    def generate(self, k):
        """
        specify the number of tables and candidates tables .
        we preferentially use non-shifted table to generate plan. If the number of sql lack, we use the tables that
        is non-shifted but has little impact on model's performance
        :return:
        """

        # join_orders is like:[[t1.c,t2.c,...],[]]
        non_shifted_join_order_lengths = self.statistic.get_non_shifted_join_order_lengths()
        all_tables = self.statistic.get_all_tables()
        shifted_lengths = self.get_shifted_length(all_tables, non_shifted_join_order_lengths)

        length_2_sqls = {}
        size_for_length = int(k / len(shifted_lengths))

        for l in shifted_lengths:
            join_orders = self.join_order_producer.generate_with_specific_length(l, size_for_length)
            if len(join_orders) > 0:
                sqls = []
                for join_order in join_orders:
                    sql = self.sql_template.generate(join_order)
                    self._check_sql(sql)
                    sqls.append(sql)
                length_2_sqls[l] = sqls
        return length_2_sqls

    def get_shifted_length(self, all_tables, join_order_lengths):
        max_length = len(all_tables)
        shifted_length_set = join_order_lengths

        res = []
        for i in range(1, max_length + 1):
            if i not in shifted_length_set:
                res.append(i)
        return res


class JoinKeySqlProducer(BaseSqlProducer):

    def generate(self, k):
        """

        :param k:
        :return:
        """
        # [(c1,c2),()]
        join_key_count = int(k / sqls_for_each_join_key)
        shifted_join_keys = list(self.get_shifted_join_keys())[:join_key_count]

        join_keys_2_sqls = {}
        for join_key in shifted_join_keys:
            sqls = []
            join_orders = self.join_order_producer.generate_with_specific_join_key(sqls_for_each_join_key, join_key)
            if join_orders is None:
                continue
            for join_order in join_orders:
                sql = self.sql_template.generate(join_order)
                self._check_sql(sql)
                sqls.append(sql)
            if len(sqls) > 0:
                join_keys_2_sqls[join_key] = sqls

        return join_keys_2_sqls

    def get_shifted_join_keys(self):
        candidate_jks = set(self.join_order_producer.get_candidate_join_keys())
        non_shifted_jks = self.statistic.get_non_shifted_join_keys()
        return candidate_jks.difference(non_shifted_jks)


class FilterSqlProducer(BaseSqlProducer):
    """
       for generating the sql with non-shifted/shifted col and shifted value interval
       """

    def generate(self, k):
        """
        require join order contain a specific table and remain table is selected from non-shifted table.
        (note that we allow only a shifted table )
        :return:
        """
        table_2_col_2_range_sqls = {}

        tables = self.statistic.get_all_tables()
        cols = []
        for t in tables:
            cols += self.statistic.get_numeric_columns(t)

        size_of_remain_cols = int(k / (histogram_bin_size * sqls_for_each_bin))
        cols = self._compress_cols(cols, size_of_remain_cols)

        for col in cols:
            table = col.split(".")[0]
            histogram: Histogram = self.statistic.get_histogram(table, col)

            vals_for_bins = histogram.pick_values_from_each_bin(1)
            for entry in vals_for_bins:
                min_val = entry[0]
                max_val = entry[1]
                vals = entry[2]

                sqls = []
                for val in vals:
                    join_orders = self.join_order_producer.generate_with_specific_table(sqls_for_each_bin, table)
                    if join_orders is None:
                        continue
                    for join_order in join_orders:
                        predicate = (col, self._pick_filter_op(table, col), val)
                        sql = self.sql_template.generate(join_order, only_predicate=predicate)
                        self._check_sql(sql)
                        sqls.append(sql)
                if len(sqls) > 0:
                    # init
                    if table not in table_2_col_2_range_sqls:
                        table_2_col_2_range_sqls[table] = {}
                    if col not in table_2_col_2_range_sqls[table]:
                        table_2_col_2_range_sqls[table][col] = []
                    table_2_col_2_range_sqls[table][col].append([min_val, max_val, sqls])
        return table_2_col_2_range_sqls

    def _pick_filter_op(self, table, col):
        ops = ["<", ">"]
        if self._is_numeric_column(table, col):
            idx = self.random.randint(0, len(ops) - 1)
            return ops[idx]
        return "="

    def _is_numeric_column(self, table, col):
        ty = self.statistic.get_column_type(col)
        if ty == "numeric":
            return True
        return False

    def _compress_cols(self, cols, k):
        cols = [c for c in cols if "id" not in c]
        self.random.shuffle(cols)
        return cols[0:min(k, len(cols))]


class TableNameSqlProducer(BaseSqlProducer):
    """
    for generating the sql with shifted table name
    """

    def generate(self, k):
        """
        require join order contain a specific table and remain table is selected from non-shifted table.
        (note that we allow only a shifted table )
        :return:
        """
        table_2_sqls = {}
        tables = self.statistic.get_shifted_tables()
        if (len(tables)) != 0:
            size_for_table = int(k / len(tables))
            for t in tables:
                join_orders = self.join_order_producer.generate_with_specific_table(size_for_table, t)
                if join_orders is None:
                    continue
                sqls = []
                for join_order in join_orders:
                    sql = self.sql_template.generate(join_order)
                    self._check_sql(sql)
                    sqls.append(sql)
                table_2_sqls[t] = sqls
        return table_2_sqls


class ScanTypeSqlProducer(BaseSqlProducer):
    """
    for generating the sql with shifted table name
    """

    def generate(self, k):
        raise RuntimeError


class SqlProduceManager:
    def __init__(self, statistic: Statistic, sql_template: SqlTemplate, db):
        self.statistic = statistic
        self.db = db
        self.join_order_producer = BruteForceJoinOrderProducer(self.statistic,db)
        self.structure_producer = StructureSqlProducer(self.join_order_producer, self.statistic, sql_template, db)
        self.shifted_table_sql_producer = TableNameSqlProducer(self.join_order_producer, self.statistic, sql_template,
                                                               db)
        self.shifted_join_key_sql_producer = JoinKeySqlProducer(self.join_order_producer, self.statistic, sql_template,
                                                                db)
        self.filter_sql_producer = FilterSqlProducer(self.join_order_producer, self.statistic, sql_template, db)

    def build(self):
        self.join_order_producer.build()

    def generate_structure_sql(self, k):
        return self.structure_producer.generate(k)

    def generate_shifted_table_sql(self, k):
        return self.shifted_table_sql_producer.generate(k)

    def generate_shifted_join_key_sql(self, k):
        return self.shifted_join_key_sql_producer.generate(k)

    def generate_filter_sqls(self, k):
        return self.filter_sql_producer.generate(k)
