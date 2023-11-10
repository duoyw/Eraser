import random

from RegressionFramework.Common.Cache import Cache
from RegressionFramework.Common.TimeStatistic import TimeStatistic
from RegressionFramework.Common.dotDrawer import PlanDotDrawer
from RegressionFramework.Plan.PgPlan import PgScanPlanNode
from RegressionFramework.Plan.Plan import PlanNode, ScanPlanNode, JoinPlanNode, Plan
from RegressionFramework.ShiftedPlanProducer.histogram import Histogram
from RegressionFramework.StaticPlanGroup import StaticPlanGroup
from RegressionFramework.config import max_col_limit, histogram_bin_size
from test_script.utils import run_query
import re
from sql_metadata import Parser


class StatisticDB:
    def __init__(self, db, bins_size=histogram_bin_size):
        self.cache = Cache("StatisticDB" + db, enable=True)

        self.db = db
        self.tables = []
        self.table_2_col_2_domain = {}
        self.tables_2_category_columns = {}
        self.bins_size = bins_size

        # the column will be added if the column's name contains 'id'
        self.table_2_id_columns = {}

        self.column_2_type = {}

        self.table_2_col_2_bin = {}

    def build(self):
        """
        statistic each column's domain for each table
        :return:
        """
        if self.cache.exist():
            # print("StatisticDB trigger cache")
            res = self.cache.read()
            self.tables = res[0]
            self.table_2_col_2_domain = res[1]
            self.tables_2_category_columns = res[2]
            self.table_2_id_columns = res[3]
            self.column_2_type = res[4]
            self.table_2_col_2_bin = res[5]
        else:
            # update self.table_2_col_2_domain
            tables = self._request_tables()
            for table in tables:
                print("cur table is {}".format(table))
                # col_types like [(t.col,category/numeric),()]
                col_types = self.request_columns(table)

                self.tables_2_category_columns[table] = set()
                self.table_2_id_columns[table] = set()
                col_2_domain = {}

                for i, col_type in enumerate(col_types):
                    print("reading domain, total cols is {}, cur col idx is {}".format(len(col_types), i))
                    col = "{}.{}".format(table, col_type[0])
                    ty = col_type[1]
                    self.column_2_type[col] = ty
                    if ty == "string":
                        self.tables_2_category_columns[table].add(col)
                        domain = self._request_column_domain(table, col, ty)
                        col_2_domain[col] = domain
                    elif "id" in col or "key" in col:
                        self.table_2_id_columns[table].add(col)
                        domain = self._request_column_domain(table, col, ty)
                        col_2_domain[col] = domain
                    else:
                        col_2_domain[col] = []

                self.table_2_col_2_domain[table] = col_2_domain

            self.tables = tables
            self._build_histogram()

            self.cache.save(
                [self.tables, self.table_2_col_2_domain, self.tables_2_category_columns, self.table_2_id_columns,
                 self.column_2_type, self.table_2_col_2_bin])

    def get_column_domain(self, table, col):
        return self.table_2_col_2_domain[table][col]

    def get_tables(self):
        return self.tables

    def get_columns(self, table):
        return list(self.table_2_col_2_domain[table].keys())

    def get_numeric_columns(self, table):
        cols = self.get_columns(table)
        res = []
        for col in cols:
            if self.column_2_type[col] == "numeric":
                res.append(col)
        return res

    def get_histogram(self, table, col):
        histogram: Histogram = self.table_2_col_2_bin[table][col]
        return histogram

    def get_column_type(self, col):
        return self.column_2_type[col]

    def get_category_columns(self, table):
        return self.tables_2_category_columns[table]

    def _build_histogram(self, ):
        print("_build_histogram")
        for table in self.tables:
            for col in self.get_columns(table):
                if not self.column_2_type[col] == "numeric":
                    continue

                res = self._request_col_domain_range(table, col)
                if res is None or len(res) == 0:
                    continue
                if table not in self.table_2_col_2_bin:
                    self.table_2_col_2_bin[table] = {}
                self.table_2_col_2_bin[table][col] = Histogram(res)

    def _request_col_domain_range(self, table, col):
        """
        :param table:
        :param col: a numeric col
        :return: a minimal and maximal value like [(floor1,size),(floor2,size)]
        """
        sql = "SELECT min({}),max({}) from {} ".format(col, col, table)
        result = self._run_query(sql)
        min_val = result[1][0][0]
        max_val = result[1][0][1]
        if min_val is None or max_val is None:
            return None

        bin_width = (max_val - min_val) / self.bins_size
        bin_width = 1 if bin_width < 1 else bin_width

        sql_for_bin = "select floor({}/{})*{} as bin_floor, count(*) from {} group by 1 order by 1;".format(col,
                                                                                                            bin_width,
                                                                                                            bin_width,
                                                                                                            table)
        result = self._run_query(sql_for_bin)
        result = [(float(row[0]), float(row[1])) for row in result[1] if row[0] is not None and row[1] is not None]
        result.append((max_val, 0))
        return result

    def _request_tables(self):
        query = "SELECT table_name  FROM information_schema.tables  WHERE table_schema = 'public';"
        result = self._run_query(query)
        return [r[0].strip() for r in result[1]]

    def request_columns(self, table):
        """
        request from DB for the domain of category column for a specific table
        :return: [(t.col,category/numeric),()]
        """
        query = "SELECT column_name, data_type FROM information_schema.columns " \
                "WHERE table_schema = 'public' AND  table_name =\'{}\';".format(table)
        result = self._run_query(query)[1]

        col_types = []
        for r in result:
            col = r[0].strip()
            col_type = self._convert_col_type(r[1].strip())
            col_types.append((col, col_type))
        return col_types

    def _request_column_domain(self, table, column, col_type):
        if col_type == "string":
            return self._request_categroy_column_domain(table, column)
        elif col_type == "numeric":
            return self._request_numeric_column_domain(table, column)
        else:
            raise RuntimeError

    def _request_categroy_column_domain(self, table, column):
        query = "SELECT {} FROM {} ORDER BY RANDOM() limit {};".format(column, table, max_col_limit)
        result = self._run_query(query)
        return set([r[0].strip() for r in result[1] if r[0] is not None])

    def _request_numeric_column_domain(self, table, column):
        # query = "SELECT {} FROM (SELECT DISTINCT  ({})  FROM {})  ORDER BY random() LIMIT {};".format(column, column,
        #                                                                                                  table,
        #                                                                                                  max_col_limit)
        query = "SELECT DISTINCT ({})  FROM {} limit {}".format(column,
                                                                table,
                                                                max_col_limit)
        result = self._run_query(query)
        return set([r[0] for r in result[1] if r[0] is not None])

    def _convert_col_type(self, col_type):
        if col_type in {"numeric", "integer", 'smallint'}:
            return "numeric"
        elif col_type in {"character varying", "character", "text"}:
            return "string"
        elif col_type in {"date", "timestamp without time zone"}:
            return "date"
        else:
            raise RuntimeError

    def _run_query(self, query):
        return run_query(query, None, self.db)


class StatisticTrain:
    def __init__(self, training_set_name, db, enable_cache=True):
        self.training_set_name = training_set_name
        self.cache = Cache("StatisticTrain_" + training_set_name, enable=enable_cache)
        self.tables = set()

        # the form: "table.col"
        self.join_cols = set()

        # {(c1,c2),()}
        self.join_keys = set()

        # filter, table->[(col,op,value),(),...]
        self.table_2_predicates = {}

        # project table->cols
        self.table_2_projectCols = {}

        self.alias_2_table = {}

        self.join_order_lengths = set()

        self.db = db

        self.plans = None

    def build(self, plans, sqls):
        self.plans = plans
        if self.cache.exist():
            # print("StatisticTrain trigger cache")
            res = self.cache.read()
            self.tables = res[0]
            self.join_cols = res[1]
            self.join_keys = res[2]
            self.table_2_predicates = res[3]
            self.table_2_projectCols = res[4]
            self.join_order_lengths = res[5]
            self.alias_2_table = res[6]

        else:
            # tables
            print("extract shifted tables")
            self.tables, self.alias_2_table = self._extract_tables(sqls)

            print("extract shifted join and predicates")
            for plan in plans:
                # join and predicates
                join_node_counts = [0]
                self._recurse_plan_collect_infos(plan.root, join_node_counts)
                self.join_order_lengths.add(join_node_counts[0])

            # projects
            print("extract shifted project cols")
            self.table_2_projectCols = self._extract_project_cols(sqls)

            self.cache.save(
                [self.tables, self.join_cols, self.join_keys, self.table_2_predicates, self.table_2_projectCols,
                 self.join_order_lengths, self.alias_2_table])


    def _request_all_columns(self, ):
        query = "SELECT column_name FROM information_schema.columns " \
                "WHERE table_schema = 'public';"
        result = run_query(query, None, db=self.db)[1]
        return [r[0] for r in result]

    def get_tables(self):
        """
        return all tables that occur in training set
        :return:
        """
        return self.tables

    def get_join_cols(self):
        return self.join_cols

    def get_join_keys(self):
        """

        :return: {(c1,c2),()}
        """
        return self.join_keys

    def get_predicates(self, table):
        if table not in self.table_2_predicates:
            return None
        return self.table_2_predicates[table]

    def get_all_cols(self):
        cols = []
        tables = self.get_tables()
        for table in tables:
            predicates = self.get_predicates(table)
            if predicates is not None:
                for p in predicates:
                    cols.append(p[0])
        return cols

    def get_project_cols(self, table):
        if table not in self.table_2_projectCols:
            return None
        return self.table_2_projectCols[table]

    def get_join_order_lengths(self):
        return self.join_order_lengths

    def _recurse_plan_collect_infos(self, node: PlanNode, join_node_counts: list):
        if isinstance(node, JoinPlanNode):
            node: JoinPlanNode = node
            join_key = list.copy(node.join_key)
            assert len(join_key) == 2

            # convert alias -> table,
            for i in range(len(join_key)):
                col = join_key[i]
                if col == "":
                    continue
                join_key[i] = self.alias_to_table_for_col(col)

            if join_key[0] != "" and join_key[1] != "":
                join_node_counts[0] = join_node_counts[0] + 1

            self.join_keys.add((join_key[0], join_key[1]))
            self.join_cols = self.join_cols.union(join_key)
        elif isinstance(node, PgScanPlanNode):
            node: PgScanPlanNode = node
            predicates = node.predicates
            for predicate in predicates:
                predicate = list.copy(list(predicate))
                predicate[0] = self.alias_to_table_for_col(predicate[0])
                table = predicate[0].split(".")[0]
                if table not in self.table_2_predicates:
                    self.table_2_predicates[table] = []
                self.table_2_predicates[table].append(predicate)
        for child in node.children:
            self._recurse_plan_collect_infos(child, join_node_counts)

    def alias_to_table_for_col(self, col):
        """

        :param col: form like table_alias.col
        :return: table.col
        """
        values = col.split(".")
        alias = values[0]
        if alias not in self.alias_2_table:
            # dynamic case, test plan will can not to find table
            return "{}.{}".format(alias, values[1])
        return "{}.{}".format(self.alias_2_table[alias], values[1])

    def _extract_tables(self, sqls):
        """
        table must contain alias. like :  link_type AS lt,     movie_keyword AS mk,
        :param sqls:
        :return:
        """
        tables = set()
        alias_2_table = {}
        for sql in sqls:
            tables_aliases = Parser(sql.lower()).tables_aliases
            alias_2_table.update(tables_aliases)
            tables = tables.union(set(tables_aliases.values()))
            if len(tables) == self._get_actual_tables_counts():
                continue
        return tables, alias_2_table

    def _get_actual_tables_counts(self):
        raise RuntimeError

    def _extract_project_cols(self, sqls):
        raise RuntimeError


class StatisticTrainJob(StatisticTrain):

    def _extract_project_cols(self, sqls):
        """
        all project cols in job  with the form min(t.col),min(t.col2)
        :param sqls:
        :return:
        """
        table_2_projectCols = {}
        for sql in sqls:
            try:
                columns_dict = Parser(sql).columns_dict
            except:
                continue
            cols = columns_dict["select"]
            for col in cols:
                # the form of col is t.c
                table = col.split(".")[0]
                if table not in table_2_projectCols:
                    table_2_projectCols[table] = set()
                table_2_projectCols[table].add("MIN({})".format(col))
        return table_2_projectCols

    def _get_actual_tables_counts(self):
        return 21


class StatisticTrainTpch(StatisticTrain):
    """
    the sql of tpch don't contain table alias. Each column with the form tableName_column. for example, col 'c_name' is
    'customer' table and col 'ps_comment' is "partsupp' table.
    to unify the process, we need to convert col to table.column
    """

    def __init__(self, training_set_name, db):
        super().__init__(training_set_name, db)
        self.alias_2_table = {
            "c": "customer",
            "l": "lineitem",
            "n": "nation",
            "o": "orders",
            "p": "part",
            "ps": "partsupp",
            "r": "region",
            "s": "supplier",
        }

    def alias_to_table_for_col(self, col):
        """

        :param col: form  table/alias.tableAlias_col
        :return: table.col
        """
        col = col.split(".")[1]
        alias = col.split("_")[0]
        return "{}.{}".format(self.alias_2_table[alias], col)

    def _extract_project_cols(self, sqls):
        """
        all project cols in job  with the form min(t.col),min(t.col2)
        :param sqls:
        :return:
        """
        all_cols = set(self._request_all_columns())
        table_2_projectCols = {}
        for sql in sqls:
            try:
                columns_dict = Parser(sql).columns_dict
            except:
                continue
            cols = columns_dict["select"]

            for col in cols:
                if col == "*" or col not in all_cols:
                    continue
                # the form of col is t.c
                if "." in col:
                    col = col.split(".")[1]
                try:
                    table = self.alias_2_table[col.split("_")[0]]
                except:
                    # some subquery is  difficult to handle
                    #  like: select  cntrycode,  count(*) as numcust,  sum(c_acctbal) as totacctbal from  (   select    substring(c_phone from 1 for 2) as cntrycode,    c_acctbal   from    customer   where    substring(c_phone from 1 for 2) in     ('34', '27', '10', '11', '12', '22', '21')    and c_acctbal > (     select      avg(c_acctbal)     from      customer     where      c_acctbal > 0.00      and substring(c_phone from 1 for 2) in       ('34', '27', '10', '11', '12', '22', '21')    )    and not exists (     select      *     from      orders     where      o_custkey = c_custkey    )  ) as custsale group by  cntrycode order by  cntrycode  limit 10;
                    continue
                if table not in table_2_projectCols:
                    table_2_projectCols[table] = set()
                table_2_projectCols[table].add("{}".format(col))
        return table_2_projectCols

    def _get_actual_tables_counts(self):
        return 8

    def _extract_tables(self, sqls):
        """
        table must exclude alias.
        :param sqls:
        :return:
        """
        tables = set()
        for sql in sqls:
            cur_tables = Parser(sql.lower()).tables
            tables = tables.union(cur_tables)
            if len(tables) == self._get_actual_tables_counts():
                continue
        return tables, self.alias_2_table


class StatisticTrainStats(StatisticTrain):

    def __init__(self, training_set_name, db):
        super().__init__(training_set_name, db)

    def _extract_project_cols(self, sqls):
        """
        all project cols in job  with the form "count(*)"
        :param sqls:
        :return:
        """
        table_2_projectCols = {}
        for table in self.tables:
            table_2_projectCols[table] = {"count(*)"}
        return table_2_projectCols

    def _get_actual_tables_counts(self):
        return 8


class Statistic:
    def __init__(self, db, training_set_name):
        self.statistic_db = StatisticDB(db)
        self.db = db
        if self.db == "imdb":
            self.statistic_train = StatisticTrainJob(training_set_name, db)
        elif self.db == "stats":
            self.statistic_train = StatisticTrainStats(training_set_name, db)
        elif self.db == "tpch":
            self.statistic_train = StatisticTrainTpch(training_set_name, db)
        else:
            raise RuntimeError

    def build(self, plans, sqls):
        self.statistic_train.build(plans, sqls)
        self.statistic_db.build()
        pass

    def get_non_shifted_tables(self):
        return self.statistic_train.get_tables()

    def get_non_shifted_join_order_lengths(self):
        return self.statistic_train.get_join_order_lengths()

    def get_all_tables(self):
        return self.statistic_db.get_tables()

    def get_shifted_tables(self):
        all_tables = self.statistic_db.get_tables()
        non_shifted_tables = self.get_non_shifted_tables()
        return set(all_tables).difference(set(non_shifted_tables))

    def get_numeric_columns(self, table):
        return self.statistic_db.get_numeric_columns(table)

    def get_histogram(self, table, col):
        return self.statistic_db.get_histogram(table, col)

    def get_non_shifted_join_cols(self):
        return self.statistic_train.get_join_cols()

    def get_non_shifted_join_keys(self):
        """
         :return: {(c1,c2),()}
        """
        return self.statistic_train.get_join_keys()

    # def get_shifted_join_orders(self):
    #     return [[]]

    def get_shifted_predicates(self, table):
        return self.statistic_train.get_predicates(table)

    def get_non_shifted_project_cols(self, table):
        return self.statistic_train.get_project_cols(table)

    def get_columns(self, table):
        return self.statistic_db.get_columns(table)

    def get_column_type(self, col):
        """

        :return: numeric, string, date
        """
        return self.statistic_db.get_column_type(col)

    def get_category_columns(self, table):
        """
        it is used for generating join keys
        :param table:
        :return:
        """
        return self.statistic_db.get_columns(table)

    def get_domains(self, table, col):
        return self.statistic_db.get_column_domain(table, col)


class StatisticTest:
    def __init__(self, statistic_train, structure_enable, join_enable, table_enable, filter_enable):
        self.statistic_train: StatisticTrain = statistic_train
        self.static_group: StaticPlanGroup = StaticPlanGroup()
        self.static_group.build(self.statistic_train.plans)
        self.structure_enable = structure_enable
        self.join_enable = join_enable
        self.table_enable = table_enable
        self.filter_enable = filter_enable
        self.existed_cols = None

    def is_shifted(self, plan: Plan):
        if self.structure_enable:
            group = self.static_group.get_group(plan)

            # shifted structure
            if group is None:
                return True

        join_keys = set()
        tables = set()
        filter_cols = set()
        self._recurse_plan_collect_infos(plan.root, join_keys, tables, filter_cols)

        if self.join_enable:
            existed_join_keys = self.statistic_train.get_join_keys()
            for join_key in join_keys:
                if join_key not in existed_join_keys:
                    return True

        if self.table_enable:
            existed_tables = self.statistic_train.get_tables()
            for table in tables:
                if table not in existed_tables:
                    return True

        if self.filter_enable:
            if self.existed_cols is None:
                self.existed_cols = set(self.statistic_train.get_all_cols())
            existed_cols = self.existed_cols
            for col in filter_cols:
                if col not in existed_cols:
                    return True

        return False

    def _recurse_plan_collect_infos(self, node: PlanNode, join_keys: set, tables: set(), filter_cols: set):
        if isinstance(node, JoinPlanNode):
            node: JoinPlanNode = node
            join_key = list.copy(node.join_key)
            assert len(join_key) == 2

            # convert alias -> table,
            for i in range(len(join_key)):
                col = join_key[i]
                if col == "":
                    continue
                try:
                    join_key[i] = self._alias_to_table_for_col(col)
                except:
                    continue

            join_keys.add((join_key[0], join_key[1]))
        elif isinstance(node, PgScanPlanNode):
            node: PgScanPlanNode = node
            predicates = node.predicates
            for predicate in predicates:
                predicate = list.copy(list(predicate))
                predicate[0] = self._alias_to_table_for_col(predicate[0])
                table = predicate[0].split(".")[0]
                tables.add(table)
                filter_cols.add(predicate[0])
        for child in node.children:
            self._recurse_plan_collect_infos(child, join_keys, tables, filter_cols)

    def _alias_to_table_for_col(self, col):
        return self.statistic_train.alias_to_table_for_col(col)
