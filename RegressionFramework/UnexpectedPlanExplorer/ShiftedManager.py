import json

from RegressionFramework.Common.Cache import Cache
from RegressionFramework.UnexpectedPlanExplorer.generateSql import SqlProduceManager
from RegressionFramework.UnexpectedPlanExplorer.sqlTemplate import SqlTemplate
from RegressionFramework.UnexpectedPlanExplorer.statistic import Statistic, StatisticTest
from RegressionFramework.config import shifted_sql_budget_ratio, alpha
from RegressionFramework.Common.utils import json_str_to_json_obj
from config import SEP
from test_script.utils import run_query


class ShiftedManager:
    """
    Unexpected plan explorer.
    In order to save source, we generate sql on three subspace, overall join relation, structure, filter subspace.
    This is equalled to set a large terminal threshold.

    Considering the related code is complex, if any user want to extend it to support other benchmark,
    but throw an exception, there are two solution.
    First, reading the code and fix the problem, you can commit the issue in github.
    Second, forcing to set the subspace evaluation result by validation set.
    Noted that, the latter method only can be used in experiment and is not reasonable for real environment.
    """

    def __init__(self, db: str, training_set_name, model, algo):
        self.model = model
        self.db = db
        self.training_set_name = training_set_name
        self.statistic = Statistic(db, training_set_name)
        self.sql_template = SqlTemplate(self.statistic, db)
        self.sql_producer = SqlProduceManager(self.statistic, self.sql_template, db)
        self.sql_cache = Cache(db + "_sql", enable=True)
        self.sql_2_plans_cache = Cache(db + "_sql_plans", enable=True)
        self.space_evaluation_cache = Cache("space_evaluation_{}_{}_{}_cache".format(algo, db, training_set_name),
                                            enable=True)

        self.structure_sqls = None
        self.table_2_sqls = None
        self.join_keys_2_sqls = None
        self.filter_table_2_col_2_range_sqls = None

        # { sql:[[first plans],[second plans ]],...}, only use first plans
        self.sql_2_plans = {}

        self.statistic_test = None

        # evaluation result, True or False
        self.delete_structure_enable = None
        self.delete_join_enable = None
        self.delete_table_enable = None
        self.delete_filter_enable = None

        self.struct_accuracy = None
        self.join_key_accuracy = None
        self.table_accuracy = None
        self.filter_accuracy = None

    def build(self, plans, sqls):
        self._generate_sqls(plans, sqls)

        sqls = self._collect_all_sqls()
        if self.sql_2_plans_cache.exist():
            self.sql_2_plans = self.sql_2_plans_cache.read()[0]

        self._generate_plans(sqls)

        self.struct_accuracy, self.join_key_accuracy, self.table_accuracy, self.filter_accuracy = self._evaluation()
        self.delete_structure_enable = True if self.struct_accuracy < alpha else False
        self.delete_join_enable = True if self.join_key_accuracy < alpha else False
        self.delete_table_enable = True if self.table_accuracy < alpha else False
        self.delete_filter_enable = True if self.filter_accuracy < alpha else False

        self.statistic_test = StatisticTest(self.statistic.statistic_train, self.delete_structure_enable,
                                            self.delete_join_enable,
                                            self.delete_table_enable,
                                            self.delete_filter_enable)

        # end
        self.sql_2_plans_cache.save([self.sql_2_plans])
        print("")

    def write_accuracy(self, struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy):
        with open("./unexpected_plan_accuracy.txt", "a") as f:
            line = "train_file is {}, structure is {}, join is {}, table is {}, filter is {} \n".format(
                self.training_set_name, struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy)
            f.write(line)
        exit()

    def is_filter(self, plan):
        return self.statistic_test.is_shifted(plan)

    def _evaluation(self):
        if self.space_evaluation_cache.exist():
            res = self.space_evaluation_cache.read()
            struct_accuracy = res[0]
            join_key_accuracy = res[1]
            table_accuracy = res[2]
            filter_accuracy = res[3]
        else:
            print("start evaluation")
            # structure
            struct_sqls = self._collect_structure_sqls()
            struct_accuracy, valid_struct_sqls_count = self._evaluate_model(struct_sqls)

            join_key_sqls = self._collect_join_key_sqls()
            join_key_accuracy, valid_join_key_sqls_count = self._evaluate_model(join_key_sqls)

            table_sqls = self._collect_table_sqls()
            table_accuracy, valid_table_sqls_count = self._evaluate_model(table_sqls)

            filter_sql = self._collect_filter_sqls()
            filter_accuracy, valid_filter_sql_count = self._evaluate_model(filter_sql)

            print(
                "struct_sql_count is {}, join_key_sql_count is {}, table_sql_count is {}, filter_sql_count is {}".format(
                    valid_struct_sqls_count, valid_join_key_sqls_count, valid_table_sqls_count, valid_filter_sql_count))
            self.space_evaluation_cache.save([struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy])

        print("struct_accuracy is {}, join_key_accuracy is {}, table_accuracy is {}, filter_accuracy is {}".format(
            struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy))

        return struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy

    def _evaluate_model(self, sqls):
        raise RuntimeError

    def _generate_sqls(self, plans, sqls):
        if self.sql_cache.exist():
            print("shifted sql trigger cache")
            res = self.sql_cache.read()
            self.structure_sqls = res[0]
            self.table_2_sqls = res[1]
            self.join_keys_2_sqls = res[2]
            self.filter_table_2_col_2_range_sqls = res[3]
            self.statistic.build_train(plans, sqls)
        else:
            self.statistic.build(plans, sqls)
            self.sql_producer.build()
            print("generating shifted sqls")
            budget = int(len(sqls) * shifted_sql_budget_ratio / 4.0)
            # generate
            print("generating structure sqls")
            self.structure_sqls = self.sql_producer.generate_structure_sql(budget)
            print("the size of structure sql is {}".format(len(self.structure_sqls)))

            print("generating table sqls")
            self.table_2_sqls = self.sql_producer.generate_shifted_table_sql(budget)
            print("the size of table sql is {}".format(len(self.table_2_sqls)))

            print("generating join key sqls")
            self.join_keys_2_sqls = self.sql_producer.generate_shifted_join_key_sql(budget)
            print("the size of join sql is {}".format(len(self.join_keys_2_sqls)))

            print("generating filter sqls")
            self.filter_table_2_col_2_range_sqls = self.sql_producer.generate_filter_sqls(budget)
            print("the size of filter sql is {}".format(len(self.filter_table_2_col_2_range_sqls)))

            self.sql_cache.save(
                [self.structure_sqls, self.table_2_sqls, self.join_keys_2_sqls, self.filter_table_2_col_2_range_sqls])

    def _collect_all_sqls(self):
        sqls = []
        sqls += self._collect_structure_sqls()
        sqls += self._collect_join_key_sqls()
        sqls += self._collect_table_sqls()
        sqls += self._collect_filter_sqls()
        assert len(set(sqls)) == len(sqls)
        return sqls

    def _generate_run_pg_plan(self, sql, hint=""):
        result = run_query("{} EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON)".format(hint) + sql,
                           None, self.db,
                           time_out=30000)
        return json.dumps(result[1][0][0])

    def _to_json(self, query_result):
        return json.dumps(query_result[1][0][0])

    def _generate_plans(self, sqls):
        print("generate plan and then run")
        for i, sql in enumerate(sqls):
            if sql not in self.sql_2_plans:
                if i % max(int(len(sqls) / len(sqls)), 1) == 0:
                    print("total sql size is {}, cur is {}".format(len(sqls), i))
                try:
                    pg_plan = self._generate_run_pg_plan(sql)
                    self.sql_2_plans[sql] = [pg_plan]
                except Exception as e:
                    print(e)
                    self.sql_2_plans[sql] = [None]

    def _collect_structure_sqls(self):
        sqls = []
        for s in list(self.structure_sqls.values()):
            sqls += s
        return sqls

    def _collect_join_key_sqls(self):
        sqls = []
        for s in list(self.join_keys_2_sqls.values()):
            sqls += s
        return sqls

    def _collect_table_sqls(self):
        sqls = []
        for s in list(self.table_2_sqls.values()):
            sqls += s
        return sqls

    def _collect_filter_sqls(self):
        sqls = []
        for col_2_range_sqls in list(self.filter_table_2_col_2_range_sqls.values()):
            for range_sqls in list(col_2_range_sqls.values()):
                for item in range_sqls:
                    sqls += item[2]
        return sqls

    def write_sqls(self):
        i = 0
        with open("stats_generate_sqls", "w") as f:
            for sql in self.sql_2_plans.keys():
                f.write("q{}{}{}\n".format(i, SEP, sql))
                i += 1


class LeroShiftedManager(ShiftedManager):

    def __init__(self, db: str, training_set_name, pair_model, algo):
        super().__init__(db, training_set_name, pair_model, algo)
        self.pair_model = pair_model

    def _is_identical_plan(self, pg_plan: str, another_plan: str):
        p1 = json.dumps(json_str_to_json_obj(pg_plan)["Plan"])
        p2 = json.dumps(json_str_to_json_obj(another_plan)["Plan"])
        return p1 == p2

    def _evaluate_model(self, sqls):
        total_count = 0
        correct_count = 0
        all_plans = []
        for sql in sqls:
            plans = self.sql_2_plans[sql]
            if plans[0] is not None:
                all_plans.append(plans[0])

        for i in range(len(all_plans)):
            for j in range(i + 1, len(all_plans)):
                p1, p2 = all_plans[i], all_plans[j]
                if self._is_smaller_from_model(p1, p2) == self._is_smaller_from_actual(p1, p2):
                    correct_count += 1
                total_count += 1

        return float(correct_count) / total_count if total_count > 0 else 0.0, total_count

    def _is_smaller_from_model(self, p1, p2):
        """
        return whether the prediction time of p1 is smaller than p2's
        :param p1:
        :param p2:
        :return:
        """
        model = self.model
        features = model.to_feature([p1, p2])
        values = model.predict(features)
        if float(values[0]) <= float(values[1]):
            return True
        return False

    def _is_smaller_from_actual(self, p1, p2):
        """
        return whether the actual time of p1 is smaller than p2's
        :param p1:
        :param p2:
        :return:
        """
        t1 = json_str_to_json_obj(p1)["Execution Time"]
        t2 = json_str_to_json_obj(p2)["Execution Time"]
        if t1 <= t2:
            return True
        return False
