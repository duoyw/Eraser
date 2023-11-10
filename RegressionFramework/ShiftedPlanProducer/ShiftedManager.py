import json

import torch
from sql_metadata import Parser

from RegressionFramework.Common.Cache import Cache
from RegressionFramework.Common.TimeStatistic import TimeStatistic
from RegressionFramework.Plan.Plan import Plan
from RegressionFramework.ShiftedPlanProducer.generateSql import SqlProduceManager
from RegressionFramework.ShiftedPlanProducer.sqlTemplate import SqlTemplate
from RegressionFramework.ShiftedPlanProducer.statistic import Statistic, StatisticTest
from RegressionFramework.config import shifted_sql_budget_ratio, shifted_space_thres
from RegressionFramework.utils import json_str_to_json_obj, cal_ratio, absolute_relative_error, \
    absolute_relative_error_with_limit
from test_script.config import SEP
from Hyperqo.plan2latency import get_hyperqo_result
from Perfguard.plan2score import get_perfguard_result
from test_script.utils import run_query


class ShiftedManager:
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

        self.statistic_test: StatisticTest = None

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

        # self.write_sqls()
        # exit()

        self._generate_plans(sqls)

        self.struct_accuracy, self.join_key_accuracy, self.table_accuracy, self.filter_accuracy = self._evaluation()
        self.table_accuracy = self.table_accuracy if self.table_accuracy > 0 else 0.5

        self.delete_structure_enable = True if self.struct_accuracy < shifted_space_thres else False
        self.delete_join_enable = True if self.join_key_accuracy < shifted_space_thres else False
        self.delete_table_enable = True if self.table_accuracy < shifted_space_thres else False
        self.delete_filter_enable = True if self.filter_accuracy < shifted_space_thres else False

        self.statistic_test = StatisticTest(self.statistic.statistic_train, self.delete_structure_enable,
                                            self.delete_join_enable,
                                            self.delete_table_enable,
                                            self.delete_filter_enable)

        # self.write_accuracy(struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy)

        # end
        self.sql_2_plans_cache.save([self.sql_2_plans])

    def write_accuracy(self, struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy):
        with open("./unexpected_plan_accuracy.txt", "a") as f:
            line = "train_file is {}, structure is {}, join is {}, table is {}, filter is {} \n".format(
                self.training_set_name, struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy)
            f.write(line)
        exit()

    def is_filter(self, plan):
        return self.statistic_test.is_shifted(plan)

    def get_subspace_result(self, confidence=None):
        if confidence is None:
            return self.delete_structure_enable, self.delete_join_enable, self.delete_table_enable, self.delete_filter_enable
        else:
            self.statistic_test.structure_enable = delete_structure_enable = True if self.struct_accuracy < confidence else False
            self.statistic_test.join_enable = delete_join_enable = True if self.join_key_accuracy < confidence else False
            self.statistic_test.table_enable = delete_table_enable = True if self.table_accuracy < confidence else False
            self.statistic_test.filter_enable = delete_filter_enable = True if self.filter_accuracy < confidence else False

            return delete_structure_enable, delete_join_enable, delete_table_enable, delete_filter_enable

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
            TimeStatistic.start("evaluation")
            struct_accuracy, valid_struct_sqls_count = self._evaluate_model(struct_sqls)
            TimeStatistic.end("evaluation")
            # TimeStatistic.report()
            # exit()

            join_key_sqls = self._collect_join_key_sqls()
            join_key_accuracy, valid_join_key_sqls_count = self._evaluate_model(join_key_sqls)

            table_sqls = self._collect_table_sqls()
            table_accuracy, valid_table_sqls_count = self._evaluate_model(table_sqls)

            filter_sql = self._collect_filter_sqls()
            filter_accuracy, valid_filter_sql_count = self._evaluate_model(filter_sql)

            self.space_evaluation_cache.save([struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy])

        return struct_accuracy, join_key_accuracy, table_accuracy, filter_accuracy

    def _evaluate_model(self, sqls):
        raise RuntimeError

    def _generate_sqls(self, plans, sqls):
        self.statistic.build(plans, sqls)
        if self.sql_cache.exist():
            # print("shifted sql trigger cache")
            res = self.sql_cache.read()
            self.structure_sqls = res[0]
            self.table_2_sqls = res[1]
            self.join_keys_2_sqls = res[2]
            self.filter_table_2_col_2_range_sqls = res[3]
        else:
            self.sql_producer.build()
            # print("generating shifted sqls")
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
        # print("generate plan and then run")
        for i, sql in enumerate(sqls):
            if sql not in self.sql_2_plans:
                # if i % max(int(len(sqls) / len(sqls)), 1) == 0:
                #     print("total sql size is {}, cur is {}".format(len(sqls), i))
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

    def _generate_plans(self, sqls):
        super()._generate_plans(sqls)
        # self._generate_another_plans(sqls)
        # for sql in sqls:
        #         pg_plan = self._generate_run_pg_plan(sql)
        #     another_plan = self._generate_another_plans(sql, pg_plan)

    def _generate_another_plans(self, sqls):
        for i, sql in enumerate(sqls):

            pg_plan = None
            # if plan is time out, and it will be set None
            plans = self.sql_2_plans[sql]
            if len(plans) > 1:
                # exist another plan
                continue
            if plans[0] is None:
                self.sql_2_plans[sql] = [None, None]
                continue

            if i % max(int(len(sqls) / len(sqls)), 1) == 0:
                print("_generate_another_plans: total sql size is {}, cur is {}".format(len(sqls), i))
            pg_plan = plans[0]

            # select valid leading hint
            tables = Parser(sql).tables
            is_find = False
            for i in range(len(tables) - 1):
                t1 = tables[i]
                t2 = tables[i + 1]
                q = "/*+  Leading({},{}) */ EXPLAIN (VERBOSE, COSTS, SUMMARY, FORMAT JSON) ".format(t1, t2) + sql
                result = run_query(q, None, self.db)
                another_plan = self._to_json(result)
                if not self._is_identical_plan(pg_plan, another_plan):
                    # run
                    is_find = True
                    try:
                        another_plan = self._generate_run_pg_plan(sql, "/*+  Leading({},{}) */".format(t1, t2))
                        self.sql_2_plans[sql] = [pg_plan, another_plan]
                    except:
                        self.sql_2_plans[sql] = [pg_plan, None]
                    break
            if not is_find:
                self.sql_2_plans[sql] = [pg_plan, None]

    def _is_identical_plan(self, pg_plan: str, another_plan: str):
        p1 = json.dumps(json_str_to_json_obj(pg_plan)["Plan"])
        p2 = json.dumps(json_str_to_json_obj(another_plan)["Plan"])
        return p1 == p2

    def _evaluate_model(self, sqls):
        if len(sqls) == 0:
            return 0, 0

        total_count = 0
        correct_count = 0
        all_plans = []
        for sql in sqls:
            plans = self.sql_2_plans[sql]
            if plans[0] is not None:
                all_plans.append(plans[0])

        left_plans = []
        right_plans = []

        for i in range(len(all_plans)):
            for j in range(i + 1, len(all_plans)):
                left_plans.append(all_plans[i])
                right_plans.append(all_plans[j])

        model_res = self._is_smaller_from_model_batch(left_plans, right_plans)
        actual_res = self._is_smaller_from_actual_batch(left_plans, right_plans)

        for i in range(0, len(model_res)):
            if model_res[i] == actual_res[i]:
                correct_count += 1

        total_count = len(model_res)

        return float(correct_count) / total_count if total_count > 0 else 0.0, total_count

    # def _evaluate_model(self, sqls):
    #     total_count = 0
    #     correct_count = 0
    #     all_plans = []
    #     for sql in sqls:
    #         plans = self.sql_2_plans[sql]
    #         if plans[0] is not None:
    #             all_plans.append(plans[0])
    #
    #     for i in range(len(all_plans)):
    #         for j in range(i + 1, len(all_plans)):
    #             p1, p2 = all_plans[i], all_plans[j]
    #             if self._is_smaller_from_model(p1, p2) == self._is_smaller_from_actual(p1, p2):
    #                 correct_count += 1
    #             total_count += 1
    #
    #     return float(correct_count) / total_count if total_count > 0 else 0.0, total_count

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
        torch.cuda.empty_cache()
        return False

    def _is_smaller_from_model_batch(self, plans1, plans2):
        """
        return whether the prediction time of p1 is smaller than p2's
        :param p1:
        :param p2:
        :return:
        """
        model = self.model
        features1 = model.to_feature(plans1)
        features2 = model.to_feature(plans2)
        values1 = model.predict(features1)
        values2 = model.predict(features2)

        res = []
        for i in range(0, len(values1)):
            res.append(values1[i] <= values2[i])
        return res

    def _is_smaller_from_actual_batch(self, plans1, plans2):
        """
        return whether the actual time of p1 is smaller than p2's
        :param p1:
        :param p2:
        :return:
        """
        res = []

        for i in range(0, len(plans1)):
            p1 = plans1[i]
            p2 = plans2[i]
            t1 = json_str_to_json_obj(p1)["Execution Time"]
            t2 = json_str_to_json_obj(p2)["Execution Time"]
            res.append(t1 <= t2)
        return res

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


class HyperqoShiftedManager(ShiftedManager):
    def _evaluate_model(self, sqls):
        if len(sqls) == 0:
            return 0, 0
        total_count = 0
        correct_count = 0

        ratios = []

        for sql in sqls:
            plans = self.sql_2_plans[sql]
            if plans[0] is not None:
                predicts = get_hyperqo_result(plans, sql, self.model)
                if predicts is None:
                    continue
                for i, p in enumerate(plans):
                    plan = json_str_to_json_obj(p)
                    ratios.append(absolute_relative_error_with_limit(predicts[i], plan["Execution Time"]))

        return float(correct_count) / total_count if total_count > 0 else 0.0, total_count

    def is_filter(self, plan):
        # tmp
        return False
        return self.statistic_test.is_shifted(plan)



class PerfguardShiftedManager(LeroShiftedManager):
    def _evaluate_model(self, sqls):
        if len(sqls) == 0:
            return 0, 0
        total_count = 0
        correct_count = 0
        all_plans = []
        for sql in sqls:
            plans = self.sql_2_plans[sql]
            if plans[0] is not None:
                all_plans.append(plans[0])
        if len(all_plans) < 2:
            return 0.0, 0
        left = []
        right = []
        for i in range(len(all_plans)):
            for j in range(len(all_plans)):
                left.append(all_plans[i])
                right.append(all_plans[j])
        left = [json_str_to_json_obj(p) for p in left]
        right = [json_str_to_json_obj(p) for p in right]
        results = get_perfguard_result(left, right, self.model)

        for i in range(len(all_plans)):
            p1 = all_plans[i]
            for j in range(len(all_plans)):
                p2 = all_plans[j]
                result = results[i * len(all_plans) + j]
                if self._is_smaller_from_actual(p1, p2) and result == 1:
                    correct_count += 1
                elif self._is_smaller_from_actual(p1, p2) == False and result == 0:
                    correct_count += 1
                total_count += 1

        return float(correct_count) / total_count if total_count > 0 else 0.0, total_count
