import sys
import unittest

import math
import numpy as np

from RegressionFramework.utils import get_beta_params, get_beta_dynamic_params

sys.path.append("test_script/")
sys.path.append("Hyperqo/")
sys.path.append("Perfguard/")

from pandas import DataFrame
from drawConfig import name_2_color, get_plt, regression_algo_name, font_size, capitalize
from RegressionFramework.Common.TimeStatistic import TimeStatistic
from RegressionFramework.Plan.Plan import json_str_to_json_obj
from RegressionFramework.Plan.PlanFactory import PlanFactory
from RegressionFramework.utils import flat_depth2_list, draw_by_agg, read_sqls, to_rgb_tuple
from RegressionFramework.RegressionFramework import RegressionFramework, LeroRegressionFramework
from RegressionFramework.config import data_base_path, model_base_path
from test_script.model import LeroModelPairWise

input_ratio = None


class LeroTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.algo = "lero"
        global input_ratio
        input_ratio = sys.argv[2] if len(sys.argv) == 3 else None
        self.regression_gap = 10

    def test_static_all_job(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_job()

    def test_static_all_stats(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_stats()

    def test_static_all_tpch(self):
        global input_ratio
        for ratio in [1, 2, 3, 4]:
            input_ratio = ratio
            self.test_tpch()

    def test_job(self):
        ratio = 2 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_job{}.log.training".format(ratio)
        sql_file_name = "job{}.txt".format(ratio)
        test_file_name = "job_test"

        self.performance(train_file_name, test_file_name, sql_file_name, "imdb", ratio=ratio)

    def test_stats(self):
        ratio = 4 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_stats{}.log.training".format(ratio)
        sql_file_name = "stats{}.txt".format(ratio)
        test_file_name = "stats_test"

        self.performance(train_file_name, test_file_name, sql_file_name, "stats", ratio=ratio)

    def test_tpch(self):
        ratio = 4 if input_ratio is None else input_ratio
        print("data ratio is {}".format(ratio))
        train_file_name = "lero_tpch{}.log.training".format(ratio)
        sql_file_name = "tpch{}.txt".format(ratio)
        test_file_name = "tpch_test"

        self.performance(train_file_name, test_file_name, sql_file_name, "tpch", ratio=ratio)

    def test_dynamic_job(self):
        train_file_name = "lero_job4.log.training"
        sql_file_name = "job4.txt"
        self.dynamic_performance(train_file_name, sql_file_name, "imdb")

    def test_dynamic_tpch(self):
        train_file_name = "lero_tpch4.log.training"
        sql_file_name = "tpch4.txt"
        self.dynamic_performance(train_file_name, sql_file_name, "tpch")

    def dynamic_performance(self, train_file_name, train_sql_file_name, db, is_dynamic_database=False, save_suffix=""):
        train_sqls = self.read_sqls(self.get_data_file_path(train_sql_file_name))
        train_plans_for_query = self.read_data(self.get_data_file_path(train_file_name))
        self._add_id_to_plan(train_plans_for_query)

        # load lero model
        print("predict")
        gap = 100
        # gap = 50
        count = math.ceil(len(train_plans_for_query) / float(gap))

        regression_results = {}
        algo_results = []
        best_results = []
        pg_results = []
        candidate_thres = [get_beta_dynamic_params(self.algo, db)]
        for i in range(0, count):
            print("cur iteration is {}".format(i))

            model_path = self.get_dynamic_model_name(train_file_name, i)
            model = self.load_model(model_path, db)

            min_pos = 0
            max_pos = max(i * gap, 1)
            tmp_train_plans_for_query = train_plans_for_query[min_pos:max_pos]
            train_plans = flat_depth2_list(tmp_train_plans_for_query)
            cur_train_sqls = train_sqls[min_pos: max_pos]
            self._add_plan_metric(train_plans_for_query[min_pos:max_pos], model, cur_train_sqls)
            regression_framework = self._init_regression_framework(train_plans, tmp_train_plans_for_query,
                                                                   cur_train_sqls, db,
                                                                   "dynamic" + train_file_name + "{}".format(i), model,
                                                                   mode="dynamic")
            regression_framework.build()

            test_plans_for_query = train_plans_for_query[i * gap:(i + 1) * gap]
            test_sqls = train_sqls[i * gap:(i + 1) * gap]

            # algo result
            algo_results += self.select_plan_by_model(model, test_plans_for_query, model_path, test_sqls, db)
            best_results += self.select_best_plan_times(test_plans_for_query)
            pg_results += self.select_pg_plan_times(test_plans_for_query)

            # algo + regression result
            predict_latencies_for_queries = [None] * len(test_plans_for_query)
            plan_id_2_confidence = {}
            for thres in candidate_thres:
                # print("cur thres {}".format(thres))
                result = self.select_plan_by_lero_model_regression(model, regression_framework, test_plans_for_query,
                                                                   predict_latencies_for_queries, thres,
                                                                   plan_id_2_confidence, test_sqls, None)
                if str(thres) not in regression_results:
                    regression_results[str(thres)] = []
                regression_results[str(thres)] += result

        if len(algo_results) != len(train_plans_for_query):
            print(len(algo_results))
            raise RuntimeError

        for thres in candidate_thres:
            algo_2_values = {
                self.algo: algo_results,
                "{}_{}".format(self.algo, regression_algo_name): regression_results[str(thres)],
                "PostgreSQL": pg_results
            }
            self.draw_scatter_chart(algo_2_values, db, "dynamic_{}_{}".format(db, self.algo, str(thres)),
                                    self.algo.capitalize(),
                                    width_inc=200, show_legend=True)

    def performance(self, train_file_name, test_file_name, train_sql_file_name, db, model_path=None, config_dict=None,
                    save_suffix="", is_dynamic_database=False, save_prefix="", forest=100,
                    model_with_generate_sql=False, ood_thres=None, ratio=None):
        train_sqls = self.read_sqls(self.get_data_file_path(train_sql_file_name))

        # read training data
        train_plans_for_query = self.read_data(self.get_data_file_path(train_file_name))

        # read test data
        test_plans_for_query = self.read_data(self.get_data_file_path(test_file_name))
        test_sqls = self.read_sqls(self.get_data_file_path(test_file_name))

        self._add_id_to_plan(test_plans_for_query)

        # load lero model
        print("load {} model".format(self.algo))
        assert model_path is None
        model_path = self.get_model_name(train_file_name, model_with_generate_sql)
        model = self.load_model(model_path, db)
        self._add_plan_metric(train_plans_for_query, model, train_sqls)
        train_plans = flat_depth2_list(train_plans_for_query)

        # building regression framework
        print("build Eraser")
        TimeStatistic.start("{}_{}_Eraser_Training".format(self.algo, db))
        regression_framework = self._init_regression_framework(train_plans, train_plans_for_query, train_sqls, db,
                                                               train_file_name, model,
                                                               config_dict=config_dict, forest=forest)
        regression_framework.build()
        TimeStatistic.end("{}_{}_Eraser_Training".format(self.algo, db))

        print("select plan for all algorithms with or without Eraser")
        # choose plans
        lero_results = self.select_plan_by_model(model, test_plans_for_query, model_path, test_sqls, db)

        predict_latencies_for_queries = [None] * len(test_plans_for_query)
        plan_id_2_confidence = {}
        candidate_thres = [get_beta_params(self.algo, db)]
        regression_results = {}
        for thres in candidate_thres:
            TimeStatistic.start("{}".format(self.algo))
            result = self.select_plan_by_lero_model_regression(model, regression_framework, test_plans_for_query,
                                                               predict_latencies_for_queries, thres,
                                                               plan_id_2_confidence, test_sqls, ood_thres=ood_thres)
            TimeStatistic.end("{}".format(self.algo))
            regression_results[str(thres)] = result

        best_results = self.select_best_plan_times(test_plans_for_query)
        pg_results = self.select_pg_plan_times(test_plans_for_query)

        if db == "imdb" and self.algo == "lero":
            for thres in candidate_thres:
                self.draw_regression_per_query(lero_results, pg_results, regression_results[str(thres)], ratio=ratio)

        self.draw(lero_results, regression_results, pg_results, best_results, candidate_thres, train_file_name, db,
                  save_suffix=save_suffix, save_prefix=save_prefix)

    def draw(self, algo_result, regression_results, pg_results, best_results, candidate_thres, train_file_name, db,
             save_prefix="", save_suffix=""):
        queries_name = ["Q{}".format(i) for i in range(len(algo_result))]
        # print .csv
        df = DataFrame(
            {capitalize(self.algo): algo_result,
             "PostgreSQL": pg_results,
             }
        )
        for thres in candidate_thres:
            df["{}-Eraser".format(capitalize(self.algo))] = regression_results[str(thres)]

        df["qn"] = queries_name

        # lero_job4.log.training -> 4
        train_ratio = train_file_name.split(".")[0][-1]
        workload = db if db != "imdb" else "job"
        name = "{}_{}_test{}".format(self.algo, workload, train_ratio)
        # df.to_csv(
        #     "RegressionFramework/fig/{}{}_performance{}.csv".format(save_prefix, name, save_suffix))

        # draw
        y_names = list(df.columns)
        y_names.remove("qn")
        # draw2(df, x_name="qn", y_names=y_names, file_name="overall")
        draw_by_agg(df, y_names=y_names, agg="mean", file_name="{}{}{}".format(save_prefix, name, save_suffix))

    def draw_scatter_chart(self, algo_2_values, db, file, title, x_ticks=None, width_inc=0, font_inc=0, x_gap=200,
                           show_legend=False):
        plt = get_plt()
        plt.figure(figsize=(10 + width_inc / 100, 10))
        for algo, values in algo_2_values.items():
            algo_alias = algo.split("-" if db == "tpcds" else "_")[0]
            color = name_2_color[algo_alias.lower()]
            # symbol = name_2_scatter_symbol[algo.lower()]
            line_dash = ":" if regression_algo_name in algo else "-"
            values = self.to_minute(values, db)
            if algo == "hyperqo_{}".format(regression_algo_name) and db.lower() == "tpch":
                # To differentiate the two curves of HyperQO-Eraser and HyperQO,
                # we made slight adjustments to the Eraser's result, causing it to have slightly worse performance
                inc_value = 0.015
                values = [v + inc_value for v in values]

            # values=list(np.log(np.array(values)))
            plt.plot(list(range(len(values))), self.accumulate(values), linestyle=line_dash, label=capitalize(algo),
                     linewidth=10,
                     color=to_rgb_tuple(color))

        cur_font_size = font_size + font_inc - 10
        plt.ylabel("E2E Execution Total Time (Mins)", fontsize=cur_font_size)
        plt.xlabel("# of queries", fontsize=cur_font_size)
        plt.yticks(size=cur_font_size)
        plt.xticks(
            [i for i in range(len(values)) if i % x_gap == 0],
            [i for i in range(len(values)) if i % x_gap == 0],
            size=cur_font_size, weight='bold')
        if show_legend:
            # plt.legend()._spacing = 0.5
            plt.legend(loc='upper center', frameon=True, handletextpad=0.3, columnspacing=0.5,
                       bbox_to_anchor=(0.33, 1.03),
                       ncol=1,
                       fontsize=cur_font_size)
        plt.grid()
        plt.tight_layout()
        plt.savefig("RegressionFramework/fig/{}.png".format(file), format="png")
        plt.show()

    def _init_regression_framework(self, train_plans, plans_for_queries, train_sqls, db, train_file_name, model,
                                   mode="static",
                                   config_dict=None, forest=100):
        return LeroRegressionFramework(train_plans, train_sqls, db, train_file_name, model, mode=mode,
                                       config_dict=config_dict, forest=forest, plans_for_queries=plans_for_queries)

    def _add_plan_metric(self, plans_for_query, model: LeroModelPairWise, sqls):
        for plans in plans_for_query:
            predict_latencies = self.predict(model, plans)
            total_count = 0
            correct_count = 0
            for i in range(len(predict_latencies)):
                p1 = plans[i]
                t1 = predict_latencies[i]
                for j in range(len(predict_latencies)):
                    # if i == j:
                    #     continue
                    p2 = plans[j]
                    t2 = predict_latencies[j]
                    if t1 <= t2 and p1["Execution Time"] <= p2["Execution Time"]:
                        correct_count += 1
                    total_count += 1
                p1["metric"] = correct_count / total_count if total_count > 0 else 0.5
                # print(p1["metric"])
                p1["predict"] = t1

    def select_plan_by_model(self, model: LeroModelPairWise, plans_for_query, model_path, sqls, db):
        times_for_query = []
        for plans in plans_for_query:
            predicts = list(self.predict(model, plans))
            idx = predicts.index(min(predicts))
            times_for_query.append(plans[idx]["Execution Time"])
        return times_for_query

    def select_plan_by_lero_model_regression(self, model: LeroModelPairWise, regression_framework: RegressionFramework,
                                             plans_for_query, latencies_for_queries, thres, plan_id_2_confidence, sqls,
                                             ood_thres=None):
        total_count = 0
        negative_count = 0
        exceed_count = 0
        valid_queries_count = 0

        select_times_for_query = []

        for i, plans in enumerate(plans_for_query):
            id_to_win_count = {}
            if latencies_for_queries[i] is None:
                latencies_for_queries[i] = model.predict(model.to_feature(plans_for_query[i]))
            predict_latencies = latencies_for_queries[i]
            query_impact_flag = False
            for j in range(len(plans)):
                plan1 = plans[j]
                win_count = 0
                for k in range(len(plans)):
                    plan2 = plans[k]
                    if j == k:
                        continue
                    if predict_latencies[j] < predict_latencies[k]:
                        total_count += 1
                        p1 = PlanFactory.get_plan_instance("pg", plan1)
                        p2 = PlanFactory.get_plan_instance("pg", plan2)
                        confidence = regression_framework.evaluate(p1, p2, ood_thres=ood_thres)
                        if confidence >= 1:
                            exceed_count += 1
                        elif confidence == -1 or confidence == -1.0:
                            negative_count += 1
                        if confidence >= thres:
                            win_count += 1
                        else:
                            query_impact_flag = True
                id_to_win_count[j] = win_count
            if query_impact_flag:
                valid_queries_count += 1

            db_win_count = id_to_win_count[0]
            id_win_count = sorted(id_to_win_count.items(), key=lambda x: x[1])
            choose_idx = self._get_idx_min_predict_latency_with_max_count(id_win_count, predict_latencies)
            choose_idx = 0 if db_win_count >= id_to_win_count[choose_idx] else choose_idx
            latency = json_str_to_json_obj(plans[choose_idx])["Execution Time"]
            select_times_for_query.append(latency)
        return select_times_for_query

    def _get_idx_min_predict_latency_with_max_count(self, id_win_count, predict_latencies):
        """
        :param id_win_count: [(id,win_count),(),...], sorted count by ascending order
        :param predict_latencies: [latency1,...]
        :return:
        """
        count = id_win_count[-1][1]
        candidate_ids = []
        for items in id_win_count:
            if items[1] == count:
                candidate_ids.append(items[0])

        choose_idx = -1
        choose_predict_latency = math.inf
        for idx in candidate_ids:
            if predict_latencies[idx] < choose_predict_latency:
                choose_idx = idx
                choose_predict_latency = predict_latencies[idx]
        return choose_idx

    def get_model_name(self, data_file_name, model_with_generate_sql=False):
        if not model_with_generate_sql:
            return model_base_path + "test_model_on_0_{}".format(data_file_name)
        else:
            return model_base_path + "test_model_on_0_{}(incluing_generated_sqls)".format(data_file_name)

    def get_dynamic_model_name(self, data_file_name: str, count):
        prefix = "dynamic"
        data_file_name = data_file_name[0:-14] + str(count) + data_file_name[-13:]
        return model_base_path + "{}/dynamic_model_{}".format(prefix, data_file_name)

    def load_model(self, model_name, db):
        lero_model = LeroModelPairWise(None)
        lero_model.load(model_name)
        return lero_model

    def predict(self, model: LeroModelPairWise, plans):
        features = model.to_feature(plans)
        return model.predict(features)

    def read_data(self, data_file_path):
        plans_for_query = []
        with open(data_file_path, "r") as f:
            for line in f.readlines():
                plans = line.split("#####")[1:]
                plans = [json_str_to_json_obj(p) for p in plans]
                plans_for_query.append(plans)
        return plans_for_query

    def get_data_file_path(self, data_file_name):
        return data_base_path + data_file_name

    def read_sqls(self, data_file_path):
        return read_sqls(data_file_path)

    def _add_id_to_plan(self, plans_for_query):
        plan_id = 0
        for plans in plans_for_query:
            for plan in plans:
                plan["id"] = plan_id
                plan_id += 1

    @classmethod
    def to_plan_objects(cls, ):
        pass

    def select_pg_plan_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(plans[0]["Execution Time"])
            else:
                res.append(plans[0]["Execution Time"])
        return res

    def select_best_plan_times(self, plans_for_query):
        res = []
        for idx, plans in enumerate(plans_for_query):
            if len(plans) >= 2:
                res.append(self.find_best_plan(plans)[1])
            else:
                res.append(plans[0]["Execution Time"])
        return res

    def find_best_plan(self, plans):
        best_plan_idx = -1
        best_latency = float('inf')
        for i, plan in enumerate(plans):
            plan = json_str_to_json_obj(plan)
            if best_latency > plan["Execution Time"]:
                best_plan_idx = i
                best_latency = plan["Execution Time"]
        return best_plan_idx, best_latency

    def accumulate(self, values):
        new_arr = []
        for i in range(len(values)):
            new_arr.append(sum(values[:i + 1]))
        return new_arr

    def to_minute(self, values, db):
        if db == "tpcds":
            return [v / 60.0 for v in values]
        return [v / 60.0 / 1000 for v in values]

    def draw_regression_per_query(self, algo_value, pg_values, lero_r_value, ratio):
        workload = "job"
        algo_2_values = {
            "Lero": self.arrange(pg_values, algo_value),
            "Lero_{}".format(regression_algo_name): self.arrange(pg_values, lero_r_value)
        }
        x = ["{}".format(v * self.regression_gap) for v in range(1, int(100 / self.regression_gap) + 1)]
        x.append(">")
        self.draw_grouped_bar_chart(x, algo_2_values, "Regression Ratios (%)",
                                    "{}_{}_regression_per_query_{}".format(self.algo, workload, ratio),
                                    24,
                                    bar_width=0.3,
                                    show_legend=True)

    def draw_grouped_bar_chart(self, x: list, algo_2_values: dict, x_title, file, y_max, show_x_title=True,
                               show_symbol=True,
                               show_legend=False, bar_width=0.12, ):
        plt = get_plt()
        plt.figure(figsize=(16, 10))
        i = 0
        values = None
        for algo, values in algo_2_values.items():
            algo_alias = algo
            # color = name_2_color[algo_alias.lower()]
            color = "rgb(255,0,0)" if regression_algo_name in algo_alias else "rgb(0,127,0)"
            symbol = ""
            values = algo_2_values[algo]
            x_vals = np.arange(len(values)) + (bar_width) * i
            label = algo.replace("_", "-")
            plt.bar(x_vals, values, width=bar_width, label=capitalize(label), color=to_rgb_tuple(color), hatch=symbol,
                    edgecolor='black',
                    linewidth=1)
            i += 1
        # Change the bar mode
        cur_font_size = font_size
        plt.ylabel("# of queries", fontsize=cur_font_size)
        plt.xlabel(x_title, fontsize=cur_font_size)
        y = list(range(0, y_max, 2))

        plt.yticks(y, y, size=cur_font_size)
        plt.xticks(
            np.arange(len(x)) + bar_width * (len(algo_2_values) / 2 - 0.5), x,
            size=cur_font_size, weight='bold')
        if show_legend:
            plt.legend(loc='upper center', frameon=False, handletextpad=0.3, columnspacing=0.5, handlelength=1.0,
                       bbox_to_anchor=(0.3, 1.05),
                       ncol=len(algo_2_values),
                       fontsize=cur_font_size - 0)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig("RegressionFramework/fig/{}.png".format(file), format="png")
        plt.show()

    def arrange(self, pg_values, algo_values):
        regression_ratios = [0] * (int(100 / self.regression_gap) + 1)
        assert len(pg_values) == len(algo_values)
        for i in range(len(pg_values)):
            pg_val = pg_values[i]
            algo_val = algo_values[i]
            ratio = (algo_val - pg_val) / pg_val * 100
            if ratio <= 0:
                continue
            elif ratio > 100:
                ratio = 101
            regression_ratios[int(ratio) // self.regression_gap] += 1
        return regression_ratios


if __name__ == '__main__':
    unittest.main()
