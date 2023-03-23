from ImportantConfig import Config
from math import e
from PGUtils import pgrunner
import torch
from KNN import KNN
import time
from NET import *
from mcts import *
import json
import psycopg2
# from multiprocessing import Pool
from multiprocessing import pool as Pool


def formatFloat(t):
    try:
        return " ".join(["{:.4f}".format(x) for x in t])
    except:
        return " ".join(["{:.4f}".format(x) for x in [t]])


config = Config()


class Timer:
    def __init__(self, ):
        from time import time
        self.timer = time
        self.startTime = {}

    def reset(self, s):
        self.startTime[s] = self.timer()

    def record(self, s):
        return self.timer() - self.startTime[s]


timer = Timer()


class Hinter:
    def __init__(self, CONNECTION_STR, save_plan_path, model, sql2vec, value_extractor, mcts_searcher=None):
        # model = TreeNet(tree_builder= tree_builder,value_network = value_network)
        self.model = model  # Net.TreeNet
        self.sql2vec = sql2vec  #
        self.value_extractor = value_extractor
        # 存储pg默认的计划的相关计划和执行时间
        self.pg_planningtime_list = []
        self.pg_runningtime_list = []  # default pg running time
        # 存储'pg'或leading_hin
        self.chosen_plan = []  # eg((leading ,pg))
        # 存储的内容和self.hinter_runtime_list一样，除非发生time_out，则还要存储max_time_out
        self.hinter_time_list = []  # final plan((eg [(leading,),(leading,pg),...]))
        # 存储执行mcts的时间
        self.mcts_time_list = []  # time for mcts
        # 存储执行MHPE的时间
        self.MHPE_time_list = []

        # 类似于 self.hinter_runtime_list
        self.hinter_planningtime_list = []  # chosen hinter running time,include the timeout
        # 存储着真实执行时间。当best_hint的预测性能大于pg默认计划的性能时，存储带有hint的query生成的计划的性能，否则存储pg默认计划的性能
        self.hinter_runtime_list = []

        self.knn = KNN(10)
        self.mcts_searcher = mcts_searcher
        self.hinter_times = 0
        self.CONNECTION_STR = CONNECTION_STR
        self.save_plan_path = save_plan_path

        # 计数
        # self.one_times = 0
        # self.two_times = 0

    def findBestHint(self, plan_json_PG, alias, sql_vec, sql):
        """ self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it':
          16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an
         1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'a
         n': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}

         alias = {'k', 'mc', 'an', 'mk', 'ci', 't', 'n', 'cn'}
         """
        # [20, 21, 3, 2, 8, 10, 7, 5]
        alias_id = [self.sql2vec.aliasname2id[a] for a in alias]
        timer.reset('mcts_time_list')
        # [(5, 7), (3, 5), (2, 7), (7, 10), (8, 10)]
        id_joins_with_predicate = [(self.sql2vec.aliasname2id[p[0]], self.sql2vec.aliasname2id[p[1]]) for p in
                                   self.sql2vec.join_list_with_predicate]
        # [(5, 7), (2, 21), (2, 10), (3, 5), (2, 5), (2, 20), (20, 21), (2, 7), (5, 10), (7, 10), (8, 10)]
        id_joins = [(self.sql2vec.aliasname2id[p[0]], self.sql2vec.aliasname2id[p[1]]) for p in self.sql2vec.join_list]
        # self.leading_length = 2
        leading_length = config.leading_length
        if leading_length == -1:
            leading_length = len(alias)
        if leading_length > len(alias):
            leading_length = len(alias)
        """
        长度为3
        [(array([2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), inf), (array([3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]), inf), (array([5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), inf)]
        """
        """
        def findCanHints(self, totalNumberOfTables, numberOfTables, queryEncode,all_joins,joins_with_predicate,nodes,depth=2):
        """
        # join_list_with_predicate = self.mcts_searcher.findCanHints(40,len(alias),sql_vec,id_joins,id_joins_with_predicate,alias_id,depth=leading_length)
        join_list_with_predicate = self.mcts_searcher.findCanHints(len(config.id2aliasname), len(alias), sql_vec,
                                                                   id_joins, id_joins_with_predicate, alias_id,
                                                                   depth=leading_length)
        # [127.86579990386963]
        self.mcts_time_list.append(timer.record('mcts_time_list'))
        # ['/*+Leading(ci t)*/', '/*+Leading(cn mc)*/', '/*+Leading(mc t)*/']
        leading_list = []
        plan_jsons = []
        plan_json_with_exe_time = []
        # plan_json_with_exe_time.extend([self.run_query("EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + sql)])

        try:
            plan_json = self.run_query("EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + sql)
            plan_json = plan_json[0][0]
        except:
            plan_json = self.run_query("EXPLAIN (VERBOSE, COSTS, FORMAT JSON, SUMMARY) " + sql)
            plan_json = plan_json[0][0]
            plan_json[0]["Execution Time"] = config.max_time_out

        plan_json_with_exe_time.append(plan_json)

        plan_jsons.extend([plan_json_PG])
        # leadings_utility_list = [0.0, 0.0, 0.0]
        leadings_utility_list = []
        # join = (array([3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.0)

        pool = Pool(10)
        for join in join_list_with_predicate:
            # ['/*+Leading(ci t)*/']
            leading_list.append(
                '/*+Leading(' + " ".join([self.sql2vec.id2aliasname[x] for x in join[0][:leading_length]]) + ')*/')
            leadings_utility_list.append(join[1])
            ##To do: parallel planning 将leading hints 和 sql结合起来
            plan_jsons.append(pgrunner.getCostPlanJson(leading_list[-1] + sql))
            try:
                plan_json = self.run_query(
                    "EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + leading_list[-1] + sql)
            except:
                plan_json = self.run_query("EXPLAIN (VERBOSE, COSTS, FORMAT JSON, SUMMARY) " + sql)
                plan_json = plan_json[0][0]
                plan_json[0]["Execution Time"] = config.max_time_out
            plan_json_with_exe_time.append(plan_json)
        # 长度为4，还加入了不用hint直接预测sql的性能
        pool.close()
        pool.join()

        with open(self.save_plan_path, 'a+') as f:
            tmp = plan_json_with_exe_time
            tmp = [json.dumps(x) for x in tmp]
            write_str = sql + config.SEP + config.SEP.join(tmp) + "\n"
            f.write(write_str)

        timer.reset('MHPE_time_list')

        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons, sql_vec=sql_vec)
        self.MHPE_time_list.append(timer.record('MHPE_time_list'))
        """
        [(0.777046799659729, 1.1121588945388794, 1.0), (0.7715609073638916, 1.112508773803711, 1.0), 
        (0.7729040384292603, 1.1113137006759644, 1.0), (0.7715591192245483, 1.1087466478347778, 1.0)]
        """
        """
        ['/*+Leading(ci t)*/', '/*+Leading(cn mc)*/', '/*+Leading(mc t)*/']
        """
        """
        [0.0, 0.0, 0.0]
        """
        # self.max_hint_num = 20 chosen_leading_pair = ((0.5299996733665466, 1.4747910499572754, 1.0), '/*+Leading(ci t)*/', 0.0)
        tmp = zip(plan_times[:config.max_hint_num], leading_list, leadings_utility_list)
        # 排序的时候不仅考虑了预测的latency，也考虑了q-error
        chosen_leading_pair = sorted(zip(plan_times[:config.max_hint_num], leading_list, leadings_utility_list),
                                     key=lambda x: x[0][0] + self.knn.kNeightboursSample(x[0]))[0]
        return chosen_leading_pair

    def run_query(self, q):

        conn = psycopg2.connect(self.CONNECTION_STR)
        conn.set_client_encoding('UTF8')
        result = None
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout TO " + str(config.max_time_out))
            cur.execute(q)
            result = cur.fetchall()
        finally:

            conn.close()
        return result

    def hinterRun(self, sql):
        # hinter的次数
        self.hinter_times += 1
        """
        不执行语句，而是直接返回查询计划，更新Planning Time为计划时间
        self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
        geqo：Genetic Query Optimizer，geqo_threshold是遗传算法的阈值，如果超过这个预知，就使用遗传优化器
        self.cur.execute("SET geqo_threshold  = 12;")
        self.cur.execute("explain (COSTS, FORMAT JSON) "+sql)
        """
        plan_json_PG = pgrunner.getCostPlanJson(sql)
        self.samples_plan_with_time = []
        # head_num = 10
        # torch.rand 生成满足正态分布的样本，尺寸为（1,config.head_num）,mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
        mask = (torch.rand(1, config.head_num, device=config.device) < 0.9).long()
        # config.cost_test_for_debug：False
        if config.cost_test_for_debug:
            self.pg_runningtime_list.append(pgrunner.getCost(sql)[0])
            self.pg_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
        else:
            # 扩充pg_runningtime_list和pg_planningtime_list
            """
            实际执行语句，然后得到执行计划
            self.cur.execute("SET geqo_threshold  = 12;")
            self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
            self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sql)
            """
            self.pg_runningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            # print(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            self.pg_planningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Planning Time'])

        # 将query转化为vector,其中alias = {'k', 'mc', 'an', 'mk', 'ci', 't', 'n', 'cn'},sql_vec = array([0., 0., 0., ..., 0., 0., 0.])
        sql_vec, alias = self.sql2vec.to_vec(sql)
        plan_jsons = [plan_json_PG]
        # 输入计划，获取均值、方差等，来预测pg原始optimization的执行时间
        # res = list(zip(mean_item,variance_item,v2_item)),plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons, sql_vec=sql_vec)

        algorithm_idx = 0

        """
        输入plan、query、query encoding和alias =  # {'k', 'mc', 'an', 'mk', 'ci', 't', 'n', 'cn'}来预测得到最好的max_hint_num个hint(考虑了latency和q-error)
        chosen_leading_pair = sorted(zip(plan_times[:config.max_hint_num],leading_list,leadings_utility_list),key = lambda x:x[0][0]+
        self.knn.kNeightboursSample(x[0]))[0]
        """
        # ((0.8268615007400513, 1.329881191253662, 1.0), '/*+Leading(cn mc)*/', inf)
        chosen_leading_pair = self.findBestHint(plan_json_PG=plan_json_PG, alias=alias, sql_vec=sql_vec, sql=sql)
        # knn_plan = 4
        # plan_times = [(0.8306915163993835, 1.3287104494339836, 1.0)]
        # chosen_leading_pair = ((0.48867878317832947, 0.34872353076934814, 1.0), '/*+Leading(ci n)*/', inf)
        # plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        # config.threshold = log(3)/log(self.max_time_out) = 0.09393664679537382
        # 找到和pg默认的plan（它的相关latency是预测的）方差相似的几个plan，然后随机返回一个plan，它的q-error当作当前plan的q-error
        knn_plan = abs(self.knn.kNeightboursSample(plan_times[0]))

        # if chosen_leading_pair[0][0]<plan_times[algorithm_idx][0]:
        #     self.one_times += 1
        # if abs(knn_plan)<config.threshold:
        #     self.two_times += 1            

        if chosen_leading_pair[0][0] < plan_times[algorithm_idx][0] and abs(
                knn_plan) < config.threshold and self.value_extractor.decode(plan_times[0][0]) > 100:
            from math import e
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0]) * 3), config.max_time_out)
            # False
            if config.cost_test_for_debug:
                leading_time_flag = pgrunner.getCost(sql=chosen_leading_pair[1] + sql)
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(
                    pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql)['Planning Time'])
            else:
                plan_json = pgrunner.getAnalysePlanJson(sql=chosen_leading_pair[1] + sql)
                leading_time_flag = (plan_json['Plan']['Actual Total Time'], plan_json['timeout'])
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(plan_json['Planning Time'])

            self.knn.insertAValue(
                (chosen_leading_pair[0], self.value_extractor.encode(leading_time_flag[0]) - chosen_leading_pair[0][0]))
            # False
            if config.cost_test_for_debug:
                self.samples_plan_with_time.append(
                    [pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql, timeout=max_time_out),
                     leading_time_flag[0], mask])
            # max_time_out是训练计划在pg中执行的相关参数，假如max_time_out很大，那么可能latency很大样本也会作为训练集，效果不好？
            else:
                self.samples_plan_with_time.append(
                    [pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql, timeout=max_time_out),
                     leading_time_flag[0], mask])
            if leading_time_flag[1]:
                if config.cost_test_for_debug:
                    pg_time_flag = pgrunner.getCost(sql=sql)
                else:
                    pg_time_flag = pgrunner.getLatency(sql=sql, timeout=300 * 1000)
                self.knn.insertAValue((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
                if self.samples_plan_with_time[0][1] > pg_time_flag[0] * 1.8:
                    self.samples_plan_with_time[0][1] = pg_time_flag[0] * 1.8
                    self.samples_plan_with_time.append([plan_json_PG, pg_time_flag[0], mask])
                else:
                    self.samples_plan_with_time[0] = [plan_json_PG, pg_time_flag[0], mask]

                self.hinter_time_list.append([max_time_out, pgrunner.getLatency(sql=sql, timeout=300 * 1000)[0]])
                self.chosen_plan.append([chosen_leading_pair[1], 'PG'])
            else:
                self.hinter_time_list.append([leading_time_flag[0]])
                self.chosen_plan.append([chosen_leading_pair[1]])
        else:
            # False
            if config.cost_test_for_debug:
                pg_time_flag = pgrunner.getCost(sql=sql)
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
            else:
                # 输入query，利用explain得到plan_json['Plan']['Actual Total Time']和plan_json['timeout']，pg_time_flag = (1499.574, False)
                """
                def
                plan_json = self.getAnalysePlanJson(sql,timeout)
                return plan_json['Plan']['Actual Total Time'],plan_json['timeout']
                """
                pg_time_flag = pgrunner.getLatency(sql=sql, timeout=300 * 1000)
                # 存下hinter_runtime_list，执行时间
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                # 存下hinter_planningtime_list，计划时间
                self.hinter_planningtime_list.append(pgrunner.getAnalysePlanJson(sql=sql)['Planning Time'])
            # 插入值：self.kvs.append(data)，（执行时间（均值、方差、v2），实际执行时间减去预测时间再编码），编码：int(np.log(2+v)/np.log(config.max_time_out)*200)/200，类似于q-error
            self.knn.insertAValue((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
            # 插入值：[plan_json_PG,pg_time_flag[0],mask]，[[{...}, 1499.574, tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')]]
            self.samples_plan_with_time.append([plan_json_PG, pg_time_flag[0], mask])
            # 存下pg_time_flag[0]，[[1499.574]]
            self.hinter_time_list.append([pg_time_flag[0]])
            # 选择了pg的计划，[['PG']]
            self.chosen_plan.append(['PG'])
        ## self.samples_plan_with_time  self.samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
        ## To do: parallel the training process 
        ## 训练TreeNet:用于提取query的特征表示，训练mcts_searcher：用于找到最好的hint
        ## 这里的train仅仅用于将训练样本记录下来
        for sample in self.samples_plan_with_time:
            target_value = self.value_extractor.encode(sample[1])  # 0.625
            self.model.train(plan_json=sample[0], sql_vec=sql_vec, target_value=target_value, mask=mask, is_train=True)
            self.mcts_searcher.train(tree_feature=self.model.tree_builder.plan_to_feature_tree(sample[0]),
                                     sql_vec=sql_vec, target_value=sample[1], alias_set=alias)
        # self.hinter_times，已经输入的query的个数；当query个数少于1000并且为10的倍数时

        # if self.hinter_times<1000 or self.hinter_times%10==0:
        #     loss=  self.model.optimize()[0]
        #     loss1 = self.mcts_searcher.optimize()
        #     if self.hinter_times<1000:
        #         loss=  self.model.optimize()[0]
        #         loss1 = self.mcts_searcher.optimize()
        #     if loss>3:
        #         loss=  self.model.optimize()[0]
        #         loss1 = self.mcts_searcher.optimize()
        #     if loss>3:
        #         loss=  self.model.optimize()[0]
        #         loss1 = self.mcts_searcher.optimize()
        # 只用前1000个训练，batch_size为10
        # if self.hinter_times%config.train_interval==0:
        #     for epoch in range(config.epochs):
        #         loss=  self.model.optimize()[0]
        #         self.model.save_model(config.model_path)
        #         self.model.load_model(config.model_path)
        #         loss1 = self.mcts_searcher.optimize()

        #     with open('log'+str(config.num)+'.txt',"a") as f:
        #         #f.write(str(self.hinter_times)+" : "+str(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])+"\n")
        #         f.write("Epoch "+ str(epoch) + " loss1:" + str(loss) + " loss2:" + str(loss1))     

        # print("Epoch {}, loss1:{}, loss2:{}".format(epoch,loss,loss1))

        # loss=  self.model.optimize()[0]
        # loss1 = self.mcts_searcher.optimize()
        # if loss>3:
        #     loss=  self.model.optimize()[0]
        #     loss1 = self.mcts_searcher.optimize()
        # if loss>3:
        #     loss=  self.model.optimize()[0]
        #     loss1 = self.mcts_searcher.optimize()

        # if self.hinter_times % config.batch_size == 0:
        #     loss=  self.model.optimize()[0]
        #     loss1 = self.mcts_searcher.optimize()

        #     self.model.memory = ReplayMemory(config.mem_size)
        #     self.mcts_searcher.memory = MCTSMemory(5000)
        print("run or cahce the training query {0}".format(self.hinter_times))
        assert len(set([len(self.hinter_runtime_list), len(self.pg_runningtime_list), len(self.mcts_time_list),
                        len(self.hinter_planningtime_list), len(self.MHPE_time_list), len(self.hinter_runtime_list),
                        len(self.chosen_plan), len(self.hinter_time_list)])) == 1
        # pg_plan_time,pg_latency,mcts_time,hinter_plan_time,MPHE_time,hinter_latency,actual_plans,actual_time
        return self.pg_planningtime_list[-1], self.pg_runningtime_list[-1], self.mcts_time_list[-1], \
               self.hinter_planningtime_list[-1], self.MHPE_time_list[-1], self.hinter_runtime_list[-1], \
               self.chosen_plan[-1], self.hinter_time_list[-1]

    def hinterTest(self, sql):
        # hinter的次数
        self.hinter_times += 1
        """
        不执行语句，而是直接返回查询计划，更新Planning Time为计划时间
        self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
        geqo：Genetic Query Optimizer，geqo_threshold是遗传算法的阈值，如果超过这个预知，就使用遗传优化器
        self.cur.execute("SET geqo_threshold  = 12;")
        self.cur.execute("explain (COSTS, FORMAT JSON) "+sql)
        """
        plan_json_PG = pgrunner.getCostPlanJson(sql)
        self.samples_plan_with_time = []
        # head_num = 10
        # torch.rand 生成满足正态分布的样本，尺寸为（1,config.head_num）,mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
        mask = (torch.rand(1, config.head_num, device=config.device) < 0.9).long()
        # config.cost_test_for_debug：False
        if config.cost_test_for_debug:
            self.pg_runningtime_list.append(pgrunner.getCost(sql)[0])
            self.pg_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
        else:
            # 扩充pg_runningtime_list和pg_planningtime_list
            """
            实际执行语句，然后得到执行计划
            self.cur.execute("SET geqo_threshold  = 12;")
            self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
            self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sql)
            """
            self.pg_runningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            # print(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            self.pg_planningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Planning Time'])

        # 将query转化为vector,其中alias = {'k', 'mc', 'an', 'mk', 'ci', 't', 'n', 'cn'},sql_vec = array([0., 0., 0., ..., 0., 0., 0.])
        sql_vec, alias = self.sql2vec.to_vec(sql)
        plan_jsons = [plan_json_PG]
        # 输入计划，获取均值、方差等，来预测pg原始optimization的执行时间
        # res = list(zip(mean_item,variance_item,v2_item)),plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons, sql_vec=sql_vec)

        algorithm_idx = 0

        """
        输入plan、query、query encoding和alias =  # {'k', 'mc', 'an', 'mk', 'ci', 't', 'n', 'cn'}来预测得到最好的max_hint_num个hint(考虑了latency和q-error)
        chosen_leading_pair = sorted(zip(plan_times[:config.max_hint_num],leading_list,leadings_utility_list),key = lambda x:x[0][0]+
        self.knn.kNeightboursSample(x[0]))[0]
        """
        # ((0.8268615007400513, 1.329881191253662, 1.0), '/*+Leading(cn mc)*/', inf)
        chosen_leading_pair = self.findBestHint(plan_json_PG=plan_json_PG, alias=alias, sql_vec=sql_vec, sql=sql)
        # knn_plan = 4
        # plan_times = [(0.8306915163993835, 1.3287104494339836, 1.0)]
        # chosen_leading_pair = ((0.48867878317832947, 0.34872353076934814, 1.0), '/*+Leading(ci n)*/', inf)
        # plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        # config.threshold = log(3)/log(self.max_time_out) = 0.09393664679537382
        # 找到和pg默认的plan（它的相关latency是预测的）方差相似的几个plan，然后随机返回一个plan，它的q-error当作当前plan的q-error
        knn_plan = abs(self.knn.kNeightboursSample(plan_times[0]))

        # if chosen_leading_pair[0][0]<plan_times[algorithm_idx][0]:
        #     self.one_times += 1
        # if abs(knn_plan)<config.threshold:
        #     self.two_times += 1            

        if chosen_leading_pair[0][0] < plan_times[algorithm_idx][0] and abs(
                knn_plan) < config.threshold and self.value_extractor.decode(plan_times[0][0]) > 100:
            from math import e
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0]) * 3), config.max_time_out)
            # False
            if config.cost_test_for_debug:
                leading_time_flag = pgrunner.getCost(sql=chosen_leading_pair[1] + sql)
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(
                    pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql)['Planning Time'])
            else:
                plan_json = pgrunner.getAnalysePlanJson(sql=chosen_leading_pair[1] + sql)
                leading_time_flag = (plan_json['Plan']['Actual Total Time'], plan_json['timeout'])
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(plan_json['Planning Time'])

            self.knn.insertAValue(
                (chosen_leading_pair[0], self.value_extractor.encode(leading_time_flag[0]) - chosen_leading_pair[0][0]))
            # False
            if config.cost_test_for_debug:
                self.samples_plan_with_time.append(
                    [pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql, timeout=max_time_out),
                     leading_time_flag[0], mask])
            # max_time_out是训练计划在pg中执行的相关参数，假如max_time_out很大，那么可能latency很大样本也会作为训练集，效果不好？
            else:
                self.samples_plan_with_time.append(
                    [pgrunner.getCostPlanJson(sql=chosen_leading_pair[1] + sql, timeout=max_time_out),
                     leading_time_flag[0], mask])
            if leading_time_flag[1]:
                if config.cost_test_for_debug:
                    pg_time_flag = pgrunner.getCost(sql=sql)
                else:
                    pg_time_flag = pgrunner.getLatency(sql=sql, timeout=300 * 1000)
                self.knn.insertAValue((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
                if self.samples_plan_with_time[0][1] > pg_time_flag[0] * 1.8:
                    self.samples_plan_with_time[0][1] = pg_time_flag[0] * 1.8
                    self.samples_plan_with_time.append([plan_json_PG, pg_time_flag[0], mask])
                else:
                    self.samples_plan_with_time[0] = [plan_json_PG, pg_time_flag[0], mask]

                self.hinter_time_list.append([max_time_out, pgrunner.getLatency(sql=sql, timeout=300 * 1000)[0]])
                self.chosen_plan.append([chosen_leading_pair[1], 'PG'])
            else:
                self.hinter_time_list.append([leading_time_flag[0]])
                self.chosen_plan.append([chosen_leading_pair[1]])
        else:
            # False
            if config.cost_test_for_debug:
                pg_time_flag = pgrunner.getCost(sql=sql)
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
            else:
                # 输入query，利用explain得到plan_json['Plan']['Actual Total Time']和plan_json['timeout']，pg_time_flag = (1499.574, False)
                """
                def
                plan_json = self.getAnalysePlanJson(sql,timeout)
                return plan_json['Plan']['Actual Total Time'],plan_json['timeout']
                """
                pg_time_flag = pgrunner.getLatency(sql=sql, timeout=300 * 1000)
                # 存下hinter_runtime_list，执行时间
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                # 存下hinter_planningtime_list，计划时间
                self.hinter_planningtime_list.append(pgrunner.getAnalysePlanJson(sql=sql)['Planning Time'])
            # 插入值：self.kvs.append(data)，（执行时间（均值、方差、v2），实际执行时间减去预测时间再编码），编码：int(np.log(2+v)/np.log(config.max_time_out)*200)/200，类似于q-error
            self.knn.insertAValue((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
            # 插入值：[plan_json_PG,pg_time_flag[0],mask]，[[{...}, 1499.574, tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')]]
            self.samples_plan_with_time.append([plan_json_PG, pg_time_flag[0], mask])
            # 存下pg_time_flag[0]，[[1499.574]]
            self.hinter_time_list.append([pg_time_flag[0]])
            # 选择了pg的计划，[['PG']]
            self.chosen_plan.append(['PG'])
        ## self.samples_plan_with_time  self.samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
        ## To do: parallel the training process 
        ## 训练TreeNet:用于提取query的特征表示，训练mcts_searcher：用于找到最好的hint
        ## 这里的train仅仅用于将训练样本记录下来
        for sample in self.samples_plan_with_time:
            target_value = self.value_extractor.encode(sample[1])  # 0.625
            self.model.train(plan_json=sample[0], sql_vec=sql_vec, target_value=target_value, mask=mask, is_train=True)
            self.mcts_searcher.train(tree_feature=self.model.tree_builder.plan_to_feature_tree(sample[0]),
                                     sql_vec=sql_vec, target_value=sample[1], alias_set=alias)

        assert len(set([len(self.hinter_runtime_list), len(self.pg_runningtime_list), len(self.mcts_time_list),
                        len(self.hinter_planningtime_list), len(self.MHPE_time_list), len(self.hinter_runtime_list),
                        len(self.chosen_plan), len(self.hinter_time_list)])) == 1
        # pg_plan_time,pg_latency,mcts_time,hinter_plan_time,MPHE_time,hinter_latency,actual_plans,actual_time
        return self.pg_planningtime_list[-1], self.pg_runningtime_list[-1], self.mcts_time_list[-1], \
               self.hinter_planningtime_list[-1], self.MHPE_time_list[-1], self.hinter_runtime_list[-1], \
               self.chosen_plan[-1], self.hinter_time_list[-1]

    def predictWithUncertaintyBatch(self, plan_jsons, sql_vec):
        # model = TreeNet(tree_builder= tree_builder,value_network = value_network)
        # 利用lstm来抽取整个query的特征表示，EQ
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        import torchfold
        fold = torchfold.Fold(cuda=True)
        res = []
        multi_list = []
        # 实际上就一个plan_json
        for plan_json in plan_jsons:
            # 利用tree-lstm将plan转化为特征Rp
            tree_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
            # 获取multi_value，即（Rp，EQ）
            multi_value = self.model.plan_to_value_fold(tree_feature=tree_feature, sql_feature=sql_feature, fold=fold)
            multi_list.append(multi_value)
        # dynamic batch
        multi_value = fold.apply(self.model.value_network, [multi_list])[0]
        # 获取均值方差，参数有head_num,即多个输出层，然后计算对应的均值和方差
        mean, variance = self.model.mean_and_variance(multi_value=multi_value[:, :config.head_num])
        # 以e为底的指数 self.var_weight = 0.00
        v2 = torch.exp(multi_value[:, config.head_num] * config.var_weight).data.reshape(-1)
        if isinstance(mean, float):
            mean_item = [mean]
        else:
            mean_item = [x.item() for x in mean]
        if isinstance(variance, float):
            variance_item = [variance]
        else:
            variance_item = [x.item() for x in variance]
        # variance_item = [x.item() for x in variance]
        if isinstance(v2, float):
            v2_item = [v2]
        else:
            v2_item = [x.item() for x in v2]
        # v2_item = [x.item() for x in v2]
        # 返回均值，方差，res对
        res = list(zip(mean_item, variance_item, v2_item))
        return res
