import torch
from math import log


class Config:
    def __init__(self, ):
        self.datafile = 'JOBqueries.workload'
        self.schemaFile = "schema.sql"
        self.user = 'woodybryant.wd'
        self.password = None
        # self.dataset = 'JOB'
        self.userName = self.user
        self.usegpu = True
        self.head_num = 10
        self.input_size = 9
        self.hidden_size = 64
        self.ip = "11.164.204.79"
        self.port = 5432
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_weight = 0.00  # for au, 0:disable,0.01:enable
        self.SEP = "#####"

        self.cost_test_for_debug = False

        # self.max_time_out = 120*1000     

        self.mem_size = 2000
        self.mcts_v = 1.1
        self.searchFactor = 4
        self.U_factor = 0.0
        self.log_file = 'log_c3_h64_s4_t3.txt'
        self.latency_file = 'latency_record.txt'
        # self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
        # self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
        self.modelpath = 'model/'
        self.offset = 20
        # job:100
        self.max_column = 100
        # job:2
        self.leading_length = 2

        # 120*1000,120*1000*100000000

        self.max_hint_num = 20
        self.try_hint_num = 3

        self.database = 'imdb'
        # self.database = 'tpch'
        # self.database = 'stats'

        if self.database == "imdb":
            # job:{0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
            # stats:{0: 'c', 1: 'b', 2: 'ph', 3: 'u', 4: 'pl', 5: 'v', 6: 'p', 7: 't'}
            self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k',
                                 9: 'lt',
                                 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt',
                                 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3',
                                 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1',
                                 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
            # self.id2aliasname = {0:'start',8: 'c', 1: 'b', 2: 'ph', 3: 'u', 4: 'pl', 5: 'v', 6: 'p', 7: 't'}
            # job:{'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
            # stats:{'c': 0, 'b': 1, 'ph': 2, 'u': 3, 'pl': 4, 'v': 5, 'p': 6, 't': 7}
            self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39,
                                 'kt2': 32,
                                 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8,
                                 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19,
                                 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12,
                                 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
            # self.aliasname2id = {'start':0,'c': 8, 'b': 1, 'ph': 2, 'u': 3, 'pl': 4, 'v': 5, 'p': 6, 't': 7}
            # 不用改
            self.max_alias_num = len(self.aliasname2id)
            # 不用改
            # job:self.max_alias_num*self.max_alias_num+self.max_column
            # stats:self.max_column+pow(len(self.id2aliasname),2)
            self.mcts_input_size = self.max_column + pow(len(self.id2aliasname), 2)
        elif self.database == "tpch":
            self.id2aliasname = {0: 'start', 1: 'c', 2: 'l', 3: 'n', 4: 'o', 5: 'p', 6: 'ps', 7: 'r', 8: 's'}
            self.aliasname2id = {'start': 0, 'c': 1, 'l': 2, 'n': 3, 'o': 4, 'p': 5, 'ps': 6, 'r': 7, 's': 8}
            # 不用改
            self.max_alias_num = len(self.aliasname2id)
            # 不用改
            # job:self.max_alias_num*self.max_alias_num+self.max_column
            # stats:self.max_column+pow(len(self.id2aliasname),2)
            self.mcts_input_size = self.max_column + pow(len(self.id2aliasname), 2)
            # 不用改
            # job:40*40+self.max_column
            # stats:self.mcts_input_size
            self.sql_size = self.mcts_input_size
            self.tpch_map = {'customer': 'c', 'lineitem': 'l', 'nation': 'n', 'orders': 'o', 'part': 'p',
                             'partsupp': 'ps',
                             'region': 'r', 'supplier': 's'}
            self.tpch_leading_hint_map = {'c': 'customer', 'l': 'lineitem', 'n': 'nation', 'o': 'orders', 'p': 'part',
                                          'ps': 'partsupp', 'r': 'region', 's': 'supplier'}
        elif self.database == "stats":
            self.id2aliasname = {0: 'start', 8: 'c', 1: 'b', 2: 'ph', 3: 'u', 4: 'pl', 5: 'v', 6: 'p', 7: 't'}

            self.aliasname2id = {'start': 0, 'c': 8, 'b': 1, 'ph': 2, 'u': 3, 'pl': 4, 'v': 5, 'p': 6, 't': 7}

            # 不用改
            self.max_alias_num = len(self.aliasname2id)
            # 不用改
            # job:self.max_alias_num*self.max_alias_num+self.max_column
            # stats:self.max_column+pow(len(self.id2aliasname),2)
            self.mcts_input_size = self.max_column + pow(len(self.id2aliasname), 2)
            # 不用改
            # job:40*40+self.max_column
            # stats:self.mcts_input_size
            self.sql_size = self.mcts_input_size
        else:
            raise RuntimeError

        # 不用改
        # job:40*40+self.max_column
        # stats:self.mcts_input_size 
        self.sql_size = self.mcts_input_size
        # bar,line
        self.draw_style = 'bar'

        self.record_file = False
        self.num = 1

        self.max_time_out = 500 * 1000
        self.threshold = log(3) / log(self.max_time_out)
        self.queries_file = 'workload/job/train/job' + str(self.num) + '.txt'
        # self.queries_file = 'workload/stats/test/stats.txt'
        # self.test_file = 'workload/stats/test/stats.txt'
        self.max_out_name = 'split' + str(self.num)
        # self.query_num = 250*self.num
        self.query_num = 250
        self.train_interval = self.query_num
        self.batch_size = self.train_interval
        self.epochs = 50

        self.save_plan_path = './plan_with_exe_time/job1.train'
        self.model_path = './model/job1'
