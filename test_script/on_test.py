import argparse

from utils import *
import os
import socket
from config import *
from multiprocessing import Pool

class PolicyEntity:
    def __init__(self, score) -> None:
        self.score = score

    def get_score(self):
        return self.score


class CardinalityGuidedEntity(PolicyEntity):
    def __init__(self, score, card_str) -> None:
        super().__init__(score)
        self.card_str = card_str


class PgHelper():
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file

    def start(self, pool_num):
        pool = Pool(pool_num)
        for fp, q in self.queries:
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, True, None, None))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()

    # def start(self, pool_num):
    #     pool = Pool(pool_num)
    #     for fp, q in self.queries:
    #      do_run_query(q, fp, [], self.output_query_latency_file, True, None, None)



class LeroHelper():
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file, 
                test_queries, model_prefix, topK) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.lero_server_path = LERO_SERVER_PATH
        """
        LERO_SERVER_PATH = "../"
        LERO_DUMP_CARD_FILE = "dump_card_with_score.txt"
        """
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def start(self, pool_num):
        # ## 例如100/5
        # lero_chunks = list(self.chunks(self.queries, self.query_num_per_chunk))

        # run_args = self.get_run_args()
        # ## 相当于batch

        # pool = Pool(pool_num)
        # for fp, q in self.queries:
        #     self.run_pairwise(q, fp, run_args, self.output_query_latency_file, self.output_query_latency_file + "_exploratory", pool)
        # print('Waiting for all subprocesses done...')
        # pool.close()
        # pool.join()

        # # for c_idx, chunk in enumerate(lero_chunks):
        model_name = self.model_prefix + "_" + str(0)
        self.load_model(model_name)
        # self.retrain(model_name)
        self.test_benchmark(self.output_query_latency_file + "_" + model_name)

    def retrain(self, model_name):
        # output_query_latency_file:lero_tpch.log
        # training_data_file:lero_tpch.log.training
        # training_data_file = self.output_query_latency_file + ".training"
        # create_training_file(training_data_file, self.output_query_latency_file, self.output_query_latency_file + "_exploratory")
        # print("retrain Lero model:", model_name, "with file", training_data_file)
        
        # cmd_str = "cd " + self.lero_server_path + " && python3.8 train.py" \
        #                                         + " --training_data " + os.path.abspath(training_data_file) \
        #                                         + " --model_name " + model_name \
        #                                         + " --training_type 1"
        # print("run cmd:", cmd_str)
        # os.system(cmd_str)

        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type":"load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        print(reply_json)
        os.system("sync")

    def test_benchmark(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
            do_run_query(q, fp, run_args, output_file, True, None, None)


    def test_benchmark_on_training_data(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.queries:
            do_run_query(q, fp, run_args, output_file, True, None, None)

    def get_run_args(self):
        run_args = []
        run_args.append("SET enable_lero TO True")
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args

    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool):
        ## 输出计划，但是没有真正执行
        explain_query(q, run_args)
        policy_entities = []
        """
        LERO_SERVER_PATH = "../"
        LERO_DUMP_CARD_FILE = "dump_card_with_score.txt"
        """
        with open(self.lero_card_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split(";") for line in lines]
            # lines = [[card_lists1,latency1],[card_lists2,latency2]]
            for line in lines:
                policy_entities.append(CardinalityGuidedEntity(float(line[1]), line[0]))

        policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
        ## 获取score最小的几个实体
        policy_entities = policy_entities[:self.topK]

        i = 0
        for entity in policy_entities:
            if isinstance(entity, CardinalityGuidedEntity):
                card_str = "\n".join(entity.card_str.strip().split(" "))
                # ensure that the cardinality file will not be changed during planning
                card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                # 将这top-k个基数列表写入pg的系统数据文件便于pg生成plan
                card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                with open(card_file_path, "w") as card_file:
                    card_file.write(card_str)

                output_file = output_query_latency_file if i == 0 else exploratory_query_latency_file
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name), output_file, True, None, None))
                i += 1

    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type":"predict", "Plan":plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']

if __name__ == "__main__":


    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if ALGO:
        assert ALGO.lower() in ALGO_LIST
        algo = ALGO.lower()
    print("algo:", algo)


    # args.query_path =  "stats_train.txt"
    query_path = TRAIN_QUERY_PATH if algo =='lero' else PG_QUERY_PATH
    print("Load queries from ", query_path)
    queries = []
    with open(query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(SEP)
            queries.append((arr[0], arr[1]))
    print("Read", len(queries), "training queries.")
    # output_query_latency_file = lero_tpch.log
    output_query_latency_file = OUTPUT_QUERY_LATENCY_FILE_LERO if algo == 'lero' else OUTPUT_QUERY_LATENCY_FILE_PG
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 10
    if POOL_NUM:
        pool_num = POOL_NUM
    print("pool_num:", pool_num)



    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)




    
    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        test_queries = []
        if TEST_QUERY_PATH is not None:
            with open(TEST_QUERY_PATH, 'r') as f:
                for line in f.readlines():
                    arr = line.strip().split(SEP)
                    test_queries.append((arr[0], arr[1]))
        print("Read", len(test_queries), "test queries.")

        query_num_per_chunk = QUERY_NUM_PER_CHUCK
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if MODEL_PREFIX:
            model_prefix = MODEL_PREFIX
        print("model_prefix:", model_prefix)

        topK = 5
        if TOPK is not None:
            topK = TOPK
        print("topK", topK)
        
        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK)
        
        # 训练和测试
        helper.start(pool_num)

        # # 让最终的模型在training_data跑一次
        # helper.test_benchmark_on_training_data(OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA)
