import argparse

from utils_nohup import *
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
                test_queries, model_prefix, topK,CONNECTION_STR) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.lero_server_path = LERO_SERVER_PATH
        self.CONNECTION_STR = CONNECTION_STR
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
        ## 例如100/5
        run_args = self.get_run_args()
        
        pool = Pool(pool_num)
        for fp, q in self.queries:
            self.run_pairwise(q, fp, run_args, self.output_query_latency_file, self.output_query_latency_file + "_exploratory", pool,self.CONNECTION_STR)
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


    def get_run_args(self):
        run_args = []
        run_args.append("SET enable_lero TO True")
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args

    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool,CONNECTION_STR):
        ## 输出计划，但是没有真正执行
        explain_query(q, run_args,CONNECTION_STR)
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
                # do_run_query(q, fp, self.get_card_test_args(card_file_name), output_file,CONNECTION_STR,True, None, None)
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name), output_file, CONNECTION_STR,True, None, None))
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

    queries = []
    test_queries = []
    query_num_per_chunk = None
    model_prefix = None
    topK = 100

    query_path_prefix = "/home/admin/wd_files/Lero/reproduce/"
    output_query_latency_file_prefix = "/home/admin/wd_files/Lero/test_script/train_and_test_plan/"

    query_path = query_path_prefix  + 'tpch_new/test/tpch.txt' 
    output_query_latency_file = output_query_latency_file_prefix + 'tpch/test/tpch_test.log'

    with open(query_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(SEP)
            queries.append((arr[0], arr[1]))
    print("Read", len(queries), "training queries.")

    pool_num = 10

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    DB = "tpch"
    PORT = 5432
    USER = "woodybryant.wd"
    PASSWORD = "Wd257538"

    CONNECTION_STR = "dbname=" + DB + " user=" + USER + " password=" + PASSWORD + " host=localhost port=" + str(PORT)
    helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK,CONNECTION_STR)
    # 训练和测试
    helper.start(pool_num)


