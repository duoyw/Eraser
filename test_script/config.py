# Postgresql conf (Please configure it according to your situation)
PORT = 5432
HOST = "localhost"
USER = "woodybryant.wd"
PASSWORD = "Wd257538"


# [important]
# the data directory of your Postgres in which the database data will live 
# you can execute "show data_directory" in psql to get it
# Please ensure this path is correct, 
# because the program needs to write cardinality files to it 
# to make the optimizer generate some specific execution plans of each query.
# /home/admin/Lero-on-PostgreSQL/postgresql-13.1/psql/data
#PG_DB_PATH = "../../data"

# Rap conf (No modification is required by default)
LERO_SERVER_PORT = 14567
LERO_SERVER_HOST = "localhost"
LERO_SERVER_PATH = "../"
LERO_DUMP_CARD_FILE = "dump_card_with_score.txt"
# Test conf (No modification is required by default)
LOG_PATH = "./log/query_latency"

TOPK = 20
POOL_NUM = 10
SEP = "#####"
QUERY_NUM_PER_CHUCK = 1000




####待修改参数####
PG_DB_PATH ="/home/admin/wd_files/HyperQO/PostgreSQL12.1_hint/psql12/data"
ALGO = "lero"
DB = "tpch"
CONNECTION_STR = "dbname=" + DB + " user=" + USER + " password=" + PASSWORD + " host=localhost port=" + str(PORT)
TIMEOUT = 500000
EPOCHS = 50


"""
tpch
"""
TRAIN_TPCH_NUM = "2"
TEST_TPCH_NUM = ""

# if algo == pg
# PG_QUERY_PATH = "../reproduce/tpch_new/train/tpch"+str(TRAIN_TPCH_NUM)+".txt"
# OUTPUT_QUERY_LATENCY_FILE_PG = "./result/pg/pg_tpch_train_"+str(TRAIN_TPCH_NUM)+".log"

PG_QUERY_PATH = "../reproduce/tpch_new/test/tpch"+str(TEST_TPCH_NUM)+".txt"
OUTPUT_QUERY_LATENCY_FILE_PG = "./result/pg/pg_tpch_test_"+str(TEST_TPCH_NUM)+".log"



# if algo = lero
TRAIN_QUERY_PATH = "../reproduce/tpch_new/train/tpch"+str(TRAIN_TPCH_NUM)+".txt"
TEST_QUERY_PATH = "../reproduce/tpch_new/test/tpch"+str(TEST_TPCH_NUM)+".txt"
MODEL_PREFIX = "tpch_test_model_on"+str(TRAIN_TPCH_NUM)
OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA = "./result/tpch/"+str(TRAIN_TPCH_NUM)+"/lero_tpch_test_on_trainging_data_"+str(TRAIN_TPCH_NUM)+".log"
OUTPUT_QUERY_LATENCY_FILE_LERO = "./result/tpch/"+str(TRAIN_TPCH_NUM)+"/lero_tpch_" + str(TEST_TPCH_NUM) + ".log"
TEST_PLAN1 = "./result/tpch/tpch_test.log"
TEST_PLAN2 = "./result/tpch/tpch_test.log_exploratory"

# draw
PG_TRAIN = "./result/pg/pg_tpch_train_"+str(TRAIN_TPCH_NUM)+".log"
PG_TEST = "./result/pg/pg_tpch_test_"+str(TEST_TPCH_NUM)+".log"
LERO_TRAIN = OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA
LERO_TEST = OUTPUT_QUERY_LATENCY_FILE_LERO + "_" + MODEL_PREFIX + '_'

# """
# job
# """
# TRAIN_JOB_NUM = 4
# TEST_JOB_NUM = ""

# # if algo == pg
# # PG_QUERY_PATH = "../reproduce/imdb_new/train/job"+str(TRAIN_JOB_NUM)+".sql"
# # OUTPUT_QUERY_LATENCY_FILE_PG = "./result/job/"+str(TRAIN_JOB_NUM)+"/pg_stats_train_"+str(TRAIN_JOB_NUM)+".log"

# PG_QUERY_PATH = "../reproduce/job_new/test/job"+str(TEST_JOB_NUM)+".txt"
# OUTPUT_QUERY_LATENCY_FILE_PG = "./result/job/pg_job_test_"+str(TEST_JOB_NUM)+".log"


# # if algo = lero
# TRAIN_QUERY_PATH = "../reproduce/job_new/train/job"+str(TRAIN_JOB_NUM)+".txt"
# TEST_QUERY_PATH = "../reproduce/job_new/test/job"+str(TEST_JOB_NUM)+".txt"
# MODEL_PREFIX = "job_test_model_on_"+str(TRAIN_JOB_NUM)
# OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA = "./result/job/"+str(TRAIN_JOB_NUM)+"/lero_job_test_on_trainging_data_"+str(TRAIN_JOB_NUM)+".log"
# OUTPUT_QUERY_LATENCY_FILE_LERO = "./result/job/"+str(TRAIN_JOB_NUM)+"/lero_job_" + str(TEST_JOB_NUM) + ".log"
# TEST_PLAN1 = "./result/job/job_test.log"
# TEST_PLAN2 = "./result/job/job_test.log_exploratory"


# # draw
# PG_TRAIN = "./result/job/"+str(TRAIN_JOB_NUM)+"/pg_job_train_"+str(TRAIN_JOB_NUM)+".log"
# PG_TEST = "./result/pg/pg_job_test_"+str(TEST_JOB_NUM)+".log"
# LERO_TRAIN = OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA
# LERO_TEST = OUTPUT_QUERY_LATENCY_FILE_LERO + "_" + MODEL_PREFIX + '_'




"""
stats
"""

TRAIN_STATS_NUM = 4
# TEST_STATS_NUM = ""

# # if algo == pg
# # PG_QUERY_PATH = "../reproduce/stats_new/train_table_sort/stats"+str(TRAIN_STATS_NUM)+".sql"
# # OUTPUT_QUERY_LATENCY_FILE_PG = "./result/stats/table_sort_"+str(TRAIN_STATS_NUM)+"/pg_stats_train_"+str(TRAIN_STATS_NUM)+".log"

# PG_QUERY_PATH = "../reproduce/stats_new/test/stats"+str(TEST_STATS_NUM)+".txt"
# OUTPUT_QUERY_LATENCY_FILE_PG = "./result/pg/"+"pg_stats_test_"+str(TEST_STATS_NUM)+".log"


# # if algo = lero
# TRAIN_QUERY_PATH = "../reproduce/stats_new/train_sort/stats"+str(TRAIN_STATS_NUM)+".txt"
# TEST_QUERY_PATH = "../reproduce/stats_new/test/stats"+str(TEST_STATS_NUM)+".txt"
# MODEL_PREFIX = "stats_test_model_on_"+str(TRAIN_STATS_NUM)
# OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA = "./result/stats/"+str(TRAIN_STATS_NUM)+"/lero_stats_test_on_trainging_data_"+str(TRAIN_STATS_NUM)+".log"
# OUTPUT_QUERY_LATENCY_FILE_LERO = "./result/stats/"+str(TRAIN_STATS_NUM)+"/lero_stats_" + str(TEST_STATS_NUM) + ".log"
# TEST_PLAN1 = "./result/stats/stats_test.log"
# TEST_PLAN2 = "./result/stats/stats_test.log_exploratory"

# # draw
# PG_TRAIN = "./result/stats/"+str(TRAIN_STATS_NUM)+"/pg_stats_train_"+str(TRAIN_STATS_NUM)+".log"
# PG_TEST = "./result/pg"+"/pg_stats_test_"+str(TEST_STATS_NUM)+".log"
# LERO_TRAIN = OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA
# LERO_TEST = OUTPUT_QUERY_LATENCY_FILE_LERO + "_" + MODEL_PREFIX + '_'






