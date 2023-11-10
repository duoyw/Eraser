# Postgresql conf (Please configure it according to your situation)
PORT = 5432
HOST = "localhost"
USER = "lianggui.wlg"
PASSWORD = "ai4db"
DB = "stats"
CONNECTION_STR = "dbname=" + DB + " user=" + USER + " password=" + PASSWORD + " host=localhost port=" + str(PORT)
TIMEOUT = 30000000
# [important]
# the data directory of your Postgres in which the database data will live 
# you can execute "show data_directory" in psql to get it
# Please ensure this path is correct, 
# because the program needs to write cardinality files to it 
# to make the optimizer generate some specific execution plans of each query.
PG_DB_PATH = "/home/lianggui.wlg/pgsql/data"

# Rap conf (No modification is required by default)
AUNCEL_SERVER_PORT = 14567
AUNCEL_SERVER_HOST = "localhost"
AUNCEL_SERVER_PATH = "../"
AUNCEL_DUMP_CARD_FILE = "dump_card_with_score.txt"

# Test conf (No modification is required by default)
LOG_PATH = "./log/query_latency"
SEP = "#####"

# path to save image,performance.csv
PROJECT_BASE_PATH = "/home/lianggui.wlg/ai4db/rap/auncel/Data/"

# path to load train and test
# DATA_BASE_PATH = "/mnt/yuchu.yc/Ai4Spark/Data/"
DATA_BASE_PATH = "../Data/Spark/"
LOG_BASE_PATH = DATA_BASE_PATH+"../model_train_log/"
