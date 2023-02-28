
import json
import copy
import pandas as pd
from config import *
from delete_time_out import *

def get_qid_latency(latency_path):
    mapping = {}
    with open(latency_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            k = arr[0]
            v = json.loads(arr[1])[0]['Execution Time'] / 1000
            mapping[k] = v
    return mapping

def get_qid(latency_path):
    qid_list = []
    with open(latency_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            k = arr[0]
            qid_list.append(k)
    return qid_list




# lero
latency_path1 = "/home/admin/wd_files/Lero/test_script/result/job/"+str(TRAIN_JOB_NUM)+ "/lero_job_.log_job_test_model_on_"+str(TRAIN_JOB_NUM)+"_0"

lero_map = get_qid_latency(latency_path1)

# pg
latency_path2 = "/home/admin/wd_files/Lero/test_script/result/pg/pg_job_test_.log"
delete_time_out_test(latency_path2)
pg_map = get_qid_latency(latency_path2)

# # best
# latency_path3 = "lero_job.log_exploratory"
# best_map = copy.deepcopy(lero_map)



# with open(latency_path3, 'r') as f:
#     for line in f.readlines():
#         arr = line.strip().split("#####")
#         k = arr[0]
#         v = json.loads(arr[1])[0]['Execution Time'] / 1000
#         best_map[k] = min(v,best_map[k],pg_map[k])

# # worst
# latency_path4 = "lero_job.log_exploratory"
# worst_map = copy.deepcopy(lero_map)
# with open(latency_path4, 'r') as f:
#     for line in f.readlines():
#         arr = line.strip().split("#####")
#         k = arr[0]
#         v = json.loads(arr[1])[0]['Execution Time'] / 1000
#         worst_map[k] = max(v,worst_map[k],pg_map[k])

lero_list = []
pg_list = []
best_list = []
worst_list = []
common_qid = list(set(get_qid(latency_path1)).intersection(set(get_qid(latency_path2))))
for key in common_qid:
    lero_list.append(lero_map[key])
    pg_list.append(pg_map[key])

    # best_list.append(best_map[qid])
    # worst_list.append(worst_map[qid])

df = pd.DataFrame(columns = ['qid','lero','pg','best','worst','lero-pg','lero-best','lero-worst','pg-worst'])
df['qid'] = list(common_qid)
df['lero'] = lero_list
df['pg'] = pg_list
df['lero-pg'] = df['lero']-df['pg']


# df['best'] = best_list
# df['worst'] = worst_list
# df['lero-best'] = df['lero']-df['best']
# df['lero-worst'] = df['lero']-df['worst']
# df['pg-worst'] = df['pg']-df['worst']

df.to_csv("/home/admin/wd_files/Lero/test_script/result/excel/job"+str(TRAIN_JOB_NUM)+".csv",index = False)
