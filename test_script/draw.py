import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import *
from delete_time_out import *
import math
import random



def get_qid(latency_path):
    qid_list = []
    with open(latency_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            k = arr[0]
            qid_list.append(k)
    return qid_list

def sum_runtime_file(fp,common_qid):
    sum_ = 0
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(SEP)
            if arr[0] in common_qid:
                sum_ += json.loads(arr[1].strip())[0]['Execution Time'] / 1000 

    return sum_
    
#####删除240～249
qid_list_remove = []
for i in range(240,250):
    qid_list_remove.append('q'+str(i))


delete_time_out_test(PG_TEST)
common_qid = list(set(get_qid(PG_TEST)).intersection(set(get_qid(LERO_TEST+str(0)))))
common_qid  = [x for x in common_qid if x not in qid_list_remove]


lero_run_time_test = []
for i in range(0, 1):
    lero_run_time_test.append(sum_runtime_file(LERO_TEST+ str(i),common_qid))

lero_run_time_test = lero_run_time_test*5
pg_run_time_test = [sum_runtime_file(PG_TEST,common_qid)] * 5

x = range(len(pg_run_time_test))
plt.figure(figsize=(12, 8))

plt.plot(x, pg_run_time_test, label="PostgreSQL", linewidth=8)
plt.plot(x, lero_run_time_test, label="Lero", linewidth=8)

plt.ylabel("Time (s)", fontsize=52)
plt.xlabel("Model id", fontsize=52)
plt.title("Test", fontsize=52)
plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], size=36)
plt.yticks(size=36)
plt.legend(fontsize=36)
plt.grid()

plt.tight_layout()
plt.savefig("./test.jpg")