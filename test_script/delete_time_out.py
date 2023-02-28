import json
from config import *
path1 =  "./result/pg/pg_stats_test_.log"



def delete_time_out_test(path1):
    lines_pg = []
    with open(path1,'r') as f:
        lines_pg = f.readlines()

    with open(path1,'w') as f:
        for line in lines_pg:
            arr = line.strip().split(SEP)
            if json.loads(arr[1])[0]['Execution Time'] < TIMEOUT:
                f.write(line)
          

    

