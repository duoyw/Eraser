import hashlib
import json
import os
from time import time
from config import *
import fcntl
import psycopg2

# 加密
def encode_str(s):
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()
        
def run_query(q, run_args,CONNECTION_STR):
    # 1 连接数据库；2 打开lero开关；3输出执行计划
    start = time()
    # CONNECTION_STR = "dbname=" + DB + " user=" + USER + " password=" + PASSWORD + " host=localhost port=" + str(PORT)
    conn = psycopg2.connect(CONNECTION_STR)
    conn.set_client_encoding('UTF8')
    result = None
    ## 方便debug
    cur = conn.cursor()
    cur.execute("select pg_backend_pid();")
    pid = cur.fetchall()
    print(pid)
    try:
        cur = conn.cursor()
        ## run_args:["SET enable_lero TO True"]
        if run_args is not None and len(run_args) > 0:
            for arg in run_args:
                # SET enable_lero TO True
                cur.execute(arg)
        # TIMEOUT = 30000000       
        cur.execute("SET statement_timeout TO " + str(TIMEOUT))
        # EXPLAIN的作用是输出pg的执行计划
        # q = "EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) " + sql
        cur.execute(q)
        result = cur.fetchall()
    finally:
        
        conn.close()
    # except Exception as e:
    #     conn.close()
    #     raise e
    
    stop = time()
    return stop - start, result

def get_history(encoded_q_str, plan_str, encoded_plan_str):
    # LOG_PATH = "./log/query_latency"
    # encoded_q_str = 'e19b3906eb705eae8d689d409f980f89'
    # encoded_plan_str = '0431aab98bb4efb6474be00798aa1ed4'
    history_path = os.path.join(LOG_PATH, encoded_q_str, encoded_plan_str)
    if not os.path.exists(history_path):
        return None
    
    # print("visit histroy path: ", history_path)
    with open(os.path.join(history_path, "check_plan"), "r") as f:
        # 比较当前lero生成的计划和文件中的历史计划，看是否存在hash冲突
        history_plan_str = f.read().strip()
        if plan_str != history_plan_str:
            print("there is a hash conflict between two plans:", history_path)
            print("given", plan_str)
            print("wanted", history_plan_str)
            return None
    
    # print("get the history file:", history_path)
    with open(os.path.join(history_path, "plan"), "r") as f:
        return f.read().strip()
    
def save_history(q, encoded_q_str, plan_str, encoded_plan_str, latency_str):
    history_q_path = os.path.join(LOG_PATH, encoded_q_str)
    # 写入原sql语句
    if not os.path.exists(history_q_path):
        os.makedirs(history_q_path)
        with open(os.path.join(history_q_path, "query"), "w") as f:
            f.write(q)
    else:
        with open(os.path.join(history_q_path, "query"), "r") as f:
            history_q = f.read()
            if q != history_q:
                print("there is a hash conflict between two queries:", history_q_path)
                print("given", q)
                print("wanted", history_q)
                return
    
    # 写入plan_str和latency_str
    history_plan_path = os.path.join(history_q_path, encoded_plan_str)
    if os.path.exists(history_plan_path):
        print("the plan has been saved by other processes:", history_plan_path)
        return
    else:
        os.makedirs(history_plan_path)
        
    with open(os.path.join(history_plan_path, "check_plan"), "w") as f:
        f.write(plan_str)
    with open(os.path.join(history_plan_path, "plan"), "w") as f:
        f.write(latency_str)
    print("save history:", history_plan_path)

def explain_query(q, run_args, CONNECTION_STR,contains_cost = False):
    q = "EXPLAIN (COSTS " + ("" if contains_cost else "False") + ", FORMAT JSON, SUMMARY) " + (q.strip().replace("\n", " ").replace("\t", " "))
    _, plan_json = run_query(q, run_args,CONNECTION_STR)
    plan_json = plan_json[0][0]
    if len(plan_json) == 2:
        # remove bao's prediction
        plan_json = [plan_json[1]]
    return plan_json

def create_training_file(training_data_file, *latency_files):


    if (os.path.exists(training_data_file)):
        os.remove(training_data_file)
    # 读取latency_files中的每一行
    lines = []
    for file in latency_files:
        with open(file, 'r') as f:
            lines += f.readlines()

    pair_dict = {}
    # latency_files中的每一行:query_name######[{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]
    for line in lines:
        arr = line.strip().split(SEP)
        if arr[0] not in pair_dict:
            pair_dict[arr[0]] = []
        pair_dict[arr[0]].append(arr[1])

    """
    每一行存储着同一个查询语句的多个查询计划的信息，不同查询计划在不同行
    [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]####[...]####[...]
    [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]####[...]####[...]
    [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]####[...]####[...]
    """
    pair_str = []
    for k in pair_dict:
        if len(pair_dict[k]) > 1:
            candidate_list = pair_dict[k]
            pair_str.append(SEP.join(candidate_list))
    str = "\n".join(pair_str)

    with open(training_data_file, 'w') as f2:
        f2.write(str)

## 将查询变成计划存下来
def do_run_query(sql, query_name, run_args, latency_file,CONNECTION_STR, write_latency_file = True, manager_dict = None, manager_lock = None):
    sql = sql.strip().replace("\n", " ").replace("\t", " ")
    # 1. run query with pg hint
    # 获取pg的执行计划
    # _ = stop - start, plan_json = result
    _, plan_json = run_query("EXPLAIN (COSTS FALSE, FORMAT JSON, SUMMARY) " + sql, run_args,CONNECTION_STR)
    """
    plan_json [{'Plan': {'Node Type': 'Aggregate', 'Strategy': 'Plain', 'Partial Mode': 'Simple', 'Parallel Aware': False, 
    'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 
    'Inner Unique': True, 'Hash Cond': '(b.userid = u.id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 
    'Parallel Aware': False, 'Relation Name': 'badges', 'Alias': 'b'}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 
    'Parallel Aware': False, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer',
     'Parallel Aware': False, 'Relation Name': 'users', 'Alias': 'u', 'Filter': '(upvotes >= 0)'}]}]}]}, 'Planning Time': 44.38}]
    """
    plan_json = plan_json[0][0]
    if len(plan_json) == 2:
        # remove bao's prediction
        plan_json = [plan_json[1]]
    planning_time = plan_json[0]['Planning Time']
    """
    cur_plan_str
    {"Node Type": "Aggregate", "Strategy": "Plain", "Partial Mode": "Simple", "Parallel Aware": false, 
    "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": false, 
    "Join Type": "Inner", "Inner Unique": true, "Hash Cond": "(b.userid = u.id)", 
    "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": false, 
    "Relation Name": "badges", "Alias": "b"}, {"Node Type": "Hash", "Parent Relationship": "Inner", 
    "Parallel Aware": false, "Plans": [{"Node Type": "Seq Scan", "Parent 
    Relationship": "Outer", "Parallel Aware": false, "Relation Name": "users", "Alias": "u", 
    "Filter": "(upvotes >= 0)"}]}]}]}
    """
    cur_plan_str = json.dumps(plan_json[0]['Plan'])
    try:
        # 2. get previous running result
        latency_json = None
        # 加密当前的计划
        encoded_plan_str = encode_str(cur_plan_str)
        # 加密原始的sql语句
        encoded_q_str = encode_str(sql)
        # 查看是否有（查询，计划）对应的历史信息，以免重新跑一遍
        previous_result = get_history(encoded_q_str, cur_plan_str, encoded_plan_str)
        if previous_result is not None:
            latency_json = json.loads(previous_result)
        else:
            if manager_dict is not None and manager_lock is not None:
                manager_lock.acquire()
                if cur_plan_str in manager_dict:
                    manager_lock.release()
                    print("another process will run this plan:", cur_plan_str)
                    return
                else:
                    manager_dict[cur_plan_str] = 1
                    manager_lock.release()

            # 3. run current query 
            run_start = time()
            try:
                # ANALYZE：执行语句并显示真正的运行时间和其他统计信息，默认值False
                # VERBOSE：显示额外的信息，尤其是计划树中每个节点的字段列表
                # COSTS：包括每个计划节点的启动成本预估和总成本消耗，也包括行数和行宽度的预估，默认值是True
                # TIMING：在输出中包含实际启动时间和每个节点花费的时间，这个参数一般在ANALYZE也启用的时候使用，缺省为TRUE
                # FORMAT：声明输出格式，，缺省为TEXT
                # 实际执行计划，获取函数执行时间和latency_json
                _, latency_json = run_query("EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) " + sql, run_args,CONNECTION_STR)
                latency_json = latency_json[0][0]
                if len(latency_json) == 2:
                    # remove bao's prediction
                    latency_json = [latency_json[1]]
            except Exception as e:
                if  time() - run_start > (TIMEOUT / 1000 * 0.9):
                    # Execution timeout
                    _, latency_json = run_query("EXPLAIN (VERBOSE, COSTS, FORMAT JSON, SUMMARY) " + sql, run_args,CONNECTION_STR)
                    latency_json = latency_json[0][0]
                    if len(latency_json) == 2:
                        # remove bao's prediction
                        latency_json = [latency_json[1]]
                    latency_json[0]["Execution Time"] = TIMEOUT
                else:
                    raise e

            latency_str = json.dumps(latency_json)
            # 将原sql、预估的plan信息cur_plan_str和将plan实际执行后的信息latency_str分别存到三个不同的文件sql、check_plan和plan
            save_history(sql, encoded_q_str, cur_plan_str, encoded_plan_str, latency_str)

        # 4. save latency
        # latency_json = [{'Plan': {...}, 'Planning Time': 41.435, 'Triggers': [...], 'Execution Time': 39.693}]
        # 计划时间和执行时间的区别
        latency_json[0]['Planning Time'] = planning_time
        if write_latency_file:
            with open(latency_file, "a+") as f:
                # 对文件加锁
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(query_name + SEP + json.dumps(latency_json) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)

        exec_time = latency_json[0]["Execution Time"]
        print("Current time: {0}, query_name: {1}, exec_time: {2}".format(time(), query_name, exec_time, flush=True))
        # print(time(), query_name, exec_time, flush=True)

    except Exception as e:
        with open(latency_file + "_error", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(query_name + "\n")
            f.write(str(e).strip() + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)