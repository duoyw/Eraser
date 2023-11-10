from perfguardConfig import Config
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json
import pickle
import numpy as np
import joblib
import os
from outer_module import feature

## prepartion
def transformer(x):
    return x.get_feature()
def left_child(x):
    return x.get_left()
def right_child(x):
    return x.get_right()
    
config = Config()

queries = []
if not config.cache:
    from outer_module import sql2fea,mcts,util,PGUtils
    pgrunner = PGUtils.PGGRunner(config.database,config.user,config.password,config.ip,config.port,need_latency_record=True,latency_file=config.latency_file)
    ## load queries
    
    with open(config.queries_file) as f:
        if ".json" in config.queries_file:
            # 'workload/JOB_static.json'
            queries = json.load(f)
        if ".txt" in config.queries_file:
            for line in f.readlines():
                arr = line.strip().split("#####")
                queries.append(arr[1])

    if (os.path.exists(config.log_file)):
        os.remove(config.log_file)

    ## generate plans for each query
    label_list = []
    plan_default = []
    plan_change = []

    for index,query in enumerate(queries[:config.query_num]):
        with open(config.log_file,"a") as f:
            f.write("**********Start the query "+str(index)+".**********"+"\n")
        if isinstance(query,list):
            query = query[0]
        leading_list = []
        latency_list = []
        plan_jsons = []


        sql2vec = sql2fea.Sql2Vec()
        mcts_searcher = mcts.MCTSHinterSearch()
        feature_generator = feature.FeatureGenerator()

        plan_json_PG = pgrunner.getCostPlanJson(query)
        plan_jsons.append(plan_json_PG)
        total_time = pgrunner.getLatency(sql=query,timeout = 300*1000)[0]
        plan_json_PG['Plan']['Actual Total Time'] = total_time
        latency_list.append(total_time)

        sql_vec,alias = sql2vec.to_vec(query)
        alias_id = [config.aliasname2id[a] for a in alias]
        
        id_joins_with_predicate = [(config.aliasname2id[p[0]],config.aliasname2id[p[1]]) for p in sql2vec.join_list_with_predicate]
        id_joins = [(config.aliasname2id[p[0]],config.aliasname2id[p[1]]) for p in sql2vec.join_list]
        leading_length = config.leading_length
        
        if leading_length==-1:
            leading_length = len(alias)
        if leading_length>len(alias):
            leading_length = len(alias)
        join_list_with_predicate = mcts_searcher.findCanHints(len(config.id2aliasname),len(alias),sql_vec,id_joins,id_joins_with_predicate,alias_id,depth=leading_length)
        
        for join in join_list_with_predicate:
            leading_list.append('/*+Leading('+" ".join([sql2vec.id2aliasname[x] for x in join[0][:leading_length]])+')*/')
            plan_jsons.append(pgrunner.getCostPlanJson(leading_list[-1]+query))

            total_time = pgrunner.getLatency(sql=leading_list[-1]+query,timeout = 300*1000)[0]
            plan_jsons[-1]['Plan']['Actual Total Time'] = total_time
            latency_list.append(total_time)

        for i in range(len(plan_jsons)):
            for j in range(i+1,len(plan_jsons)):
                plan_default.append(plan_jsons[i])
                plan_change.append(plan_jsons[j])
                label_list.append(0 if latency_list[i]>=latency_list[j] else 1)

else:
    print("**********cache**********")
    # Execution Time
    with open(config.cache_file) as f:
        for line in f.readlines():
            arr = line.strip().split("%-*-%")
            queries.append(json.loads(arr[1]))



# for i in range(len(plan_jsons)):
#     for j in range(i+1,len(plan_jsons)):
#         plan_default.append(plan_jsons[i])
#         plan_change.append(plan_jsons[j])
#         label_list.append(0 if latency_list[i]>=latency_list[j] else 1)


feature_generator.fit(plan_default + plan_change)
X1, _ = feature_generator.transform(plan_default)
X2, _ = feature_generator.transform(plan_change)

features1 = util.prepare_trees(X1, transformer, left_child, right_child, cuda=False, device=None)
features2 = util.prepare_trees(X2, transformer, left_child, right_child, cuda=False, device=None)
#[6,30,26,2] batch_size,node_num,feature,default or change
features = np.stack((features1,features2),axis = 3)

joblib.dump(features,config.path_data)
joblib.dump(label_list,config.path_label)
# features = joblib.load(config.path_data)
# label = joblib.load(config.path_label)





