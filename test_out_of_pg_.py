import collections
from model import LeroModel
from test_script.config import *
import argparse
import json
from test_script.config import *

def load_model_(model_path):
    lero_model = LeroModel(None)
    lero_model.load(model_path)
    return lero_model

def get_plan_list(explore1,explore2):
    lines = []
    plan_dict= collections.defaultdict(list)

    with open(explore1,'r') as f:
        tmp = f.readlines()
        lines.extend(tmp)

    with open(explore2,'r') as f:
        tmp = f.readlines()
        lines.extend(tmp)
    for line in lines:
        arr= line.strip().split(SEP)
        qid = arr[0]
        plan = arr[1]
        if json.loads(plan)[0]['Execution Time'] < TIMEOUT:
            plan_dict[qid].append(plan)
    return plan_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--model_path",
                        metavar="PATH")
    parser.add_argument("--explore1",
                        metavar="PATH")
    parser.add_argument("--explore2",
                        metavar="PATH")
    parser.add_argument("--output_file",
                    metavar="PATH")


    


#########################

    args = parser.parse_args()
    # args.model_path = '/home/admin/wd_files/Lero-on-PostgreSQL/lero/job_test_model_on_4_0'
    # args.explore1 = '/home/admin/wd_files/Lero-on-PostgreSQL/lero/test_script/result/job/job_test.log'
    # args.explore2 = '/home/admin/wd_files/Lero-on-PostgreSQL/lero/test_script/result/job/job_test.log_exploratory'
    # args.output_file = '/home/admin/wd_files/Lero-on-PostgreSQL/lero/test_script/result/job/4/lero_job_.log_job_test_model_on_4_0'



    lero_model = load_model_(args.model_path)

    ## 读取plan_dict
    plan_dict = get_plan_list(args.explore1,args.explore2)
    best_dict = dict()

    ## 预测分数
    for key in plan_dict.keys():
        best_plan = None
        best_score = float('inf')
        for plan in plan_dict[key]:
            local_features, _ = lero_model._feature_generator.transform([plan])
            y = lero_model.predict(local_features)
            assert y.shape == (1, 1)
            y = y[0][0]
            if y<best_score:
                best_score = y
                best_plan = plan
        
        best_dict[key] = best_plan

    with open(args.output_file,'w') as f:
        for key in best_dict:
            f.write(key + SEP + best_dict[key] + '\n')
    print("done!")