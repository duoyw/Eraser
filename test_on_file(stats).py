from model import LeroModel
from test_script.config import *

def test_benchmark_on_training_data(output_file):
    ## 加载模型
    model_path = MODEL_PREFIX + "_" + str(0)
    lero_model = load_model_(model_path)

    ## 整合plan
    plan_dict = create_plan_dict(OUTPUT_QUERY_LATENCY_FILE_LERO[2:],OUTPUT_QUERY_LATENCY_FILE_LERO[2:] + '_exploratory')
    
    ## 预测
    chose_dict = {}
    for key in plan_dict.keys():
        best_score = 0
        plan_chosen = None
        for plan_json in plan_dict[key]:
            local_features, _ = lero_model._feature_generator.transform([plan_json])
            y = lero_model.predict(local_features)
            assert y.shape == (1, 1)
            y = y[0][0]
            if y < best_score:
                best_score = y
                plan_chosen = plan_json        
        chose_dict[key] = plan_chosen

    with open(output_file, 'w') as f:
        for key in chose_dict.keys():
            f.write(key + SEP + chose_dict[key])


    # print(chose_dict)


def load_model_(model_path):
    lero_model = LeroModel(None)
    lero_model.load(model_path)
    return lero_model

def create_plan_dict(*latency_files):
    # 读取latency_files中的每一行
    lines = []
    for file in latency_files:
        with open(file, 'r') as f:
            lines += f.readlines()
            
    plan_dict = {}
    for line in lines:
        arr = line.strip().split(SEP)
        if arr[0] not in plan_dict:
            plan_dict[arr[0]] = []
        plan_dict[arr[0]].append(arr[1])
    return plan_dict

if __name__ == "__main__":
    test_benchmark_on_training_data(OUTPUT_QUERY_LATENCY_FILE_ON_TRAINING_DATA[2:])


