from get_data import *
import collections
import json
import perfguardConfig

config = perfguardConfig.Config()


def load_model(model_path):
    model = PerfGuard.load_model(model_path)
    return model


def perfguard_predict(plan1, plan2, model: PerfGuard):
    plan_dict = {
        "p1": plan1,
        "p2": plan2
    }

    get_data_ = Get_Dataset_Test(plan_dict)
    features1, features2 = get_data_.get_features()
    adjaceny_matrix_list_x1, adjaceny_matrix_list_x2 = get_data_.get_two_adjaceny_matrix()
    predict = model(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
    return predict


def test():
    plan_dict = collections.defaultdict(list)
    with open(config.pg_plan_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(config.SEP)
            plan_dict[arr[0]].append(json.loads(arr[1]))

    with open(config.lero_plan_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split(config.SEP)
            if arr[0] in plan_dict.keys():
                plan_dict[arr[0]].append(json.loads(arr[1]))

    """
    get testing data
    """

    qids = []
    for qid in plan_dict.keys():
        qids.append(qid)

    get_data_ = Get_Dataset_Test(plan_dict)

    model_path = 'model_pth/' + config.data + '_' + str(config.data_num)
    model = get_data_.load_model(model_path)
    features1, features2 = get_data_.get_features()
    adjaceny_matrix_list_x1, adjaceny_matrix_list_x2 = get_data_.get_two_adjaceny_matrix()
    predict = model(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)

    """
    metric
    """
    predict = predict.cpu().detach().numpy().tolist()
    predict_label = [1 if x > config.threshold else 0 for x in predict]
    # print(predict_label)

    label_dict = dict(zip(qids, predict_label))
    # print(label_dict)
    # true_label = label
    # p = precision_score(true_label,predict_label)
    # r = recall_score(true_label,predict_label)
    # f1 = f1_score(true_label,predict_label)
    # print("the precision is {0} \n the recall is {1} \n the f1-score is {2}".format(p,r,f1))

    perfguard_dict = {}
    for qid in label_dict.keys():
        if len(plan_dict[qid]) == 2:
            if label_dict[qid] == 1:
                # print(plan_dict[0])
                perfguard_dict[qid] = plan_dict[qid][0]
            else:
                perfguard_dict[qid] = plan_dict[qid][1]

    with open(config.perfguard_path, 'w') as f:
        for qid in perfguard_dict.keys():
            f.write(qid + config.SEP + json.dumps(perfguard_dict[qid]) + '\n')
    print("done!")


if __name__ == '__main__':
    test()
