from ImportantConfig import Config

config = Config()
from sql2fea import TreeBuilder, value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN
import json
from KNN import KNN
import torch
from PGUtils import pgrunner


class Plans2BestPlan:
    def __init__(self, model_path):
        # model
        tree_builder = TreeBuilder()
        self.sql2vec = Sql2Vec()
        value_network = SPINN(head_num=config.head_num, input_size=7 + 2, hidden_size=config.hidden_size, table_num=50,
                              sql_size=config.sql_size).to(config.device)
        net = TreeNet(tree_builder=tree_builder, value_network=value_network)
        self.model = net
        self.model.load_model(model_path)

        self.knn = KNN(10)
        self.value_extractor = value_extractor

    def get_best_plan_qo(self, sql, plan_list):
        # get best plan qo
        sql_vec, alias = self.sql2vec.to_vec(sql)
        chosen_leading_pair = self.findBestHint(plan_list, sql_vec)

        # plan_json  = pgrunner.getAnalysePlanJson(sql = chosen_leading_pair[1]+sql)
        # leading_time_flag = (plan_json['Plan']['Actual Total Time'],plan_json['timeout'])

        time = chosen_leading_pair[1]['Execution Time']

        # insert value to knn
        self.knn.insertAValue((chosen_leading_pair[0], self.value_extractor.encode(time) - chosen_leading_pair[0][0]))
        return chosen_leading_pair[1]

    def findBestHint(self, plan_list, sql_vec):
        plans = []
        for plan in plan_list:
            p = json.loads(plan)
            if type(p) == list:
                p = p[0]
            plans.append(p)
        plan_list = plans

        # plan_list.extend([plan_list]) #长度为4，还加入了不用hint直接预测sql的性能
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_list, sql_vec=sql_vec)
        # self.max_hint_num = 20 chosen_leading_pair = ((0.5299996733665466, 1.4747910499572754, 1.0), '/*+Leading(ci t)*/', 0.0)
        # 排序的时候不仅考虑了预测的latency，也考虑了q-error
        tmp = zip(plan_times, plan_list)
        chosen_leading_pair = sorted(tmp, key=lambda x: x[0][0] + self.knn.kNeightboursSample(x[0]))[0]
        return chosen_leading_pair

    def predictWithUncertaintyBatch(self, plan_jsons, sql_vec):
        # model = TreeNet(tree_builder= tree_builder,value_network = value_network)
        # 利用lstm来抽取整个query的特征表示，EQ
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        import torchfold
        fold = torchfold.Fold(cuda=True)
        res = []
        multi_list = []
        # 实际上就一个plan_json
        for plan_json in plan_jsons:
            # 利用tree-lstm将plan转化为特征Rp
            tree_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
            # 获取multi_value，即（Rp，EQ）
            multi_value = self.model.plan_to_value_fold(tree_feature=tree_feature, sql_feature=sql_feature, fold=fold)
            multi_list.append(multi_value)
        # dynamic batch
        multi_value = fold.apply(self.model.value_network, [multi_list])[0]
        # 获取均值方差，参数有head_num,即多个输出层，然后计算对应的均值和方差
        mean, variance = self.model.mean_and_variance(multi_value=multi_value[:, :config.head_num])
        # 以e为底的指数 self.var_weight = 0.00
        v2 = torch.exp(multi_value[:, config.head_num] * config.var_weight).data.reshape(-1)
        if isinstance(mean, float):
            mean_item = [mean]
        else:
            mean_item = [x.item() for x in mean]
        if isinstance(variance, float):
            variance_item = [variance]
        else:
            variance_item = [x.item() for x in variance]
        # variance_item = [x.item() for x in variance]
        if isinstance(v2, float):
            v2_item = [v2]
        else:
            v2_item = [x.item() for x in v2]
        # v2_item = [x.item() for x in v2]
        # 返回均值，方差，res对
        res = list(zip(mean_item, variance_item, v2_item))
        return res


def load_hyperqo_best_plan_model(model_path):
    return Plans2BestPlan(model_path)


def get_hyperqo_best_plan(plans, sql, model):
    best_plan_qo = model.get_best_plan_qo(sql, plans)
    return best_plan_qo['Execution Time']


if __name__ == "__main__":
    pass
    # # read plans
    # test_path = "/home/admin/wd_files/HyperQO/plan_with_exe_time/job.test"
    # sql2plans = {}
    # with open(test_path,'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         arr = line.strip().split(config.SEP)
    #         sql = arr[0]
    #         plans = arr[1:]
    #         sql2plans[sql] = plans
    #
    # # load model
    # model_path = './model/job1'
    # plans2best_plan = Plans2BestPlan(model_path)
    #
    # # test
    # hyperqo_chosen_plan = {}
    # for sql in sql2plans.keys():
    #     plan_list = sql2plans[sql]
    #     best_plan_qo = plans2best_plan.get_best_plan_qo(sql,plan_list)
    #     # print(best_plan_qo)
    #     hyperqo_chosen_plan[sql] = best_plan_qo
    #
    # sum_latency_pg = 0
    # sum_latency_qo = 0
    # for qid in hyperqo_chosen_plan.keys():
    #     sum_latency_pg += json.loads(sql2plans[qid][0])[0]['Execution Time']/1000
    #     sum_latency_qo += hyperqo_chosen_plan[qid]['Execution Time']/1000
    # print(sum_latency_pg)
    # print(sum_latency_qo)
