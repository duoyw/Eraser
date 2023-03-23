from Hinter import *
from sql2fea import Sql2Vec, TreeBuilder, value_extractor, ValueExtractor
from NET import TreeNet
from TreeLSTM import SPINN
from ImportantConfig import Config

config = Config()


class Plan2Latency():
    def __init__(self, path):
        tree_builder = TreeBuilder()
        self.sql2vec = Sql2Vec()
        value_network = SPINN(head_num=config.head_num, input_size=7 + 2, hidden_size=config.hidden_size, table_num=50,
                              sql_size=config.sql_size).to(config.device)
        net = TreeNet(tree_builder=tree_builder, value_network=value_network)
        self.model = net
        net.load_model(path)

    def predict(self, plan_jsons, sql):
        sql_vec, _ = self.sql2vec.to_vec(sql)
        # 转换为[{}]
        # 输入计划，获取均值、方差等，来预测pg原始optimization的执行时间
        # res = list(zip(mean_item,variance_item,v2_item)),plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        latencies = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons, sql_vec=sql_vec)
        return latencies

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


def load_hyperqo_model(model_path):
    return Plan2Latency(model_path)


def get_hyperqo_result(plans: list, sql, model: Plan2Latency):
    # [(mean,var,other),...]
    results = model.predict(plans, sql)
    value_extractor = ValueExtractor()
    latencies = [r[0] for r in results]
    return [value_extractor.decode(l) for l in latencies]


if __name__ == "__main__":
    pass
    # # read plans
    # test_path = "/home/admin/wd_files/HyperQO/plan_with_exe_time/job.test"
    # sql2plans = {}
    # with open(test_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         arr = line.strip().split(config.SEP)
    #         sql = arr[0]
    #         plans = arr[1:]
    #         sql2plans[sql] = plans
    #
    # # predict
    # model_path = config.model_path
    # latency_predictor = Plan2Latency(model_path)
    # plan_json = None
    # sql = None
    # for sql in sql2plans.keys():
    #     plans = sql2plans[sql]
    #     for plan in plans:
    #         plan = json.loads(plan)[0]
    #         latency = latency_predictor.predict(plan, sql)
    #         print("the latency is {}.".format(latency))
