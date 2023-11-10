from Hinter import *
from PGUtils import PGGRunner
from sql2fea import Sql2Vec, TreeBuilder
from NET import TreeNet
from TreeLSTM import SPINN

config = Config()


class ValueExtractor:
    def __init__(self, offset=config.offset, max_value=20):
        self.offset = offset
        self.max_value = max_value

    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self, v):
        return int(np.log(2 + v) / np.log(config.max_time_out) * 200) / 200.

    def decode(self, v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v * np.log(config.max_time_out))  # -self.offset

    def cost_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def cost_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost

    def latency_encode(self, v, min_latency, max_latency):
        return (v - min_latency) / (max_latency - min_latency)

    def latency_decode(self, v, min_latency, max_latency):
        return (max_latency - min_latency) * v + min_latency

    def rows_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def rows_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost


class Plan2Latency:
    def __init__(self, DB, path):
        tree_builder = TreeBuilder()
        self.pgrunner = PGGRunner(DB, config.user, config.password, config.ip, config.port, need_latency_record=False,
                                  latency_file=config.latency_file)
        self.sql2vec = Sql2Vec(self.pgrunner)
        value_network = SPINN(head_num=config.head_num, input_size=7 + 2, hidden_size=config.hidden_size, table_num=50,
                              sql_size=config.sql_size).to(config.device)
        net = TreeNet(tree_builder=tree_builder, value_network=value_network)
        self.model = net
        net.load_model(path)

    def predict(self, plan_jsons, sql):

        sql_vec, alias = None, None
        try:
            sql_vec, alias = self.sql2vec.to_vec(sql)
        except:
            return None

        sql_vec, _ = self.sql2vec.to_vec(sql)
        # 转换为[{}]
        # plan_jsons = [plan_json]
        # 输入计划，获取均值、方差等，来预测pg原始optimization的执行时间
        # res = list(zip(mean_item,variance_item,v2_item)),plan_times = [(0.43368837237358093, 0.9064075398809958, 1.0)]
        output = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons, sql_vec=sql_vec)
        if output == None:
            return None
        latency_list = [latency[0] for latency in output]
        # latency = output[0][0]
        return latency_list

    def predictWithUncertaintyBatch(self, plan_jsons, sql_vec):
        # model = TreeNet(tree_builder= tree_builder,value_network = value_network)
        # 利用lstm来抽取整个query的特征表示，EQ
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        import torchfold
        fold = torchfold.Fold(cuda=True)
        res = []
        multi_list = []

        # debug_tree_features = []

        # 实际上就一个plan_json
        for plan_json in plan_jsons:
            # 利用tree-lstm将plan转化为特征Rp
            tree_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
            # debug_tree_features.append(tree_feature)
            # 获取multi_value，即（Rp，EQ）
            if tree_feature == None:
                return None
            multi_value = self.model.plan_to_value_fold(tree_feature=tree_feature, sql_feature=sql_feature, fold=fold)
            multi_list.append(multi_value)
        # if debug_tree_features[0][1].equal(debug_tree_features[1][1]):
        #     print("the same")
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


def load_hyperqo_model(model_path, db):
    return Plan2Latency(db,model_path)


def get_hyperqo_result(plans: list, sql, model: Plan2Latency):
    # [(mean,var,other),...]
    results = model.predict(plans, sql)
    if results is None:
        return None
    value_extractor = ValueExtractor()
    # latencies = [r[0] for r in results]
    latencies = results

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
