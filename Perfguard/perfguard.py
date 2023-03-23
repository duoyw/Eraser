import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from Perfguard.outer_module import util
from Perfguard.perfguardConfig import Config
from RegressionFramework.Plan.Plan import Plan
from RegressionFramework.Plan.PlanFactory import PlanFactory

config = Config()


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")


def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    batch_size = A.shape[0]
    final_list = []
    for i in range(batch_size):
        tmp_A = A[i, :, :].squeeze()
        I = np.eye(tmp_A.shape[0])
        A_hat = tmp_A + I  # add self-loops
        D_hat_diag = np.sum(A_hat, axis=1)
        D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
        D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
        D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
        final = np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        final_list.append(final)
    final_result = np.stack(final_list, axis=0)
    return final_result


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # bias = False is also ok.
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None

    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class Attention_Plus(nn.Module):
    def __init__(self, embed_dim, k_dim):
        super(Attention_Plus, self).__init__()
        self.linear1 = nn.Linear(embed_dim, k_dim)
        self.linear2 = nn.Linear(embed_dim, k_dim)

    def forward(self, plan1, plan2):
        # B*M*F，B*M‘*F
        W_s1 = self.linear1(plan1)
        W_s2 = self.linear2(plan2)
        # B*M*M'
        # print(W_s1.shape,W_s2.shape)
        E = torch.sigmoid(torch.bmm(W_s1, W_s2.permute(0, 2, 1)))
        Y = torch.bmm(E, plan2)
        X = (torch.mul(plan1, Y) + Y - plan1) / 2.0
        return X, Y


class NeuralTensorNetwork(nn.Module):

    def __init__(self, embedding_size, tensor_dim, dropout=0.5):
        super(NeuralTensorNetwork, self).__init__()
        self.tensor_dim = tensor_dim
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(tensor_dim)

    def forward(self, emb1, emb2):
        R = self.tensor_Linear(emb1, emb2, self.T1, self.W1)
        R = self.tanh(R)
        R = self.dropout(R)
        return R

    def tensor_Linear(self, emb1, emb2, tensor_layer, linear_layer):
        # b*1*d
        batch_size, _, emb_size = emb1.size()
        emb1_2 = torch.cat((emb1, emb2), dim=2)
        # b*1*tensor_dim
        linear_product = linear_layer(emb1_2).view(batch_size, -1)

        # 2. Tensor Production
        # b*1*d  d*(d*k) -> b*1*(d*k)
        tensor_product = emb1.view(batch_size, emb_size).mm(tensor_layer.view(emb_size, -1)).view(batch_size, 1,
                                                                                                  emb_size, -1)
        # |tensor_product| = (batch_size, unknown_dim * tensor_dim)
        # 1*k*d
        tensor_product = tensor_product.view(batch_size, -1, emb_size).bmm(emb2.view(batch_size, emb_size, _)).view(
            batch_size, -1)
        tensor_product = tensor_product.contiguous()

        # 3. Summation
        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)
        return self.bn(result)


class PerfGuard(nn.Module):
    def __init__(self, input_dim, embed_dim, tensor_dim, p):
        super(PerfGuard, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, embed_dim)
        self.dropout = nn.Dropout(p)
        self.attention = Attention_Plus(embed_dim=embed_dim, k_dim=embed_dim)
        self.ntn = NeuralTensorNetwork(embedding_size=embed_dim, tensor_dim=tensor_dim)
        self.linear = nn.Linear(tensor_dim, 1)
        self._feature_generator = None

    def forward(self, A1, A2, X1, X2):
        A1 = torch.from_numpy(preprocess_adj(A1)).cuda(config.device)
        X1 = self.dropout(torch.from_numpy(X1).cuda(config.device))
        F1 = torch.bmm(A1, X1)
        output_gcn1 = self.gcn_layer1(F1)

        A2 = torch.from_numpy(preprocess_adj(A2)).cuda(config.device)
        X2 = self.dropout(torch.from_numpy(X2).cuda(config.device))
        F2 = torch.bmm(A2, X2)
        output_gcn2 = self.gcn_layer1(F2)

        # B*M*N 
        X, Y = self.attention(output_gcn1, output_gcn2)

        """
        将X和Y转换为B*1*N'的形式（注意改变ntn的输入维度）
        要么把第一唯独的都拼接在一起
        要么均值池化
        """
        X = torch.mean(X, axis=1).view(X.shape[0], 1, -1)
        Y = torch.mean(Y, axis=1).view(Y.shape[0], 1, -1)
        ntn_output = self.ntn(X, Y)
        # final_output = self.linear(ntn_output)
        # print(ntn_output[0,:]==ntn_output[1,:])
        final_output = torch.sigmoid(self.linear(ntn_output).view(-1))

        return final_output

    # def save(self, path):
    #     os.makedirs(path, exist_ok=True)

    #     torch.save(self.state_dict(), _nn_path(path))

    #     with open(_feature_generator_path(path), "wb") as f:
    #         joblib.dump(self._feature_generator, f)
    #     with open(_input_feature_dim_path(path), "wb") as f:
    #         joblib.dump(self._input_feature_dim, f)


class PerfGuardModel:
    def __init__(self, feature_generator=None) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None

    def load_model(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self._net = PerfGuard(self._input_feature_dim, config.embd_dim, config.tensor_dim, config.dropout)
        if config.CUDA:
            self._net = self._net.cuda(config.device)
            self._net = torch.nn.DataParallel(self._net, device_ids=config.GPU_LIST)
            self._net.load_state_dict(torch.load(_nn_path(path)))
            self._net.eval()
        else:
            self._net.load_state_dict(torch.load(
                _nn_path(path), map_location=torch.device('cpu')))

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def to_features(self, plans):
        features, _ = self._feature_generator.transform(plans)
        # plan_num,node_num,feature
        features = util.prepare_trees(features, self.transformer, self.left_child, self.right_child, cuda=False,
                                      device=None)
        return np.array(features)

    def transformer(self, x):
        return x.get_feature()

    def left_child(self, x):
        return x.get_left()

    def right_child(self, x):
        return x.get_right()

    def predict(self, plans1, plans2):
        features1 = self.to_features(plans1)
        features2 = self.to_features(plans2)
        adjaceny_matrix_list_x1 = self.get_two_adjaceny_matrix(plans1)
        adjaceny_matrix_list_x2 = self.get_two_adjaceny_matrix(plans2)
        predict = self._net(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
        return predict

    def get_two_adjaceny_matrix(self, plans):
        # self.MAX_NODE_NUM = self.x1.shape[1]
        node_nums = []
        for plan in plans:
            node_num = self.get_plan_node_count(plan)
            node_nums.append(node_num)
        max_node_num = max(node_nums)
        # plan_num,node_num,node_num
        adjaceny_matrix_list_x1 = [self.get_adjaceny_matrix(plan_json[0]['Plan'], max_node_num) for plan_json in
                                   [plans]]
        adjaceny_matrix_list_x1 = np.array(adjaceny_matrix_list_x1)
        return adjaceny_matrix_list_x1

    def get_adjaceny_matrix(self, plan_json, max_node_num):
        self.node_index = 0
        adjacecy_matrix = [[0] * max_node_num for _ in range(max_node_num)]
        self.dfs(plan_json, 0, adjacecy_matrix)
        return adjacecy_matrix

    def dfs(self, plan_json, node_index, adjacecy_matrix):
        self.node_index += 1
        if 'Plans' in plan_json:
            plan_json = plan_json['Plans']
            for plan_json_ in plan_json:
                adjacecy_matrix[node_index][self.node_index] = 1
                adjacecy_matrix[self.node_index][node_index] = 1
                self.dfs(plan_json_, self.node_index, adjacecy_matrix)
        else:
            return

    def get_plan_node_count(self, plan: str):
        plan: Plan = PlanFactory.get_plan_instance("pg", plan)
        return len(plan.get_all_nodes())
