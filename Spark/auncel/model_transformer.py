import math
import os
from datetime import datetime
from time import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from pandas import DataFrame
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from QueryFormer.model.database_util import collator, Batch, Encoding
from QueryFormer.model.dataset import PlanTreeDataset
from QueryFormer.model.model import QueryFormer, cal_hidden_dim
from Spark.auncel.model_config import TransformerConfig, TIMESTAMP
from model import AuncelModel
from Spark.auncel.test_script.config import DATA_BASE_PATH, LOG_BASE_PATH
from feature import SampleEntity
from TreeConvolution.util import prepare_trees

CUDA = torch.cuda.is_available()
# CUDA = False
# GPU_LIST = [0]
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if CUDA else "cpu")


def collate_pairwise_fn(x):
    x1 = []
    x2 = []
    labels = []
    y1 = []
    y2 = []

    for data1, data2 in x:
        x1.append(data1[0])
        x2.append(data2[0])
        labels.append(1.0 if data1[1] >= data2[1] else 0.0)
        y1.append(data1[1])
        y2.append(data2[1])
    return x1, x2, labels, y1, y2


class AuncelTransformerNet(nn.Module):
    def __init__(self, encoding: Encoding) -> None:
        super(AuncelTransformerNet, self).__init__()
        self._cuda = False
        self.device = None

        emb_size = 64
        use_hist = TransformerConfig.use_hist
        use_one_hot = TransformerConfig.use_one_hot

        query_former = QueryFormer(emb_size=emb_size, ffn_dim=128, head_size=12,
                                   dropout=0.0, n_layers=8,
                                   use_sample=False, use_hist=use_hist, encoding=encoding, use_one_hot=use_one_hot)
        hidden_dim = query_former.get_output_dim()
        self.tree_conv = nn.Sequential(query_former
                                       ,
                                       nn.Linear(hidden_dim, 32),
                                       nn.LeakyReLU(),
                                       nn.Linear(32, 1)
                                       )

    def forward(self, trees):
        return self.tree_conv(trees)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class AuncelModelTransformerPairWise(AuncelModel):
    def __init__(self, encoding, histogram, normalizer, table_sample) -> None:
        super().__init__(None)
        self.encoding = encoding
        self.histogram = histogram
        self.normalizer = normalizer
        self.table_sample = table_sample

    def load(self, path):
        path = os.path.join(DATA_BASE_PATH, path)
        with open(self._transformer_info_path(path), "rb") as f:
            infos = joblib.load(f)
            self.encoding = infos[0]
            self.histogram = infos[1]
            self.normalizer = infos[2]
            self.table_sample = infos[3]

        self._net = AuncelTransformerNet(self.encoding)
        if CUDA:
            self._net.load_state_dict(torch.load(self._nn_path(path)))
        else:
            self._net.load_state_dict(torch.load(
                self._nn_path(path), map_location=torch.device('cpu')))
        self._net.eval()

    def save(self, path):
        path = os.path.join(DATA_BASE_PATH, path)
        os.makedirs(path, exist_ok=True)

        if CUDA:
            torch.save(self._net.module.state_dict(), self._nn_path(path))
        else:
            torch.save(self._net.state_dict(), self._nn_path(path))

        with open(self._transformer_info_path(path), "wb") as f:
            joblib.dump([self.encoding, self.histogram, self.normalizer, self.table_sample], f)

    def _transformer_info_path(self, base):
        return os.path.join(base, "info")

    def to_feature(self, targets, dataset=None):
        plan_df = DataFrame({"id": list(range(0, len(targets))), "json": targets})
        assert len(targets) == 1
        return PlanTreeDataset(plan_df, self.encoding, self.histogram, self.normalizer, self.table_sample, dataset)[0][
            0]

    # def fit(self, dataset1: PlanTreeDataset, dataset2: PlanTreeDataset):
    #
    #     # batch size
    #     batch_size = 64
    #     if CUDA:
    #         batch_size = batch_size * len(GPU_LIST)
    #
    #     pairs = []
    #     for i in range(len(dataset1)):
    #         pairs.append((dataset1[i], dataset2[i]))
    #
    #     dataset = DataLoader(pairs,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=collate_pairwise_fn)
    #
    #     #  determine the initial number of channels
    #     self._net = AuncelTransformerNet()
    #     if CUDA:
    #         self._net = self._net.cuda(device)
    #         self._net = torch.nn.DataParallel(
    #             self._net, device_ids=GPU_LIST)
    #         self._net.cuda(device)
    #
    #     optimizer = None
    #     lr = 1e-3
    #     if CUDA:
    #         # optimizer = torch.optim.Adam(self._net.module.parameters())
    #         optimizer = torch.optim.Adam(self._net.module.parameters(), lr=lr)
    #         # optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
    #     else:
    #         optimizer = torch.optim.Adam(self._net.parameters(),lr=lr)
    #
    #     # bce_loss_fn = Weight_BCE_Loss()
    #     bce_loss_fn = torch.nn.BCELoss()
    #
    #     losses = []
    #     sigmoid = nn.Sigmoid()
    #     start_time = time()
    #     for epoch in range(1000):
    #         loss_accum = 0
    #         for x1, x2, label, y1, y2 in dataset:
    #             x1 = self.collator(x1)
    #             x2 = self.collator(x2)
    #
    #             if CUDA:
    #                 x1.to(device)
    #                 x2.to(device)
    #
    #             y_pred_1 = self._net(x1.to_list())
    #             y_pred_2 = self._net(x2.to_list())
    #             weight = np.abs(np.array(y1) - np.array(y2))
    #
    #             prob_y = sigmoid(y_pred_1 - y_pred_2)
    #
    #             label_y = torch.tensor(np.array(label).reshape(-1, 1))
    #             if CUDA:
    #                 label_y = label_y.cuda(device)
    #
    #             # loss = bce_loss_fn(weight, prob_y, label_y)
    #             loss = bce_loss_fn(prob_y, label_y)
    #             loss_accum += loss.item()
    #
    #             # if CUDA:
    #             #     optimizer.module.zero_grad()
    #             #     loss.backward()
    #             #     optimizer.module.step()
    #             # else:
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #         loss_accum /= len(dataset)
    #         losses.append(loss_accum)
    #         print("Epoch", epoch, "training loss:", loss_accum)
    #     print("training time:", time() - start_time, "batch size:", batch_size)

    def fit(self, dataset1: PlanTreeDataset, dataset2: PlanTreeDataset):
        writer = SummaryWriter(LOG_BASE_PATH + "/{}".format(TIMESTAMP), flush_secs=1)

        # batch size
        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        pairs = []
        for i in range(len(dataset1)):
            pairs.append((dataset1[i], dataset2[i]))

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        #  determine the initial number of channels
        self._net = AuncelTransformerNet(self.encoding)
        if CUDA:
            self._net = self._net.cuda(device)
            self._net = torch.nn.DataParallel(
                self._net, device_ids=GPU_LIST)
            self._net.cuda(device)

        optimizer = None
        lr = 1e-3
        if CUDA:
            # optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = torch.optim.Adam(self._net.module.parameters(), lr=lr)
            # optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)

        # bce_loss_fn = Weight_BCE_Loss()
        bce_loss_fn = torch.nn.BCELoss()
        # mse_loss_fn = torch.nn.MSELoss()
        mse_loss_fn = Weight_MSE_Loss()

        losses = []
        sigmoid = nn.Sigmoid()
        start_time = time()

        # writer.add_graph(self._net, input_to_model=[self.extract_ele(dataset1)[0]])
        for epoch in range(100):
            loss_accum = 0
            mse_loss_accum = 0
            bce_loss_accum = 0
            for x1, x2, label, y1, y2 in dataset:
                x1 = self.collator(x1)
                x2 = self.collator(x2)

                if CUDA:
                    x1.to(device)
                    x2.to(device)

                y_pred_1 = self._net(x1.to_list())
                y_pred_2 = self._net(x2.to_list())

                # label_y1 = torch.tensor(np.array(y1).reshape(-1, 1))

                # calculate MSE
                mse_label_y = torch.tensor(np.array([y1, y2]).reshape(-1, 1))
                y_pred = torch.cat((y_pred_1, y_pred_2))
                if CUDA:
                    mse_label_y = mse_label_y.cuda(device)
                # mse_loss = mse_loss_fn(y_pred, mse_label_y)
                weight = np.abs(np.array([*y1, *y2]))
                mse_loss = mse_loss_fn(weight, y_pred, mse_label_y)
                mse_loss_accum += mse_loss.item()

                # calculate BCE
                bce_label_y = torch.tensor(np.array(label).reshape(-1, 1))
                if CUDA:
                    bce_label_y = bce_label_y.cuda(device)
                prob_y = sigmoid(y_pred_1 - y_pred_2)
                weight = np.abs(np.array(y1) - np.array(y2))
                bce_loss = bce_loss_fn(prob_y, bce_label_y)

                # bce_loss = bce_loss_fn(weight, prob_y, bce_label_y)

                bce_loss_accum += bce_loss.item()

                optimizer.zero_grad()
                mse_loss.backward()
                # bce_loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            mse_loss_accum /= len(dataset)
            bce_loss_accum /= len(dataset)
            writer.add_scalar("mse_loss_accum", mse_loss_accum, global_step=epoch)
            writer.add_scalar("bce_loss_accum", bce_loss_accum, global_step=epoch)
            losses.append(loss_accum)
            print("Epoch", epoch, "training mse_loss_accum:", mse_loss_accum)
            print("Epoch", epoch, "training bce_loss_accum:", bce_loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)
        writer.close()

    def collator(self, datas):
        x = torch.cat([s['x'] for s in datas])
        attn_bias = torch.cat([s['attn_bias'] for s in datas])
        rel_pos = torch.cat([s['rel_pos'] for s in datas])
        heights = torch.cat([s['heights'] for s in datas])

        return Batch(attn_bias, rel_pos, heights, x)

    def extract_ele(self, dataset1):
        data = dataset1[0]
        batch = self.collator([data[0]])
        if CUDA:
            batch.to(device)
        return batch.to_list(), data[1]

    def predict(self, x):
        if CUDA:
            self._net = self._net.cuda(device)

        if not isinstance(x, list):
            x = [x]

        tree = None
        # todo
        # if CUDA:
        #     tree = self._net.module.build_trees(x)
        # else:
        #     tree = self._net.build_trees(x)
        x = self.collator(x)
        x.to(device)
        x = x.to_list()
        pred = self._net(x).cpu().detach().numpy()
        return pred


class Weight_BCE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight, predict_y, label_y):
        loss = - (label_y * torch.log(predict_y) +
                  (1 - label_y) * torch.log(1 - predict_y))
        weight = torch.tensor(weight).cuda()
        loss = torch.mul(weight, loss)
        loss = loss.sum()
        return loss


class Weight_MSE_Loss(nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, weight, predict_y, label_y):
        weight = torch.tensor(weight).cuda(predict_y.device)
        loss: torch.tensor = super().forward(predict_y, label_y) * weight
        loss = loss.mean()
        return loss
