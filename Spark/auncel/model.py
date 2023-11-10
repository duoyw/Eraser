import os
from time import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch import tensor
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Spark.auncel.model_config import TIMESTAMP
from Spark.auncel.test_script.config import DATA_BASE_PATH, LOG_BASE_PATH
from auncel.TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees
from feature import SampleEntity, Normalizer

CUDA = torch.cuda.is_available()
CUDA = False
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
# GPU_LIST = [2]

torch.set_default_tensor_type(torch.DoubleTensor)
# device = torch.device("cuda:4" if CUDA else "cpu")
device = torch.device("cuda:0" if CUDA else "cpu")


def transformer(x: SampleEntity):
    return x.get_feature()


def left_child(x: SampleEntity):
    return x.get_left()


def right_child(x: SampleEntity):
    return x.get_right()


class AuncelNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(AuncelNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        # self.tree_conv = nn.Sequential(
        #     BinaryTreeConv(self.input_feature_dim, 256),
        #     # TreeLayerNorm(),
        #     # TreeActivation(nn.LeakyReLU()),
        #     # BinaryTreeConv(512, 256),
        #     TreeLayerNorm(),
        #     TreeActivation(nn.LeakyReLU()),
        #     BinaryTreeConv(256, 128),
        #     TreeLayerNorm(),
        #     TreeActivation(nn.LeakyReLU()),
        #     BinaryTreeConv(128, 64),
        #     TreeLayerNorm(),
        #     DynamicPooling(),
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 1)
        # )

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 512),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class AuncelNetDistance(AuncelNet):
    def __init__(self, input_feature_dim) -> None:
        super(AuncelNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 512),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
        )


class AuncelModel():
    def __init__(self, feature_generator) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None

    def inverse_norm(self, name, value):
        norm: Normalizer = self._feature_generator.normalizer
        return norm.inverse_norm(value, name)

    def inverse_norm_no_log(self, name, value):
        norm: Normalizer = self._feature_generator.normalizer
        return norm.inverse_norm_no_log(value, name)

    def norm_no_log(self, name, value):
        norm: Normalizer = self._feature_generator.normalizer
        return norm.norm_no_log(value, name)

    def to_feature(self, targets, dataset=None):
        local_features, _ = self._feature_generator.transform(targets)
        return local_features

    def load(self, path):
        path = os.path.join(DATA_BASE_PATH, path)
        with open(self._input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self._net = AuncelNet(self._input_feature_dim)
        if CUDA:
            self._net.load_state_dict(torch.load(self._nn_path(path)))
            self._net = self._net.cuda(device)
            self._net = torch.nn.DataParallel(
                self._net, device_ids=GPU_LIST)
        else:
            self._net.load_state_dict(torch.load(
                self._nn_path(path), map_location=torch.device('cpu')))
        self._net.eval()
        with open(self._feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        path = os.path.join(DATA_BASE_PATH, path)
        os.makedirs(path, exist_ok=True)

        if CUDA:
            torch.save(self._net.module.state_dict(), self._nn_path(path))
        else:
            torch.save(self._net.state_dict(), self._nn_path(path))

        with open(self._feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(self._input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, X, Y, pre_training=False):
        if isinstance(Y, list):
            Y = np.array(Y)
            Y = Y.reshape(-1, 1)

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        pairs = []
        for i in range(len(Y)):
            pairs.append((X[i], Y[i]))
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=self.collate_fn)

        if not pre_training:
            # # determine the initial number of channels
            input_feature_dim = len(X[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = AuncelNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())

        loss_fn = torch.nn.MSELoss()
        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda(device)

                tree = None
                if CUDA:
                    tree = self._net.module.build_trees(x)
                else:
                    tree = self._net.build_trees(x)

                y_pred = self._net(tree)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()

                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)

    def collate_fn(self, x):
        trees = []
        targets = []

        for tree, target in x:
            trees.append(tree)
            targets.append(target)

        targets = torch.tensor(targets)
        return trees, targets

    def predict(self, x):
        if not isinstance(x, list):
            x = [x]

        tree = None
        # todo
        if CUDA:
            tree = self._net.module.build_trees(x)
        else:
            tree = self._net.build_trees(x)

        pred = self._net(tree).cpu().detach().numpy()

        pred = self.inverse_norm_no_log("Execution Time", pred)
        return pred

    def predict_confidence(self, x):
        if not isinstance(x, list):
            x = [x]

        tree = None
        # todo
        if CUDA:
            tree = self._net.module.build_trees(x)
        else:
            tree = self._net.build_trees(x)

        pred = self._net(tree).cpu().detach().numpy()

        return pred

    @classmethod
    def _nn_path(cls, base):
        return os.path.join(base, "nn_weights")

    def _feature_generator_path(self, base):
        return os.path.join(base, "feature_generator")

    def _input_feature_dim_path(self, base):
        return os.path.join(base, "input_feature_dim")


class AuncelModelPairWise(AuncelModel):
    def __init__(self, feature_generator) -> None:
        super().__init__(feature_generator)

    # def fit(self, X1, X2, Y1, Y2, pre_training=False):
    #     assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
    #     if isinstance(Y1, list):
    #         Y1 = np.array(Y1)
    #         Y1 = Y1.reshape(-1, 1)
    #     if isinstance(Y2, list):
    #         Y2 = np.array(Y2)
    #         Y2 = Y2.reshape(-1, 1)
    #
    #     # # determine the initial number of channels
    #     if not pre_training:
    #         input_feature_dim = len(X1[0].get_feature())
    #         print("input_feature_dim:", input_feature_dim)
    #
    #         self._net = AuncelNet(input_feature_dim)
    #         self._input_feature_dim = input_feature_dim
    #         if CUDA:
    #             self._net = self._net.cuda(device)
    #             self._net = torch.nn.DataParallel(
    #                 self._net, device_ids=GPU_LIST)
    #             self._net.cuda(device)
    #
    #     pairs = []
    #     for i in range(len(X1)):
    #         pairs.append((X1[i], X2[i], Y1[i], Y2[i], 1.0 if Y1[i] >= Y2[i] else 0.0))
    #
    #     batch_size = 64
    #     if CUDA:
    #         batch_size = batch_size * len(GPU_LIST)
    #
    #     dataset = DataLoader(pairs,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=collate_pairwise_fn)
    #
    #     optimizer = None
    #     if CUDA:
    #         optimizer = torch.optim.Adam(self._net.module.parameters())
    #         optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
    #     else:
    #         optimizer = torch.optim.Adam(self._net.parameters())
    #
    #     # bce_loss_fn = torch.nn.BCELoss()
    #     bce_loss_fn = Weight_BCE_Loss()
    #
    #     losses = []
    #     sigmoid = nn.Sigmoid()
    #     start_time = time()
    #     for epoch in range(100):
    #         loss_accum = 0
    #         for x1, x2, label, y1, y2 in dataset:
    #             tree_x1, tree_x2 = None, None
    #             if CUDA:
    #                 tree_x1 = self._net.module.build_trees(x1)
    #                 tree_x2 = self._net.module.build_trees(x2)
    #             else:
    #                 tree_x1 = self._net.build_trees(x1)
    #                 tree_x2 = self._net.build_trees(x2)
    #
    #             # pairwise
    #             y_pred_1 = self._net(tree_x1)
    #             y_pred_2 = self._net(tree_x2)
    #             weight_by_time = np.abs(np.array(y1) - np.array(y2))
    #             diff = y_pred_1 - y_pred_2
    #             prob_y = sigmoid(diff)
    #
    #             label_y = torch.tensor(np.array(label).reshape(-1, 1))
    #             if CUDA:
    #                 label_y = label_y.cuda(device)
    #
    #             loss = bce_loss_fn(weight_by_time, prob_y, label_y)
    #             # loss = bce_loss_fn(prob_y, label_y)
    #             loss_accum += loss.item()
    #
    #             if CUDA:
    #                 optimizer.module.zero_grad()
    #                 loss.backward()
    #                 optimizer.module.step()
    #             else:
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #
    #         loss_accum /= len(dataset)
    #         losses.append(loss_accum)
    #
    #         print("Epoch", epoch, "training loss:", loss_accum)
    #     print("training time:", time() - start_time, "batch size:", batch_size)

    def fit(self, X1, X2, Y1, Y2, pre_training=False):
        writer = SummaryWriter(LOG_BASE_PATH + "/{}".format(TIMESTAMP), flush_secs=1)
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # # determine the initial number of channels
        if not pre_training:
            input_feature_dim = len(X1[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = self.get_auncel_net(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            if CUDA:
                self._net = self._net.cuda(device)
                self._net = torch.nn.DataParallel(
                    self._net, device_ids=GPU_LIST)
                self._net.cuda(device)

        pairs = []
        for i in range(len(X1)):
            pairs.append((X1[i], X2[i], Y1[i], Y2[i], 1.0 if Y1[i] >= Y2[i] else 0.0))

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=self.collate_pairwise_fn)

        optimizer = None
        if CUDA:
            optimizer = torch.optim.Adam(self._net.module.parameters())
            optimizer = nn.DataParallel(optimizer, device_ids=GPU_LIST)
        else:
            optimizer = torch.optim.Adam(self._net.parameters())

        # mse_loss_fn = torch.nn.MSELoss()
        # bce_loss_fn = Weight_BCE_Loss()
        loss_fn = self.get_loss()
        # prob_loss_fn = ProbLoss()

        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            second_loss_accum = 0

            for x1, x2, label, y1, y2 in dataset:
                tree_x1, tree_x2 = None, None
                if CUDA:
                    tree_x1 = self._net.module.build_trees(x1)
                    tree_x2 = self._net.module.build_trees(x2)
                else:
                    tree_x1 = self._net.build_trees(x1)
                    tree_x2 = self._net.build_trees(x2)

                # pairwise
                y_pred_1 = self._net(tree_x1)
                y_pred_2 = self._net(tree_x2)

                loss = self.cal_loss(loss_fn, y_pred_1, y_pred_2, label)

                loss_accum += loss.item()
                second_loss_accum += self.cal_second_loss(y_pred_1, y_pred_2, label)
                if CUDA:
                    optimizer.module.zero_grad()
                    loss.backward()
                    optimizer.module.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            loss_accum /= len(dataset)
            second_loss_accum /= len(dataset)
            losses.append(loss_accum)
            writer.add_scalar("loss_accum", loss_accum, global_step=epoch)
            print("Epoch:{}, training loss:{}, second_loss:{}".format(epoch, loss_accum, second_loss_accum))
        print("training time:", time() - start_time, "batch size:", batch_size)
        writer.close()

    def get_auncel_net(self, input_feature_dim):
        return AuncelNet(input_feature_dim)

    def get_loss(self):
        return torch.nn.BCELoss()

    def cal_loss(self, loss_fn, predict_y1, predict_y2, label_y):
        sigmoid = nn.Sigmoid()
        prob_y = sigmoid(predict_y1 - predict_y2)

        label_y = torch.tensor(np.array(label_y).reshape(-1, 1))
        if CUDA:
            label_y = label_y.cuda(device)

        return loss_fn(prob_y, label_y)

    def cal_second_loss(self, predict_y1, predict_y2, label_y):
        sigmoid = nn.Sigmoid()
        prob_y = sigmoid(predict_y1 - predict_y2)
        y = [1 if f > 0.5 else 0 for f in prob_y.detach().cpu()]
        count = np.sum(np.array(y).reshape(-1, 1) == np.array(label_y).reshape(-1, 1))
        return count / len(y)

    def collate_pairwise_fn(self, x):
        trees1 = []
        trees2 = []
        labels = []
        run_times1 = []
        run_times2 = []

        for tree1, tree2, run_time1, run_time2, label in x:
            trees1.append(tree1)
            trees2.append(tree2)
            labels.append(label)
            run_times1.append(run_time1)
            run_times2.append(run_time2)
        return trees1, trees2, labels, run_times1, run_times2


class AuncelModelPairConfidenceWise(AuncelModelPairWise):
    def __init__(self, feature_generator, diff_thres) -> None:
        super().__init__(feature_generator)
        self.diff_thres = diff_thres

    # def get_auncel_net(self, input_feature_dim):
    #     # return AuncelNetDistance(input_feature_dim)
    #     return AuncelNet(input_feature_dim)
    #
    # def get_loss(self):
    #     return nn.BCELoss
    #     return ContrastiveLoss(self.diff_thres)
    #     # return FakeContrastiveLoss(self.diff_thres)
    #
    # def cal_loss(self, loss_fn, predict_y1, predict_y2, label_y):
    #     label_y = torch.tensor(np.array(label_y).reshape(-1, 1))
    #     if CUDA:
    #         label_y = label_y.cuda(device)
    #     loss = loss_fn(predict_y1, predict_y2, label_y)
    #     return loss
    #
    # def cal_second_loss(self, predict_y1, predict_y2, label_y):
    #     dist = torch.nn.functional.pairwise_distance(predict_y1, predict_y2)
    #     y = dist < self.diff_thres
    #     y = [1 if f else 0 for f in y.detach().cpu()]
    #     count = np.sum(np.array(y).reshape(-1, 1) == np.array(label_y).reshape(-1, 1))
    #     return count / len(y)

    def collate_pairwise_fn(self, x):
        trees1 = []
        trees2 = []
        labels = []
        accuracies1 = []
        accuracies2 = []

        for tree1, tree2, accuracy1, accuracy2, _ in x:
            trees1.append(tree1)
            trees2.append(tree2)
            accuracies1.append(accuracy1)
            accuracies2.append(accuracy2)
            if abs(accuracy1 - accuracy2) <= self.diff_thres:
                labels.append(1.0)
            else:
                labels.append(0.0)
        return trees1, trees2, labels, accuracies1, accuracies2

    def predict_pair(self, x1, x2):
        if not isinstance(x1, list):
            x1 = [x1]
        if not isinstance(x2, list):
            x2 = [x2]

        # todo
        if CUDA:
            tree1 = self._net.module.build_trees(x1)
            tree2 = self._net.module.build_trees(x2)
        else:
            tree1 = self._net.build_trees(x1)
            tree2 = self._net.build_trees(x2)
        y_pred_1 = self._net(tree1)
        y_pred_2 = self._net(tree2)
        sigmoid = nn.Sigmoid()
        prob_y = sigmoid(y_pred_1 - y_pred_2)
        return [1 if v.item() > 0.5 else 0 for v in prob_y.detach().cpu().numpy()]

    def is_same_buckets(self, x1, x2):
        sigmoid = nn.Sigmoid()
        prob_y = sigmoid(torch.tensor(x1) - torch.tensor(x2))
        assert len(x1) == 1
        return 1 if prob_y.item() >= 0.5 else 0


class Weight_BCE_Loss(nn.BCELoss):
    def __init__(self):
        super().__init__()

    def forward(self, weight, predict_y: tensor, label_y):
        weight = torch.tensor(weight).cuda(predict_y.device)
        return binary_cross_entropy(predict_y, label_y, weight=weight, reduction=self.reduction)


class ProbLoss(nn.Module):
    def forward(self, weight, predict_y: tensor, label_y):
        weight = torch.tensor(weight).cuda(predict_y.device)
        loss = label_y * (weight * (1 - predict_y)) + (1 - label_y) * (predict_y * weight)
        return loss.mean()
        # weight=torch.tensor(weight).cuda(predict_y.device)
        # return binary_cross_entropy(predict_y, label_y, weight=weight, reduction=self.reduction)


class FakeContrastiveLoss(nn.Module):
    """
    # y*(a-b)^2+(1-y)(max(diff^2-(a-b)^2,0)^2)
    """

    def __init__(self, diff_thres):
        super().__init__()
        self.diff_thres = diff_thres

    def forward(self, predict_y1: tensor, predict_y2: tensor, label_y):
        dist = torch.pow((predict_y1 - predict_y2), 2)
        abs_dist = torch.abs(predict_y1 - predict_y2)
        # margin=torch.pow(self.diff_thres,2)

        e1 = label_y * dist
        e2 = (1 - label_y) * torch.pow(torch.clamp(self.diff_thres - abs_dist, min=0.0), 2)
        loss = e1 + e2
        return loss.mean()
        # weight=torch.tensor(weight).cuda(predict_y.device)
        # return binary_cross_entropy(predict_y, label_y, weight=weight, reduction=self.reduction)


class ContrastiveLoss(nn.Module):
    """
    # y*L2(a,b)+(1-y)(max(diff-abs(a-b),0)^2)
    """

    def __init__(self, diff_thres):
        super().__init__()
        self.diff_thres = diff_thres

    # def forward(self, predict_y1: tensor, predict_y2: tensor, label_y):
    #     dist = torch.nn.functional.pairwise_distance(predict_y1, predict_y2)
    #     e1 = label_y * torch.pow(dist, 2)
    #     e2 = (1 - label_y) * torch.pow(torch.clamp(self.diff_thres - dist, min=0.0), 2)
    #     loss = e1 + e2
    #     return loss.mean()

    def forward(self, predict_y1: tensor, predict_y2: tensor, label_y):
        dist = torch.nn.functional.pairwise_distance(predict_y1, predict_y2)
        e1 = label_y * torch.pow(torch.clamp(dist - self.diff_thres, min=0.0), 2)
        e2 = (1 - label_y) * torch.pow(torch.clamp(self.diff_thres - dist, min=0.0), 2)
        loss = e1 + e2
        return loss.mean()
