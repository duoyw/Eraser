import torch
# from datasets import load_data
from perfguard import PerfGuard
from get_data import *
import os
import perfguardConfig

config = PerfguardConfig.Config()


def train():
    """
    get training data
    """
    get_data_ = Get_Dataset(config.train_plan_path)
    features1, features2 = get_data_.get_features()
    label = get_data_.get_labels()
    adjaceny_matrix_list_x1, adjaceny_matrix_list_x2 = get_data_.get_two_adjaceny_matrix()

    """
    model training
    """
    model = PerfGuard(features1.shape[2], config.embd_dim, config.tensor_dim, config.dropout).cuda(config.device)
    model = torch.nn.DataParallel(model, device_ids=config.GPU_LIST)

    optimizer = torch.optim.Adam(model.parameters(), config.init_lr)
    # optimizer = torch.nn.DataParallel(optimizer, device_ids=config.GPU_LIST)

    Loss = torch.nn.BCELoss()
    model.train()
    for epoch in range(config.epochs):
        final_output = model(adjaceny_matrix_list_x1, adjaceny_matrix_list_x2, features1, features2)
        """
        gcn的输入：MxD维的特征矩阵features和MxM维的邻接矩阵adj（M是节点数，D是每个节点的输入特征数）
        gcn的输出：MxF维的特征矩阵outputs
        """

        loss = Loss(final_output, torch.tensor(label).float().cuda(config.device))
        print("Epoch {}, loss {}".format(epoch + 1, loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model_path = 'model_pth/' + config.data + '_' + str(config.data_num)
    # if os.path.exists(model_path):
    #     os.removedirs(model_path)
    # 保存模型

    get_data_.save(model, model_path, features1.shape[2])
    # torch.save(model.state_dict(), 'model_pth/'+config.data+str(config.data_num)+'.pth')


if __name__ == '__main__':
    train()
