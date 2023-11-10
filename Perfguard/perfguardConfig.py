import torch


class Config():
    def __init__(self):
        # cuda
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu")
        self.GPU_LIST = [0]
        # self.GPU_LIST = [0,1,2,3,4,5,6,7],need batch_size * 8
        self.SEP = "#####"
        # parameter
        self.init_lr = 0.1
        self.epochs = 500
        # the output dim of gcn
        self.embd_dim = 30
        # the out put dim of ntn
        self.tensor_dim = 10
        self.dropout = 0.

        # it is set in load_perfguard_model by database
        self.threshold = 0.6

        # data path
        self.data = 'tpch'
        self.data_num = 2
        self.train_plan_path = '../result/' + self.data + '/' + str(
            self.data_num) + '/lero_' + self.data + '_.log.training'

        self.lero_plan_path = '../result/' + self.data + '/' + str(
            self.data_num) + '/lero_' + self.data + '_.log_' + self.data + '_test_model_on_' + str(self.data_num) + '_0'
        self.pg_plan_path = '../result/' + self.data + '/' + str(self.data_num) + '/pg_' + self.data + '_test_.log'
        self.perfguard_path = '../result/' + self.data + '/' + str(
            self.data_num) + '/perfguard_' + self.data + '_test_.log'
