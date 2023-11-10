import os.path
import unittest

import numpy as np

from auncel.model import AuncelModel
from auncel.model_config import ModelType, model_type
from auncel.model_transformer import AuncelModelTransformerPairWise
from auncel.test_script.config import DATA_BASE_PATH
from auncel.train import train
from auncel.utils import get_plans_with_accuracy_file_path, get_confidence_model_name, get_group_plans_file_path
from model_config import confidence_model_type


class MyTestCase(unittest.TestCase):
    training_type = 6

    def test_1(self):
        a=np.array([1,2,3,4,5])
        b=np.array([1,2,3,4,5])
        c=np.array([1,4,3,4,5])
        print("a==b is{},a=c is{}".format((a==b).all(),(a==c).all()))

    def test_stats(self):
        # train_dataset_name = "stats10NodeTrainDatasetQ1-500_0929"
        # train_dataset_name = "stats10NodeTrainDatasetQ1-1000_1002"
        # train_dataset_name = "stats10NodeTestDataset_0928"
        # train_dataset_name = "stats10Q1000_train0911_wo508"
        # train_dataset_name = "stats10Q146_test0910_wo136"
        # train_dataset_name = "stats10Q10"
        train_dataset_name = "stats10Q50"
        self.train(train_dataset_name, "stats")

    def test_tpcds(self):
        # train_dataset_name = "tpcdsQuery50_100_train_0914"
        # train_dataset_name = "tpcdsQuery40_100_train_0914"
        # train_dataset_name = "tpcdsQuery30_100_train_0914"
        # train_dataset_name = "tpcdsQuery20_100_train_0914"
        train_dataset_name = "tpcdsQuery10_100_train_0914"
        # train_dataset_name = "tpcdsQuery10_20_train_0914"
        # train_dataset_name = "tpcdsQuery1_10_train"
        # train_dataset_name = "tpcdsQuery10_20"
        self.train(train_dataset_name, "tpcds")

    def test_tpcds_by_leaf_groups(self):
        # train_dataset_name = "tpcdsQuery50_100_train_0914"
        # train_dataset_name = "tpcdsQuery40_100_train_0914"
        # train_dataset_name = "tpcdsQuery30_100_train_0914"
        # train_dataset_name = "tpcdsQuery20_100_train_0914"
        # train_dataset_name = "tpcdsQuery10_100_train_0914"
        train_dataset_name = "tpcdsQuery10_20_train_0914"
        # train_dataset_name = "tpcdsQuery10_20"
        self.train(train_dataset_name, "tpcds", True)

    def test_tpcds_loop(self):
        dataset_names = ["tpcdsQuery50_100_train_0914", "tpcdsQuery40_100_train_0914", "tpcdsQuery30_100_train_0914",
                         "tpcdsQuery20_100_train_0914", "tpcdsQuery10_100_train_0914"]
        # dataset_names = ["tpcdsQuery30_100_train_0914",
        #                  "tpcdsQuery20_100_train_0914", "tpcdsQuery10_100_train_0914"]
        for train_dataset_name in dataset_names:
            self.train(train_dataset_name, "tpcds")

    def train(self, train_set_name, dataset_name, is_cross_plan_by_group=False):
        train_data_path = get_plans_with_accuracy_file_path(train_set_name)
        if confidence_model_type == ModelType.TREE_CONV:
            if is_cross_plan_by_group:
                self.training_type = 7
                train_data_path = get_group_plans_file_path(train_set_name, model_type)
            else:
                self.training_type = 6
        elif confidence_model_type == ModelType.MSE_TREE_CONV:
            self.training_type = 8
        else:
            raise RuntimeError

        train(train_data_path, dataset_name=dataset_name, model_name=self.get_model_name(train_set_name),
              training_type=self.training_type)

    def get_data_path(self, file_name):
        return os.path.join(DATA_BASE_PATH, file_name)

    def get_model_name(self, dataset_name):
        return get_confidence_model_name(dataset_name, confidence_model_type.name.lower())

    def get_model(self):
        if model_type == ModelType.TRANSFORMER:
            return AuncelModelTransformerPairWise(None, None, None, None)
        elif model_type == ModelType.TREE_CONV:
            return AuncelModel(None)
        elif model_type == ModelType.MSE_TREE_CONV:
            return AuncelModel(None)
        else:
            raise RuntimeError

    if __name__ == '__main__':
        unittest.main()
