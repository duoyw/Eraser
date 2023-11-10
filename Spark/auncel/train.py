import argparse
import math

from Common.LoadData import _load_accuracy_pairwise_plans_cross_plan_with_filter
from feature import *
from model import AuncelModel, AuncelModelPairWise, AuncelModelPairConfidenceWise
from model_config import confidence_model_accuracy_diff_thres
from sparkFeature import SparkFeatureGenerator
from train_transformer import train_with_transformer_pairwise
from utils import _load_pairwise_plans, _load_pointwise_plans, _load_accuracy_pairwise_plans, \
    _load_accuracy_pointwise_plans, \
    _load_accuracy_pairwise_plans_cross_plan


def compute_rank_score(path, pretrain=False, rank_score_type=0):
    X, Y = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.split("#####")
            if pretrain:
                arr = [(json.loads(p)[0]['Plan']['Total Cost'], p)
                       for p in arr]
            else:
                arr = [(json.loads(p)[0]['Execution Time'], p) for p in arr]
            sorted_arr = sorted(arr, key=lambda x: x[0])

            for i in range(len(sorted_arr)):
                X.append(sorted_arr[i][1])
                if rank_score_type == 0:
                    # 1. x^2
                    print("X^2")
                    Y.append(float((i + 1) ** 2))
                elif rank_score_type == 1:
                    # 2. x^4
                    print("X^4")
                    Y.append(float((i + 1) ** 4))
                elif rank_score_type == 2:
                    # 3. e^x
                    print("e^X")
                    Y.append(float(math.exp(i + 1)))
                elif rank_score_type == 3:
                    # 3. x^1
                    print("X^1")
                    Y.append(float((i + 1)))
    return X, Y


def training_pairwise(tuning_model_path, model_name, training_data_file, pretrain=False, dataset_name=None,
                      data_limit_ratio=None):
    X1, X2 = _load_pairwise_plans(training_data_file, data_limit_ratio)

    tuning_model = tuning_model_path is not None
    auncel_model = None

    if tuning_model:
        auncel_model = AuncelModelPairWise(None)
        auncel_model.load(tuning_model_path)
        feature_generator = auncel_model._feature_generator
    else:
        feature_generator = SparkFeatureGenerator(dataset_name)
        feature_generator.fit(X1 + X2)

    Y1, Y2 = None, None

    if pretrain:
        Y1 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X1]
        Y2 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X2]
        X1, _ = feature_generator.transform(X1)
        X2, _ = feature_generator.transform(X2)
    else:
        X1, Y1 = feature_generator.transform(X1)
        X2, Y2 = feature_generator.transform(X2)
    print("Training data set size = " + str(len(X1)))

    if not tuning_model:
        assert auncel_model == None
        auncel_model = AuncelModelPairWise(feature_generator)
    auncel_model.fit(X1, X2, Y1, Y2, tuning_model)

    print("saving model...")
    auncel_model.save(model_name)


def training_pairwise_accuracy(tuning_model_path, model_name, training_data_file, pretrain=False, dataset_name=None,
                               is_cross_plan_by_group=False):
    """
    distinguish whether two plans are similar in accuracy
    :param tuning_model_path:
    :param model_name:
    :param training_data_file:
    :param pretrain:
    :param dataset_name:
    :return:
    """
    if is_cross_plan_by_group:
        # plans1, plans2 = _load_accuracy_pairwise_plans_cross_plan(training_data_file, 5)
        plans1, plans2 = _load_accuracy_pairwise_plans_cross_plan_with_filter(training_data_file, 2)
    else:
        plans1, plans2 = _load_accuracy_pairwise_plans(training_data_file)
    print("Training data set size = " + str(len(plans1)))

    tuning_model = tuning_model_path is not None
    auncel_model = None

    feature_generator = SparkFeatureGenerator(dataset_name)
    feature_generator.fit(plans1 + plans2)

    X1, _ = feature_generator.transform(plans1)
    Y1 = feature_generator.transform_confidence_y(plans1)

    X2, _ = feature_generator.transform(plans2)
    Y2 = feature_generator.transform_confidence_y(plans2)
    # X1, Y1, X2, Y2 = filter_by_same_accuracy(X1, Y1, X2, Y2)
    # X1, Y1, X2, Y2 = filter_same_x_different_y(X1, Y1, X2, Y2)

    print("Filter Training data set size = " + str(len(X1)))
    if not tuning_model:
        assert auncel_model is None
        auncel_model = AuncelModelPairConfidenceWise(feature_generator, confidence_model_accuracy_diff_thres)
    auncel_model.fit(X1, X2, Y1, Y2, tuning_model)

    print("saving model...")
    auncel_model.save(model_name)


def filter_by_same_accuracy(X1, Y1, X2, Y2):
    v = zip(X1, Y1, X2, Y2)
    v = list(filter(lambda x: x[1] != x[3], v))
    X1, Y1, X2, Y2 = zip(*v)
    return X1, Y1, X2, Y2


def is_different_entity(x):
    return not (str(x[0]) == str(x[2]))


def filter_same_x_different_y(X1, Y1, X2, Y2):
    origin_size = len(X1)
    v = zip(X1, Y1, X2, Y2)
    res = []

    for i, a in enumerate(list(v)):
        if is_different_entity(a):
            res.append(v)
    # v = list(filter(is_different_entity, v))
    assert len(list(v)) > 0
    X1, Y1, X2, Y2 = zip(*v)
    print("filter_same_x_different_y ratio is {}".format(1 - len(X1) / float(origin_size)))
    return X1, Y1, X2, Y2


def training_pointwise_accuracy(tuning_model_path, model_name, training_data_file, pretrain=False, dataset_name=None):
    """
    use model to predict confidence
    :param tuning_model_path:
    :param model_name:
    :param training_data_file:
    :param pretrain:
    :param dataset_name:
    :return:
    """
    plans = _load_accuracy_pointwise_plans(training_data_file)

    tuning_model = tuning_model_path is not None
    auncel_model = None

    feature_generator = SparkFeatureGenerator(dataset_name)
    feature_generator.fit(plans)

    X, _ = feature_generator.transform(plans)
    Y = feature_generator.transform_confidence_y(plans)

    print("Training data set size = " + str(len(X)))

    if not tuning_model:
        assert auncel_model is None
        auncel_model = AuncelModel(feature_generator)
    auncel_model.fit(X, Y, tuning_model)

    print("saving model...")
    auncel_model.save(model_name)


def training_with_rank_score(tuning_model_path, model_name, training_data_file, pretrain=False, rank_score_type=0):
    X, Y = compute_rank_score(training_data_file, pretrain, rank_score_type)

    tuning_model = tuning_model_path is not None
    auncel_model = None
    if tuning_model:
        auncel_model = AuncelModel(None)
        auncel_model.load(tuning_model_path)
        feature_generator = auncel_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X)

    # replace lantency with rank score
    local_features, _ = feature_generator.transform(X)
    assert len(local_features) == len(Y)
    print("Training data set size = " + str(len(local_features)))

    if not tuning_model:
        assert auncel_model == None
        auncel_model = AuncelModel(feature_generator)

    auncel_model.fit(local_features, Y, tuning_model)

    print("saving model...")
    auncel_model.save(model_name)


def training_pointwise(tuning_model_path, model_name, training_data_file, dataset_name=None):
    X = _load_pointwise_plans(training_data_file)

    tuning_model = tuning_model_path is not None
    auncel_model = None
    if tuning_model:
        auncel_model = AuncelModel(None)
        auncel_model.load(tuning_model_path)
        feature_generator = auncel_model._feature_generator
    else:
        feature_generator = SparkFeatureGenerator(dataset_name)
        feature_generator.fit(X)

    local_features, y = feature_generator.transform(X)
    assert len(local_features) == len(y)
    print("Training data set size = " + str(len(local_features)))

    if not tuning_model:
        assert auncel_model == None
        auncel_model = AuncelModel(feature_generator)

    auncel_model.fit(local_features, y, tuning_model)

    print("saving model...")
    auncel_model.save(model_name)


def train(training_data, dataset_name=None, training_type=0, model_name=None, pretrain_model_name=None,
          rank_score_training_type=0, data_limit_ratio=None):
    print("training_type:", training_type)

    print("training_data:", training_data)

    print("model_name:", model_name)

    print("pretrain_model_name:", pretrain_model_name)

    # print("rank_score_training_type:", rank_score_training_type)

    if training_type == 0:
        print("training_pointwise")
        training_pointwise(pretrain_model_name, model_name, training_data, dataset_name)
    elif training_type == 1:
        print("training_pairwise")
        training_pairwise(pretrain_model_name, model_name,
                          training_data, False, dataset_name, data_limit_ratio=data_limit_ratio)
    elif training_type == 2:
        print("training_with_rank_score")
        training_with_rank_score(
            pretrain_model_name, model_name, training_data, False, rank_score_training_type)
    elif training_type == 3:
        print("pre-training_pairwise")
        training_pairwise(pretrain_model_name, model_name,
                          training_data, True)
    elif training_type == 4:
        print("pre-training_with_rank_score")
        training_with_rank_score(
            pretrain_model_name, model_name, training_data, True, rank_score_training_type)
    elif training_type == 5:
        print("pre-train_with_transformer_pairwise")

        train_with_transformer_pairwise(
            pretrain_model_name, model_name, training_data, dataset_name, False)
    elif training_type == 6:
        print("training_pairwise_accuracy")
        training_pairwise_accuracy(
            pretrain_model_name, model_name, training_data, False, dataset_name)
    elif training_type == 7:
        print("training_pairwise_accuracy_cross_plan_by_group")
        training_pairwise_accuracy(
            pretrain_model_name, model_name, training_data, False, dataset_name, is_cross_plan_by_group=True)
    elif training_type == 8:
        print("training_pointwise_accuracy")
        training_pointwise_accuracy(
            pretrain_model_name, model_name, training_data, False, dataset_name)
    else:
        raise Exception()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--training_data",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--training_type", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrain_model_name", type=str)
    parser.add_argument("--rank_score_training_type", type=int)

    args = parser.parse_args()

    training_type = 0
    if args.training_type is not None:
        training_type = args.training_type
    print("training_type:", training_type)

    training_data = None
    if args.training_data is not None:
        training_data = args.training_data
    print("training_data:", training_data)

    model_name = None
    if args.model_name is not None:
        model_name = args.model_name
    print("model_name:", model_name)

    pretrain_model_name = None
    if args.pretrain_model_name is not None:
        pretrain_model_name = args.pretrain_model_name
    print("pretrain_model_name:", pretrain_model_name)

    rank_score_training_type = 0
    if args.rank_score_training_type is not None:
        rank_score_training_type = args.rank_score_training_type
    print("rank_score_training_type:", rank_score_training_type)

    train(training_data, training_type, model_name, pretrain_model_name, rank_score_training_type)
