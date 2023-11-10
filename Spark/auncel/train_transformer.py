import os.path

from pandas import DataFrame

from auncel.HistogramManager import HistogramManager
from auncel.QueryFormer.model.database_util import Encoding
from auncel.QueryFormer.model.dataset import PlanTreeDataset
from auncel.model_config import TransformerConfig
from auncel.model_transformer import AuncelModelTransformerPairWise
from auncel.sparkFeature import SparkFeatureGenerator, OP_TYPES, JOIN_TYPES
import pandas as pd

from auncel.test_script.config import DATA_BASE_PATH
from auncel.utils import _load_pairwise_plans, join_key_identifier, json_str_to_json_obj

config = TransformerConfig


def build_encoding(histogram, feature_generator: SparkFeatureGenerator):
    tables = ["NA"] + feature_generator.get_tables()
    # tables = feature_generator.get_tables()

    join_keys = [join_key_identifier(None, None)] + [join_key_identifier(keys[0], keys[1]) for keys in
                                                     feature_generator.get_join_keys()]

    # join_keys = [join_key_identifier(keys[0], keys[1]) for keys in
    #              feature_generator.get_join_keys()] + [join_key_identifier(None, None)]


    node_types = OP_TYPES

    column_min_max_vals = feature_generator.get_col_min_max_vals()
    cols = ["NA"] + list(column_min_max_vals.keys())
    cols.sort()
    col2idx = dict(zip(cols, range(len(cols))))

    return Encoding(node_types=node_types, join_keys=join_keys, tables=tables,
                    column_min_max_vals=column_min_max_vals, col2idx=col2idx)


def build_norm(feature_generator):
    return feature_generator.normalizer


def train_with_transformer_pairwise(tuning_model_path, model_name, training_data_file, dataset_name,
                                    pretrain=False):
    plans1, plans2 = _load_pairwise_plans(training_data_file)

    histogram_file_path = os.path.join(DATA_BASE_PATH, "{}_hist.txt".format(dataset_name))

    # statistic plan info
    feature_generator = SparkFeatureGenerator(dataset_name)
    feature_generator.fit(plans1 + plans2)

    plan_df_1 = DataFrame({"id": list(range(0, len(plans1))), "json": plans1})
    plan_df_2 = DataFrame({"id": list(range(0, len(plans2))), "json": plans2})

    histogram = HistogramManager(histogram_file_path)
    # draw_dot_spark_plan(json_str_to_json_obj(plans1[0])["Plan"])

    encoding = build_encoding(histogram, feature_generator)
    normalizer = build_norm(feature_generator)

    table_sample = None

    dataset1 = PlanTreeDataset(plan_df_1, encoding, histogram, normalizer, table_sample, dataset_name)

    dataset2 = PlanTreeDataset(plan_df_2, encoding, histogram, normalizer, table_sample, dataset_name)

    auncel_model = AuncelModelTransformerPairWise(encoding, histogram, normalizer, table_sample)
    auncel_model.fit(dataset1, dataset2)

    auncel_model.save(model_name)

    # evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'])
