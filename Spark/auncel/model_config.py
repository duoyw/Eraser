# IGNORE_TYPES = ["org.apache.spark.sql.execution.ProjectExec", "org.apache.spark.sql.execution.FilterExec"]
import os
from datetime import datetime
from enum import Enum

from Common.PlanConfig import SparkNodeConfig
from Spark.auncel.test_script.config import DATA_BASE_PATH

# from auncel.test_script.config import DATA_BASE_PATH

IGNORE_TYPES = ["org.apache.spark.sql.execution.ProjectExec"]
FILTER_TYPES = ["org.apache.spark.sql.execution.FilterExec"]

OPERATOR_TYPE = {
    "org.apache.spark.sql.catalyst.expressions.EqualTo": "=",
    "org.apache.spark.sql.catalyst.expressions.LessThan": "<",
    "org.apache.spark.sql.catalyst.expressions.LessThanOrEqual": "<",
    "org.apache.spark.sql.catalyst.expressions.GreaterThan": ">",
    "org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual": ">"
}

ALIAS_TO_TABLE = {
    "stats": {
        "u": "users",
        "p": "posts",
        "pl": "postLinks",
        "ph": "postHistory",
        "c": "comments",
        "v": "votes",
        "b": "badges",
        "t": "tags"
    },
    "tpcds": {
        "u": "users",
        "p": "posts",
        "pl": "postLinks",
        "ph": "postHistory",
        "c": "comments",
        "v": "votes",
        "b": "badges",
        "t": "tags"
    }
}

LITERAL_TYPE = "org.apache.spark.sql.catalyst.expressions.Literal"
IN_TYPE = "org.apache.spark.sql.catalyst.expressions.In"
INSET_TYPE = 'org.apache.spark.sql.catalyst.expressions.InSet'
ATTRIBUTE_REFERENCE_TYPE = "org.apache.spark.sql.catalyst.expressions.AttributeReference"

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

db_node_config = SparkNodeConfig


class ModelType(Enum):
    TREE_CONV = 0,
    TRANSFORMER = 1
    MSE_TREE_CONV = 2

    def __eq__(self, *args, **kwargs):
        return self.name == args[0].name and self.value == args[0].value


class ConfidenceEstimateType(Enum):
    STATIC = 0,
    ADAPTIVE = 1,
    SINGLE_MODEL = 2,
    ADAPTIVE_MODEL = 3,
    SimilarModelEstimate = 4,


class Feature_Type(Enum):
    COMMON = 0,
    COMPLETE = 1


class DbType(Enum):
    SPARK = 0

    def __eq__(self, *args, **kwargs):
        return self.name == args[0].name


# the model that is used to choose best plan
model_type = ModelType.TREE_CONV

# will change in test
feature_type = Feature_Type.COMPLETE

# the method that is used to estimate confidence
confidence_estimate_type = ConfidenceEstimateType.ADAPTIVE

# for ConfidenceEstimateType.ADAPTIVE, the model == model_type, but for SINGLE_MODEL,  the model is MSE_TREE_CONV
confidence_model_type = ModelType.MSE_TREE_CONV

db_type = DbType.SPARK

uncertainty_threshold = 0.8
enable_uncertainty = True
# whether to differentiate join type, if true: hash join,merge join,... if false: join
diff_join_type = True

max_predicate_num = 5
is_predict = True

CACHE_FILE_PATH = os.path.join(DATA_BASE_PATH, "cache")

confidence_model_accuracy_diff_thres = 0.1

max_bias_ratio = 2.0
valid_bias_ratio = 1.5


def set_predict_status(status: bool):
    global is_predict
    is_predict = status


class GroupEnable:
    struct_enable = True
    scan_type_enable = False
    table_name_enable = False
    join_type_enable = False
    join_key_enable = False
    filter_enable = False
    filter_col_enable = False
    filter_op_enable = False
    filter_value_enable = False


class TransformerConfig:
    use_hist = True
    use_one_hot = False
    max_predicate_num = max_predicate_num
