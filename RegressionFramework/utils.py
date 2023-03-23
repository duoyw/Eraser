import json

from pandas import DataFrame
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from test_script.config import SEP


def json_str_to_json_obj(json_data):
    if isinstance(json_data, str):
        origin = json_data
        json_data = json_data.strip().strip("\\n")
        json_obj = json.loads(json_data)
        if type(json_obj) == list:
            assert len(json_obj) == 1
            json_obj = json_obj[0]
            assert type(json_obj) == dict
        return json_obj
    return json_data


def cal_ratio(predict, actual):
    predict = 0 if predict < 0 else predict
    return max(0.0, min(predict / actual, 2.0))


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def flat_depth2_list(targets: list):
    """
    :param targets: [[],[],[]]
    :return: []
    """
    res = []
    for values in targets:
        res += values
    return res


def draw_by_agg(df: DataFrame, y_names: list, agg: str, file_name):
    x_names = []
    y_values = []
    for y_name in y_names:
        x_names.append(y_name)
        values = np.array(list(df[y_name]))
        if agg == "mean":
            y = np.mean(values)
        elif agg == "sum":
            y = np.sum(values)
        else:
            raise RuntimeError
        y_values.append(y)
    new_df = DataFrame({
        "x": x_names,
        "y": y_values
    })
    fig = px.bar(new_df, x="x", y="y", text_auto=True)
    fig.show()
    fig.write_image("RegressionFramework/fig/{}.png".format(file_name))


def read_sqls(file_path):
    sqls = []
    with open(file_path) as f:
        line = f.readline()
        while line is not None and line != "":
            sqls.append(line.split(SEP)[1])
            line = f.readline()
    return sqls


def join(separate: str, target: list):
    res = []
    for t in target:
        res.append(str(t))

    return separate.join(res)
