import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame
import numpy as np

from Spark.auncel.test_script.config import PROJECT_BASE_PATH


def draw(x_name, spark_plans, auncel_plans, best_plans, file_name):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_name,
        y=spark_plans,
        name="sparkPlans",
        text=spark_plans,
    ))
    fig.add_trace(go.Bar(
        x=x_name,
        y=auncel_plans,
        name="auncel_plans",
        text=auncel_plans,
    ))
    fig.add_trace(go.Bar(
        x=x_name,
        y=best_plans,
        name="best_plans",
        text=best_plans,

    ))
    # fig.update_layout(yaxis_range=[0, 10])

    fig.show()

    fig.write_image(PROJECT_BASE_PATH + "{}.png".format(file_name))


def draw2(df: DataFrame, x_name, y_names: list, file_name):
    fig = px.bar(df, x=x_name, y=y_names, text_auto=True)
    fig.show()
    fig.write_image(PROJECT_BASE_PATH + "{}.png".format(file_name))


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
    fig.update_layout(
        xaxis_title="Algorithms",
        yaxis_title="Execution Time (ms)"
    )
    fig.show()
    fig.write_image("../RegressionFramework/fig/{}.png".format(file_name))
