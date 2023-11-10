from matplotlib import font_manager, pyplot as plt

font_size = 55
width_quarter = 500
regression_algo_name = "Eraser"

name_2_symbol = {
    "postgresql": "",
    "lero+unobserved explore": "",
    "lero+segment": "",
    "lero": "x",
    "lero_{}".format(regression_algo_name.lower()): "\\",
    "lero-{}".format(regression_algo_name.lower()): "\\",
    "perfguard": "x",
    "perfguard_{}".format(regression_algo_name.lower()): "\\",
    "perfguard-{}".format(regression_algo_name.lower()): "\\",
    "hyperqo": "x",
    "hyperqo_{}".format(regression_algo_name.lower()): "\\",

    "algo": "",
    "{}".format(regression_algo_name): "/",
}

name_2_scatter_symbol = {
    "postgresql": "diamond",
    "lero": "triangle-up",
    "lero_1": "triangle-up",
    "lero_3": "triangle-up",
    "lero_system": "triangle-down",
    "lero_1_system": "triangle-down",
    "lero_3_system": "triangle-down",
    "perfguard": "triangle-up",
    "perfguard_system": "triangle-down",
    "hyperqo": "triangle-up",
    "hyperqo_system": "triangle-down",
}

name_2_color = {
    "0.25": "rgb(20,81,124)",
    "0.5": "rgb(47,127,193)",
    "unseen": "rgb(47,127,193)",
    "structure": "rgb(47,127,193)",
    "0.75": "rgb(150,195,125)",
    "filter": "rgb(150,195,125)",
    "1.0": "rgb(243,210,102)",
    "postgresql": "rgb(216,56,58)",
    "spark": "rgb(216,56,58)",
    "join": "rgb(216,56,58)",
    "existed": "rgb(216,56,58)",

    "lero": "rgb(47,127,193)",
    "lero_".format(regression_algo_name): "rgb(216,56,58)",
    "lero-".format(regression_algo_name): "rgb(216,56,58)",
    "lero-{}".format(regression_algo_name): "rgb(216,56,58)",
    "lero-{}".format(regression_algo_name.lower()): "rgb(216,56,58)",
    "hyperqo": "rgb(150,195,125)",
    "perfguard": "rgb(243,210,102)",

    "lero-generated-sql": "rgb(243,210,102)",
    "lero-{}-generated-sql".format(regression_algo_name.lower()): "rgb(243,210,102)",
    "lero+": "rgb(243,210,102)",
    "lero_system_stats": "rgb(216,56,58)",
    "lero_system_job": "rgb(243,210,102)",
    "imdb": "rgb(243,210,102)",
    "stats": "rgb(216,56,58)",
    "tpch": "rgb(150,195,125)",

    "imdb-0.25": "rgb(243,210,102)",
    "imdb-1.0": "rgb(216,56,58)",

    "stats-0.25": "rgb(243,210,102)",
    "stats-1.0": "rgb(216,56,58)",

    "lero+unobserved explore": "rgb(150,195,125)",
    "lero+explorer": "rgb(150,195,125)",
    "lero-explorer": "rgb(150,195,125)",
    "lero+segment": "rgb(147,148,231)",
    "lero-segment": "rgb(147,148,231)",
    "lero+both": "rgb(243,210,102)",
    "lero-{}".format(regression_algo_name.lower()): "rgb(243,210,102)",

    "lero_1": "rgb(47,127,193)",
    "lero_1_{}".format(regression_algo_name.lower()): "rgb(150,195,125)",
    "lero_3": "rgb(47,127,193)",
    "lero_3_{}".format(regression_algo_name.lower()): "rgb(150,195,125)",
    "lero_4": "rgb(47,127,193)",
    "lero_4_{}".format(regression_algo_name.lower()): "rgb(150,195,125)",
}


def get_plt():
    path = 'Linux-Libertine.ttf'
    font_manager.fontManager.addfont(path)
    prop = font_manager.FontProperties(fname=path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['mathtext.default'] = 'regular'
    return plt


def capitalize(s: str):
    return s[0].upper() + s[1:]
