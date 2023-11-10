from copy import copy

from Common.PlanConfig import SparkNodeConfig
from Common.PlanFactory import PlanFactory
from auncel.model_config import FILTER_TYPES, db_type
from utils import json_str_to_json_obj


class DotDrawer:
    def __init__(self) -> None:
        super().__init__()
        self.nodes = {}
        self.edge = {}

    def add_node(self, node_id, label):
        self.nodes[node_id] = label

    def add_edge(self, from_id: str, to_id: str, label):
        key = (from_id, to_id)
        self.edge[key] = label

    def get_dot_str(self):
        res = "digraph { \n rankdir=Tb \n"

        # add node
        for node_id, node_label in self.nodes.items():
            res += "\"{}\" [label=\"{}\"  ]\n".format(node_id, node_label)

        # add edge
        for ids, edge_label in self.edge.items():
            res += "\"{}\"->\"{}\"[label= \" {} \"] \n".format(ids[0], ids[1], edge_label)

        res += "\n }"
        return res


class GroupTreeDotDrawer:
    dot_node_id = 0

    @classmethod
    def get_plan_dot_str(cls, plan_group):
        dot_drawer = DotDrawer()

        i = 1
        for k, v in plan_group.key_to_static_root.items():
            dot_drawer.add_node(-i, "struct_{}".format(i))
            root = v
            cls.add_unique_id(root)
            dot_drawer.add_edge(-i, root.id, "")
            cls._recurse(dot_drawer, root)
            i += 1

        return dot_drawer.get_dot_str()

    @classmethod
    def _recurse(cls, dot_drawer: DotDrawer, parent):
        action = parent.split_action
        dot_drawer.add_node(parent.id, "{},size={}".format(action.__class__.__name__, len(parent.plans)))
        if not parent.is_leaf():
            dot_drawer.add_edge(parent.id, parent.left_child.id, "yes_{}".format(action.name()))
            dot_drawer.add_edge(parent.id, parent.right_child.id, "no_{}".format(action.name()))
            cls._recurse(dot_drawer, parent.left_child)
            cls._recurse(dot_drawer, parent.right_child)

    @classmethod
    def add_unique_id(cls, root):
        def recurse(parent):
            cls.dot_node_id += 1
            parent.id = cls.dot_node_id
            if not parent.is_leaf():
                recurse(parent.left_child)
                recurse(parent.right_child)

        recurse(root)


class SparkPlanDotDrawer:
    dot_node_id = 0

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_plan_dot_str(cls, plan):
        if isinstance(plan, str):
            plan = json_str_to_json_obj(plan)
        dot_drawer = DotDrawer()
        prefix_id = 0

        def fill(plan_node):
            node_id = cls._get_node_key(plan_node)
            node_label = cls._get_node_label(plan_node)
            dot_drawer.add_node(node_id, node_label)
            children = cls._get_child(plan_node)

            for child in children:
                edge_label = cls._get_edge_info(plan_node, child)
                dot_drawer.add_edge(cls._get_node_key(child), node_id, edge_label)
                fill(child)

        plan = copy(plan)
        if "Plan" in plan:
            plan = plan["Plan"]
        cls.add_unique_id(plan)
        fill(plan)
        return dot_drawer.get_dot_str()

    @classmethod
    def add_unique_id(cls, plan):
        def recurse(plan_node):
            plan_node["dot_id"] = cls.dot_node_id
            children = cls._get_child(plan_node)
            for child in children:
                cls.dot_node_id += 1
                recurse(child)

        recurse(plan)
        cls.dot_node_id = 0
        return plan

    @classmethod
    def _get_node_label(cls, plan_node):
        node_type = plan_node["class"].split(".")[-1]
        row = 0.0 if "rowCount" not in plan_node else plan_node["rowCount"]
        width = plan_node["sizeInBytes"]

        label = "{}, row={}, width={}".format(node_type, row, width)
        if plan_node["class"] == "org.apache.spark.sql.execution.FileSourceScanExec":
            table = extract_table_name(plan_node)
            label += ", table is {}".format(table)

        if plan_node["class"] in FILTER_TYPES:
            node = PlanFactory.get_plan_node_instance(db_type, plan_node)
            label += ", {}".format(node.get_identifier())

        if plan_node["class"] in SparkNodeConfig.JOIN_TYPES:
            node = PlanFactory.get_plan_node_instance(db_type, plan_node)
            label += ", {}".format(node.get_identifier())

        return label

    @classmethod
    def _get_node_key(cls, plan_node):
        node_type = plan_node["class"].split(".")[-1]
        node_id = plan_node["dot_id"]
        return "id={}:{}".format(node_id, node_type)

    @classmethod
    def _get_child(cls, plan_node):
        child = []
        if "Plans" in plan_node:
            child += plan_node["Plans"]
        return child

    @classmethod
    def _get_edge_info(cls, parent, child):
        return ""


def extract_table_name(file_scan_operator):
    if "tableIdentifier" not in file_scan_operator:
        raise RuntimeError("please input file_scan_operator")
    return file_scan_operator["tableIdentifier"]["table"]


def draw_dot_spark_plan(plan):
    return SparkPlanDotDrawer.get_plan_dot_str(plan)
