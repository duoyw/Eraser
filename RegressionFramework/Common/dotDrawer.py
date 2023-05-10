from RegressionFramework.Plan.PgPlan import PgScanPlanNode
from RegressionFramework.Plan.Plan import PlanNode


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


class PlanDotDrawer:
    dot_node_id = 0

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_plan_dot_str(cls, plan):
        dot_drawer = DotDrawer()

        def fill(plan_node):
            node_id = plan_node.node_id
            node_label = cls._get_node_label(plan_node)
            dot_drawer.add_node(node_id, node_label)
            children = cls._get_child(plan_node)

            for child in children:
                edge_label = cls._get_edge_info(plan_node, child)
                dot_drawer.add_edge(child.node_id, node_id, edge_label)
                fill(child)

        fill(plan.root)
        return dot_drawer.get_dot_str()

    @classmethod
    def _get_node_label(cls, plan_node):
        node_type = plan_node.node_type
        label = "{}".format(node_type)
        if plan_node.is_filter_node():
            label += ", {}".format(plan_node.get_identifier())
        elif plan_node.is_scan_node():
            label += ", table is {}".format(plan_node.get_table_name())
            if isinstance(plan_node, PgScanPlanNode):
                predicates_str:str="{}".format(plan_node.predicates)
                label += ", predicate is {}".format(predicates_str.replace("\"",""))

        elif plan_node.is_join_node():
            label += ", {}".format(plan_node.get_identifier())

        return label

    @classmethod
    def _get_node_key(cls, plan_node):
        node_type = plan_node.node_type
        node_id = plan_node.node_id
        return "".format(node_id)

    @classmethod
    def _get_child(cls, plan_node):
        return plan_node.children

    @classmethod
    def _get_edge_info(cls, parent, child):
        return ""

