from RegressionFramework.Plan.PgPlan import PgScanPlanNode
from RegressionFramework.Plan.Plan import Plan, PlanNode, FilterPlanNode, JoinPlanNode, ScanPlanNode, ProjectPlanNode


class PlansManager:
    def __init__(self, plans):
        super().__init__()
        self.plans = plans
        self.plan_id_to_node_id_to_node = {}

        for plan in self.plans:
            plan: Plan = plan
            plan_id = plan.plan_id
            if plan_id not in self.plan_id_to_node_id_to_node:
                self.plan_id_to_node_id_to_node[plan_id] = {}

            nodes = plan.get_all_nodes()
            for node in nodes:
                node: PlanNode = node
                node_id = node.get_node_id()
                if node_id not in self.plan_id_to_node_id_to_node[plan_id]:
                    self.plan_id_to_node_id_to_node[plan_id][node_id] = node
                else:
                    raise RuntimeError

        self.all_nodes = []
        for node_id_to_node in self.plan_id_to_node_id_to_node.values():
            self.all_nodes += node_id_to_node.values()

        # node_id to col to op to values for all plans
        self.plan_id_node_id_to_filter_predicate = {}

        self.plan_id_to_node_id_to_join_types = {}
        self.plan_id_to_node_id_to_join_keys = {}

        self.plan_id_to_node_id_to_scan_types = {}
        self.plan_id_to_node_id_to_table_names = {}

        self.plan_id_to_node_id_to_project_cols = {}

        for plan in plans:
            self._recurse_plan(plan.plan_id, plan.root)

    def _recurse_plan(self, plan_id, node: PlanNode):
        node_id = node.get_node_id()

        if isinstance(node, FilterPlanNode):
            raise RuntimeError("not exist")
        elif isinstance(node, PgScanPlanNode):
            node: PgScanPlanNode = node
            if plan_id not in self.plan_id_node_id_to_filter_predicate:
                self.plan_id_node_id_to_filter_predicate[plan_id] = {}

            if node_id not in self.plan_id_node_id_to_filter_predicate[plan_id]:
                self.plan_id_node_id_to_filter_predicate[plan_id][node_id] = {}

            for col, op, value in node.predicates:
                if col not in self.plan_id_node_id_to_filter_predicate[plan_id][node_id]:
                    self.plan_id_node_id_to_filter_predicate[plan_id][node_id][col] = {}
                if op not in self.plan_id_node_id_to_filter_predicate[plan_id][node_id][col]:
                    self.plan_id_node_id_to_filter_predicate[plan_id][node_id][col][op] = set()
                self.plan_id_node_id_to_filter_predicate[plan_id][node_id][col][op].add(value)

        if isinstance(node, JoinPlanNode):
            node: JoinPlanNode = node
            self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_join_types, node.join_type)
            self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_join_keys, node.get_join_key_str())

        if isinstance(node, ScanPlanNode):
            node: ScanPlanNode = node
            self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_scan_types, node.scan_type)
            self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_table_names, node.table_name)

        if isinstance(node, ProjectPlanNode):
            node: ProjectPlanNode = node
            if node.empty():
                self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_project_cols, None)
            else:
                for col in node.project_cols:
                    self._init_dict_or_add(plan_id, node_id, self.plan_id_to_node_id_to_project_cols, col)

        for child in node.children:
            self._recurse_plan(plan_id, child)

    def _init_dict_or_add(self, plan_id, node_id, plan_id_to_node_id_to_values, value):
        if plan_id not in plan_id_to_node_id_to_values:
            plan_id_to_node_id_to_values[plan_id] = {}
        if node_id not in plan_id_to_node_id_to_values[plan_id]:
            plan_id_to_node_id_to_values[plan_id][node_id] = []
        if value is not None:
            plan_id_to_node_id_to_values[plan_id][node_id].append(value)

    def get_all_filter_infos(self, plan_id, node_id):
        return self.plan_id_node_id_to_filter_predicate[plan_id][node_id]

    def get_all_join_keys(self, plan_id, node_id):
        return self.plan_id_to_node_id_to_join_keys[plan_id][node_id]

    def get_all_join_types(self, plan_id, node_id):
        return self.plan_id_to_node_id_to_join_types[plan_id][node_id]

    def get_all_table_types(self, plan_id, node_id):
        return self.plan_id_to_node_id_to_scan_types[plan_id][node_id]

    def get_all_table_names(self, plan_id, node_id):
        return self.plan_id_to_node_id_to_table_names[plan_id][node_id]

    def get_all_project_cols(self, plan_id, node_id):
        return self.plan_id_to_node_id_to_project_cols[plan_id][node_id]

    def get_node(self, plan_id, plan_node_id):
        return self.plan_id_to_node_id_to_node[plan_id][plan_node_id]

    def get_nodes(self, plan_node_id):
        res = []
        for node_id_to_node in self.plan_id_to_node_id_to_node.values():
            res.append(node_id_to_node[plan_node_id])
        return res
