from Plan import Plan, PlanNode, JoinPlanNode, ScanPlanNode, FilterPlanNode


class PlanClassify:
    def __init__(self, init_plans):
        self.join_keys = set()
        self.join_types = set()
        self.table_names = set()
        self.filter_cols = set()
        self.structs = set()
        self.init_plans = init_plans

        for plan in init_plans:
            infos = self._get_plan_info(plan.root)
            self.structs.add(infos[0])
            self.join_types.update(infos[1])
            self.join_keys.update(infos[2])
            self.table_names.update(infos[3])
            self.filter_cols.update(infos[4])
        print()

    def classify(self, plans: list):
        # exist same plan in self.init_plans and cur plan contain new features
        same_new = []
        same_old = []
        diff_new = []
        diff_old = []

        for plan in plans:
            plan: Plan = plan
            struct, join_types, join_keys, table_names, filter_cols = self._get_plan_info(plan.root)
            if struct in self.structs:
                if self._is_new_feature(join_types, join_keys, table_names, filter_cols):
                    same_new.append(plan)
                else:
                    same_old.append(plan)
            else:
                if self._is_new_feature(join_types, join_keys, table_names, filter_cols):
                    diff_new.append(plan)
                else:
                    diff_old.append(plan)
        return same_new, same_old, diff_new, diff_old

    def _is_new_feature(self, join_types, join_keys, table_names, filter_cols):
        if len(join_types.difference(self.join_types)) > 0 or len(
                join_keys.difference(self.join_keys)) > 0 or len(
            table_names.difference(self.table_names)) > 0 or len(filter_cols.difference(self.filter_cols)) > 0:
            return True
        return False

    def _get_plan_info(self, node: PlanNode):
        struct = ""
        join_types = set()
        join_keys = set()
        table_names = set()
        filter_cols = set()

        if node.is_join_node():
            join_type, join_key = self._get_join_infos(node)
            join_types.update(join_type)
            join_keys.update(join_key)
        elif node.is_scan_node():
            scan_type, table_name = self._get_scan_infos(node)
            table_names.update(table_name)
        elif node.is_filter_node():
            cols = self._get_filter_infos(node)
            filter_cols.update(cols)

        struct += node.node_type + "_"

        if not node.is_leaf():
            for child in node.children:
                infos = self._get_plan_info(child)
                struct += infos[0]
                join_types.update(infos[1])
                join_keys.update(infos[2])
                table_names.update(infos[3])
                filter_cols.update(infos[4])
        return struct, join_types, join_keys, table_names, filter_cols

    def _get_join_infos(self, node: JoinPlanNode):
        return node.join_type, node.join_key

    def _get_scan_infos(self, node: ScanPlanNode):
        return node.scan_type, node.table_name

    def _get_filter_infos(self, node: FilterPlanNode):
        predicates = node.predicates
        cols = []
        for p in predicates:
            cols.append(p[0])
        return cols
