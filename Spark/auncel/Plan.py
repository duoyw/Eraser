import json


# from Common.DotDrawer import draw_dot_spark_plan
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


class PlanNode:
    def __init__(self, node_json, node_id):
        super().__init__()
        self.node_json = node_json
        self.node_id = node_id
        self.children = []
        self.node_type = self.get_node_type(self.node_json)

    def is_filter_node(self):
        return False

    def is_scan_node(self):
        return False

    def is_join_node(self):
        return False

    def get_identifier(self):
        return ""

    def is_leaf(self):
        return len(self.children)==0

    @classmethod
    def get_node_type(cls, node_json):
        raise RuntimeError

    def _parse_join_key(self):
        raise RuntimeError

    def get_join_type(self):
        raise RuntimeError

    def get_node_id(self):
        return self.node_id

    def get_scan_type(self):
        raise RuntimeError

    def get_table_name(self):
        raise RuntimeError


class FilterPlanNode(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

        # [[col,op,value],[col,op,value],...]
        self.predicates, self.tables = self.parse_predicates()

    def is_filter_node(self):
        return True

    def parse_predicates(self):
        raise RuntimeError

    def get_identifier(self):
        return "{}".format(self.predicates)


class ScanPlanNode(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

        # [(col,op,value),(col,op,value),...]
        self.scan_type = self.get_node_type(self.node_json)
        self.table_name = self.get_table_name()

    def is_scan_node(self):
        return True

    def get_scan_type(self):
        raise RuntimeError

    def get_table_name(self):
        raise RuntimeError

    def get_identifier(self):
        return "{}_{}".format(self.table_name, self.scan_type)


class JoinPlanNode(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

        self.join_type = self.get_join_type()
        # [left_keys_str,right_keys_str]
        self.join_key = self._parse_join_key()

    def get_join_key_str(self):
        return "{}_{}".format(self.join_key[0], self.join_key[1])

    def is_join_node(self):
        return True

    def get_join_type(self):
        raise RuntimeError

    def _parse_join_key(self):
        raise RuntimeError

    def get_identifier(self):
        return "{}_{}".format(self.join_key, self.join_type)


class ProjectPlanNode(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)
        self.project_cols = self._parse_project_cols()

    def _parse_project_cols(self):
        raise RuntimeError

    def empty(self):
        return len(self.project_cols) == 0

    def get_identifier(self):
        return "{}".format(self.project_cols)


class OtherPlanNode(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)


class Plan:
    def __init__(self, plan_json, plan_id, predict=None):
        super().__init__()
        if isinstance(plan_json,str):
            plan_json=json_str_to_json_obj(plan_json)
        self.plan_json = plan_json
        self.predict = predict
        self.execution_time = self._get_execution_time()
        self.plan_id = plan_id

        self.node_id_to_node = {}
        self.root = self._to_plan_node(plan_json["Plan"], 0, self.node_id_to_node)[0]
        if predict is not None:
            self.plan_json["predict"] = predict
        if "predict" in self.plan_json:
            self.predict = self.plan_json["predict"]


    def get_plan_json_str(self):
        return json.dumps(self.plan_json)

    def _get_execution_time(self):
        return self.plan_json["Execution Time"]

    def _to_plan_node(self, node_json, node_id, node_id_to_node):
        raise RuntimeError

    def get_all_nodes(self):
        return list(self.node_id_to_node.values())

    def draw_dot(self):
        pass
        # return draw_dot_spark_plan(self.plan_json)


def json_str_to_json_obj(json_data):
    json_data = json_data.strip().strip("\\n")
    json_obj = json.loads(json_data)
    if type(json_obj) == list:
        assert len(json_obj) == 1
        json_obj = json_obj[0]
        assert type(json_obj) == dict
    return json_obj
