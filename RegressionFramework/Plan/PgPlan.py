import queue
import time
import re
from RegressionFramework.Plan.Plan import Plan, ScanPlanNode, PlanNode, FilterPlanNode, JoinPlanNode, ProjectPlanNode
from RegressionFramework.Plan.PlanConfig import PgNodeConfig
from RegressionFramework.utils import is_number


class PgPlan(Plan):
    def __init__(self, plan_json, plan_id, predict=None):
        super().__init__(plan_json, plan_id, predict)

    @classmethod
    def to_node(cls, node_json, node_id=None):
        node_type = cls.get_node_type(node_json)
        if node_type in PgNodeConfig.SCAN_TYPES:
            plan_node = PgScanPlanNode(node_json, node_id)
        elif node_type in PgNodeConfig.JOIN_TYPES:
            plan_node = PgJoinPlanNode(node_json, node_id)
        # elif node_type in SparkNodeConfig.PROJECT_TYPES:
        #     plan_node = SparkProjectPlanNode(node_json, node_id)
        else:
            plan_node = PgOtherPlanNode(node_json, node_id)
        return plan_node


    def _to_plan_node(self, node_json, node_id, node_id_to_node):
        plan_node = self.to_node(node_json, node_id)

        assert node_id not in node_id_to_node
        node_id_to_node[node_id] = plan_node

        cur_max_node_id = node_id
        if "Plans" in node_json:
            children = node_json["Plans"]
            for child in children:
                child_node, cur_max_node_id = self._to_plan_node(child, cur_max_node_id + 1,
                                                                 node_id_to_node)
                plan_node.children.append(child_node)
        return plan_node, cur_max_node_id

    @classmethod
    def get_node_type(cls, node_json):
        return PgPlanNodeMixIn.get_node_type(node_json)


class PgPlanNodeMixIn(PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)

    @classmethod
    def get_node_type(cls, node_json):
        return node_json["Node Type"]


class PgScanPlanNode(PgPlanNodeMixIn, ScanPlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)
        self.predicates = self.extract_filter_predicate()

    def get_scan_type(self):
        return self.get_node_type(self.node_json)

    def get_table_name(self):
        if "Relation Name" not in self.node_json:
            raise RuntimeError("please input file_scan_operator")
        return self.node_json["Relation Name"]

    def extract_filter_predicate(self):
        #  "Filter": "((it1.info)::text = 'admissions'::text)",
        # "Filter": "((t.production_year >= 1916) AND (t.production_year <= 2009))",
        # "(mi_idx.info > '10'::text)",
        predicates = []
        if "Filter" in self.node_json:
            iter = self.node_json["Filter"][1:-1]
            entries = self.split_to_entries(iter)
            for e in entries:
                # ci.note LIKE '%r%' -> "(ci.note ~~ '%r%'::text)"
                # '(mc.note IS NOT NULL)'
                # "(mi.info = ANY ('{Thriller,Australia:G,Animation,German,"RAT:1.37 : 1",Short}'::text[]))"
                # "p.score >= '-1'::integer"
                if "~~" in e or "NULL" in e or "ANY" in e or "ALL" in e or "NOT" in e:
                    continue
                if "(p.score >= '-1'::integer)" == e:
                    e = "(p.score >= -1::integer)"
                if "(p.score >= '-2'::integer)" == e:
                    e = "(p.score >= -2::integer)"
                if "(p.score >= '-4'::integer)" == e:
                    e = "(p.score >= -4::integer)"
                if "(p.score >= '-3'::integer)" == e:
                    e = "(p.score >= -3::integer)"

                col, value, op = self.extract_from_entry(e)
                # if type(value) == str and '..' in value:
                #     continue
                value = value.replace("$", "")

                if is_number(value):
                    value = float(value)

                if col == "mi_idx.info" or col == "mi.info":
                    # mi_idx.info mix number and text, and it is difficult to distinguish
                    value = str(value)
                    # print("{}     {}     {}".format(col, op, value))

                predicates.append((col, op, value))
                # print((col, op, value))

        return predicates

    def _clean(self, t: str):
        return t.strip("\'").strip("\"").strip("(").strip(")")

    def split_to_entries(self, iter):
        """
        :param iter: like :"(mc.note IS NOT NULL) AND ((mc.note ~~ '%(USA)%'::text) OR (mc.note ~~ '%(worldwide)%'::text))"
        :return:
        """

        buff = queue.Queue()
        buff.put(iter)
        res = []
        while not buff.empty():
            entry = buff.get()
            sep = None
            if " AND " in entry:
                sep = " AND "
            elif " OR " in entry:
                sep = " OR "
            if sep is not None:
                es = entry.split(sep, 1)
                assert len(es) == 2
                buff.put(es[0])
                buff.put(es[1])
            else:
                res.append(entry)
        return res

    def extract_from_entry(self, entry):
        """
        :param entry: (it1.info)::text = 'admissions'::text or (t.production_year >= 1916) or (mi_idx.info > '10'::text)
        :return:
        """
        entry = self.delete_bracket(entry)
        pattern = "(.*) (>|>=|=|<|<=|<>) (.*)"
        match = re.match(pattern, entry)
        if match is None:
            raise RuntimeError("match is None")
        col = self.delete_for_text(self._clean(match.group(1)))
        op = self._clean(match.group(2))
        value = self._clean(self.delete_for_text(self._clean((match.group(3)))))
        return self.delete_bracket(col, False).strip(), self.delete_bracket(value, False).strip(), op.strip()

    def delete_for_text(self, target):
        """
        if target contain ::text, then delete
        :return:
        """
        target = target.strip()
        if target.endswith("::text"):
            return target[0:-6]
        if target.endswith("::bpchar"):
            return target[0:-8]
        if target.endswith("::integer"):
            return target[0:-9]
        if target.endswith("::numeric"):
            return target[0:-9]
        if target.endswith("::date"):
            return target[0:-6]
        if target.endswith("::timestamp without time zone"):
            return target[0:-len("::timestamp without time zone")]
        return target

    def delete_bracket(self, target, is_pair=True):
        """

        :param target:
        :param is_pair: if true, delete only if two bracket
        :return:
        """
        target = target.strip()
        if len(target) > 0 and target[0] == "(" and target[-1] == ")":
            target = target[1:-1]

        if not is_pair:
            target = target.strip("(").strip(")")
        return target


class PgFilterPlanNode(PgPlanNodeMixIn, FilterPlanNode):
    def parse_predicates(self):
        raise RuntimeError


class PgJoinPlanNode(PgPlanNodeMixIn, JoinPlanNode):
    def get_join_type(self):
        return self.node_type

    def _parse_join_key(self):
        node = self.node_json
        left_keys = []
        right_keys = []
        #   "Hash Cond": "(mi_idx.info_type_id = it2.id)",

        if "Hash Cond" in node:
            join_str = node["Hash Cond"]
            # '((lineitem.l_suppkey = supplier.s_suppkey) AND (customer.c_nationkey = supplier.s_nationkey))'
            if "AND" in join_str:
                join_str = join_str.split("AND")[0][1:]
            keys = join_str.split("=")
        elif "Join Filter" in node:
            join_str = node["Join Filter"]
            if "bpchar" not in join_str:
                keys = node["Join Filter"].split("=")
            else:
                keys = "NONE = NONE"
        else:
            keys = "NONE = NONE"
        left_keys.append(keys[0].strip()[1:])
        right_keys.append(keys[1].strip()[:-1])

        return ".".join(left_keys), ".".join(right_keys)


class PgProjectPlanNode(PgPlanNodeMixIn, ProjectPlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)
        self.project_cols = self._parse_project_cols()

    def _parse_project_cols(self):
        raise RuntimeError

    def empty(self):
        return len(self.project_cols) == 0

    def get_identifier(self):
        return "{}".format(self.project_cols)


class PgOtherPlanNode(PgPlanNodeMixIn, PlanNode):
    def __init__(self, node_json, node_id):
        super().__init__(node_json, node_id)
