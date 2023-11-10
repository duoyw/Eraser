import torch
from torch.utils.data import Dataset
from collections import deque

from Common.PlanConfig import SparkNodeConfig
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *
from auncel.Common.TimeStatistic import TimeStatistic
from auncel.model_config import ALIAS_TO_TABLE
from auncel.utils import json_str_to_json_obj, extract_table_name
from auncel.Common.GlobalVariable import GlobalVariable
from pandas import DataFrame


class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, encoding, hist_file, normalizer, table_sample, dataset_name):

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.dataset_name = dataset_name

        self.length = len(json_df)
        # train = train.loc[json_df['id']]

        nodes = [json_str_to_json_obj(plan)['Plan'] for plan in json_df['json']]
        self.execution_time = [json_str_to_json_obj(plan)['Execution Time'] for plan in json_df['json']]

        self.execution_time = torch.from_numpy(
            np.array([normalizer.norm(t, 'Execution Time') for t in self.execution_time]))
        self.labels = self.execution_time

        idxs = list(json_df['id'])

        # for mem collection
        self.treeNodes = []
        self.collated_dicts = []
        TimeStatistic.start("total")
        for i, node in zip(idxs, nodes):
            self.collated_dicts.append(self.js_node2dict(i, node))
            if i % 500 == 0:
                print("collated_dicts size is {}, total is {}".format(i, len(nodes)))
        TimeStatistic.end("total")
        # TimeStatistic.print()
        print("cache_visit_count is {}, no_cache_visit_count is{}".format(
            str(GlobalVariable.get("model_cache_visit_count")),
            str(GlobalVariable.get("model_no_cache_visit_count"))))

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)

        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.collated_dicts[idx], self.labels[idx]

    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            TimeStatistic.start("floyd_warshall_rewrite")

            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
            TimeStatistic.end("floyd_warshall_rewrite")

        rel_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features': torch.FloatTensor(features),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
        }

    def topo_sort(self, root_node):
        #        nodes = []
        adj_list = []  # from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    def traversePlan(self, plan, idx, encoding):  # bfs accumulate plan

        nodeType = plan['class']
        typeId = encoding.encode_type(nodeType)
        card = None  # plan['Actual Rows']
        filters, table = formatFilter(plan, use_filter=self.hist_file is not None)
        if table is not None:
            table=table[0]
        table = ALIAS_TO_TABLE[self.dataset_name][table] if table in ALIAS_TO_TABLE[self.dataset_name] else table

        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, table)

        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)

        self.treeNodes.append(root)

        if plan["class"] in SparkNodeConfig.SCAN_TYPES:
            root.table = extract_table_name(plan)
            root.table_id = encoding.encode_table(root.table)

        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order


def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])

    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    if node.table_id != 0 and table_sample is not None:
        sample = table_sample[node.query_id][node.table]
    else:
        sample = np.zeros(1000)

    # return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))
