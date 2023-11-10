#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset

data_path = './data/imdb/'


class Args:
    pass


hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)

encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']
checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

from model.util import seed_everything

seed_everything()

args = checkpoint['args']

model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                    dropout=args.dropout, n_layers=args.n_layers,
                    use_sample=True, use_hist=True,
                    pred_hid=args.pred_hid)

model.load_state_dict(checkpoint['model'])

device = 'cuda:0'
_ = model.to(device).eval()

to_predict = 'cost'

methods = {
    'get_sample': get_job_table_sample,
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': device,
    'bs': 512,
}


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror, 90)
    e_mean = np.mean(qerror)
    print("Median: {}".format(e_50))
    print("90th percentile: {}".format(e_90))
    print("Mean: {}".format(e_mean))
    return


def get_corr(ps, ls):  # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))

    return corr


def evaluate(model, ds, bs, norm, device):
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i, min(i + bs, len(ds)))])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

    print_qerror(norm.unnormalize_labels(cost_predss), ds.costs)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    print('Corr: ', corr)

    return


def eval_workload(workload, methods):
    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv('./data/imdb/workloads/{}.csv'.format(workload), sep='#', header=None)
    workload_csv.columns = ['table', 'join', 'predicate', 'cardinality']

    ds = PlanTreeDataset(plan_df, workload_csv,
                         methods['encoding'], methods['hist_file'], methods['cost_norm'], \
                         methods['cost_norm'], 'cost', table_sample)

    evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'])
    return


eval_workload('job-light', methods)
# eval_workload('synthetic', methods)


