# Adapted from https://github.com/THUDM/HGB/blob/master/NC/benchmark/scripts/data_loader.py
import json
import os
from typing import Sequence, Dict, Tuple, Union

import torch as th
import dgl
import numpy as np
import pandas as pd
from dgl import DGLGraph
from pandas.core.dtypes.common import is_string_dtype
from sklearn.preprocessing import MultiLabelBinarizer


def load_graph(path) -> tuple[DGLGraph, np.ndarray, np.ndarray]:
    nodes_df = _load_nodes(path)
    nfeat = {ntype: th.Tensor(np.vstack(ndf['node_attr'].values)) for ntype, ndf in nodes_df.groupby('node_type')}

    hg_dicts = _load_links(path)
    hg = dgl.heterograph(hg_dicts, num_nodes_dict=nodes_df['node_type'].value_counts().to_dict())

    hg.ndata['feat'] = nfeat

    node_mask, all_labels = _load_labels(path, ('label.dat', 'label.dat.test'))
    return hg, all_labels, node_mask


def _load_labels(path: str, names: Sequence[str]) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    dfs = [pd.read_csv(os.path.join(path, name), sep='\t', header=None, index_col='node_id',
                       names=['node_id', 'node_name', 'node_type', 'node_label']) for name in names]
    df = pd.concat(dfs)
    df = df.sort_values(by=['node_id'])

    if is_string_dtype(df['node_label']) and df['node_label'].str.contains(',').any():
        labels = df['node_label'].str.split(',').values
        labels = MultiLabelBinarizer().fit_transform(labels)
    else:
        labels = df['node_label'].values

    return df.index.values, labels


def _load_links(path) -> dict[tuple[str, str, str], np.ndarray]:
    def _parse_etype(etype: str) -> tuple[str, str, str]:
        etype = etype.replace('>', '')
        if len(etype.split('-')) == 3:
            return tuple(etype.split('-'))
        utype, vtype = etype.split('-')
        return utype, utype[0] + vtype[0], vtype

    def _reset_index(node_ids: np.ndarray) -> np.ndarray:
        return node_ids - np.min(node_ids)

    info = _load_info(path)
    etypes = {int(k): _parse_etype(v['meaning']) for k, v in info['link.dat']['link type'].items()}

    edges_df = pd.read_csv(os.path.join(path, 'link.dat'), sep='\t', header=None,
                           names=['from', 'to', 'etype', 'weight'])
    edges_df['etype'] = edges_df['etype'].map(etypes)
    hg_dict = {etype: tuple(edges.values.T) for etype, edges in edges_df.groupby('etype')[['from', 'to']]}
    return {etype: (_reset_index(u), _reset_index(v)) for etype, (u, v) in hg_dict.items()}


def _load_nodes(path) -> pd.DataFrame:
    info = _load_info(path)
    ntypes = {int(k): v for k, v in info['node.dat']['node type'].items()}

    df = pd.read_csv(os.path.join(path, 'node.dat'), sep='\t', header=None, index_col='node_id',
                     names=['node_id', 'node_name', 'node_type', 'node_attr'])
    feat = df['node_attr'].dropna().apply(lambda x: list(map(float, x.split(','))))
    df['node_attr'] = feat
    df['node_type'] = df['node_type'].map(ntypes)
    return df


def _load_info(path) -> Dict[str, Dict[str, Dict]]:
    with open(os.path.join(path, 'info.dat')) as f:
        info_dict = json.load(f)
    return info_dict
