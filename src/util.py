from typing import Sequence

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F


def _propagate(g, metapath, node_feats: Sequence[th.Tensor] = None):
    adj = g.adj(etype=metapath[0], transpose=False, ctx=g.device)

    node_feats = [None] * len(metapath) if node_feats is None else node_feats

    for etype, node_feat in zip(metapath[1:], node_feats):
        adj_ = g.adj(etype=etype, transpose=False, ctx=g.device)
        if node_feat is not None:
            adj_vals = F.embedding(adj_.coalesce().indices()[0], node_feat.unsqueeze(1)).squeeze()
            adj_ = th.sparse_coo_tensor(adj_.coalesce().indices(), adj_vals, device=g.device)
            # Hacky workaround for adj_ * node_feat that does work with autograd
        adj = th.sparse.mm(adj, adj_)
    # transpose needed for edge indexing
    return adj.T.coalesce()


def get_metagraph(g: dgl.DGLHeteroGraph, metapath: Sequence[str]) -> dgl.DGLHeteroGraph:
    adj = _propagate(g, metapath)
    indices = adj.indices()[0].cpu().numpy(), adj.indices()[1].cpu().numpy()

    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    new_g = dgl.convert.heterograph({(srctype, '_E', dsttype): indices}, {srctype: adj.shape[0], dsttype: adj.shape[1]},
                                    idtype=g.idtype, device=g.device)

    # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    # copy dstnode features
    if srctype != dsttype:
        new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)

    return new_g


def get_informed_metagraph(g: dgl.DGLHeteroGraph, metapath: Sequence[str],
                           node_embeddings: Sequence[th.Tensor]) -> dgl.DGLHeteroGraph:
    assert len(node_embeddings) == len(metapath) - 1, 'need node embeddings for each node type in the metapath'

    new_g = get_metagraph(g, metapath)

    dims = [int(ne.shape[1]) for ne in node_embeddings]
    assert len(np.unique(dims)) == 1, 'all node embeddings must have the same shape'

    edge_feats = th.stack([_propagate(g, metapath, [ne.T[dim] for ne in node_embeddings]).values()
                           for dim in range(dims[0])], dim=-1)

    new_g.edata['feat'] = edge_feats.to(new_g.device)
    return new_g


import networkx as nx


def get_all_metapaths(g: dgl.DGLHeteroGraph, metapath):
    edge_types = [g.to_canonical_etype(et)[1] for et in metapath]

    g_ = dgl.edge_type_subgraph(g, etypes=edge_types)
    g_hom = dgl.to_homogeneous(g_)
    g_nx = dgl.to_networkx(g_hom)

    src_ntype = g_.ntypes.index(g.to_canonical_etype(metapath[0])[0])
    target_ntype = g_.ntypes.index(g.to_canonical_etype(metapath[-1])[-1])

    metagraph = dgl.metapath_reachable_graph(g, metapath)

    source_nodes = g_hom.nodes()[g_hom.ndata['_TYPE'] == src_ntype]
    target_nodes = g_hom.nodes()[g_hom.ndata['_TYPE'] == target_ntype]

    source_nodes = metagraph.edges()[0].numpy() + np.array(source_nodes.min())
    target_nodes = metagraph.edges()[1].numpy() + np.array(target_nodes.min())

    metapaths = []
    for source_node, target_node in zip(source_nodes, target_nodes):
        if source_node == target_node:
            pass  # print(source_node, target_node)
        metapaths.append(nx.all_simple_paths(g_nx, source_node, target_node, cutoff=len(metapath)))

    return list(map(list, metapaths))

