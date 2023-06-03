from typing import Tuple, Optional, Sequence, Dict, Union

import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

from src import util


class SemanticAttention(nn.Module):
    def __init__(self, in_size: int, hidden_size: Optional[int] = 128):
        super(SemanticAttention, self).__init__()
        if hidden_size is None or hidden_size < 1:
            self.project = nn.Linear(in_size, 1, bias=False)
        else:
            self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(),
                                         nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = th.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


class SparseGraphConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, sparsity: float = .5, **kwargs):
        super().__init__()
        self.sparsity = sparsity
        self.in_feats, self.out_feats = in_feats, out_feats
        self.gc = GraphConv(in_feats, out_feats, **kwargs)
        if sparsity > 0:
            weight_ = th.flatten(self.gc.weight).clone()
            indices = th.randperm(len(weight_))[:int(sparsity * self.gc.weight.numel())]
            weight_[indices] = 0
            self.gc.weight = th.nn.Parameter(th.reshape(weight_, self.gc.weight.shape))

    def forward(self, graph, feat, weight=None, edge_weight=None):
        return self.gc(graph, feat, weight=weight, edge_weight=edge_weight)


class MGATConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, num_heads: int, etype_arg: str = dgl.ETYPE,
                 negative_slope: float = 0.2, activation=None, feat_drop=0., attn_drop=0., v2: bool = False,
                 share_weights: bool = True, bias: bool = True, num_etypes: Optional[int] = None,
                 edge_norm: bool = True, use_type_embeddings: bool = False, attn_agg=th.sum,
                 sparse_type_embeddings: bool = False, **__):
        super().__init__()
        self.edge_norm = edge_norm
        self.in_feats, self.out_feats = in_feats, out_feats
        self.v2, self.share_weights = v2, share_weights
        self.use_type_embeddings = use_type_embeddings
        self.attn_agg = attn_agg
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.etype_arg = etype_arg
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self.leaky_relu = th.nn.LeakyReLU(negative_slope)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.attn_l = th.nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = th.nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = th.nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc_e = nn.Linear(num_heads, out_feats * num_heads, bias=False)
        if use_type_embeddings:
            self.fc_et = nn.Linear(out_feats, out_feats * num_heads, bias=False)
            self.attn_et = th.nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_dst = nn.Linear(self._in_src_feats, out_feats * num_heads,
                                bias=False) if not share_weights else self.fc_src

        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)

        self.activation = activation
        if not use_type_embeddings:
            self.edge_type_embeddings_ = None
        elif not sparse_type_embeddings:
            self.edge_type_embeddings_ = nn.Embedding(num_etypes, self.out_feats)
        else:
            shape = self.out_feats // num_etypes
            self.edge_type_embeddings_ = nn.Embedding.from_pretrained(
                th.block_diag(*[th.ones(shape) for _ in range(num_etypes)]), freeze=False)

        self.reset_parameters()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self.use_type_embeddings:
            nn.init.xavier_normal_(self.attn_et, gain=gain)
            nn.init.xavier_normal_(self.fc_et.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, graph: dgl.DGLGraph, feat: th.Tensor, edge_embeddings: Optional[th.FloatTensor] = None,
                res_edge_attn: Optional[th.Tensor] = None, get_attention: bool = False):
        r"""

        Description
        -----------
        Compute graph attentive convolution with masked edge attention for i.e. metagraphs with salient edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            if self.use_type_embeddings:
                edge_types = graph.edata[self.etype_arg].int().squeeze()
                edge_type_embeddings = self.edge_type_embeddings_(edge_types)
                edge_type_embeddings = self.fc_et(edge_type_embeddings).view(-1, self.num_heads, self.out_feats)
                edge_type_embeddings = (edge_type_embeddings * self.attn_et).sum(dim=-1).unsqueeze(-1)

            h_src = h_dst = self.feat_drop(feat)
            feat_src = self.fc_src(h_src).view(-1, self.num_heads, self.out_feats)

            if self.share_weights:
                feat_dst = feat_src
            else:
                feat_dst = self.fc_dst(h_dst).view(-1, self.num_heads, self.out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            # apply activation to attention and mask with edge embedding (e.g. type or meta edge feature)
            edge_attn = self.leaky_relu(graph.edata.pop('e')) if self.v2 else graph.edata.pop('e')

            edge_attn_components = [edge_attn] if res_edge_attn is None else [edge_attn, res_edge_attn]
            if edge_embeddings is not None:
                assert tuple(edge_embeddings.shape) == (graph.number_of_edges(),
                                                        self.num_heads), f'need {graph.number_of_edges(), self.num_heads} but got {edge_embeddings.shape}'
                edge_embeddings = self.fc_e(edge_embeddings).view(-1, self.num_heads, self.out_feats)
                edge_embeddings = edge_embeddings * self.attn_e
                edge_attn_components.append(edge_embeddings.sum(dim=-1).unsqueeze(-1))
            if self.use_type_embeddings:
                edge_attn_components.append(edge_type_embeddings)

            edge_attn = self.attn_agg(th.stack(edge_attn_components), dim=0) if len(edge_attn_components) > 1 else \
                edge_attn

            if self.edge_norm and self.etype_arg in graph.edata:
                edge_types = graph.edata[self.etype_arg].int().squeeze()
                # bincount on CPU is deterministic
                etype_counts = th.bincount(edge_types.cpu()).to(graph.device)
                etype_weights = th.sum(etype_counts) / (len(etype_counts) * etype_counts)
                edge_weights = nn.Embedding.from_pretrained(etype_weights.unsqueeze(-1))(edge_types)
                edge_attn *= edge_weights.unsqueeze(-1)

            if not self.v2:
                edge_attn = self.leaky_relu(edge_attn)
            # GATv2 does activation AFTER masking

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, edge_attn))

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.bias is not None:
                rst = rst + self.bias.view(1, self.num_heads, self.out_feats)

            rst = self.activation(rst) if self.activation else rst
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class GeneralMPConv(nn.Module):
    """
        Superclass for general metapath attention convolution layers. Does the graph handling.
        Arguments
        ---------
        meta_paths : list of metapaths, each as a list of edge types
        self_loops : add self loops to the graph
    """

    def __init__(self, meta_paths, in_size: int, num_heads: int, self_loops: bool = False):
        super(GeneralMPConv, self).__init__()
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.self_loops = self_loops
        self._cached_graph, self._cached_combined_graph = None, None
        self.in_size, self.num_heads = in_size, num_heads

    def get_graph(self, g, combine: bool = True):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g

            meta_graphs = [dgl.metapath_reachable_graph(g, meta_path) for meta_path in self.meta_paths]

            if not combine:
                if len(self.meta_paths) == 1:
                    self._cached_combined_graph = meta_graphs[0]
                else:
                    self._cached_combined_graph = {mp: mg for mp, mg in zip(self.meta_paths, meta_graphs)}
                return self._cached_combined_graph

            heterograph_dicts = {('', ''.join(mp), ''): metagraph.edges() for
                                 mp, metagraph in zip(self.meta_paths, meta_graphs)}

            if self.self_loops:
                relevant_nodes = th.unique(th.cat([mg.nodes() for mg in meta_graphs]))
                heterograph_dicts[('', 'SELF', '')] = (relevant_nodes, relevant_nodes)

            heterograph: dgl.DGLHeteroGraph = dgl.heterograph(heterograph_dicts)

            self._cached_combined_graph = dgl.to_homogeneous(heterograph)

        return self._cached_combined_graph


class InformedMPConv(GeneralMPConv):
    """
        Superclass for general metapath attention convolution layers. Does the graph handling.
        Arguments
        ---------
        meta_paths : list of metapaths, each as a list of edge types
        self_loops : add self loops to the graph
    """

    def __init__(self, meta_paths, in_size: int, num_heads: int, node_feat_size: int, self_loops: bool = False,
                 static_embeddings: bool = True, convolve_embeddings: bool = True, num_layers: int = 2, hidden_size=8,
                 sparsity:float = 0):
        super(InformedMPConv, self).__init__(in_size=in_size, num_heads=num_heads, meta_paths=meta_paths,
                                             self_loops=self_loops)
        if convolve_embeddings:
            self.edge_embedder_layers = th.nn.ModuleList()
            for i in range(num_layers):
                out_size = hidden_size if i < num_layers - 1 else num_heads
                conv_layer = SparseGraphConv(node_feat_size, out_size, bias=False, sparsity=sparsity)
                self.edge_embedder_layers.append(conv_layer)
                node_feat_size = out_size

            if static_embeddings:
                for param in self.edge_embedder_layers.parameters():
                    param.requires_grad = False

        self.static_embeddings = static_embeddings
        self.convolve_embeddings = convolve_embeddings

    def _get_edge_emb(self, g: dgl.DGLHeteroGraph, h: th.Tensor, metapath):
        edge_types = [g.to_canonical_etype(etype)[0] for etype in metapath[1:]]
        node_types = np.array(g.ntypes)
        etype_ids = [np.where(node_types == etype)[0][0] for etype in edge_types]

        g_: dgl.DGLGraph = dgl.add_self_loop(dgl.to_homogeneous(g))

        if self.convolve_embeddings:
            for conv_layer in self.edge_embedder_layers:
                h: th.Tensor = conv_layer(g_, h)

        if self.static_embeddings:
            return [h[g_.ndata['_TYPE'] == etype_id].detach() for etype_id in etype_ids]
        else:
            return [h[g_.ndata['_TYPE'] == etype_id] for etype_id in etype_ids]

    def get_graph(self, g, node_embeddings: Optional[Sequence[th.Tensor]], combine: bool = True):
        if self._cached_graph is None or self._cached_graph is not g or not self.static_embeddings:
            self._cached_graph = g
            meta_graphs = [util.get_informed_metagraph(g, mp, ne) for mp, ne in zip(self.meta_paths, node_embeddings)]

            if not combine:
                self._cached_combined_graph = {mp: mg for mp, mg in zip(self.meta_paths, meta_graphs)}
                return self._cached_combined_graph

            heterograph_dicts = {('', ''.join(mp), ''): metagraph.edges() for mp, metagraph in
                                 zip(self.meta_paths, meta_graphs)}

            if self.self_loops:
                relevant_nodes = th.unique(th.cat([mg.nodes() for mg in meta_graphs]))
                heterograph_dicts[('', 'SELF', '')] = (relevant_nodes, relevant_nodes)

            heterograph: dgl.DGLHeteroGraph = dgl.heterograph(heterograph_dicts)

            heterograph.edata['feat'] = {''.join(mp): mg.edata['feat'] for mp, mg in
                                         zip(self.meta_paths, meta_graphs)}

            self._cached_combined_graph = dgl.to_homogeneous(heterograph, edata=['feat'])
        return self._cached_combined_graph



class HAConv(GeneralMPConv):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size: int, out_size: int, num_heads: int, dropout: float, activation=F.elu,
                 base_conv_cls=MGATConv, **conv_kwargs):
        super(HAConv, self).__init__(in_size=in_size, num_heads=num_heads, meta_paths=meta_paths)

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(base_conv_cls(in_size, out_size, num_heads, feat_drop=dropout,
                                                 attn_drop=dropout, activation=activation, allow_zero_in_degree=True,
                                                 **conv_kwargs))
        self.semantic_attention = SemanticAttention(in_size=out_size * num_heads)

    def forward(self, g: dgl.DGLHeteroGraph, h: th.Tensor, get_attention: bool = False,
                res_edge_attn: Optional[Sequence[th.Tensor]] = None):
        semantic_embeddings, edge_attns = [], []

        g = self.get_graph(g, combine=False)

        if len(self.meta_paths) == 1:
            return self.gat_layers[0](g, h).flatten(1)

        for i, meta_path in enumerate(self.meta_paths):
            g_ = g[meta_path]
            res = self.gat_layers[i](g_, h, res_edge_attn=None if res_edge_attn is None else res_edge_attn[i],
                                     get_attention=get_attention)
            if get_attention:
                res, edge_attn = res
                edge_attns.append(edge_attn)
            semantic_embeddings.append(res.flatten(1))
        semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        if get_attention:
            return self.semantic_attention(semantic_embeddings), edge_attns
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class IHAConv(InformedMPConv):
    """
    Informed HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size: int, out_size: int, num_heads: int, node_feat_size: int,
                 dropout: float = 0.6, static_embeddings: bool = True, base_conv_cls=MGATConv, num_layers: int = 2,
                 convolve_embeddings: bool = True, activation=F.elu, semantic_attn_size: int = 128,
                 sparsity: float = 0, **layer_kwargs):
        super(IHAConv, self).__init__(meta_paths=meta_paths, in_size=in_size, num_heads=num_heads,
                                      static_embeddings=static_embeddings, num_layers=num_layers,
                                      node_feat_size=node_feat_size, convolve_embeddings=convolve_embeddings,
                                      sparsity=sparsity)
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(base_conv_cls(in_size, out_size, num_heads, feat_drop=dropout, attn_drop=dropout,
                                                 allow_zero_in_degree=True, activation=activation, **layer_kwargs))
        if type(semantic_attn_size) == int or semantic_attn_size is None:
            self.semantic_attention = SemanticAttention(in_size=out_size * num_heads, hidden_size=semantic_attn_size)
        else:
            self.semantic_attention = None
            assert semantic_attn_size in ('mean', 'max'), 'Please choose mean or max for meta-path fusion'
        self.num_heads = num_heads
        self.semantic_attn_size = semantic_attn_size
        self.out_size = out_size

    def forward(self, g: dgl.DGLHeteroGraph, h: Union[th.Tensor, Dict[str, th.Tensor]], all_node_feat: th.Tensor,
                get_attention: bool = False, res_edge_attn: Optional[Sequence[th.Tensor]] = None):
        def _fuse(embs: Sequence[th.Tensor]):
            embs = th.stack(embs, dim=1)  # (N, M, D * K)
            if self.semantic_attention is not None:
                return self.semantic_attention(embs)
            if self.semantic_attn_size == 'mean':
                return th.mean(embs, dim=-2)
            if self.semantic_attn_size == 'max':
                return th.max(embs, dim=-2)[0]

        embeddings_dict = {ntype: [] for ntype in g.ntypes}
        res_edge_attns = [None] * len(self.gat_layers) if res_edge_attn is None else res_edge_attn

        if self._cached_combined_graph is None or self._cached_graph != g or not self.static_embeddings:
            node_embeddings = [self._get_edge_emb(g, all_node_feat, mp) for mp in self.meta_paths]
        else:
            node_embeddings = None
        mgs: Dict[Tuple[str], dgl.DGLHeteroGraph] = self.get_graph(g, node_embeddings=node_embeddings, combine=False)

        for i, mp in enumerate(self.meta_paths):
            g_: dgl.DGLHeteroGraph = mgs[mp]
            edge_embeddings = g_.edata['feat']
            ntype = g.to_canonical_etype(mp[-1])[-1]
            if len(h) > g_.number_of_nodes():
                h_ = h[dgl.to_homogeneous(g).ndata[dgl.NTYPE] == g.ntypes.index(ntype)]
            else:
                h_ = h

            res = self.gat_layers[i](g_, h_, edge_embeddings, res_edge_attn=res_edge_attns[i],
                                     get_attention=get_attention)
            if get_attention:
                res, res_edge_attn = res
                res_edge_attns[i] = res_edge_attn

            embeddings_dict[ntype].append(res.flatten(1))

        embeddings_dict = {k: _fuse(v) for k, v in embeddings_dict.items() if v}
        if len(embeddings_dict) == 1 and len(h) < g.number_of_nodes():
            semantic_embeddings = list(embeddings_dict.values())[0]
        else:
            semantic_embeddings = th.zeros((g.number_of_nodes(), self.out_size * self.num_heads), device=g.device)
            for ntype, embs in embeddings_dict.items():
                semantic_embeddings[dgl.to_homogeneous(g).ndata[dgl.NTYPE] == g.ntypes.index(ntype)] = embs

        if get_attention:
            return semantic_embeddings, res_edge_attns
        return semantic_embeddings  # (N, D * K)

