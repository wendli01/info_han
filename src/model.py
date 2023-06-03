"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import datetime
import os
from typing import Tuple, Optional, Sequence, Dict, Union

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src import layers


def _check_metapaths(metapaths: Sequence[Sequence[str]], g: dgl.DGLHeteroGraph):
    def _check_metapath(mp: Sequence[str], g: dgl.DGLHeteroGraph):
        cmp = [g.to_canonical_etype(etype) for etype in mp]
        assert cmp[0][0] == cmp[-1][-1], 'The first and last node type of the metapath should be the same.'
        for i in range(len(cmp) - 1):
            assert cmp[i][-1] == cmp[i + 1][0], \
                'The source node type of the {}-th edge type should be the same as the destination ' \
                'node type of the {}-th edge type.'.format(i, i + 1)

    for mp in metapaths:
        _check_metapath(mp, g)


class HAN(nn.Module):
    def __init__(self, metapaths: Sequence[str], in_sizes: Dict[str, int], hidden_size: int, out_size: int,
                 num_heads: Tuple[int], dropout: float, conv_cls=layers.HAConv, softmax: bool = False,
                 normalize: bool = False, full_feature_set: bool = False, residual: bool = True,
                 feat_proj_size: int = None, project_all_features: bool = False, nfeat_attr: str = 'feat',
                 **conv_kwargs):
        super(HAN, self).__init__()
        self.normalize = normalize
        self.residual = residual
        self.softmax = softmax
        self.metapaths = metapaths
        self.hidden_size, self.out_size, self.num_heads = hidden_size, out_size, num_heads
        self.layers = nn.ModuleList()
        if self.normalize:
            self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size * nh) for nh in num_heads[:-1]])
        self.in_sizes = in_sizes
        self.full_feature_set = full_feature_set
        self.feat_proj_size = feat_proj_size
        self.project_all_features = project_all_features
        self.nfeat_attr = nfeat_attr

        in_size = feat_proj_size if feat_proj_size is not None else list(in_sizes.values())[0]
        layer_in_size = in_size
        activation = None if self.residual else F.elu
        for layer_num_heads in num_heads:
            if issubclass(conv_cls, layers.InformedMPConv):
                conv_cls: layers.InformedMPConv
                layer = conv_cls(metapaths, layer_in_size, hidden_size, num_heads=layer_num_heads, dropout=dropout,
                                 node_feat_size=in_size, activation=activation, **conv_kwargs)

            elif issubclass(conv_cls, layers.GeneralMPConv):
                conv_cls: layers.GeneralMPConv
                layer = conv_cls(metapaths, layer_in_size, hidden_size, num_heads=layer_num_heads, dropout=dropout,
                                 **conv_kwargs, activation=activation)

            else:
                conv_cls: GATConv
                layer = conv_cls(layer_in_size, hidden_size, num_heads=layer_num_heads, feat_drop=dropout,
                                 attn_drop=dropout, activation=activation, **conv_kwargs)

            self.layers.append(layer)
            layer_in_size = hidden_size * layer_num_heads

        if hasattr(self.layers[-1], 'num_heads') or isinstance(self.layers[-1], GATConv):
            layer_output_size = hidden_size * num_heads[-1]
        else:
            layer_output_size = hidden_size
        self.predict = nn.Linear(layer_output_size, out_size)
        if self.softmax:
            self.predict = nn.Sequential(self.predict, nn.Softmax(dim=1))

        if self.residual:
            self.res_fc = nn.Linear(in_size, layer_output_size, bias=False)

        self.feat_proj_layers_ = nn.ModuleList()
        for in_size in in_sizes.values():
            if in_size == 1:
                self.feat_proj_layers_.append(None)
            elif self.project_all_features or in_size != self.feat_proj_size:
                self.feat_proj_layers_.append(nn.Linear(in_size, self.feat_proj_size))
            else:
                self.feat_proj_layers_.append(nn.Identity())

    def forward(self, g, node_mask=None, y=None, predict: bool = True, train_mask=None,
                filter_node_types: bool = True):
        target_ntype = g.to_canonical_etype(self.metapaths[0][0])[0]
        target_ntype_id = np.where(np.array(g.ntypes) == target_ntype)[0][0]
        is_ntype = dgl.to_homogeneous(g).ndata['_TYPE'] == target_ntype_id

        proj_nfeat = {}
        for proj, ntype in zip(self.feat_proj_layers_, g.ndata[self.nfeat_attr]):
            if proj is None:
                proj_nfeat[ntype] = th.zeros([len(g.ndata[self.nfeat_attr][ntype]), self.feat_proj_size],
                                             device=g.device)
            else:
                proj_nfeat[ntype] = proj(g.ndata[self.nfeat_attr][ntype])

        all_node_feat = th.vstack(list(proj_nfeat.values()))
        h = all_node_feat if self.full_feature_set else proj_nfeat[target_ntype]
        del proj_nfeat

        for layer_ind, gnn in enumerate(self.layers):
            resval = h
            if isinstance(gnn, layers.InformedMPConv):
                gnn(g, th.flatten(h, 1), all_node_feat=all_node_feat, get_attention=False)
            else:
                h = gnn(g, th.flatten(h, 1), get_attention=False)

            if self.residual:
                if len(h) > len(resval):
                    h = h[is_ntype]

                h = h + (self.res_fc(resval) if resval.shape != h.shape else resval)

            h = F.elu(h)

            if self.normalize and layer_ind < len(self.norm_layers):
                h = self.norm_layers[layer_ind](h)

        if filter_node_types and len(h) != is_ntype.sum():
            h = h[is_ntype]

        if node_mask is not None:
            h = h[node_mask]

        if len(h.shape) == 3:
            h = th.max(h, 1)[0]

        if not predict:
            return h

        return self.predict(h) if self.softmax else th.sigmoid(self.predict(h))


def score(logits, labels, loss_fn):
    with th.no_grad():
        loss = loss_fn(logits, labels)
    _, indices = th.max(logits, dim=1)
    prediction = indices.long().cpu().numpy() if len(labels.shape) == 1 else (logits > 0.5).cpu().numpy()
    labels = labels.long().cpu().numpy()

    accuracy = accuracy_score(labels, prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return loss, accuracy, micro_f1, macro_f1


class EarlyStopping(object):
    def __init__(self, patience: int = 100, verbose: bool = False):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Reverting from validation loss {round(loss, 4)} to {round(self.best_loss, 4)}")
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        th.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model, remove_checkpoint: bool = True):
        """Load the latest checkpoint."""
        model.load_state_dict(th.load(self.filename))
        if remove_checkpoint:
            os.remove(self.filename)


class HAGNN():
    def __init__(self, metapaths, full_feature_set: bool = False, feat_proj_size: int = None,
                 hidden_size: int = 8, nb_epochs: int = 300, lr: float = 0.005, weight_decay: float = 0.001,
                 num_heads: Tuple[int] = (8, 8), dropout: float = 0.6, device: str = 'cuda:7', verbose: bool = False,
                 conv_cls: Union[layers.InformedMPConv, layers.GeneralMPConv] = layers.HAConv,
                 random_state=42, patience: int = 100, val_size: Optional[float] = 0.2, nfeat_attr: str = 'feat',
                 label_smoothing: Optional[float] = None, **model_kwargs):
        self.optimizer_ = None
        self.stopper_ = None
        self.scheduler_ = None
        self.metapaths = metapaths
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.conv_cls = conv_cls
        self.model_kwargs = model_kwargs
        self.random_state = random_state
        self.model_ = None
        self.patience = patience
        self.val_size = val_size
        self.multi_label_ = None
        self.full_feature_set = full_feature_set
        self.feat_proj_size = feat_proj_size
        self.nfeat_attr = nfeat_attr
        self.training_history_, self.run_ = [], -1
        self.label_smoothing = 0 if label_smoothing is None else label_smoothing

    def partial_fit(self, graph: dgl.DGLHeteroGraph, y_train: np.ndarray, nb_epochs, node_ids=None):
        if self.metapaths is not None:
            _check_metapaths(self.metapaths, graph)

        self.multi_label_ = len(y_train.shape) > 1

        y_train = th.tensor(y_train).float() if self.multi_label_ else th.tensor(y_train).long()

        node_ids = range(len(y_train)) if node_ids is None else node_ids
        if self.val_size is not None and self.val_size > 0:
            train_id, val_id, y_train, y_val = train_test_split(range(len(node_ids)), y_train,
                                                                random_state=self.random_state, test_size=self.val_size,
                                                                stratify=None if self.multi_label_ else y_train.cpu())
            y_val = y_val.to(self.device)
        else:
            train_id = range(len(node_ids))

        num_classes = y_train.shape[1] if self.multi_label_ else len(th.unique(y_train))
        if self.feat_proj_size is not None and self.feat_proj_size > 0:
            feature_sizes = {ntype: feat.shape[1] for ntype, feat in graph.ndata[self.nfeat_attr].items()}
            feat_proj_size = self.feat_proj_size
        else:
            feature_sizes = {ntype: graph.ndata[self.nfeat_attr][ntype].shape[1] for ntype, feat in
                             graph.ndata[self.nfeat_attr].items()}
            target_node_type = graph.to_canonical_etype(self.metapaths[0][0])[0]
            feat_proj_size = feature_sizes[target_node_type]

        if self.model_ is None:
            self.model_ = HAN(metapaths=self.metapaths, in_sizes=feature_sizes, hidden_size=self.hidden_size,
                              out_size=num_classes, num_heads=self.num_heads, dropout=self.dropout,
                              feat_proj_size=feat_proj_size, conv_cls=self.conv_cls, nfeat_attr=self.nfeat_attr,
                              project_all_features=self.feat_proj_size is not None and self.feat_proj_size > 0,
                              full_feature_set=self.full_feature_set, **self.model_kwargs).to(self.device)

        if self.verbose:
            total_trainable_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            print("total trainable params: {:d}".format(total_trainable_params))

        loss_fn = th.nn.BCELoss() if self.multi_label_ else \
            th.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        if self.stopper_ is None:
            self.stopper_ = EarlyStopping(patience=self.patience, verbose=self.verbose)
        if self.optimizer_ is None:
            self.optimizer_ = th.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        graph, y_train = graph.to(self.device), y_train.to(self.device)

        for epoch in range(nb_epochs):
            if self.label_smoothing and self.multi_label_:
                with th.no_grad():
                    y_train_ = y_train.clamp(self.label_smoothing, 1 - self.label_smoothing)
            else:
                y_train_ = y_train
            self.model_.train()
            logits = self.model_(graph, node_mask=node_ids)

            loss = loss_fn(logits[train_id], y_train_)

            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()
            self.model_.eval()

            # print info in each batch
            _, train_acc, train_micro_f1, train_macro_f1 = score(logits[train_id], y_train, loss_fn)

            self.training_history_.append(dict(epoch=epoch, loss=loss.detach().cpu().item(), train_acc=train_acc,
                                               train_macro_f1=train_macro_f1, train_micro_f1=train_micro_f1,
                                               run=self.run_))

            if self.val_size is not None and self.val_size > 0:
                val_loss, val_acc, val_micro_f1, val_macro_f1 = score(logits[val_id], y_val, loss_fn)
                early_stop = self.stopper_.step(val_loss.data.item(), val_acc, self.model_)

                if early_stop:
                    break

            if self.val_size is not None and self.val_size > 0:
                self.training_history_[-1]['val_macro_f1'] = val_macro_f1
                self.training_history_[-1]['val_micro_f1'] = val_micro_f1
                self.training_history_[-1]['val_acc'] = val_acc
                self.training_history_[-1]['val_loss'] = val_loss.detach().cpu().item()

            if self.verbose:
                s = "Epoch {:d} | loss: {:.4f} | train_acc: {:.3f} | train_macro_f1: {:.3f}".format(
                    epoch + 1, loss, train_acc, train_micro_f1, train_macro_f1)
                if self.val_size is not None and self.val_size > 0:
                    s += " | val_macro_f1: {:.3f}".format(val_macro_f1) + " | val_loss: {:.4f}".format(val_loss)

                print(s)
            self.scheduler_.step()

        return self

    def fit(self, graph: dgl.DGLHeteroGraph, y_train: np.ndarray, node_ids=None):
        self._set_random_state()
        self.model_, self.stopper_, self.scheduler_, self.optimizer_ = None, None, None, None
        self.run_ += 1
        self.partial_fit(graph, y_train, node_ids=node_ids, nb_epochs=self.nb_epochs)

        if self.val_size is not None and self.val_size > 0:
            self.stopper_.load_checkpoint(self.model_)

        th.cuda.empty_cache()
        return self

    def predict_proba(self, graph: dgl.DGLHeteroGraph):
        self.model_.eval()
        with th.no_grad():
            return self.model_(graph.to(self.device))

    def predict(self, graph: dgl.DGLHeteroGraph):
        probas = self.predict_proba(graph)
        return (probas > 0.5).cpu().numpy() if self.multi_label_ else probas.argmax(dim=1).cpu().numpy()

    def transform(self, graph: dgl.DGLHeteroGraph, filter_node_types: bool = True) -> np.ndarray:
        self.model_.eval()
        with th.no_grad():
            return self.model_(graph.to(self.device), predict=False)

    def _set_random_state(self):
        def _get_rng(random_state):
            if random_state is None:
                return np.random.RandomState()
            if type(random_state) is np.random.RandomState:
                return random_state
            return np.random.RandomState(random_state)

        self.rng_ = _get_rng(self.random_state)
        th.use_deterministic_algorithms(True)
        if self.device != 'cpu':
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if self.random_state is not None:
            th.cuda.manual_seed_all(self.random_state)
            dgl.random.seed(self.random_state)
            th.manual_seed(self.random_state)
