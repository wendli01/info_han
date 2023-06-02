# Soft and Informed Heterogeneous Attention Networks for Meta-Path Based Learning

This repository accompanies the CIKM 2023 paper [Soft and Informed Heterogeneous Attention Networks for Meta-Path Based Learning]().

It contains all code as well as experimental setups described in the paper including results with all visualizations as standalone `jupyter` notebooks.


If you use code, data or any results in this repository, please cite:

```bibtex
@inproceedings{
}
```

## Experiments

Complete experiments are stored in the notebooks for [node classification](experiments/node_classification.ipynb) and [ablation studies](experiments/ablation_studies.ipynb).

## Dataset

We use datasets from the [Heterogeneous Graph Benchmark](https://github.com/THUDM/HGB) introduced in [Are we really making much progress?: Revisiting, benchmarking and refining heterogeneous graph neural networks](https://dl.acm.org/doi/abs/10.1145/3447548.3467350) by Lv et al.

## Installation


Installation via the provided conda envirionment is encouraged.

> `conda env create -f info_han.yml`


To replicate the experiments, [`jupyter`](https://jupyter.org/install) needs to be installed as well, e.g. with


> `conda install -c conda-forge notebook`
> 
> or 
> 
> `pip install jupyterlab`


## Usage


All models and transformers are implemented as `sklearn` compatible Estimators.


```python
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src import model, hgb_data, baseline, evaluate, layers


imdb_graph, imdb_labels, imdb_mask = hgb_data.load_graph('data/IMDB')

ihan = model.HAGNN([["md", "dm"], ["ma", "am"]], conv_cls = layers.IHAConv, normalize=True, feat_proj_size=64, 
                   label_smoothing=0.05, num_heads=(16,16,16), semantic_attn_size=64)

train_ind, test_ind = train_test_split(range(len(imdb_labels)))
ihan.fit(imdb_graph, imdb_labels[train_ind], node_ids=imdb_mask[train_ind])
y_pred = ihan.predict(imdb_graph)[imdb_mask][test_ind]
f1_score(y[test_ind], y_pred, average='macro')
```
