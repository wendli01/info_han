import time
from typing import Sequence, Union, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def cv_score_node_classification(estimator, graph, y, node_mask: Optional[np.ndarray] = None,
                                 cv=StratifiedShuffleSplit(random_state=42, train_size=.3),
                                 scorers=(accuracy_score, f1_micro, f1_macro), verbose: Union[bool, int] = False):
    scores = []
    if node_mask is None:
        node_mask = np.arange(len(y))
    if node_mask.dtype == bool:
        node_mask = np.arange(len(y))[node_mask]

    for random_seed, (train_ind, test_ind) in enumerate(cv.split(y, y)):
        if hasattr(estimator, 'random_state'):
            estimator.random_state = random_seed
        start = time.time()
        estimator.fit(graph, y[train_ind], node_ids=node_mask[train_ind])
        fit_time, start = time.time() - start, time.time()
        y_pred = estimator.predict(graph)[node_mask][test_ind]
        test_time = time.time() - start
        scores.append({scorer.__name__: scorer(y[test_ind], y_pred) for scorer in scorers})
        scores[-1]['fit time'], scores[-1]['test time'] = fit_time, test_time
        if verbose > 0:
            print(', '.join([k + ': ' + str(round(v, 3)) for k, v in scores[-1].items()]))

    return pd.DataFrame(scores)


def evaluate_node_classification_scenarios(estimator, graph, y, node_mask: Optional[np.ndarray] = None,
                                           train_sizes=(.3,), random_state=42,
                                           verbose: Union[bool, int] = False, n_splits: int = 10,
                                           cv_cls=StratifiedShuffleSplit) -> pd.DataFrame:
    cvs = [cv_cls(random_state=random_state, train_size=train_size, n_splits=n_splits) for train_size in
           train_sizes]
    score_dfs = []
    for cv in cvs:
        score_df: pd.DataFrame = cv_score_node_classification(estimator, graph, y, node_mask=node_mask, cv=cv,
                                                              verbose=verbose - 1)
        score_df['train_size'] = cv.train_size
        score_dfs.append(score_df)
        if verbose > 0:
            print(str(cv.train_size) + ':', score_df.aggregate([np.mean, np.std]).round(4))
    return pd.concat(score_dfs)


def evaluate_node_classifications(estimators: Sequence,
                                  datasets: Union[Sequence[Tuple[dgl.DGLHeteroGraph, np.ndarray]],
                                  Sequence[Tuple[dgl.DGLHeteroGraph, np.ndarray, np.ndarray]]],
                                  dataset_names, verbose: Union[bool, int] = False, **kwargs) -> pd.DataFrame:
    score_dfs = []

    for estimator, dataset, dataset_name in zip(estimators, datasets, dataset_names):
        if len(dataset) == 2:
            (graph, y, node_mask) = dataset + (None,)
        else:
            graph, y, node_mask = dataset

        cv_cls = ShuffleSplit if len(y.shape) > 1 and y.shape[1] > 1 else StratifiedShuffleSplit

        score_df = evaluate_node_classification_scenarios(estimator, graph, y, node_mask, verbose=verbose - 1,
                                                          cv_cls=cv_cls, **kwargs)
        score_df['dataset'] = dataset_name
        if verbose > 0:
            print(dataset_name + ':', score_df.groupby('train_size').aggregate([np.mean, np.std]).round(4))

        score_dfs.append(score_df)

    return pd.concat(score_dfs)
