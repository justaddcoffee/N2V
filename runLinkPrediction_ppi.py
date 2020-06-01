import argparse
from embiggen import Graph, GraphFactory, Embiggen
from embiggen.utils import write_embeddings, serialize, deserialize
from embiggen.neural_networks import MLP, FFNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from cache_decorator import Cache
from sanitize_ml_labels import sanitize_ml_labels
from deflate_dict import deflate
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import time
from tqdm.auto import tqdm
from typing import Dict, List
import compress_json


def parse_args():
    """Parses arguments.

    """
    parser = argparse.ArgumentParser(description="Run link Prediction.")

    parser.add_argument('--pos_train', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_train_edges_max_comp_graph',
                        help='Input positive training edges path')

    parser.add_argument('--pos_valid', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_validation_edges_max_comp_graph',
                        help='Input positive validation edges path')

    parser.add_argument('--pos_test', nargs='?',
                        default='tests/data/ppismall_with_validation/pos_test_edges_max_comp_graph',
                        help='Input positive test edges path')

    parser.add_argument('--neg_train', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_train_edges_max_comp_graph',
                        help='Input negative training edges path')

    parser.add_argument('--neg_valid', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_validation_edges_max_comp_graph',
                        help='Input negative validation edges path')

    parser.add_argument('--neg_test', nargs='?',
                        default='tests/data/ppismall_with_validation/neg_test_edges_max_comp_graph',
                        help='Input negative test edges path')

    parser.add_argument('--output_file', nargs='?',
                        default='output_results.json',
                        help='path to the output file which contains results of link prediction')

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('edges_embedding_method', nargs='?', default='hadamard',
                        help='Embeddings embedding method of the positive training graph. '
                             'It can be hadamard, weightedL1, weightedL2 or average')

    parser.add_argument('--embedding_size', type=int, default=200,
                        help='Number of dimensions which is size of the embedded vectors. Default is 200.')

    parser.add_argument('--walks_length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--walks_number', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--context_window', type=int, default=3,
                        help='Context size for optimization. Default is 3.')

    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of training epochs')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='node2vec p hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='node2vec q hyperparameter. Default is 1.')

    parser.add_argument('--classifier', nargs='?', default='LR',
                        help="Binary classifier for link prediction, it should be either LR, RF, SVM, MLP, FFNN")

    parser.add_argument('--embedding_model', nargs='?', default='Skipgram',
                        help="word2vec model (Skipgram, CBOW, GloVe)")

    return parser.parse_args()


def read_graphs(*paths: List[str], **kwargs: Dict) -> List[Graph]:
    """Return Graphs at given paths.

    These graphs are expected to be without a header.

    Parameters
    -----------------------
    *paths: List[str],
        List of the paths to be loaded.
        Notably, only the first one is fully preprocessed for random walks.
    **kwargs: Dict,

    """

    factory = GraphFactory()
    return [
        factory.read_csv(
            path,
            edge_has_header=False,
            start_nodes_column=0,
            end_nodes_column=1,
            weights_column=2,
            random_walk_preprocessing=i == 0,
            **kwargs
        )
        for i, path in tqdm(enumerate(paths), desc="Loading graphs")
    ]


def get_classifier_model(classifier: str, **kwargs: Dict):
    """Return choen classifier model.

    Parameters
    ------------------
    classifier:str,
        Chosen classifier model. Can either be:
            - LR for LogisticRegression
            - RF for RandomForestClassifier
            - MLP for Multi-Layer Perceptron
            - FFNN for Feed Forward Neural Network
    **kwargs:Dict,
        Keyword arguments to be passed to the constructor of the model.

    Raises
    ------------------
    ValueError,
        When given classifier model is not supported.

    Returns
    ------------------
    An instance of the selected model.
    """
    if classifier == "LR":
        return LogisticRegression(**kwargs)
    if classifier == "RF":
        return RandomForestClassifier(**kwargs)
    if classifier == "MLP":
        return MLP(**kwargs)
    if classifier == "FFNN":
        return FFNN(**kwargs)

    raise ValueError(
        "Given classifier model {} is not supported.".format(classifier)
    )


def performance_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return performance report for given predictions and ground truths.

    Parameters
    ------------------------
    y_true: np.ndarray,
        The ground truth labels.
    y_pred: np.ndarray,
        The labels predicted by the classifier.

    Returns
    ------------------------
    Dictionary with the performance metrics, including AUROC, AUPRC, F1 Score,
    and accuracy.
    """
    # TODO: add confusion matrix
    metrics = roc_auc_score, average_precision_score, f1_score
    report = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in metrics
    }
    report[sanitize_ml_labels(accuracy_score.__name__)] = accuracy_score(
        y_true, np.round(y_pred).astype(int)
    )
    return report


def main(args):
    """
    The input files are positive training, positive test, negative training and negative test edges. The code
    reads the files and create graphs in Graph format. Then, the positive training graph is embedded.
    Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Result of link prediction
    """

    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = read_graphs(
        args.pos_train,
        args.pos_valid,
        args.pos_test,
        args.neg_train,
        args.neg_valid,
        args.neg_test,
        return_weight=1/args.p,
        explore_weight=1/args.q
    )

    embedding = Embiggen()
    embedding.fit(
        pos_train,
        walks_number=args.walks_number,
        walks_length=args.walks_length,
        embedding_model=args.embedding_model,
        epochs=args.epochs,
        embedding_size=args.embedding_size,
        context_window=args.context_window,
        edges_embedding_method=args.edges_embedding_method
    )

    X_train, y_train = embedding.transform(pos_train, neg_train)
    X_test, y_test = embedding.transform(pos_test, neg_test)
    X_valid, y_valid = embedding.transform(pos_valid, neg_valid)

    classifier_model = get_classifier_model(
        args.classifier,
        **(
            dict(input_shape=X_train.shape[1])
            if args.classifier in ("MLP", "FFNN")
            else {}
        )
    )

    if args.classifier in ("MLP", "FFNN"):
        classifier_model.fit(X_train, y_train, X_test, y_test)
    else:
        classifier_model.fit(X_train, y_train)

    return dict(
        train=performance_report(y_train, classifier_model.predict(X_train)),
        test=performance_report(y_test, classifier_model.predict(X_test)),
        valid=performance_report(y_valid, classifier_model.predict(X_valid)),
    )


if __name__ == "__main__":
    args = parse_args()
    report = main(args)
    compress_json.dump(report, args.output_file, json_kwargs=dict(indent=4))
