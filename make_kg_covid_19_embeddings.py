from typing import Dict
from embiggen.transformers import GraphPartitionTransformer
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen.embedders import CBOW, SkipGram, GloVe
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sanitize_ml_labels import sanitize_ml_labels


def report(y_true, y_pred) -> Dict:
    metrics = (roc_auc_score, average_precision_score, accuracy_score)
    return {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in metrics
    }


paths = [
    "neg_test_edges",
    "neg_train_edges",
    "neg_valid_edges",
    "pos_train_edges",
    "pos_test_edges",
    "pos_valid_edges",
]
_graphs = {
    path: EnsmallenGraph(
        edge_path=f"./edges/{path}.tsv",
        sources_column="subject",
        destinations_column="object",
        edge_types_column="edge_label",
        default_edge_type="biolink:interacts_with",
        directed=False,
    )
    for path in paths
}

X = tf.ragged.constant(
    _graphs["pos_train_edges"].walk(20, 100))
embedder_model = SkipGram()
embedder_model.fit(
    X,
    _graphs["pos_train_edges"].get_nodes_number())

transformer_model = GraphPartitionTransformer()
transformer_model.fit(embedder_model.embedding)
X_train, y_train = transformer_model.transform(
    _graphs["pos_train_edges"],
    _graphs["neg_train_edges"]
)
X_test, y_test = transformer_model.transform(
    _graphs["pos_test_edges"],
    _graphs["neg_test_edges"]
)
X_validation, y_validation = transformer_model.transform(
    _graphs["pos_validation_edges"],
    _graphs["neg_validation_edges"]
)

forest = RandomForestClassifier(max_depth=20)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
y_validation_pred = forest.predict(X_validation)

print({
    "train": report(y_train, y_train_pred),
    "test": report(y_test, y_test_pred),
    "validation": report(y_validation, y_validation_pred),
})
