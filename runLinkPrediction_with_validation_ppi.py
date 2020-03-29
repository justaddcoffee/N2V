import argparse
from xn2v import CSFGraph
from xn2v import N2vGraph
from xn2v.word2vec import SkipGramWord2Vec
from xn2v.word2vec import ContinuousBagOfWordsWord2Vec
from xn2v import LinkPredictionWithValidation
from xn2v.utils import write_embeddings
import sys


# import os
# import logging
#
# handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "link_prediction.log"))
# formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s')
# handler.setFormatter(formatter)
# log = logging.getLogger()
# log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
# log.addHandler(handler)


def parse_args():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser(description="Run link Prediction using training/validation and test sets.")

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

    parser.add_argument('--embed_graph', nargs='?', default='embedded_graph.embedded',
                        help='Embeddings path of the positive training graph')

    parser.add_argument('--edge_embed_method', nargs='?', default='hadamard',
                        help='Embeddings embedding method of the positive training graph. '
                             'It can be hadamard, weightedL1, weightedL2 or average')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--gamma', type=float, default=1,
                        help='hyperparameter for jumping from one network to another network '
                             'in heterogeneous graphs. Default is 1.')

    parser.add_argument('--useGamma', dest='useGamma', action='store_false', help="True if the graph is heterogeneous, "
                                                                                  "False if the graph is homogeneous.")
    parser.set_defaults(useGamma=False)
    parser.add_argument('--classifier', nargs='?', default='LR',
                        help="Binary classifier for link prediction, it should be either LR, RF or SVM")

    parser.add_argument('--type', nargs='?', default='homogen',
                        help="Type of graph which is either homogen for homogeneous graph or heterogen for "
                             "heterogeneous graph")

    parser.add_argument('--w2v-model', nargs='?', default='Skipgram',
                        help="word2vec model. It can be either Skipgram or CBOW")

    parser.add_argument('--num_steps', type=int, default=100000,
                        help='number of steps for GD.  Default is 100000.')

    return parser.parse_args()


def learn_embeddings(walks, pos_train_graph, w2v_model):
    """
    Learn embeddings by optimizing the Skipgram or CBOW objective using SGD.
    """

    worddictionary = pos_train_graph.get_node_to_index_map()
    reverse_worddictionary = pos_train_graph.get_index_to_node_map()

    if w2v_model == "Skipgram":
        model = SkipGramWord2Vec(walks, worddictionary=worddictionary,
                                 reverse_worddictionary=reverse_worddictionary, num_steps=args.num_steps)
    elif w2v_model == "CBOW":
        model = ContinuousBagOfWordsWord2Vec(walks, worddictionary=worddictionary,
                                             reverse_worddictionary=reverse_worddictionary, num_steps=args.num_steps)
    else:
        print("[ERROR] enter Skipgram or CBOW")
        sys.exit(1)

    model.train(display_step=100)

    write_embeddings(args.embed_graph, model.embedding, reverse_worddictionary)


def linkpred(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph):
    """
    :param pos_train_graph: positive training graph
    :param pos_valid_graph: positive validation graph
    :param pos_test_graph: positive test graph
    :param neg_train_graph: negative training graph
    :param neg_valid_graph: negative validation graph
    :param neg_test_graph: negative test graph
    :return: Metrics of logistic regression as the results of link prediction
    """
    lp = LinkPredictionWithValidation(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph,
                                      neg_valid_graph,
                                      neg_test_graph, args.embed_graph, args.edge_embed_method, args.classifier,
                                      args.type)

    lp.prepare_training_validation_test_labels()
    lp.predict_links()
    lp.output_classifier_results()
    lp.output_edge_node_information()
    lp.predicted_ppi_links()
    lp.predicted_ppi_non_links()


def read_graphs():
    """
    Reads pos_train, pos_validation, pos_test, neg_train, neg_validation and neg_test edges with CSFGraph
    :return: pos_train, pos-validation, pos_test, neg_train, neg_validation and neg_test graphs in CSFGraph format
    """

    pos_train_graph = CSFGraph(args.pos_train)
    pos_valid_graph = CSFGraph(args.pos_valid)
    pos_test_graph = CSFGraph(args.pos_test)
    neg_train_graph = CSFGraph(args.neg_train)
    neg_valid_graph = CSFGraph(args.neg_valid)
    neg_test_graph = CSFGraph(args.neg_test)
    return pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph


def main(args):
    """
    The input files are positive training, positive validation, positive test, negative training, negative validation
     and negative test edges. The code reads the files and create graphs in CSFGraph format.
     Then, the positive training graph is embedded.
     Finally, link prediction is performed.

    :param args: parameters of node2vec and link prediction
    :return: Results of link prediction
    """
    print("[INFO]: p={}, q={}, classifier= {}, useGamma={}, word2vec_model={}, num_steps={}"
          .format(args.p, args.q, args.classifier, args.useGamma, args.w2v_model, args.num_steps))
    pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph = read_graphs()
    pos_train_g = N2vGraph(pos_train_graph, args.p, args.q, args.gamma, args.useGamma)
    walks = pos_train_g.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, pos_train_graph, args.w2v_model)
    linkpred(pos_train_graph, pos_valid_graph, pos_test_graph, neg_train_graph, neg_valid_graph, neg_test_graph)


if __name__ == "__main__":
    args = parse_args()
    main(args)
