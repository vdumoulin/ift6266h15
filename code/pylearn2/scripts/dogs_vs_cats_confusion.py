"""
Computes confusion matrices the unofficial train/valid/test split of the Dogs
vs. Cats train set.

usage: dogs_vs_cats_confusion.py [-h] model_path

positional arguments:
  model_path  path to the pickled model

optional arguments:
  -h, --help  show this help message and exit
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2015, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import argparse
import numpy
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.config import yaml_parse


def compute_confusion_matrix(model, dataset):
    """
    Computes confusion matrix of a given model on a given DogsVsCats dataset
    instance

    Parameters
    ----------
    model : pylearn2.models.mlp.MLP
        Trained model
    dataset : DogsVsCats
        An instance of the DogsVsCats dataset

    Returns
    -------
    rval : numpy.ndarray
        Confusion matrix
    """
    X = model.get_input_space().make_theano_batch('X')
    Y = model.get_output_space().make_theano_batch('Y')
    Y_hat = model.fprop(X)
    f = theano.function([X], Y_hat)

    iterator = dataset.iterator(batch_size=8, mode='sequential',
                                data_specs=model.cost_from_X_data_specs())
    confusion_matrix = numpy.zeros((dataset.y_labels, dataset.y_labels))
    for np_X, np_Y in iterator:
        np_Y_hat = f(np_X)
        numpy.add.at(confusion_matrix,
                     [np_Y.argmax(axis=1), np_Y_hat.argmax(axis=1)], 1)
    return confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the pickled model")
    args = parser.parse_args()

    model = serial.load(args.model_path)

    train_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    train_proxy.keywords['start'] = 0
    train_proxy.keywords['stop'] = 20000
    train_set = yaml_parse._instantiate(train_proxy)
    train_confusion_matrix = compute_confusion_matrix(model, train_set)
    print "Train confusion matrix is "
    print train_confusion_matrix

    valid_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    valid_proxy.keywords['start'] = 20000
    valid_proxy.keywords['stop'] = 22500
    valid_set = yaml_parse._instantiate(valid_proxy)
    valid_confusion_matrix = compute_confusion_matrix(model, valid_set)
    print "Valid confusion matrix is "
    print valid_confusion_matrix

    test_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    test_proxy.keywords['start'] = 22500
    test_proxy.keywords['stop'] = 25000
    test_set = yaml_parse._instantiate(test_proxy)
    test_confusion_matrix = compute_confusion_matrix(model, test_set)
    print "Test confusion matrix is  "
    print test_confusion_matrix
