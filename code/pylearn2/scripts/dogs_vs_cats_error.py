"""
Computes error rates on the unofficial train/valid/test split of the Dogs vs.
Cats train set
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2015, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

import argparse
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.config import yaml_parse


def compute_error(model, dataset):
    """
    Computes error rate of a given model on a given DogsVsCats dataset instance

    Parameters
    ----------
    model : pylearn2.models.mlp.MLP
        Trained model
    dataset : DogsVsCats
        An instance of the DogsVsCats dataset

    Returns
    -------
    rval : float
        Error rate
    """
    X = model.get_input_space().make_theano_batch('X')
    Y = model.get_output_space().make_theano_batch('Y')
    Y_hat = model.fprop(X)
    errors = T.neq(Y_hat.argmax(axis=1), Y.argmax(axis=1)).sum()
    f = theano.function([X, Y], errors)

    iterator = dataset.iterator(batch_size=8, mode='sequential',
                                data_specs=model.cost_from_X_data_specs())
    errs = 0.0
    for np_X, np_Y in iterator:
        errs += f(np_X, np_Y)
    return errs / dataset.get_num_examples()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the pickled model")
    args = parser.parse_args()

    model = serial.load(args.model_path)

    train_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    train_proxy.keywords['start'] = 0
    train_proxy.keywords['stop'] = 20000
    train_set = yaml_parse._instantiate(train_proxy)
    print "Train error rate is " + str(compute_error(model, train_set))

    valid_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    valid_proxy.keywords['start'] = 20000
    valid_proxy.keywords['stop'] = 22500
    valid_set = yaml_parse._instantiate(valid_proxy)
    print "Valid error rate is " + str(compute_error(model, valid_set))

    test_proxy = yaml_parse.load(model.dataset_yaml_src, instantiate=False)
    test_proxy.keywords['start'] = 22500
    test_proxy.keywords['stop'] = 25000
    test_set = yaml_parse._instantiate(test_proxy)
    print "Test error rate is  " + str(compute_error(model, test_set))
