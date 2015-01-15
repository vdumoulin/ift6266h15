"""
Solution to the first assignment of IFT6266 - Winter 2015
"""
import cPickle
import gzip
import itertools
import numpy

numpy.random.seed(1234)


def compute_H(X, W, b):
    """
    Computes H = sigmoid(X . W + b) corresponding to the hidden unit
    activations of a one-hidden-layer MLP classifier

    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    """
    return 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W) + b)))


def compute_Y(H, V, d):
    """
    Computes Y = softmax(H . V + d) corresponding to the output probabilities
    of a one-hidden-layer MLP classifier

    Parameters
    ----------
    H : numpy.ndarray
        Batch of hidden unit activations of shape (batch_size, num_hid)
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )
    """
    # Numerically stable version of the softmax activation function
    activations = numpy.dot(H, V) + d
    Y_tilde = numpy.exp(activations - activations.max(axis=1, keepdims=True))
    return Y_tilde / Y_tilde.sum(axis=1, keepdims=True)


def compute_loss(Y, T):
    """
    Computes the binary cross-entropy loss of an MLP classifier

    Parameters
    ----------
    Y : numpy.ndarray
        Batch of output probabilities of shape (batch_size, num_classes)
    T : numpy.ndarray
        Batch of one-hot encoded targets of shape (batch_size, num_classes)
    """
    return -(T * numpy.log(Y)).sum(axis=1).mean(axis=0)


def compute_error(Y, T):
    """
    Computes the misclassification rate of an MLP classifier

    Parameters
    ----------
    Y : numpy.ndarray
        Batch of output probabilities of shape (batch_size, num_classes)
    T : numpy.ndarray
        Batch of one-hot encoded targets of shape (batch_size, num_classes)
    """
    return numpy.not_equal(T.argmax(axis=1), Y.argmax(axis=1)).mean(axis=0)


def fprop(X, W, b, V, d):
    """
    Does the forward-prop on an MLP classifier

    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )

    Returns
    -------
    Y : numpy.ndarray
        Batch of probability vectors of shape (batch_size, num_classes)
    """
    H = compute_H(X, W, b)
    return compute_Y(H, V, d)


def compute_analytical_gradients(X, T, W, b, V, d):
    """
    Analytically computes the gradient of the loss function of an MLP
    classifier with respect to its parameters

    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    T : numpy.ndarray
        Batch of one-hot encoded targets of shape (batch_size, num_classes)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )

    Returns
    -------
    rval : tuple of numpy.ndarray
        Tuple of gradients for W, b, V and d respectively
    """
    # The gradient expressions implemented in this function assume that T
    # is a one-hot encoded batch.
    numpy.testing.assert_allclose(
        T.sum(axis=1), numpy.ones_like(T.sum(axis=1)))

    H = compute_H(X, W, b)
    Y = compute_Y(H, V, d)

    # Batch version of numpy.outer(h, y - t)
    V_grad = numpy.dot(H.T, Y - T) / H.shape[0]
    # Batch version of y - t
    d_grad = (Y - T).mean(axis=0)
    # Batch version of numpy.outer(x, numpy.dot(y - t, V.T) * h * (1 - h))
    W_grad = numpy.dot(X.T, numpy.dot(Y - T, V.T) * H * (1 - H)) / X.shape[0]
    # Batch version of numpy.dot(y - t, V.T) * h * (1 - h)
    b_grad = (numpy.dot(Y - T, V.T) * H * (1 - H)).mean(axis=0)

    return (W_grad, b_grad, V_grad, d_grad)


def compute_numerical_gradients(X, T, W, b, V, d, delta=1e-5):
    """
    Approximates the gradient of the loss function of an MLP classifier with
    respect to its parameters using a finite differences methods

    Parameters
    ----------
    X : numpy.ndarray
        Batch of examples of shape (batch_size, num_vis)
    T : numpy.ndarray
        Batch of one-hot encoded targets of shape (batch_size, num_classes)
    W : numpy.ndarray
        Weight matrix of shape (num_vis, num_hid)
    b : numpy.ndarray
        Bias vector of shape (num_hid, )
    V : numpy.ndarray
        Weight matrix of shape (num_hid, num_classes)
    d : numpy.ndarray
        Bias vector of shape (num_classes, )

    Returns
    -------
    rval : tuple of numpy.ndarray
        Tuple of approximate gradients for W, b, V and d respectively
    """
    H = compute_H(X, W, b)
    Y = compute_Y(H, V, d)
    loss = compute_loss(Y, T)

    d_tilde = numpy.copy(d)
    d_grad = numpy.zeros_like(d)
    for i in xrange(d.shape[0]):
        d_tilde[i] = d[i] + delta
        Y_tilde = compute_Y(H, V, d_tilde)
        loss_tilde = compute_loss(Y_tilde, T)
        d_grad[i] = (loss_tilde - loss) / delta
        d_tilde[i] = d[i]

    V_tilde = numpy.copy(V)
    V_grad = numpy.zeros_like(V)
    for i, j in itertools.product(xrange(V.shape[0]), xrange(V.shape[1])):
        V_tilde[i, j] = V[i, j] + delta
        Y_tilde = compute_Y(H, V_tilde, d)
        loss_tilde = compute_loss(Y_tilde, T)
        V_grad[i, j] = (loss_tilde - loss) / delta
        V_tilde[i, j] = V[i, j]

    b_tilde = numpy.copy(b)
    b_grad = numpy.zeros_like(b)
    for i in xrange(b.shape[0]):
        b_tilde[i] = b[i] + delta
        H_tilde = compute_H(X, W, b_tilde)
        Y_tilde = compute_Y(H_tilde, V, d)
        loss_tilde = compute_loss(Y_tilde, T)
        b_grad[i] = (loss_tilde - loss) / delta
        b_tilde[i] = b[i]

    W_tilde = numpy.copy(W)
    W_grad = numpy.zeros_like(W)
    for i, j in itertools.product(xrange(W.shape[0]), xrange(W.shape[1])):
        W_tilde[i, j] = W[i, j] + delta
        H_tilde = compute_H(X, W_tilde, b)
        Y_tilde = compute_Y(H_tilde, V, d)
        loss_tilde = compute_loss(Y_tilde, T)
        W_grad[i, j] = (loss_tilde - loss) / delta
        W_tilde[i, j] = W[i, j]

    return (W_grad, b_grad, V_grad, d_grad)


def verify_gradients():
    """
    Does gradient checking of a toy MLP on a toy dataset
    """
    num_examples = 10
    num_vis = 50
    num_hid = 25
    num_classes = 5
    X = numpy.random.uniform(low=0, high=1, size=(num_examples, num_vis))
    W = numpy.random.uniform(low=-0.1, high=0.1, size=(num_vis, num_hid))
    b = numpy.random.uniform(low=-0.1, high=0.1, size=(num_hid,))
    V = numpy.random.uniform(low=-0.1, high=0.1, size=(num_hid, num_classes))
    d = numpy.random.uniform(low=-0.1, high=0.1, size=(num_classes,))
    T = numpy.zeros((num_examples, num_classes))
    for i, t in enumerate(T):
        T[i, i % num_classes] = 1

    analytical_grads = compute_analytical_gradients(X, T, W, b, V, d)
    numerical_grads = compute_numerical_gradients(X, T, W, b, V, d, delta=1e-7)

    atol = 1e-5
    numpy.testing.assert_allclose(
        analytical_grads[0], numerical_grads[0], atol=atol)
    numpy.testing.assert_allclose(
        analytical_grads[1], numerical_grads[1], atol=atol)
    numpy.testing.assert_allclose(
        analytical_grads[2], numerical_grads[2], atol=atol)
    numpy.testing.assert_allclose(
        analytical_grads[3], numerical_grads[3], atol=atol)


def one_hot_encode(y, num_classes):
    """
    Performs a one-hot encoding of a batch of integer targets

    Parameters
    ----------
    y : numpy.ndarray
        Batch of integer targets of shape (batch_size, )
    num_classes : int
        Number of classes

    Returns
    -------
    Y : numpy.ndarray
        One-hot encoded matrix of shape (batch_size, num_classes) corresponding
        to y
    """
    Y = numpy.zeros((y.shape[0], num_classes))
    for i, c in enumerate(y):
        Y[i, c] = 1
    return Y


if __name__ == "__main__":
    verify_gradients()

    # Note: Momentum and learning rate decay aren't strictly necessary to
    #       complete the assignment, they're implemented simply to speed up
    #       things a bit

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        # Dataset-specific parameters
        num_vis = 784
        num_hid = 500
        num_classes = 10

        # Hyperparameters
        batch_size = 100
        num_epochs = 50
        momentum_coefficient = 0.95
        learning_rate = 0.01
        learning_rate_decay = 1 - 1e-5
        min_learning_rate = 1e-5

        # Parameter initialization
        W = numpy.random.uniform(low=-0.01, high=0.01, size=(num_vis, num_hid))
        b = numpy.zeros(num_hid)
        V = numpy.random.uniform(low=-0.01, high=0.01,
                                 size=(num_hid, num_classes))
        d = numpy.zeros(num_classes)

        W_momentum = numpy.zeros_like(W)
        b_momentum = numpy.zeros_like(b)
        V_momentum = numpy.zeros_like(V)
        d_momentum = numpy.zeros_like(d)

        parameters = [W, b, V, d]
        momentum = [W_momentum, b_momentum, V_momentum, d_momentum]

        # Unpack data. Targets are one-hot encoded
        train_set, valid_set, test_set = cPickle.load(f)
        train_X, train_y = train_set
        train_y = one_hot_encode(train_y, num_classes)
        valid_X, valid_y = valid_set
        valid_y = one_hot_encode(valid_y, num_classes)
        test_X, test_y = test_set
        test_y = one_hot_encode(test_y, num_classes)

        num_batches = train_set[0].shape[0] / batch_size
        # Just make sure that we don't accidentally forget some examples
        assert train_set[0].shape[0] % batch_size == 0

        # Training loop
        for epoch in xrange(1, num_epochs + 1):
            for i in xrange(num_batches):
                start = batch_size * i
                stop = batch_size * (i + 1)
                X = train_X[start: stop]
                T = train_y[start: stop]
                gradients = compute_analytical_gradients(X, T, *parameters)
                momentum = [momentum_coefficient * m - learning_rate * g
                            for m, g in zip(momentum, gradients)]
                parameters = [p + m for p, m in zip(parameters, momentum)]
                learning_rate = max(min_learning_rate,
                                    learning_rate * learning_rate_decay)

            train_error = compute_error(fprop(train_X, *parameters), train_y)
            valid_error = compute_error(fprop(valid_X, *parameters), valid_y)
            test_error = compute_error(fprop(test_X, *parameters), test_y)
            print "Epoch " + str(epoch) + ":"
            print "    Train error : " + str(train_error)
            print "    Valid error : " + str(valid_error)
            print "    Test error  : " + str(test_error)
