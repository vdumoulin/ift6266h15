import gzip
import cPickle
import numpy
import theano
import theano.tensor as tensor
from solution import one_hot_encode


# Load data
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)
    train_X, train_y = train_set
    valid_X, valid_y = valid_set
    test_X, test_y = test_set
train_y = one_hot_encode(train_y, 10)
valid_y = one_hot_encode(valid_y, 10)
test_y = one_hot_encode(test_y, 10)

# Instantiate symbolic variables
X = tensor.matrix('X')
T = tensor.matrix('T')
W = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(784, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(500, 10)), 'V')
c = theano.shared(numpy.zeros(10))
params = [W, b, V, c]

# Build computation graph
H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.softmax(tensor.dot(H, V) + c)
loss = -(T * tensor.log(Y)).sum(axis=1).mean()
misclass = tensor.neq(T.argmax(axis=1), Y.argmax(axis=1)).mean()

grads = tensor.grad(loss, params)

# Compile function
updates = dict((param, param - 0.01 * grad)
               for param, grad in zip(params, grads))
f = theano.function(inputs=[X, T], updates=updates)
g = theano.function(inputs=[X, T], outputs=[loss, misclass])

# Call function with numerical values
batch_size = 100
num_batches = train_X.shape[0] / batch_size
for epoch in xrange(10):
    for i in xrange(num_batches):
        numpy_X = train_X[batch_size * i: batch_size * (i + 1)]
        numpy_T = train_y[batch_size * i: batch_size * (i + 1)]
        f(numpy_X, numpy_T)
    print "Epoch " + str(epoch + 1) + ":"
    print "    Train loss/misclass: %0.2f/%0.2f" % tuple(g(train_X, train_y))
    print "    Valid loss/misclass: %0.2f/%0.2f" % tuple(g(valid_X, valid_y))
    print "    Test  loss/misclass: %0.2f/%0.2f" % tuple(g(test_X, test_y))
