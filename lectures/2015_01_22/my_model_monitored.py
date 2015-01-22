import numpy
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import sharedX


class MLPCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()


class MLP(Model):
    def __init__(self, nvis, nhid, nclasses):
        super(MLP, self).__init__()

        self.nvis, self.nhid, self.nclasses = nvis, nhid, nclasses

        self.W = sharedX(numpy.random.normal(scale=0.01,
                                             size=(self.nvis, self.nhid)),
                         name='W')
        self.b = sharedX(numpy.zeros(self.nhid), name='b')
        self.V = sharedX(numpy.random.normal(scale=0.01,
                                             size=(self.nhid, self.nclasses)),
                         name='V')
        self.c = sharedX(numpy.zeros(self.nclasses), name='c')
        self._params = [self.W, self.b, self.V, self.c]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    def fprop(self, inputs):
        H = T.nnet.sigmoid(T.dot(inputs, self.W) + self.b)
        return T.nnet.softmax(T.dot(H, self.V) + self.c)

    def get_monitoring_data_specs(self):
        space = CompositeSpace([self.get_input_space(),
                                self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.fprop(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()

        return OrderedDict([('error', error)])
