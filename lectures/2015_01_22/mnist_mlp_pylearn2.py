from pylearn2.costs.cost import MethodCost
from pylearn2.datasets.mnist import MNIST
from pylearn2.models.mlp import MLP, Sigmoid, Softmax
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter

train_set = MNIST(which_set='train', start=0, stop=50000)
valid_set = MNIST(which_set='train', start=50000, stop=60000)
test_set = MNIST(which_set='test')

model = MLP(nvis=784,
            layers=[Sigmoid(layer_name='h', dim=500, irange=0.01),
                    Softmax(layer_name='y', n_classes=10, irange=0.01)])

algorithm = SGD(batch_size=100, learning_rate=0.01,
                learning_rule=Momentum(init_momentum=0.5),
                monitoring_dataset={'train': train_set,
                                    'valid': valid_set,
                                    'test': test_set},
                cost=MethodCost('cost_from_X'),
                termination_criterion=EpochCounter(10))

train = Train(dataset=train_set, model=model, algorithm=algorithm,
              save_path="mnist_example.pkl", save_freq=1,
              extensions=[MomentumAdjustor(start=5, saturate=6,
                                           final_momentum=0.95)])

train.main_loop()
