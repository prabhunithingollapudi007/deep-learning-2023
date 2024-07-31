from .Base import BaseLayer
import numpy as np
import copy

class FullyConnected(BaseLayer):
    '''
    Implement a class for a fully connected layer of Feed-forwards neural networks (Child class of BaseLayer)
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.rng = np.random.default_rng(1234567)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizre = None
        self.weights = self.initialize_weights()
        self.input_tensor = None
        self._gradient_weights = None

    def initialize_weights(self):
        lower_bound = 0.0
        upper_bound = 1.0 # not inclusive
        # size is input + 1 for bias and output size
        return self.rng.uniform(lower_bound, upper_bound, (self.input_size + 1, self.output_size))

    def initialize(self, weights_initializer, bias_initializer):
        """ Re initializing weights """
        weight_size = (self.input_size, self.output_size)
        bias_size = (1, self.output_size)
        fan_in = self.input_size
        fan_out = self.output_size
        weights = weights_initializer.initialize(weight_size, fan_in, fan_out)
        bias = bias_initializer.initialize(bias_size, fan_in, fan_out)
        # add bias by using vstack option
        self.weights = np.vstack((weights, bias))

    def forward(self, input_tensor):
        # Adding a bias term to the input tensor
        self.input_tensor = np.column_stack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        # implementing weights * inputs
        self.output = np.dot(self.input_tensor, self.weights)
        return self.output

    def caluculate_gradient_weights(self, error_tensor):
        return np.dot(self.input_tensor.T, error_tensor)
    
    def backward(self, error_tensor):
        self._gradient_weights = self.caluculate_gradient_weights(error_tensor)
        if self.optimizer is not None:
            self.weights = self._weight_optimizer.calculate_update(self.weights, self._gradient_weights)
        self.error_tensor = np.dot(error_tensor, self.weights.T)[:, :-1]
        return self.error_tensor # remove the bias terms
    
    @property
    def optimizer(self):
        return self._optimizre

    @optimizer.setter
    def optimizer(self, _optimizre):
        self._optimizre = _optimizre


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def optimizer(self):
        return self._optimizre

    @optimizer.setter
    def optimizer(self, _optimizre):
        self._optimizre = _optimizre
        # create a copy of the optimizer for weights and bias
        self._weight_optimizer = copy.deepcopy(_optimizre)
        self._bias_optimizer = copy.deepcopy(_optimizre)

