import numpy as np
import copy
from .Base import BaseLayer
from . import Helpers

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True

        self.initialize()
        self.epsilon = np.finfo(float).eps
        self.decay = 0.8
        self.cumulated_mean = None
        self.cumulated_var = None
        self.input_tensor = None
        self.input_tensor_shape = None
        self._optimizre = None

    def forward(self, input_tensor):
        """ normalize the input tensor """

        compressed_to_vector = False
        if len(input_tensor.shape) == 4:
            # reshape the input tensor to vector
            # to handle the convolutional layer case
            input_tensor = self.reformat(input_tensor)
            compressed_to_vector = True

        self.input_tensor = input_tensor

        mu_bat = np.mean(input_tensor, axis=0)
        var_bat = np.var(input_tensor, axis=0)
        if self.testing_phase:
            # use the cumulated mean and variance
            # during testing phase
            self.input_tensor_normalized = (input_tensor - self.cumulated_mean) / np.sqrt(self.cumulated_var + self.epsilon)

        else:
            # during  training phase
            # update the mean and variance
            # keep a running average of the mean and variance

            if (self.cumulated_mean is None) and (self.cumulated_var is None):
                self.cumulated_mean = mu_bat
                self.cumulated_var = var_bat

            else :
                self.cumulated_mean = self.decay * self.cumulated_mean + (1 - self.decay) * mu_bat
                self.cumulated_var = self.decay * self.cumulated_var + (1 - self.decay) * var_bat

            self.input_tensor_normalized = (input_tensor - mu_bat) / np.sqrt(var_bat + self.epsilon)

        if compressed_to_vector:
            # convert the input tensor back to image
            return self.reformat(self.weights * self.input_tensor_normalized + self.bias)

        return self.weights * self.input_tensor_normalized + self.bias

    def backward(self, error_tensor):

        converted_to_vector = False
        if len(error_tensor.shape) == 4:
            # reshape the input tensor to vector
            # to handle the convolutional layer case
            converted_to_vector = True
            error_tensor = self.reformat(error_tensor)
        
        # compute the gradients
        # as per the pdf
        self.gradient_weights = np.sum(error_tensor * self.input_tensor_normalized, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # compute using the helper function
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, 
                            self.weights, self.cumulated_mean, self.cumulated_var)

        if self._optimizre is not None:
            # update the weights and bias using the optimizer
            self.weights = self._weight_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        if converted_to_vector:
            return self.reformat(gradient_input)

        return gradient_input


    def reformat(self, input_tensor):
        # reshape image to vector
        # and vector to image
        # we can reshape the B H M N tensor to B H M N from pdf
        # transpose from B H M N to B M N H
        transpose_shape = (0, 2, 1)
        if len(input_tensor.shape) == 2:
            # original shape is (num_samples * width * height, num_channels)
            num_samples, num_channels, width, height = self.input_tensor_shape
            # perform reverse operations
            input_tensor = input_tensor.reshape(num_samples, width * height, num_channels)
            input_tensor = np.transpose(input_tensor, transpose_shape)
            return input_tensor.reshape(num_samples, num_channels, width, height)
        else:
            num_samples, num_channels, width, height = input_tensor.shape
            self.input_tensor_shape = input_tensor.shape
            input_tensor = input_tensor.reshape(num_samples, num_channels, width * height)
            # change the axis to (num_samples, width * height, num_channels)
            # so that each channel is a feature
            # transpose the input tensor 
            input_tensor = np.transpose(input_tensor, transpose_shape)
            return input_tensor.reshape(num_samples * width * height, num_channels)
            

    def initialize(self, weights_initializer=None, bias_initializer=None):
        # initialize the weights and bias
        # as per the pdf
        # do not use the initializer
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    @property
    def weight(self):
        return self.weights

    @weight.setter
    def weight(self, weights):
        self.weights = weights

    @property
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def optimizer(self):
        return self._optimizre

    @optimizer.setter
    def optimizer(self, _optimizre):
        self._optimizre = _optimizre
        # create a copy of the optimizer for weights and bias
        self._weight_optimizer = copy.deepcopy(_optimizre)
        self._bias_optimizer = copy.deepcopy(_optimizre)

