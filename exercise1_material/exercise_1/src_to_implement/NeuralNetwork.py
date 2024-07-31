# NeuralNetwork.py

import copy
from Layers import *
from Optimization import *
class NeuralNetwork():
    def __init__(self, _optimizre):
        self._optimizre = _optimizre
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.lable_tensor = None

    def forward(self):
        """ be careful with data layer .next method because it refreshes the data
        only use it and do not refresh lable_tensor when not required """
        input_tensor, self.lable_tensor = self.data_layer.next()
        output_tensor = input_tensor

        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        loss_value = self.loss_layer.forward(output_tensor, self.lable_tensor)
        return loss_value

    def backward(self):
        error_tensor = self.loss_layer.backward(self.lable_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self._optimizre)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            """ forward and backward
            save loss from forward """
            loss_value = self.forward()            
            self.backward()
            self.loss.append(loss_value)

    def test(self, input_tensor):
        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        return output_tensor