import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.f_x = None 

    def forward(self, input_tensor):
        # forward pass f(x) = 1 / (1 + e^(-x))
        self.f_x = 1 / (1 + np.exp(-input_tensor))
        return self.f_x

    def backward(self, error_tensor):
        # backward pass f'(x) = f(x) * (1 - f(x))
        return self.f_x * (1 - self.f_x) * error_tensor