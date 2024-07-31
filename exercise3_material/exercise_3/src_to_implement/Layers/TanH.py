import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.f_x = None 

    def forward(self, input_tensor):
        # forward pass f(x) = tanh(x)
        self.f_x = np.tanh(input_tensor)
        return self.f_x

    def backward(self, error_tensor):
        # backward pass f'(x) = 1 - tanh(x)^2 = 1 - f(x)^2
        return (1 - np.power(self.f_x, 2)) * error_tensor