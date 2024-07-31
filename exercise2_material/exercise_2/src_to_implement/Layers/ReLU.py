from .Base import *
import numpy as np

class ReLU(BaseLayer):
    '''
    Implementing the ReLU activation function, also implementing the loss gradient for backpropogation
    '''
    def __init__(self):
        # Do not need to change the trainable parameter 
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Implementing ReLU Activation function to the input tensor
        self.output_tensor = np.where(self.input_tensor > 0, self.input_tensor, 0)
        return self.output_tensor

    def backward(self, error_tensor):
        # Implementing loss gradient for input tensors based on the derivative of the ReLU activation fn i.e. 1, if positive, else zero
        gradient_relu = np.where(self.input_tensor > 0, 1, 0)
        return gradient_relu * error_tensor