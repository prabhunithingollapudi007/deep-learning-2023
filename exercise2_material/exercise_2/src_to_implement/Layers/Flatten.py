from .Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):

    """ Flatten layer  """

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        """ Use numpy function to flatten to array and return 1D array """
        """ first parameter is the batch size """
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]
        reshaped_length = input_tensor.size // batch_size
        return input_tensor.reshape(batch_size, reshaped_length)

    def backward(self, error_tensor):
        """ Reshape and return in backward propagation
        same like unflatten """
        return np.reshape(error_tensor, self.input_shape)