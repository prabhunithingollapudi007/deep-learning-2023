from .Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    """
    Dropout Layer
    """

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.drop_out_mask = None 

    def forward(self, input_tensor):
        """
        Forward pass for dropout layer
        """
        rng = np.random.default_rng(123567)
        keep_prob = 1 - self.probability
        # array of 0s and 1s, 1s with probability 1 - keep_prob as 1 0 0 or 1 1 0
        temp_array = (rng.random(input_tensor.shape) > (keep_prob)).astype(int)
        self.drop_out_mask = (temp_array) / (self.probability)
        if self.testing_phase:
            return input_tensor
        return input_tensor * self.drop_out_mask

    def backward(self, error_tensor):
        """
        Backward pass for dropout layer
        """
        # preserve zero positions by multiplying with dropout mask
        return error_tensor * self.drop_out_mask
