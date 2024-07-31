from .Base import *
import numpy as np

class SoftMax(BaseLayer):
    '''
    Implementing the SoftMax activation function
    '''
    def __init__(self):
        # Do not need to change the trainable parameter
        super().__init__()

    def forward(self, input_tensor):
        # Unnormalized probablitites
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        # Normalized probablities
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        #saving it in outputp tensor, to be used in backward pass
        self.output_tensor = probablities
        return self.output_tensor

    def backward(self, error_tensor):

        softmax_values = self.output_tensor
        # Reshaping output tensor from 1D to 2D to perform matrix multiplication
        jacobian_matrix = - softmax_values[:, :, np.newaxis] * softmax_values[:, np.newaxis, :] 
        indices = np.arange(softmax_values.shape[1])
        jacobian_matrix[:, indices, indices] += softmax_values

        # Multiply the Jacobian matrix by the error tensor to get the gradient
        gradient = np.matmul(error_tensor[:, np.newaxis, :], jacobian_matrix)

        return gradient.squeeze()
