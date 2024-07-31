import numpy as np
from scipy.signal import correlate, convolve
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    '''
    Implementation of the Convolution layer
    '''
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        # since it is a trainable layer
        self.trainable = True
        # Maintaining the stride data type as tuple
        self.stride_shape = (stride_shape, stride_shape) if isinstance(stride_shape, int) else (stride_shape[0], stride_shape[0]) if len(stride_shape) == 1 else stride_shape
        self.conv_shape = convolution_shape
        self.num_kernels = num_kernels
        # Initializing few paramaters for further use
        self.weights = np.random.random([num_kernels, *self.conv_shape])
        self.bias = np.random.random(self.num_kernels)
        self.optimizer_bias = None
        self.optimizer = None

    def forward(self, input_tensor):
        '''
        Implementing the forward pass of the convolution layer
        '''
        self.input_tensor = input_tensor
        # Conv 1D - batches, channels, y(width)
        # Conv 2d - batches, channels, y(width) X x(height)
        batch_size, channels = self.input_tensor.shape[0], self.input_tensor.shape[1]

        if len(self.input_tensor.shape) == 3:
            Conv_1D = True # True, if its 1D Convolution
        else:
            Conv_1D = False

        # Calculating output tensor shape
        # For input_tensor.shape is (batch_size, num_channels, height, width), then input_tensor.shape[2:] would give you (height, width).
        output_tensor = np.zeros([batch_size, self.num_kernels, *self.input_tensor.shape[2:]])

        for batch in range(batch_size):# iterating through each image(tensor)
            for kernel in range(self.num_kernels):# Through all kernels
                for channel in range(channels):# through channels
                    # accumulating correlation operation through each channel to get output tensor
                    output_tensor[batch, kernel] += correlate(self.input_tensor[batch, channel], self.weights[kernel, channel], mode= "same")
                # Adding bias
                output_tensor[batch, kernel] += self.bias[kernel]
        
        # maintaing the output tensor shape
        if Conv_1D == True:
            output_tensor = output_tensor[:,:, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:,:, ::self.stride_shape[0], ::self.stride_shape[1]]

        self.output_tensor = output_tensor
        return output_tensor
    
    def backward(self, error_tensor):
        '''
        Implementing the backward pass of the layer
        '''
        #initializing with the correct dimensions
        output_error_tensor, grad_weights  = np.zeros(self.input_tensor.shape), np.zeros(self.weights.shape)
        self._gradient_weights, self._gradient_bias  = np.zeros_like(self.weights), np.zeros_like(self.bias)

        # Conv 1D - batches, channels, y(width)
        # Conv 2d - batches, channels, y(width) X x(height)
        batches, channels = error_tensor.shape[0], self.weights.shape[1]
        
        # 1D Conv Flag
        if len(error_tensor.shape) == 3:
            Conv_1D = True
        else:
            Conv_1D = False

        # Run for each image in the batch
        for b in range(batches):

            error_strided_tensor = np.zeros((self.num_kernels, *self.input_tensor.shape[2:]))
            
            for k in range(error_tensor.shape[1]):
                err_b_k = error_tensor[b, k, :]  
                
                if Conv_1D:
                    error_strided_tensor[k, :: self.stride_shape[0]] = err_b_k
                else:
                    error_strided_tensor[
                        k, :: self.stride_shape[0], :: self.stride_shape[1]
                    ] = err_b_k

            # Gradient with respect to the input
            for c in range(channels):
                err = convolve(
                    error_strided_tensor, np.flip(self.weights, 0)[:, c, :], mode="same"
                )

                midchannel = int(err.shape[0] / 2)
                output_error_tensor[b, c, :] = err[midchannel, :]

            # Gradient with respect to the weights
            for k in range(self.num_kernels):
                self._gradient_bias[k] += np.sum(error_tensor[b, k, :])

                for c in range(self.input_tensor.shape[1]):
                    input_image = self.input_tensor[b, c, :]

                    if Conv_1D:
                        # padding
                        pad_x = self.conv_shape[1] / 2
                        px = (int(np.floor(pad_x)), int(np.floor(pad_x - 0.5)))
                        padded_image = np.pad(input_image, px)
                    else:
                        # Padding input (2 D)
                        pad_x, pad_y = (self.conv_shape[1] / 2), (self.conv_shape[2] / 2)
                        px = (int(np.floor(pad_x)), int(np.floor(pad_x - 0.5)))
                        py = (int(np.floor(pad_y)), int(np.floor(pad_y - 0.5)))
                        padded_image = np.pad(input_image, (px, py))
                        

                    grad_weights[k, c, :] = correlate(
                        padded_image, error_strided_tensor[k, :], mode="valid"
                    )
            # Adding up all
            self._gradient_weights += grad_weights

        # Update weights
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        if self._optimizer_bias:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return output_error_tensor


    def initialize(self, weights_initializer, bias_initializer):
        _fan_in, _fan_out = np.prod(self.conv_shape), np.prod(self.conv_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, _fan_in, _fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, _fan_in, _fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optmzr):
        self._optimizer = optmzr

    @property
    def optimizer_bias(self):
        return self._optimizer_bias
    
    @optimizer_bias.setter
    def optimizer_bias(self, optmzr_bias):
        self._optimizer_bias = optmzr_bias