from .Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    """ Perform pooling operation """
    """ Used Max Pooling as the stratergy """

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self._maxima_locations = None
    
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        # for each batch, go through each sample
        # for each sample , go through each channel
        # for each channel, apply pooling
        
        # input shape is similar to (batch_size, #channels, y, x)
        
        existing_width, existing_height = input_tensor.shape[2:]
        pooling_width, pooling_height = self.pooling_shape
        stride_width, stride_height = self.stride_shape

        new_width = ((existing_width - pooling_width) // stride_width) + 1
        new_height = ((existing_height - pooling_height) // stride_height) + 1
        maxima_locations = []
        output_arr = []
        for sample in input_tensor:
            channel_arr = []
            channel_maxima_indices = []
            for channel in sample:
                pooled_values = []
                pooled_values_idx = []
                for x in range(0, existing_width, stride_width):
                    for y in range(0, existing_height, stride_height):
                        # edge case handling
                        if y > existing_height - pooling_height or x > existing_width - pooling_width:
                            continue
                        window_array = channel[x:x + pooling_width, y:y + pooling_height]
                        max_val = np.max(window_array)
                        pooled_values.append(max_val)
                        # get the index of max value and store it
                        local_max_x, local_max_y = np.unravel_index(np.argmax(window_array), window_array.shape)
                        pooled_values_idx.append((local_max_x + x, local_max_y + y))
                # append to channel arraz the max values of sliding window
                channel_arr.append(np.array(pooled_values).reshape(new_width, new_height))
                channel_maxima_indices.append(pooled_values_idx)
            # append sample channel values to output
            output_arr.append(channel_arr)
            maxima_locations.append(channel_maxima_indices)
        self._maxima_locations = (maxima_locations)
        # return np array
        return np.array(output_arr)

    def backward(self, error_tensor):
        # self._maxima_locations contains elements (x_idx, y_idx)
        return_tensor = np.zeros(self.input_shape)

        existing_width, existing_height = return_tensor.shape[2:]

        sample_idx = 0
        for sample in return_tensor:
            channel_idx = 0
            for _ in sample:
                channel_error = error_tensor[sample_idx][channel_idx]
                max_indices = self._maxima_locations[sample_idx][channel_idx]
                # work on current channel
                for x in range(0, existing_width):
                    for y in range(0, existing_height):
                        # if current index is in max_indices, add all the errors to the return tensor

                        for idx, max_index_item in enumerate(max_indices):
                            flattened_error = channel_error.flatten()
                            if (x, y) == max_index_item:
                                # add the corresponding error for (x, y) coordinate 
                                # from the flattened error tensor
                                return_tensor[sample_idx][channel_idx][x][y] += flattened_error[idx]

                channel_idx += 1
            sample_idx += 1
        return return_tensor