from Layers import *
import numpy as np
def test_expected_output():
        input_shape = (1, 4, 4)
        pool = Pooling.Pooling((1, 1), (2, 2))
        batch_size = 1
        numbers = [9, 8, 0, 5, 3, 5, 1, 1, 1, 1, 6, 3, 5, 2, 6, 3]
        input_tensor = np.array(numbers).reshape(1, 1, 4, 4)
        output_tensor = pool.forward(input_tensor)
        numbers = [6, 4, 3, 2, 5, 4, 7, 1, 2]
        error_tensor = np.array(numbers).reshape(1, 1, 3, 3)
        result = pool.backward(error_tensor)

test_expected_output()


