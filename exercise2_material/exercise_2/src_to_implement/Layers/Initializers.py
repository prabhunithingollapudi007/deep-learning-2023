import numpy as np
import math

class Constant:
    '''
    Implementing the constnat initilizer fn
    '''
    def __init__(self, k=0.1):
        self.k = k

    def  initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.k)
        

class UniformRandom:
    '''
    Implementing the uniform distribution initilizer fn
    '''
    def __init__(self):
        self.rng = np.random.default_rng(12345)

    def  initialize(self, weights_shape, fan_in, fan_out):
        # Generates the uniform distribution in the interval [0,1)
        return self.rng.uniform(0, 1, weights_shape)

class Xavier:
    '''
    Implementing the Xavier/Glorot initilizer fn
    '''
    def __init__(self):
        self.mean = 0
        self.rng = np.random.default_rng(12345)

    def  initialize(self, weights_shape, fan_in, fan_out):
        # Using the formula sqrt(2 / (fan_in + fan_out)) for a Xavier initialization of normal distribution
        sd = math.sqrt((2 / (fan_out + fan_in)))
        return self.rng.normal(self.mean, sd, weights_shape)

class He:
    '''
    Implementing the He initilizer fn
    '''
    def __init__(self):
        self.mean = 0
        self.rng = np.random.default_rng(123456)

    def  initialize(self, weights_shape, fan_in, fan_out):
        """ Use the formula for He Initialization for Normal distribution """
        sd = math.sqrt((2 / (fan_in)))
        return self.rng.normal(self.mean, sd, weights_shape)