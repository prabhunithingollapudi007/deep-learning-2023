import numpy as np

class Optimizer:
    '''
    Parent Class for all the optimizers
    '''
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        '''
        Initializing the constructor
        '''
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        Function to update the weights according to stochastic gradient descent
        '''

        if self.regularizer is not None:
            # if regularizer is present, add the gradient of regularizer to the gradient of loss
            weight_tensor -= (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))


        return weight_tensor - (self.learning_rate * gradient_tensor)

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.V = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        """ return updated weights as per document """
        if self.V is None:
            # handle the 0th case
            self.V = - (self.learning_rate * gradient_tensor)
        else :
            self.V = self.momentum_rate * self.V - (self.learning_rate * gradient_tensor)

        if self.regularizer is not None:
            # if regularizer is present, add the damped weight to the gradient
            weight_tensor -= (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))

        return weight_tensor + self.V

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.V = None
        self.R = None
        self.itr = 1
    
    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.regularizer is not None:
            # if regularizer is present, add the damped weight to the gradient
            weight_tensor -= (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))

        G  = gradient_tensor
        if self.V is None:
            # handle the 0th case
            self.V = (1 - self.mu) * G
        else:
            self.V = self.mu * self.V + (1 - self.mu) * G
        if self.R is None:
            # handle the 0th case
            self.R = (((1 - self.rho) * G) * G)
        else:
            self.R = self.rho * self.R +  (((1 - self.rho) * G) * G)
        # corrections
        v_cap = self.V / (1 - self.mu ** self.itr)
        r_cap = self.R / (1 - self.rho ** self.itr)
        self.itr += 1
        # applied exactly as per the sheet formula
        return weight_tensor - self.learning_rate * v_cap / (np.sqrt(r_cap) + np.finfo(float).eps)
