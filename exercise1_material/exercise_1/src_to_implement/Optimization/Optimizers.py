class Sgd:
    def __init__(self, learning_rate):
        '''
        Initializing the constructor
        '''
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        Function to update the weights according to stochastic gradient descent
        '''
        return weight_tensor - (self.learning_rate * gradient_tensor)