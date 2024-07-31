import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # alpha * w
        return self.alpha * weights

    def norm(self, weights):
        # alpha * ||w||^2
        return self.alpha * np.sum(weights ** 2)
    
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # alpha * sign(w)
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # alpha * ||w||^1
        return self.alpha * np.sum(np.abs(weights) ** 1)

        
