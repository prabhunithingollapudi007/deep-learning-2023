import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = []

    def forward(self, prediction_tensor, label_tensor):
        # saving it for backward pass
        self.prediction_tensor = prediction_tensor
        # Calculating the loss and adding epislon to ensure numerical stability
        loss = np.sum(-np.log(self.prediction_tensor[label_tensor == 1] + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        # Backward pass for CrossEntropyLoss
        # The derivative of CrossEntropyLoss with respect to its input is (prediction - label)
        error_tensor = -(label_tensor / (self.prediction_tensor + np.finfo(float).eps))

        return error_tensor
    