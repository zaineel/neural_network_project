import numpy as np

class ReLU:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, X):
        """
        Apply ReLU activation: f(x) = max(0, x)
        """
        self.input = X
        self.output = np.maximum(0, X)
        return self.output
    
    def backward(self, dY):
        """
        Backward pass for ReLU
        """
        dX = dY.copy()
        dX[self.input <= 0] = 0
        return dX

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, X):
        """
        Apply Sigmoid activation: f(x) = 1 / (1 + exp(-x))
        """
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, dY):
        """
        Backward pass for Sigmoid
        """
        sigmoid_output = self.output
        return dY * sigmoid_output * (1 - sigmoid_output)