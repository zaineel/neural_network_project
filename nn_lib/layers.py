import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights with small random values
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        # For storing values needed in backpropagation
        self.input = None
        self.output = None
        
        # For storing gradients
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward pass: Compute Y = X * W + b
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output
    
    def backward(self, dY):
        """
        Backward pass: Compute gradients
        
        Args:
            dY: Gradient from next layer, shape (batch_size, output_size)
            
        Returns:
            dX: Gradient with respect to input, shape (batch_size, input_size)
        """
        batch_size = self.input.shape[0]
        
        # Compute gradients
        self.dW = np.dot(self.input.T, dY) / batch_size
        self.db = np.sum(dY, axis=0, keepdims=True) / batch_size
        
        # Compute gradient with respect to input (for previous layer)
        dX = np.dot(dY, self.weights.T)
        
        return dX
    
    def update_params(self, learning_rate):
        """Update weights and biases using gradients"""
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db