import numpy as np

class MSELoss:
    """Mean Squared Error Loss"""
    
    def __init__(self):
        self.diff = None
    
    def forward(self, y_pred, y_true):
        """
        Compute MSE loss: (1/n) * Σ(y_pred - y_true)²
        """
        self.diff = y_pred - y_true
        loss = np.mean(np.square(self.diff))
        return loss
    
    def backward(self):
        """
        Compute gradient of MSE loss
        """
        batch_size = self.diff.shape[0]
        return 2 * self.diff / batch_size

class CrossEntropyLoss:
    """Cross Entropy Loss for classification"""
    
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def softmax(self, X):
        """
        Compute softmax: exp(x_i) / Σexp(x_j)
        """
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, logits, y_true):
        """
        Compute cross entropy loss after softmax
        
        Args:
            logits: Raw output from the last layer (batch_size, num_classes)
            y_true: One-hot encoded or class indices (batch_size, num_classes)
                    or (batch_size,) for class indices
        """
        batch_size = logits.shape[0]
        self.y_pred = self.softmax(logits)
        
        # Handle the case when y_true contains class indices
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true_indices = y_true.astype(int).flatten()
            self.y_true = np.zeros_like(self.y_pred)
            for i in range(batch_size):
                self.y_true[i, y_true_indices[i]] = 1
        else:
            self.y_true = y_true
        
        # Compute cross entropy loss
        epsilon = 1e-15  # Small value to avoid log(0)
        log_probs = -np.log(self.y_pred + epsilon)
        loss = np.sum(self.y_true * log_probs) / batch_size
        return loss
    
    def backward(self):
        """
        Compute gradient of cross entropy loss
        """
        batch_size = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / batch_size