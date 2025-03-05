class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
    
    def add(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
    
    def set_loss(self, loss_function):
        """Set the loss function"""
        self.loss_function = loss_function
    
    def forward(self, X):
        """Forward pass through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dY):
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
        return dY
    
    def update_params(self, learning_rate):
        """Update parameters in all layers"""
        for layer in self.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)
    
    def train_step(self, X, y, learning_rate):
        """Perform one training step (forward + backward + update)"""
        # Forward pass
        predictions = self.forward(X)
        
        # Compute loss
        loss = self.loss_function.forward(predictions, y)
        
        # Backward pass
        grad = self.loss_function.backward()
        self.backward(grad)
        
        # Update parameters
        self.update_params(learning_rate)
        
        return loss, predictions