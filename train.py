import numpy as np
import matplotlib.pyplot as plt
from nn_lib.layers import LinearLayer
from nn_lib.activations import ReLU, Sigmoid
from nn_lib.losses import CrossEntropyLoss
from nn_lib.network import NeuralNetwork
from data.data_loader import load_mnist, create_batches

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

# Define a function to train a model with given parameters
def train_model(hidden_layer_sizes, learning_rate, batch_size, epochs):
    """
    Train a neural network model with the given parameters
    
    Args:
        hidden_layer_sizes: List of integers representing the size of each hidden layer
        learning_rate: Learning rate for gradient descent
        batch_size: Batch size for mini-batch gradient descent
        epochs: Number of training epochs
        
    Returns:
        model: Trained neural network model
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
    """
    # Initialize network
    model = NeuralNetwork()
    
    # Add input layer
    input_size = X_train.shape[1]  # 784 for MNIST
    output_size = 10  # 10 classes for digits 0-9
    
    # Add layers to the network
    prev_size = input_size
    for i, layer_size in enumerate(hidden_layer_sizes):
        model.add(LinearLayer(prev_size, layer_size))
        model.add(ReLU())
        prev_size = layer_size
    
    # Add output layer
    model.add(LinearLayer(prev_size, output_size))
    
    # Set loss function
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Create mini-batches
        batches = create_batches(X_train, y_train, batch_size)
        
        # Train on mini-batches
        epoch_losses = []
        for X_batch, y_batch in batches:
            loss, _ = model.train_step(X_batch, y_batch, learning_rate)
            epoch_losses.append(loss)
        
        # Record average training loss
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_predictions = model.forward(X_val)
        val_loss = loss_function.forward(val_predictions, y_val)
        val_losses.append(val_loss)
        
        # Calculate validation accuracy
        predicted_classes = np.argmax(val_predictions, axis=1)
        true_classes = np.argmax(y_val, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    return model, train_losses, val_losses, val_accuracies

# Define model configurations to test
model_configs = [
    {
        "name": "Model 1 - Small",
        "hidden_layer_sizes": [128],
        "learning_rate": 0.1,
        "batch_size": 64,
        "epochs": 10
    },
    {
        "name": "Model 2 - Medium",
        "hidden_layer_sizes": [256, 128],
        "learning_rate": 0.05,
        "batch_size": 64,
        "epochs": 10
    },
    {
        "name": "Model 3 - Large",
        "hidden_layer_sizes": [512, 256, 128],
        "learning_rate": 0.01,
        "batch_size": 128,
        "epochs": 10
    },
    {
        "name": "Model 4 - Small with Different Learning Rate",
        "hidden_layer_sizes": [128],
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10
    }
]

# Train all models
results = []
for config in model_configs:
    print(f"\nTraining {config['name']}...")
    model, train_losses, val_losses, val_accuracies = train_model(
        config["hidden_layer_sizes"],
        config["learning_rate"],
        config["batch_size"],
        config["epochs"]
    )
    
    # Store results
    results.append({
        "config": config,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    })

# Find the best model based on validation accuracy
best_model_idx = np.argmax([r["val_accuracies"][-1] for r in results])
best_model = results[best_model_idx]
print(f"\nBest model: {best_model['config']['name']} with validation accuracy: {best_model['val_accuracies'][-1]:.4f}")

# Evaluate best model on test set
test_predictions = best_model["model"].forward(X_test)
test_loss = CrossEntropyLoss().forward(test_predictions, y_test)
predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
test_accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot training curves
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
for result in results:
    plt.plot(result["train_losses"], label=result["config"]["name"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot validation loss
plt.subplot(1, 3, 2)
for result in results:
    plt.plot(result["val_losses"], label=result["config"]["name"])
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot validation accuracy
plt.subplot(1, 3, 3)
for result in results:
    plt.plot(result["val_accuracies"], label=result["config"]["name"])
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()