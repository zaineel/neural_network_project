import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained neural network model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        
    Returns:
        accuracy: Test accuracy
        predictions: Raw model predictions
    """
    # Get predictions
    predictions = model.forward(X_test)
    
    # Convert to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Display confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(10)]))
    
    return accuracy, predictions

# Usage in main:
if __name__ == "__main__":
    # Import necessary modules
    from nn_lib.network import NeuralNetwork
    from data.data_loader import load_mnist
    
    # Load the saved model and test data
    # For simplicity, you'd need to add code to save/load models
    # Or use the best model from training directly
    
    # Example usage after loading
    # accuracy, predictions = evaluate_model(model, X_test, y_test)
    # print(f"Test Accuracy: {accuracy:.4f}")