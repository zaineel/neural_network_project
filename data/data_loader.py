import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_mnist(test_size=0.2, val_size=0.2, random_state=42):
    """
    Load and prepare MNIST dataset
    
    Returns:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets (one-hot encoded)
    """
    # Load data from sklearn
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Split into train and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    # One-hot encode the labels
    def one_hot_encode(y, num_classes=10):
        encoded = np.zeros((y.shape[0], num_classes))
        for i, label in enumerate(y):
            encoded[i, label] = 1
        return encoded
    
    y_train_encoded = one_hot_encode(y_train)
    y_val_encoded = one_hot_encode(y_val)
    y_test_encoded = one_hot_encode(y_test)
    
    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

def create_batches(X, y, batch_size):
    """
    Create mini-batches from the dataset
    
    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches