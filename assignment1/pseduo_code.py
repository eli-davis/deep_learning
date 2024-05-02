# B"H


'''
prompt: inputs: 256 length vectors, weights: 10x256, output vector length 10. simple loop using cross entropy loss to update the weights of this classifier

To implement a simple classifier with a cross-entropy loss and update the weights through a loop, we can create a model where:

Inputs are 256-length vectors.
Weights are a 10x256 matrix (for 10 classes, each class having a weight vector of length 256).
Output is a vector of length 10, representing the probabilities of each class.

multi-class logistic regression classifier.
- Compute the probabilities using the softmax function, which generalizes logistic regression to multiple classes.
- The softmax function converts the raw class scores (logits) into probabilities by comparing the exponentiated logits of each class against the sum of exponentiated logits for all classes.

Here’s a basic outline of what we'll implement:

- Initialize the weights.
- Loop over the dataset:
  - Compute the logits.
  - Apply the softmax function to get probabilities.
  - Compute the cross-entropy loss.
  - Compute the gradient of the loss with respect to the weights.
  - Update the weights using gradient descent.
  - Iterate until convergence or for a fixed number of epochs.
'''

import numpy as np

def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_true, probabilities):
    # Assuming y_true is one-hot encoded
    log_probs = -np.log(probabilities[range(len(probabilities)), y_true.argmax(axis=1)])
    return np.mean(log_probs)

def compute_gradients(X, y_true, probabilities):
    # y_true: one-hot encoded true labels
    # probabilities: predicted probabilities from softmax
    errors = probabilities - y_true
    return np.dot(X.T, errors) / len(X)

def update_weights(weights, gradients, learning_rate):
    return weights - learning_rate * gradients

# Initialize variables
num_features = 256
num_classes = 10
learning_rate = 0.01
epochs = 500
batch_size = 32

# Randomly initialize weights
weights = 0.01 * np.random.randn(num_classes, num_features)

# Dummy data generation (replace with actual data)
X = np.random.randn(1000, num_features)  # 1000 examples, 256 features each
y_indices = np.random.randint(num_classes, size=1000)
y = np.eye(num_classes)[y_indices]  # Convert indices to one-hot encoded labels

# Training loop
for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        
        # Compute logits (scores)
        logits = np.dot(X_batch, weights.T)
        
        # Compute probabilities
        probabilities = softmax(logits)
        
        # Compute loss
        loss = cross_entropy_loss(y_batch, probabilities)
        
        # Compute gradients
        gradients = compute_gradients(X_batch, y_batch, probabilities)
        
        # Update weights
        weights = update_weights(weights, gradients.T, learning_rate)
        
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Output the final weights and loss
print(f"Final Loss: {loss}")
print("Weights:", weights)
