import numpy as np
import json
from tensorflow.keras.datasets import mnist
import pickle

LEARNING_RATE = 0.1
ERROR_THRESHOLD = 0.016
LAYERS = [784, 250, 10]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
def get_weights_and_bias(input_size, hidden_layer_sizes, output_size):
    weights = []
    biases = []

    # Weights and biases between input layer and first hidden layer
    prev_size = input_size
    for size in hidden_layer_sizes:
        weights.append(np.random.rand(size, prev_size) - 0.5)  # Random weights
        biases.append(np.random.rand(size) - 0.5)  # Random biases
        prev_size = size

    # Weights and biases between last hidden layer and output layer
    weights.append(np.random.rand(output_size, prev_size) - 0.5)
    biases.append(np.random.rand(output_size) - 0.5)

    return weights, biases

def backpropagation_training(X, y, layers, learning_rate, error_threshold):
    weights, biases = get_weights_and_bias(layers[0], layers[1:-1], layers[-1])
    error_history = []  # To store total error per epoch
    epoch = 0

    while True:
        total_error = 0  # Initialize total error for this epoch
        epoch += 1
        for xi, target in zip(X, y):
            # Feedforward phase
            a = xi
            outputs = [a]

            for i in range(len(weights)):
                z = np.dot(a, np.array(weights[i]).T) + biases[i]
                a = sigmoid(z)
                outputs.append(a)

            y_pred = a
            error = target - y_pred
            lms_error = np.sum(error ** 2)
            total_error += lms_error

            # Backpropagation phase
            deltas = [None] * len(weights)
            deltas[-1] = error * sigmoid_derivative(y_pred)

            for i in reversed(range(len(weights) - 1)):
                delta_next = deltas[i + 1]
                weight_next = weights[i + 1]
                deltas[i] = np.dot(delta_next, weight_next) * sigmoid_derivative(outputs[i + 1])

            # Weight and bias updates
            for i in range(len(weights)):
                a_prev = outputs[i]
                delta = deltas[i]
                weights[i] += learning_rate * np.outer(delta, a_prev)
                biases[i] += learning_rate * delta

        average_error = total_error / len(X)
        error_history.append(average_error)
        print(f"Epoch {epoch}: Total error = {average_error}")

        if average_error < error_threshold:
            # Save error history to a JSON file when training is complete
            with open('error_history.json', 'w') as f:
                json.dump(error_history, f)
            return weights, biases
        

def predict_image(image, weights, biases):
    a = image
    for i in range(len(weights)):
        z = np.dot(a, np.array(weights[i]).T) + biases[i]
        a = sigmoid(z)
    return a

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    y_train_one_hot = np.zeros((y_train.size, 10))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1

    y_test_one_hot = np.zeros((y_test.size, 10))
    y_test_one_hot[np.arange(y_test.size), y_test] = 1

    weights, biases = backpropagation_training(x_train, y_train_one_hot, LAYERS, LEARNING_RATE, ERROR_THRESHOLD)
    print("Training complete!")

    with open('digit_recognition_model.pkl', 'wb') as f:
        pickle.dump({'weights': weights, 'biases': biases}, f)
