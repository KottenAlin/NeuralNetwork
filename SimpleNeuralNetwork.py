import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os

print("NeuralNetwork.py is imported")


def plot_data(data, labels, plot_type='circle', radius = 0.5):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='bwr', alpha=0.5)
    
    if plot_type == 'circle':
        # Parameters
        circle = plt.Circle((0, 0), radius, color='blue', fill=False)
        plt.gca().add_artist(circle)
    elif plot_type == 'linear':
        line = plt.Line2D([-1, 1], [-0.5, 0.5], color='blue', linestyle='--')
        plt.gca().add_artist(line)
    elif plot_type == 'quadratic':
        x = np.linspace(-1, 1, 100)
        y = x**2
        plt.plot(x, y, color='blue', linestyle='--')
    
    plt.title('Training Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Generate training data
def generate_data(data_type, num_points, radius=0.5):
    data = []
    labels = []
    for _ in range(num_points):
        x, y = np.random.uniform(-1, 1, 2)
        if data_type == 'circle':
            label = 0 if x**2 + y**2 <= radius**2 else 1
        elif data_type == 'linear':
            label = 0 if y >= 0.5 * x + 0.5 else 1
        elif data_type == 'quadratic':
            label = 0 if y >= x**2 else 1
        else:
            raise ValueError("Invalid data_type. Choose from 'circle', 'linear', or 'quadratic'.")
        data.append([x, y])
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Load MNIST dataset

'''(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Use a subset of the data for training
data = train_images[:num_points] # use the first 1000 images
labels = train_labels[:num_points] '''

# Define the neural network

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=2, learning_rate=0.02):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.uniform(-1, 1, (input_size, hidden_size)))
        self.biases.append(np.random.uniform(-1, 1, hidden_size))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.weights.append(np.random.uniform(-1, 1, (hidden_size, hidden_size)))
            self.biases.append(np.random.uniform(-1, 1, hidden_size))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.uniform(-1, 1, (hidden_size, output_size)))
        self.biases.append(np.random.uniform(-1, 1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, data):
        self.activations = [data]
        self.z_values = []
        activation = data
        
        # Forward through all layers
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight) + bias
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)
        return activation

    def cost(self, labels, outputs):
        return np.square(labels.reshape(-1, 1) - outputs)

    def backward(self, labels):
        m = labels.shape[0] # number of samples
        
        # Calculate output layer delta
        delta = (self.activations[-1] - labels.reshape(-1, 1)) * self.relu_derivative(self.z_values[-1])
        
        # Initialize gradients
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        
        # Backward through all layers
        for i in reversed(range(len(self.weights))):
            grad_weights[i] = np.dot(self.activations[i].T, delta) / m
            grad_biases[i] = np.mean(delta, axis=0)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train(self, data, labels, epochs):
        losses = []
        for i in range(epochs):
            outputs = self.forward(data)
            self.backward(labels)
            loss = np.mean(self.cost(labels, outputs))
            losses.append(loss)
            if (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss}")
        return losses

    def testdata(self, data, labels):
        for dataPoint in data:
            prediction = self.forward(dataPoint)
            color = plt.cm.RdYlGn(prediction)
        
        predictions = self.forward(data)
        plt.figure()
        plt.hist(predictions, bins=20, color='blue', alpha=0.7)
        plt.title('Prediction Histogram')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.show(block=False)
        plt.title('Predictions')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def save(self, filename):
        with open(filename, 'wb') as file:
            for weight in self.weights:
                np.save(file, weight)
            for bias in self.biases:
                np.save(file, bias)

  
def betterPlotNetwork(NN):
    
    # plot the network weights and biases

    num_layers = len(NN.weights)
    fig, axes = plt.subplots(2, num_layers, figsize=(5 * num_layers, 8))

    for i in range(num_layers):
        # Plot weights
        ax = axes[0, i]
        im = ax.imshow(NN.weights[i], aspect='auto', cmap='viridis')
        ax.set_title(f'Weights Layer {i} to {i+1}')
        fig.colorbar(im, ax=ax)

        # Plot biases
        ax = axes[1, i]
        ax.bar(range(len(NN.biases[i])), NN.biases[i])
        ax.set_title(f'Biases Layer {i+1}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Bias Value')

    plt.tight_layout()
    plt.show()

    # Clear the screen

def main(num_points=1000, data_type='circle', hidden_size=24, hidden_layers=2, learning_rate=0.02, epochs=5000):

    data, labels = generate_data(data_type, num_points)

    plot_data(data, labels, data_type)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    input_size = 2
    output_size = 1
    NN = NeuralNetwork(input_size, hidden_size, output_size, hidden_layers, learning_rate) # create a neural network with 2 input neurons, 20 hidden neurons, and 1 output neuron
    
    output = NN.forward(data)
    plot_data(data, output, data_type)
    
    # NN.testdata(data, labels)
    losses = NN.train(data, labels, epochs)
    plot_data(data, NN.forward(data), data_type)
    NN.testdata(data, labels)

    # Plot the training loss
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    betterPlotNetwork(NN) # plot the network weights and biases
    

# NN.testdata(data, labels)
#plotNetwork(NN)

if __name__ == '__main__':
    main()

# Test the neural network


