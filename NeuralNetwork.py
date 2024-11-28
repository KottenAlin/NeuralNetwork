import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os
#from tensorflow.keras.datasets import mnist

print("NeuralNetwork.py is imported")

hej = 1

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Generate training data
def generate_circle_data(radius, num_points):
    data = []
    labels = []
    for _ in range(num_points):
        x, y = np.random.uniform(-1, 1, 2)
        label = 0 if x**2 + y**2 <= radius**2 else 1
        data.append([x, y])
        labels.append(label)
    
    return np.array(data), np.array(labels)

def generate_linear_data(num_points):
    data = []
    labels = []
    for _ in range(num_points):
        x, y = np.random.uniform(-1, 1, 2)
        label = 0 if y >= 0.5 * x + 0.5 else 1
        data.append([x, y])
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Parameters
radius = 0.5
num_points = 1000
data, labels = generate_circle_data(radius, num_points)

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

# Generate and normalize data


def plot_data(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='bwr', alpha=0.5)
    circle = plt.Circle((0, 0), radius, color='blue', fill=False)
    #line = plt.Line2D([-1, 1], [0, 1], color='blue', linestyle='--')
    
    plt.gca().add_artist(circle)
    #plt.gca().add_artist(line)
    plt.title('Training Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    

    
#print("First data point:", data[0], "Label:", labels[0])


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=1 , learning_rate=0.1,  ):
        self.input_size = input_size # inputs size is 2
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_input_hidden = np.random.uniform(-1, 1, hidden_size)

        self.hidden_layers_weights = [np.random.uniform(-1, 1, (hidden_size, hidden_size)) for _ in range(hidden_layers - 1)]
        self.hidden_layers_biases = [np.random.uniform(-1, 1, hidden_size) for _ in range(hidden_layers - 1)]

        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden_output = np.random.uniform(-1, 1, output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, data): 
        # data -> (1000, 2) weights_input_hidden -> (2, 20) Bias_input_hidden -> (20,)
        self.hidden_layer = self.sigmoid(np.dot(data, self.weights_input_hidden) + self.bias_input_hidden) 
        for i in range(len(self.hidden_layers_weights)):
            self.hidden_layer = self.sigmoid(np.dot(self.hidden_layer, self.hidden_layers_weights[i]) + self.hidden_layers_biases[i])
        # hidden_layer -> (1000, 20) weights_hidden_output -> (20, 1) Bias_hidden_output -> (1,)
        output = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_hidden_output)
        # output -> (1000, 1)
        return output

    def cost(self, labels, outputs):
        return np.square(labels.reshape(-1, 1) - outputs)

    def backward(self, data, labels, output):
        """
        Perform one iteration of gradient descent to update the weights and biases of the neural network.
        Parameters:
        data (numpy.ndarray): Input data for the neural network.
        labels (numpy.ndarray): True labels corresponding to the input data.
        output (numpy.ndarray): Output from the neural network's forward pass.
        Returns:
        None
        The function calculates the error between the predicted output and the true labels, then computes the gradients
        for the output layer and the hidden layer. It updates the weights and biases of the network using these gradients
        and the learning rate. """
        
        #TODO: implements lin_trans in feedforward (this is redundant)
        trans_data = data
        lin_trans = []
        # trans_data -> (1000, 2) weights_input_hidden -> (2, 20) Bias_input_hidden -> (20,)
        lin_trans_input = np.dot(trans_data, self.weights_input_hidden) + self.bias_input_hidden
        trans_data = self.sigmoid(lin_trans_input)
        # lin_trans_input -> (1000, 20)
        for i in range(self.hidden_layers - 1):
            lin_trans[i] = np.dot(trans_data, self.hidden_layers_weights[i]) + self.hidden_layers_biases[i]
            trans_data = self.sigmoid(lin_trans[i]) 
        # trans_data -> (1000, 20) weights_hidden_output -> (20, 1) Bias_hidden_output -> (1,)

        lin_trans_output = np.dot(trans_data, self.weights_hidden_output) + self.bias_hidden_output
        
        d_output_bias = np.mean(2 * (labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans_output)), axis=0) #
        
        d_output_weights = np.mean(2 * (labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans_output)) * trans_data, axis=0) # sigmoid(lin_trans[-1])) 
        
        
        d_hidden_bias = []
        d_hidden_weights = []
                        # Backpropagate through hidden layers
        multi = 1
        for i in reversed(range(self.hidden_layers -1)): # iterate backwards through hidden layers
            multi = multi * self.sigmoid_derivative(self.sigmoid(lin_trans[i])) * self.hidden_layers_weights[i+1]
            d_hidden_bias[i] = 2 * (labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans[i])) * multi
            d_hidden_weights[i] = 2 * (labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans[i]))* multi * self.sigmoid(lin_trans[i-1])
            
        # lin_trans_input -> (1000, 20) weights_input_hidden -> (2, 20) Bias_input_hidden -> (20,)
        #TODO use np.dot instead of * for matrix multiplication
        if self.hidden_layers > 1:
            multi = multi * self.sigmoid_derivative(self.sigmoid(lin_trans_input))* self.hidden_layers_weights[0]
        else:
            multi = multi * np.dot(self.sigmoid_derivative(self.sigmoid(lin_trans_input)),self.weights_input_hidden.T)
        
        d_input_bias = 2 * np.mean((labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans_input)) * multi)
        d_input_weights = 2 * np.mean((labels.reshape(-1, 1) - output) * self.sigmoid_derivative(self.sigmoid(lin_trans_input)) * multi * data)
        
        self.bias_hidden_output += d_output_bias * self.learning_rate # update bias for output layer
        self.weights_hidden_output += d_output_weights.reshape(-1, 1)* self.learning_rate # update weights for output layer
        for i in range(self.hidden_layers - 1):
            self.hidden_layers_biases[i] += d_hidden_bias[i] * self.learning_rate
            self.hidden_layers_weights[i] += d_hidden_weights[i] * self.learning_rate
        self.weights_input_hidden += d_input_weights * self.learning_rate
        self.bias_input_hidden += d_input_bias * self.learning_rate
        

    def train(self, data, labels, epochs):
        total_error = 0
        #plot_data(data, labels)
        outputs = self.forward(data)
        #plot_data(data, outputs)
        #print ("data", data[0])
        #print("Outputs:", outputs)
        #print("error1", self.cost(labels, outputs)) 

        for i in range(10):
            outputs = self.forward(data)
            
            self.backward(data, labels, outputs)

        #print("error2", self.cost(labels, self.forward(data)))
        print("mean error", np.mean(self.cost(labels, self.forward(data))))


        outputs = self.forward(data)
        plot_data(data, outputs)
        '''error = self.backward(data, labels, output)
        total_error += np.sum(error)
        average_error = total_error / len(data)

        print(f"Average Error: {average_error}")
        print(f"Epoch {i+1}/{epochs}, Error: {error}", end='\r') # print the error '''

        

                
    
    def predict(self, data):
        return self.forward(data)
    
    def testdata(self, data, labels):
        for dataPoint in data:
            prediction = self.predict(dataPoint)
            color = plt.cm.RdYlGn(prediction)
            #print("Prediction:", prediction)
            #plt.scatter(dataPoint[0], dataPoint[1], color=color, alpha=0.5)

            '''if prediction == labels[np.where(data == dataPoint)[0][0]]:
                print("Prediction is correct" + str(prediction) + " " + str(labels[np.where(data == dataPoint)[0][0]]))
            else:
                print("Prediction is incorrect" + str(prediction) + " " + str(labels[np.where(data == dataPoint)[0][0]]))
                '''
        
        predictions = self.predict(data)
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



def plotNetwork(NeuralNetwork):
    fig, ax = plt.subplots()
    layer_sizes = [NeuralNetwork.input_size, NeuralNetwork.hidden_size, NeuralNetwork.output_size]
    v_spacing = (1.0 / float(max(layer_sizes)))
    h_spacing = (1.0 / float(len(layer_sizes) - 1))

    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing), v_spacing / 4.0,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if i == 0:
                ax.text(i * h_spacing - 0.05, layer_top - j * v_spacing, f'Input {j+1}', ha='right', va='center')
            elif i == len(layer_sizes) - 1:
                ax.text(i * h_spacing + 0.05, layer_top - j * v_spacing, f'Output {j+1}', ha='left', va='center')

    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)
                weight = NeuralNetwork.weights_input_hidden[j, k] if i == 0 else NeuralNetwork.weights_hidden_output[j, k]
                ax.text((i + 0.5) * h_spacing, (layer_top_a - j * v_spacing + layer_top_b - k * v_spacing) / 2.0,
                        f'{weight:.2f}', ha='center', va='center', color='red')

    ax.axis('off')
    plt.show()

    # Clear the screen
os.system('cls' if os.name == 'nt' else 'clear')
#plot_data(data, labels)
input_length = 2
NN = NeuralNetwork(input_length, 20, 1) # create a neural network with 2 input neurons, 20 hidden neurons, and 1 output neuron
# NN.testdata(data, labels)
NN.train(data, labels, 100)
# NN.testdata(data, labels)
#plotNetwork(NN)






    

        


# Test the neural network


