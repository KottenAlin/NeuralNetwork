import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Generate data
np.random.seed(42)
n_points = 200

# Generate random points in a square
X = np.random.uniform(-1.5, 1.5, size=(n_points, 2))

# Calculate distances from origin
distances = np.sqrt(np.sum(X**2, axis=1))

# Create labels: 1 if inside unit circle, 0 if outside
y = (distances <= 1).astype(int).reshape(-1, 1)

# Normalize the input data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Create and train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1)

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
losses = []
for epoch in range(epochs):
    # Forward pass
    predictions = nn.forward(X)
    
    # Calculate loss
    loss = -np.mean(y * np.log(predictions + 1e-8) + 
                    (1-y) * np.log(1 - predictions + 1e-8))
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    losses.append(loss)
    
    # Backward pass
    nn.backward(X, y, learning_rate)

# Create visualization grid
x_min, x_max = -2, 2
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Normalize grid points
grid_points = (grid_points - np.mean(X, axis=0)) / np.std(X, axis=0)

# Make predictions on grid
Z = nn.forward(grid_points)
Z = Z.reshape(xx.shape)

# Plot results
plt.figure(figsize=(18, 5))

# Plot training process
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot decision boundary
plt.subplot(1, 3, 2)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')

# Plot original points
original_X = X * np.std(X, axis=0) + np.mean(X, axis=0)
plt.scatter(original_X[y.ravel() == 0, 0], original_X[y.ravel() == 0, 1], 
           c='red', marker='x', label='Outside Circle')
plt.scatter(original_X[y.ravel() == 1, 0], original_X[y.ravel() == 1, 1], 
           c='blue', marker='o', label='Inside Circle')

# Plot data points with color gradient based on output
plt.subplot(1, 3, 3)
predicted_colors = nn.forward(X).ravel()
plt.scatter(original_X[:, 0], original_X[:, 1], c=predicted_colors, cmap='coolwarm', edgecolor='k')
plt.colorbar(label='Output')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points with Output Color Gradient')
plt.axis('equal')

# Plot true circle
circle = plt.Circle((0, 0), 1, fill=False, color='black', 
                   linestyle='--', label='True Circle')
plt.gca().add_artist(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Decision Boundary')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.show()