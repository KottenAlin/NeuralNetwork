import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Generate the data
np.random.seed(42)
n_points = 200

# Generate random points in a square from -1.5 to 1.5
X = np.random.uniform(-1.5, 1.5, size=(n_points, 2))

# Calculate distance from origin for each point
distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

# Label points: 1 if inside unit circle (distance <= 1), 0 otherwise
y = (distances <= 1).astype(int)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the neural network
model = MLPClassifier(
    hidden_layer_sizes=(10, 5),  # Two hidden layers with 10 and 5 neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_scaled, y)

# Create a grid of points for visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Scale the grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)

# Make predictions on the grid
Z = model.predict(grid_points_scaled)
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure(figsize=(10, 8))

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')

# Plot training points
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', label='Outside Circle')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Inside Circle')

# Plot the true circle
circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', label='True Circle')
plt.gca().add_artist(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Circle Classification')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Print model performance
training_accuracy = model.score(X_scaled, y)
print(f"Training accuracy: {training_accuracy:.3f}")

plt.show()