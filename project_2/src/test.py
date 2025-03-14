# %%
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from neural_network import apply_rotate, apply_bias, apply_activation
from visualise import plot_scatter_points
from sklearn.metrics import accuracy_score

# %%
class Layer:
    """
    A single layer in the neural network that applies geometric transformations
    """
    def __init__(self, rotation_angle=0.0, bias=0.0, activation='abs', columns=None):
        self.rotation_angle = rotation_angle
        self.bias = bias
        self.activation = activation
        self.columns = columns
    
    def forward(self, input_data):
        # Apply the three transformations in sequence
        output = apply_rotate(input_data, self.rotation_angle, self.columns)
        output = apply_bias(output, self.bias, self.columns[0])
        output = apply_activation(output, self.activation, self.columns)
        return output

# %%
class NeuralNetwork:
    """
    A neural network composed of multiple layers using geometric transformations
    """
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
        
    def forward(self, X):
        """Forward pass through all layers"""
        output = X.copy()
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def predict(self, X):
        """
        Make binary predictions using the final network output:
        """
        final_output = self.forward(X)
        activation = self.layers[-1].activation

        if activation == "sigmoid":
            threshold = 0.5
        else:
            threshold = 0

        activated_output = apply_activation(final_output[:, 0], activation)
        return (activated_output > threshold).astype(int)

    def evaluate(self, X, y):
        """Calculate accuracy of the model"""
        predictions = self.predict(X)
        accuracy = accuracy_score(y.flatten(), predictions)
        return accuracy
    
    def visualize_current_transformation(self, X, y):
        """
        Visualize the data after passing through all layers currently in the model
        """
        # Apply all current transformations
        transformed_X = self.forward(X)
        
        # Combine with labels for visualization
        transformed_data = np.hstack((transformed_X, y))
        
        # Create meaningful title
        num_layers = len(self.layers)
        if num_layers > 0:
            last_layer = self.layers[-1]
            title = f"After {num_layers} Layer(s)\nLast layer: Rotation={last_layer.rotation_angle:.2f}, Bias={last_layer.bias:.2f}, Activation={last_layer.activation}"
        else:
            title = "Original Data"
        
        # Plot the transformed data
        plot_scatter_points(transformed_data, [0, 0], title)
        plt.show()

# %%
# Step 1: Load and visualize data
print("Loading data...")
data_loader = DataLoader()
data = data_loader.get_train_test_val_split(
    'Libian_desert_data.csv', 
    scale_features=False,
    train_ratio=0.9,
    test_ratio=0.1,
    val_ratio=0.0,
)

X_train, y_train = data['train']
X_test, y_test = data['test']

print("Visualizing original data...")
train_data = np.hstack((X_train, y_train))
plot_scatter_points(train_data, [0, 0])

# %% Step 2: Build a network layer by layer with visualization
print("Building neural network and visualizing transformations...")
model = NeuralNetwork()

def rotate_bias_activate_plot(model, X, y, rotation_angle=0, bias=0, activation='abs', columns=[0, 1]):
    """
    Rotates and applies bias (with no activation), plots,
    then applies the specified activation, and plots again.
    """
    model.add_layer(Layer(rotation_angle=rotation_angle, bias=bias, activation='none', columns=columns))
    model.visualize_current_transformation(X, y)
    
    model.add_layer(Layer(rotation_angle=0, bias=0, activation=activation, columns=columns))
    model.visualize_current_transformation(X, y)

print("Adding layer 1...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0, bias=-300, activation='abs')

print("Adding layer 2...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0.44*np.pi, bias=420, activation='abs')

print("Adding layer 3...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=-0.08*np.pi, bias=-130, activation='abs')

print("Adding layer 4...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=-0.05*np.pi, bias=-65, activation='abs')

print("Adding layer 5...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0.08*np.pi, bias=-250, activation='abs')

print("Adding layer 6...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0.1*np.pi, bias=-10, activation='abs')

print("Adding layer 7...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0, bias=-60, activation='abs')

print("Adding layer 8...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=-0.12*np.pi, bias=-80, activation='abs')

print("Adding layer 9...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0.1*np.pi, bias=20, activation='abs')

print("Adding layer 10...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0, bias=-35, activation='abs')

print("Adding layer 11...")
rotate_bias_activate_plot(model, X_train, y_train, rotation_angle=0.25*np.pi, bias=150, activation='none')

# Step 3: Evaluate the model
print("Evaluating model on training dataset...")
train_accuracy = model.evaluate(X_train, y_train)
print(f"Training accuracy: {train_accuracy:.4f}")

print("Evaluating model on testing dataeset...")
test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# %%
