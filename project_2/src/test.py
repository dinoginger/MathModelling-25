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
        output = apply_rotate(input_data, self.rotation_angle, self.columns)
        
        bias_cols = [self.columns[0]] if self.columns is not None else None
        output = apply_bias(output, self.bias, bias_cols)
        output = apply_activation(output, self.activation, bias_cols)
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
        result = (activated_output > threshold).astype(int)
        num_ones = np.sum(result == 1)
        num_zeros = np.sum(result == 0)
        print(f"Number of 1's: {num_ones}")
        print(f"Number of 0's: {num_zeros}")
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
    scale_features = False,
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

print("Adding layer 1...")
model.add_layer(Layer(rotation_angle=-0.42 * np.pi, bias=-54, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 2...")
model.add_layer(Layer(rotation_angle=0.05*np.pi, bias=-26, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 3...")
model.add_layer(Layer(rotation_angle=0.12*np.pi, bias=-70, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 4...")
model.add_layer(Layer(rotation_angle=0.0*np.pi, bias=-35, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 5...")
model.add_layer(Layer(rotation_angle=-0.14*np.pi, bias=-40, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 6...")
model.add_layer(Layer(rotation_angle=0.05*np.pi, bias=-30, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 7...")
model.add_layer(Layer(rotation_angle=-0.17*np.pi, bias=-14, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 8...")
model.add_layer(Layer(rotation_angle=-0.34*np.pi, bias=80, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 9...")
model.add_layer(Layer(rotation_angle=0.17*np.pi, bias=-60, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 10...")
model.add_layer(Layer(rotation_angle=-0.20*np.pi, bias=-12, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 11...")
model.add_layer(Layer(rotation_angle=-0.53*np.pi, bias=280, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 12...")
model.add_layer(Layer(rotation_angle=-0.0*np.pi, bias=-30, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 13...")
model.add_layer(Layer(rotation_angle=-0.08*np.pi, bias=3, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 14...")
model.add_layer(Layer(rotation_angle=0.08*np.pi, bias=-50, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 15...")
model.add_layer(Layer(rotation_angle=0.07*np.pi, bias=-40, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 16...")
model.add_layer(Layer(rotation_angle=-0.09*np.pi, bias=-5, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 17...")
model.add_layer(Layer(rotation_angle=0.09*np.pi, bias=-30, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 18...")
model.add_layer(Layer(rotation_angle=0.05*np.pi, bias=-33, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 19...")
model.add_layer(Layer(rotation_angle=-0.3*np.pi, bias=-20, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 20...")
model.add_layer(Layer(rotation_angle=-0.15*np.pi, bias=-5, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer 21...")
model.add_layer(Layer(rotation_angle=-0.20*np.pi, bias=23, activation='abs', columns=[0, 1]))
#model.visualize_current_transformation(X_train, y_train)

print("Adding layer final layer for visualisation.")
model.add_layer(Layer(rotation_angle=0.25*np.pi+np.pi, bias=55, activation='none', columns=[0, 1]))
model.visualize_current_transformation(X_train, y_train)

# Step 3: Evaluate the model
print("Evaluating model on training dataset...")
train_accuracy = model.evaluate(X_train, y_train)
print("Evaluating model on training dataset...")
test_accuracy = model.evaluate(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# %%
