import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from neural_network import apply_rotate, apply_bias, apply_activation
from visualise import plot_scatter_points
from sklearn.metrics import accuracy_score

class Layer:
    """
    A single layer in the neural network that applies geometric transformations
    """
    def __init__(self, rotation_angle=0.0, bias=0.0, activation='abs', columns=None):
        self.rotation_angle = rotation_angle
        self.bias = bias
        self.activation = activation
        self.columns = columns  # Which columns to apply transformations to
    
    def forward(self, input_data):
        # Apply the three transformations in sequence
        output = apply_rotate(input_data, self.rotation_angle, self.columns)
        output = apply_bias(output, self.bias, self.columns)
        output = apply_activation(output, self.activation, self.columns)
        return output

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

def main():
    """Main function to run the experiment"""
    
    # Step 1: Load and visualize data
    print("Loading data...")
    data_loader = DataLoader()
    data = data_loader.get_train_test_val_split(
        'Libian_desert_data.csv', 
        scale_features=True,
        train_ratio=0.9,
        test_ratio=0.1,
        val_ratio=0.0,
    )
    
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    
    #print("Visualizing original data...")
    train_data = np.hstack((X_train, y_train))
    #plot_scatter_points(train_data, [0, 0])
    
    # Step 2: Build an 8-layer network
    print("Building neural network with 8 layers...")
    model = NeuralNetwork()
    
    # Add 8 layers with carefully chosen parameters
    # These parameters can be tuned for optimal performance
    model.add_layer(Layer(rotation_angle=np.pi/2, bias=0, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/8, bias=-0.2, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/6, bias=0.1, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/3, bias=-0.1, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/12, bias=0.2, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/5, bias=-0.3, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=np.pi/10, bias=0.15, activation='abs', columns=[0, 1]))
    model.add_layer(Layer(rotation_angle=0, bias=-0.4, activation='none', columns=[0, 1]))
    
    # Step 3: Evaluate the model
    print("Evaluating model...")
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()