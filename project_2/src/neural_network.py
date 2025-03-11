import numpy as np
import math
from typing import Optional
from data_loader import *
from visualise import *

def apply_rotate(input_array: np.ndarray, angle: float) -> np.ndarray:

    rotator = np.array([[math.cos(angle),-math.sin(angle),0],
                  [math.sin(angle),math.cos(angle),0],
                  [0,0,1]])
    
    return np.transpose(rotator @ np.transpose(input_array))

def apply_bias(input_array: np.ndarray, b: float, columns = None) -> np.ndarray:
    if columns is None:
        input_array += b
    else:
        input_array[:, columns] += b
    return input_array

def apply_activation(input_array: np.ndarray, activation: str = 'abs', columns = None) -> Optional[np.ndarray]:
    if activation == 'abs':
        if columns is None:
            return np.abs(input_array)
        else:
            input_array_copy = input_array.copy()
            for column_index in columns:
                input_array_copy[:, column_index] = np.abs(input_array_copy[:, column_index])
            return input_array_copy
    else:
        print('invalid activation function')
        return None

# Load the data
data_loader = DataLoader()
data = data_loader.get_train_test_val_split('Libian_desert_data.csv')

# Extract the data sets - each contains (X, y) tuple
X_train, y_train = data['train']
X_test, y_test = data['test']
X_val, y_val = data['val']

# Combined datasets
train_data = np.hstack((X_train, y_train))
test_data = np.hstack((X_test, y_test))
val_data = np.hstack((X_val, y_val))

# parameters
angle_to_rotate = math.pi/8
bias = 200
activation_functio = 'abs'

# application
rotated_test_data = apply_rotate(test_data, angle_to_rotate)
biased_test_data = apply_bias(rotated_test_data, bias, columns=[0])
activated_test_data = apply_activation(biased_test_data, activation='abs', columns=[0])

# Plotting
plot_scatter_points(activated_test_data, [0, 0])