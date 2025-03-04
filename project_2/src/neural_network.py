import numpy as np
import math
from typing import Optional
from data_loader import load_to_np
from visualise import plotpoints

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

# data
test_data = load_to_np('Libian_desert_data.csv')

# parameters
angle_to_rotate = 0
bias = 0
activation_functio = 'abs'

# application
rotated_data = apply_rotate(test_data, angle_to_rotate)
biased_data = apply_bias(rotated_data, bias, columns=[0])
activated_data = apply_activation(biased_data, activation='abs', columns=[0])

plotpoints(activated_data, canvas_size=(1200, 1200), transpose=(10, 10))