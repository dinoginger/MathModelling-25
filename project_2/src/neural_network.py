import numpy as np
import math
from typing import Optional, List, Tuple, Dict
from data_loader import *
from visualise import *

def apply_rotate(input_array: np.ndarray, angle: float, columns = None) -> np.ndarray:

    if columns is None:
        input_array_copy = input_array.copy()
        rotator = np.array([[math.cos(angle),-math.sin(angle),0],
                  [math.sin(angle),math.cos(angle),0],
                  [0,0,1]])
        
    else:
        input_array_copy = input_array[:, columns].copy()
        rotator = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
    
    return np.transpose(rotator @ np.transpose(input_array_copy))

def apply_bias(input_array: np.ndarray, b: float, columns = None) -> np.ndarray:
    if columns is None:
        input_array += b
    else:
        input_array[:, columns] += b
    return input_array

def apply_activation(input_array: np.ndarray, activation: str = 'abs', columns=None) -> Optional[np.ndarray]:
    def _abs_func(x):
        return np.abs(x)
    def _relu_func(x):
        return np.maximum(0, x)
    def _sigmoid_func(x):
        return 1 / (1 + np.exp(-x))

    if activation == 'abs':
        act_func = _abs_func
    elif activation == 'relu':
        act_func = _relu_func
    elif activation == 'sigmoid':
        act_func = _sigmoid_func
    else:
        print('Invalid activation function')
        return None

    if columns is None:
        return act_func(input_array)
    else:
        result = input_array.copy()
        for c in columns:
            result[:, c] = act_func(result[:, c])
        return result

data_loader = DataLoader()
data = data_loader.get_train_test_val_split('Libian_desert_data.csv', scale_features=True, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)

X_train, y_train = data['train']
X_test, y_test = data['test'] 
X_val, y_val = data['val']

train_data = np.hstack((X_train, y_train))

rotated_features = apply_rotate(X_train, angle=math.pi/4, columns=[0,1])
rotated_train_data = np.column_stack((rotated_features, y_train))
plot_scatter_points(rotated_train_data, [0, 0])