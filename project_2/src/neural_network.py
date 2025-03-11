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
    def _no_activation(x):
        return x

    if activation == 'abs':
        act_func = _abs_func
    elif activation == 'relu':
        act_func = _relu_func
    elif activation == 'sigmoid':
        act_func = _sigmoid_func
    elif activation == 'none':
        act_func = _no_activation
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