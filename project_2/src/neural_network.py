import numpy as np
import math
from data_loader import load_to_np
from visualise import plotpoints

def apply_rotate(input, angle):

    input = np.transpose(input)

    R = np.array([[math.cos(angle),-math.sin(angle),0],
                  [math.sin(angle),math.cos(angle),0],
                  [0,0,1]])

    output = np.transpose(R @ input)

    return output

def apply_bias(input, b):
    
    
    input[:,1] += b
   
    output = input

    return output

def apply_activation(input, activation='abs'): 
    if activation == 'abs':
        input[:,1] = np.abs(input[:,1])
        output = input
    else:
        print('invalid activation function')
    
    return output

test_data = load_to_np('Libian_desert_data.csv')



plotpoints(apply_activation(apply_bias(test_data,100)),(1200,800),(0,0))