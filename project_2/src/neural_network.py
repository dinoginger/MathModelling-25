import numpy as np
import math
from data_loader import load_to_np

def apply_rotate(input, angle):

    input = np.transpose(input)

    R = np.array([[math.cos(angle),-math.sin(angle),0],
                  [math.sin(angle),math.cos(angle),0],
                  [0,0,1]])

    output = np.transpose(R @ input)

    return output

def apply_bias(input, b):
    
    input = np.transpose(input)
    input[:,0:1] += b
    input = np.transpose(input)
    output = input

    return output

def apply_activation(input, activation='abs'): 
    if activation == 'abs':
        output = np.abs(input)
    else:
        print('invalid activation function')
    
    return output

test_data = load_to_np('Libian_desert_data.csv')

print(test_data.shape)
print(apply_bias(test_data,np.random.randint(100,size=(240000))))


#print(test_data)

#print(apply_rotate(test_data,math.pi))
#print(apply_activation(test_data))
