<<<<<<< Updated upstream
=======
import numpy as np
import math
from matplotlib import pyplot as plt
from neural_network import apply_activation, apply_bias, apply_rotate
from visualise import plotpoints
from data_loader import load_to_np

test_data = load_to_np('Libian_desert_data.csv')

def layers(data, no_layers: int, operations: np.array):

    for i in range(no_layers):
        rotated_data= apply_rotate(data,operations[i,1])
        biased_data = apply_bias(rotated_data, operations[i,0], columns=[0])
        data = apply_activation(biased_data, activation='abs', columns=[0])

        plt.figure()

        plotpoints(data, canvas_size=(800, 800))

        plt.show(block=False)
        
    plt.show()

    


#bias = np.random.randint(400, size=(5))
#rotation = np.random.randint(math.pi, size=(5))
#op = np.column_stack((bias,rotation))

op = np.array([[300,np.pi/2],
              [300,0],
              [100,0],
              [100,0],
              [100,0]])

layers(test_data,1,op)


    

>>>>>>> Stashed changes
