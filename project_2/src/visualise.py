import numpy as np
from matplotlib import pyplot as plt
from data_loader import load_to_np

def plotpoints(data,canvas_size,transpose):

    canvas = np.zeros((canvas_size[0],canvas_size[1],3))
    blue = np.array([0,0,150])
    green = np.array([0,150,0])

    for i in range(len(data)):
        x = int(data[i,0])
        y = int(data[i,1])
        
        if x + transpose[1] >= 0 and x + transpose[1] < canvas_size[1]:
            if y + transpose[0] >= 0 and y + transpose[0] < canvas_size[0]:
                c = canvas[y + transpose[0],x + transpose[1]]
                d = data[i,2]
                if np.all(c == 0) and d == 0:
                    canvas[y + transpose[0],x + transpose[1]] = blue
                if np.all(c == 0) and d == 1:
                    canvas[y + transpose[0],x + transpose[1]] = green
                


    plt.imshow(canvas,interpolation='nearest')
    plt.show()


test_data = load_to_np('Libian_desert_data.csv')
