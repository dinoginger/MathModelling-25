<<<<<<< Updated upstream
=======
import numpy as np
from matplotlib import pyplot as plt
from data_loader import load_to_np

def plotpoints(data,canvas_size):

    canvas = np.zeros((canvas_size[0],canvas_size[1],3))
    blue = np.array([0,0,150])
    green = np.array([0,150,0])
    yellow = np.array([100,100,0])
    red = np.array([150,0,0])

    transpose = np.array([0,0])
    transpose[0] = int((canvas_size[0] - int(np.average(data[:,0])))/2)
    transpose[1] = int((canvas_size[1] - int(np.average(data[:,1])))/2)

    print(transpose)

    for i in range(len(data)):
        x = int(data[i,0])
        y = int(data[i,1])

        if x + transpose[0] >= 0 and x + transpose[0] < canvas_size[0]:
            if y + transpose[1] >= 0 and y + transpose[1] < canvas_size[1]:
                c = canvas[x + transpose[0],y + transpose[1]]
                d = data[i,2]
                if np.all(c == 0) and d == 0:
                    canvas[x + transpose[0],y + transpose[1]] = blue
                if np.all(c == 0) and d == 1:
                    canvas[x + transpose[0],y + transpose[1]] = green
                if np.all(c == green) and d == 0:
                    canvas[x + transpose[0],y + transpose[1]] = red
                if np.all(c == blue) and d == 1:
                    canvas[x + transpose[0],y + transpose[1]] = red

    
    
    canvas[transpose[0],:] = yellow

    plt.imshow(canvas,interpolation='nearest',extent=[-transpose[1],canvas_size[1]-transpose[1],canvas_size[0]-transpose[0],-transpose[0]])
    #plt.show()


test_data = load_to_np('Libian_desert_data.csv')
>>>>>>> Stashed changes
