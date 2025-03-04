import numpy as np
from matplotlib import pyplot as plt
from data_loader import load_to_np

def plotpoints(data, canvas_size, transpose, colors=None):

    canvas = np.zeros((canvas_size[0], canvas_size[1], 3))
    # Define default colors if not provided
    if colors is None:
        blue = np.array([0, 0, 150])
        green = np.array([0, 150, 0])
        colors = {0: blue, 1: green}

    for i in range(len(data)):
        x = int(data[i, 0])
        y = int(data[i, 1])
        d = int(data[i, 2]) # Class label

        # Check if x and y are within canvas bounds after transpose
        if x + transpose[1] >= 0 and x + transpose[1] < canvas_size[1]:
            if y + transpose[0] >= 0 and y + transpose[0] < canvas_size[0]:
                c = canvas[y + transpose[0], x + transpose[1]]

                if d in colors: # Use colors from the colors dictionary
                    point_color = colors[d]
                    if np.all(c == 0): # Only color if the pixel is currently empty
                        canvas[y + transpose[0], x + transpose[1]] = point_color
                # If class label not in colors, you could add a default color here if needed

    plt.imshow(canvas, interpolation='nearest')
    plt.show()