import numpy as np
from matplotlib import pyplot as plt
from data_loader import *

def plot_scatter_points(data, offset, title=None):
    # Extract coordinates and classes
    x_coords = data[:, 0] + offset[1]
    y_coords = data[:, 1] + offset[0]
    classes = data[:, 2]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot class 0 points (blue)
    mask_class0 = (classes == 0)
    plt.scatter(x_coords[mask_class0], y_coords[mask_class0], 
                c='blue', s=10, label='Class 0', alpha=0.1)
    
    # Plot class 1 points (green)
    mask_class1 = (classes == 1)
    plt.scatter(x_coords[mask_class1], y_coords[mask_class1], 
                c='green', s=10, label='Class 1', alpha=0.1)
    
    plt.axvline(x=0, color='red', linestyle='-', linewidth=4)
    plt.axhline(y=0, color='red', linestyle='-', linewidth=4)
    
    # Title stuff
    if title is not None:
        plt.title(title)
    else:
        plt.title("Neural Network Data Visualization")
    plt.legend()
    
    # Set fixed axis limits instead of equal aspect ratio
    plt.xlim(-600, 600)
    plt.ylim(-600, 600)
    
    plt.grid(True, alpha=0.3)
    plt.show()