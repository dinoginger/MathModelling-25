import numpy as np
import os

os.chdir(r'C:\Users\askha\Documents\Math Mod\MathModelling-25\project_2\data')

def load_to_np(input_data):
    data = np.genfromtxt(input_data,delimiter=',')
    return data