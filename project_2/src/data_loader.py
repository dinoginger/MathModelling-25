import numpy as np
import os

os.chdir(r'/Users/linusjuni/Documents/General Engineering/6. Semester/Mathematical Modelling/Assignments/MathModelling-25/project_2/data')

def load_to_np(input_data):
    data = np.genfromtxt(input_data,delimiter=',')
    return data