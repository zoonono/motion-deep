import numpy as np
from scipy.interpolate import interp2d

def interp_double(arr):
    x = np.array(list(range(1, arr.shape[1] + 1)))
    y = np.array(list(range(1, arr.shape[0] + 1)))
    i = interp2d(x, y, arr)
    x2 = double_length(len(x))
    y2 = double_length(len(y))
    return i(x2, y2)
    
def double_length(x):
    return np.array(list(range(x * 2 - 1, x * (x * 2 - 1) + 1,  
            x - 1))) / (x * 2 - 1)
            
def dbl(arr):
    nrr = np.zeros(np.array(arr.shape) * np.array((1,1,2,2)))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nrr[i][j] = interp_double(arr[i][j])
    return nrr