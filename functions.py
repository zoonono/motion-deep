import numpy as np
from scipy.interpolate import interp2d

def double(arr):
    x = np.array(list(range(1, arr.shape[1] + 1)))
    y = np.array(list(range(1, arr.shape[0] + 1)))
    i = interp2d(x, y, arr)
    x2 = np.array(list(range(len(x) * 2, len(x) * len(x) * 2 + 1,  
            len(x) - 1))) / (len(x) * 2)
    y2 = np.array(list(range(len(y) * 2, len(y) * len(y) * 2 + 1,  
            len(y) - 1))) / (len(y) * 2)
    return i(x2, y2)