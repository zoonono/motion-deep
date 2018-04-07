import numpy as np
from scipy.interpolate import interp2d
import torch
from torch.autograd import Variable

def interp_double(arr, kind = 'linear'):
    x = np.array(list(range(1, arr.shape[1] + 1)))
    y = np.array(list(range(1, arr.shape[0] + 1)))
    i = interp2d(x, y, arr, kind = kind)
    x2 = double_length(len(x))
    y2 = double_length(len(y))
    return i(x2, y2)
    
def zeroorder_double(arr, kind = None):
    arr2 = np.zeros(np.array(arr.shape) * 2)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr2[2*i:2*i+2, 2*j:2*j+2] = arr[i,j]
    return arr2
    
def double_length(x):
    return np.array(list(range(x * 2 - 1, x * (x * 2 - 1) + 1,  
            x - 1))) / (x * 2 - 1)
            
def double_weight(arr, kind = 'linear'):
    nrr = np.zeros(np.array(arr.shape) * np.array((1,1,2,2)) 
                   + np.array((0,0,1,1)))
    if kind == 'zero':
        double_func = zeroorder_double
    else:
        double_func = interp_double
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            nrr[i][j] = zero_pad(double_func(arr[i][j], kind))
    return nrr
    
def d_out(d_in, padding, kernel, stride):
    return 1 + int((d_in + 2 * padding - kernel)/stride)

def zero_pad(arr, n = 1):
    return np.pad(arr, ((0, n),), 'constant')
    
def compute_loss(dataset, criterion):
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example['image'], example['label']
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0) # add batch dim
        output = net(image)
        avg += (criterion(output, label).data[0] - avg) / (i + 1)
    return avg